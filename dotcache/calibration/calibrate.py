"""Main calibration entry point. Runs Phases 1-3 sequentially.

Usage:
    python -m dotcache.calibration.calibrate --model NousResearch/Meta-Llama-3.1-8B
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from dotcache.calibration.calibrated_profile import CalibratedProfile
from dotcache.calibration.execution_floor import (
    diagnose_floor_failures,
    measure_execution_floor,
)
from dotcache.calibration.prompt_sets import (
    build_calibration_prompts,
    build_validation_prompts,
)
from dotcache.calibration.sequential_sweep import sequential_epsilon_calibration
from dotcache.calibration.validate import validate_profile


DEFAULT_CTX_BUCKETS = [4096, 8192, 16384, 32768]
DEFAULT_TARGET_COS = 0.999


def run_calibration(
    model,
    tokenizer,
    *,
    model_name: str = "",
    ctx_buckets: list[int] = None,
    target_cos: float = DEFAULT_TARGET_COS,
    prompts_per_bucket: int = 4,
    validation_prompts_per_bucket: int = 2,
    output_dir: str = "configs",
    device: str = "cuda",
) -> CalibratedProfile:
    """Run full 3-phase calibration."""
    if ctx_buckets is None:
        ctx_buckets = DEFAULT_CTX_BUCKETS

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    print(f"{'='*60}")
    print(f"Calibrating: {model_name}")
    print(f"  Layers: {num_layers}, Heads: {num_heads}")
    print(f"  Context buckets: {[c//1024 for c in ctx_buckets]}K")
    print(f"  Target cos: {target_cos}")
    print(f"{'='*60}")

    # Build prompt sets
    cal_prompts = build_calibration_prompts(ctx_buckets, prompts_per_bucket)
    val_prompts = build_validation_prompts(ctx_buckets, validation_prompts_per_bucket)
    print(f"Calibration prompts: {len(cal_prompts)}")
    print(f"Validation prompts: {len(val_prompts)}")

    # ── Phase 1: Execution Floor ──
    print(f"\n{'='*60}")
    print("PHASE 1: Execution Floor Measurement")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    floor = measure_execution_floor(model, tokenizer, cal_prompts, ctx_buckets, device)
    t_phase1 = time.perf_counter() - t0
    print(f"Phase 1 done in {t_phase1:.1f}s")

    # Report floor failures
    failures = diagnose_floor_failures(floor, ctx_buckets, target_cos)
    if failures:
        print(f"\n  Floor failures ({len(failures)} heads below target):")
        for f in failures[:10]:
            print(f"    {f['ctx_bucket']//1024}K L{f['layer']} H{f['head']}: "
                  f"floor={f['floor_cos']:.6f}")
        if len(failures) > 10:
            print(f"    ... and {len(failures)-10} more")
    else:
        print(f"\n  All heads above target floor ({target_cos})")

    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2: Sequential Epsilon Calibration ──
    print(f"\n{'='*60}")
    print("PHASE 2: Sequential Epsilon Calibration")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    epsilon = sequential_epsilon_calibration(
        model, tokenizer, cal_prompts, ctx_buckets, floor,
        target_cos=target_cos, device=device,
    )
    t_phase2 = time.perf_counter() - t0
    print(f"Phase 2 done in {t_phase2:.1f}s")

    gc.collect()
    torch.cuda.empty_cache()

    # Determine V-format per layer (placeholder: all fp16 for now)
    v_format = ["fp16"] * num_layers

    # Build profile
    profile = CalibratedProfile(
        epsilon=epsilon,
        execution_floor=floor,
        v_format=v_format,
        ctx_buckets=ctx_buckets,
        model_name=model_name,
        calibration_date=datetime.now().isoformat(),
        num_prompts=len(cal_prompts),
        target_cos=target_cos,
    )

    # ── Phase 3: Validation ──
    print(f"\n{'='*60}")
    print("PHASE 3: End-to-End Validation")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    val_result = validate_profile(model, tokenizer, profile, val_prompts, device)
    t_phase3 = time.perf_counter() - t0
    print(f"Phase 3 done in {t_phase3:.1f}s")

    print(f"\n  Passed: {val_result['passed']}")
    print(f"  Global cos_min: {val_result['global_cos_min']:.6f}")
    print(f"  Margin: {val_result['margin']:.6f}")
    print(f"  Violations: {val_result['num_violations']}")
    for ctx, stats in val_result["per_bucket"].items():
        print(f"  {ctx//1024}K: cos_min={stats['cos_min']:.6f}, skip={stats['skip_rate']:.1%}")

    # If validation fails, tighten failing epsilons
    if not val_result["passed"]:
        print(f"\n  Tightening {len(val_result['violations'])} failing heads...")
        for v in val_result["violations"]:
            bi = profile._find_bucket(v["bucket"])
            lid = v["layer"]
            hid = v["head"]
            old_eps = profile.epsilon[bi, lid, hid]
            profile.epsilon[bi, lid, hid] = old_eps / 10  # tighten by 10x
            if profile.epsilon[bi, lid, hid] < 1e-8:
                profile.epsilon[bi, lid, hid] = 0.0  # disable skip

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    npz_path = out_path / f"{safe_name}_calibrated.npz"
    profile.save(str(npz_path))
    print(f"\nProfile saved: {npz_path}")
    print(f"Total time: {t_phase1+t_phase2+t_phase3:.1f}s")
    print(f"\n{profile.summary()}")

    return profile


def main():
    parser = argparse.ArgumentParser(description="Calibrate epsilon profile")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--target-cos", type=float, default=DEFAULT_TARGET_COS)
    parser.add_argument("--ctx-buckets", type=int, nargs="+", default=DEFAULT_CTX_BUCKETS)
    parser.add_argument("--prompts-per-bucket", type=int, default=4)
    parser.add_argument("--output-dir", default="configs")
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), token=token,
    ).to("cuda")
    model.eval()
    print(f"Model: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    run_calibration(
        model, tokenizer,
        model_name=args.model,
        ctx_buckets=args.ctx_buckets,
        target_cos=args.target_cos,
        prompts_per_bucket=args.prompts_per_bucket,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
