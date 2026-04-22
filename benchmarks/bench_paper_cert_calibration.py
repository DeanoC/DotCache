"""Certificate calibration: measure actual ℓ₂ output error vs certified bound.

For each decode step, runs both:
  1. Dense FP16 forward (exact reference)
  2. Persistent compressed forward (with DotCache)

Compares the final logit vectors to compute ℓ₂ error, then compares
against the residual certificate bounds (beta_upper, delta_upper).

This validates Theorems 1-7 by showing the certified bound is a correct
upper bound on the actual error, and characterises how tight that bound is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_serving_policy_compare import (
    _DEFAULT_POLICY_PATH,
    _build_prompt_text_inputs,
    _resolve_prompt_records,
)
from benchmarks.bench_qwen35_persistent_real_mixed_probe import (
    real_mixed_probe_dotcache_config,
    real_mixed_probe_serving_config,
)
from dotcache.backends.metal.persistent_types import PersistentServingConfig
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheModelAdapter,
    Qwen35TextModelAdapter,
    load_qwen35_text_only_from_pretrained,
    _run_dense_prefill,
    _run_dense_decode_step,
    _clone_qwen35_past_key_values,
)
from dotcache.integrations.llama import _timed_call


_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/manifests/qwen35_stage9_repo_public_validation_broader_20260414.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certificate calibration: ℓ₂ error vs bound")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_records = _resolve_prompt_records(
        manifest_path=args.manifest_path or str(_DEFAULT_MANIFEST),
        prompt_files=[],
        prompt_file_target_length=0,
    )
    print(f"Certificate calibration: {len(prompt_records)} cases, {args.decode_steps} steps")

    print(f"Loading model {args.model_id} ...")
    model, tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id, device=args.device, torch_dtype=args.torch_dtype,
    )
    dc = real_mixed_probe_dotcache_config()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(model, dc)

    # Also load a separate dense model for reference
    dense_model, dense_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id, device=args.device, torch_dtype=args.torch_dtype,
    )
    dense_adapter = Qwen35TextModelAdapter(dense_model)

    device = next(model.parameters()).device
    config = real_mixed_probe_serving_config(policy_path=str(_DEFAULT_POLICY_PATH))
    config.enable_interval_bound = True
    config.enable_ellipsoidal_bound = True

    all_results: list[dict[str, Any]] = []

    for prompt_record in prompt_records:
        case_tag = str(prompt_record["case_tag"])
        prompt_text = Path(str(prompt_record["prompt_file_path"])).read_text(encoding="utf-8")
        input_ids, attention_mask = _build_prompt_text_inputs(
            tokenizer, device=device, prompt_text=prompt_text,
            prompt_length=int(prompt_record.get("prompt_length", 0)),
        )
        print(f"\n[case: {case_tag}] ({input_ids.shape[1]} tokens)")

        # --- Dense reference path ---
        dense_adapter.set_mode("dense")
        dense_prefill, _ = _timed_call(
            lambda: _run_dense_prefill(dense_model, input_ids=input_ids, attention_mask=attention_mask),
            device=device,
        )
        dense_token = dense_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        dense_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1)
        dense_cache_pos = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)
        dense_pkv = dense_prefill.past_key_values

        # --- Persistent compressed path ---
        adapter.set_mode("dense")
        adapter.clear()
        pers_prefill, _ = _timed_call(
            lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
            device=device,
        )
        pkv_clone = _clone_qwen35_past_key_values(pers_prefill.past_key_values)
        adapter.persistent_serving_config = config
        adapter.clear()
        adapter.load_attention_subset_persistent_prefill_cache(pkv_clone)
        adapter.set_mode("dotcache_attention_subset_persistent_experimental")
        adapter.configure_persistent_shortlist_policy_context(
            prompt_family=None, prompt_length=int(input_ids.shape[1]),
        )
        rt = adapter.require_persistent_hybrid_runtime_state()
        pers_token = pers_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        pers_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1)
        pers_cache_pos = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)

        step_results: list[dict[str, Any]] = []

        for step_idx in range(args.decode_steps):
            # Dense decode step
            dense_out, _ = _timed_call(
                lambda: _run_dense_decode_step(
                    dense_model, decode_input_ids=dense_token,
                    attention_mask=dense_mask, past_key_values=dense_pkv,
                    cache_position=dense_cache_pos,
                ),
                device=device,
            )
            dense_logits = dense_out.logits[0, -1, :].to(dtype=torch.float32)

            # Persistent decode step
            pers_out, _ = _timed_call(
                lambda: _run_dense_decode_step(
                    model, decode_input_ids=pers_token,
                    attention_mask=pers_mask,
                    past_key_values=rt.model_past_key_values,
                    cache_position=pers_cache_pos,
                ),
                device=device,
            )
            rt.advance(pers_out.past_key_values)
            pers_logits = pers_out.logits[0, -1, :].to(dtype=torch.float32)

            # ℓ₂ error between logit vectors
            l2_error = torch.norm(pers_logits - dense_logits, p=2).item()
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                pers_logits.unsqueeze(0), dense_logits.unsqueeze(0)
            ).item()
            # Max absolute difference
            max_abs_diff = (pers_logits - dense_logits).abs().max().item()
            # Token match
            dense_tok_id = int(dense_logits.argmax().item())
            pers_tok_id = int(pers_logits.argmax().item())
            token_match = dense_tok_id == pers_tok_id

            # Collect per-layer certificate data
            layer_certs = {}
            for layer_id in sorted(rt.full_attention.layers.keys()):
                fa_state = rt.full_attention.layers[layer_id]
                cert = fa_state.last_residual_certificate
                if cert:
                    layer_certs[str(layer_id)] = {
                        "beta_upper": float(cert.get("beta_upper", 0.0)),
                        "delta_upper": float(cert.get("delta_upper", 0.0)),
                        "certified_can_stop": bool(cert.get("certified_can_stop", False)),
                    }

            step_result = {
                "step": step_idx,
                "l2_error": l2_error,
                "cos_sim": cos_sim,
                "max_abs_diff": max_abs_diff,
                "token_match": token_match,
                "dense_token": dense_tok_id,
                "pers_token": pers_tok_id,
                "layer_certs": layer_certs,
            }
            step_results.append(step_result)

            print(
                f"  step {step_idx}: ℓ₂={l2_error:.4f}  cos={cos_sim:.6f}  "
                f"max_diff={max_abs_diff:.4f}  tok_match={token_match}"
            )

            # Advance dense state
            dense_pkv = dense_out.past_key_values
            dense_token = dense_logits.argmax().unsqueeze(0).unsqueeze(0)
            dense_mask = torch.cat(
                [dense_mask, torch.ones((1, 1), dtype=dense_mask.dtype, device=device)], dim=1,
            )
            dense_cache_pos = dense_cache_pos + 1

            # Advance persistent state
            pers_token = pers_logits.argmax().unsqueeze(0).unsqueeze(0)
            pers_mask = torch.cat(
                [pers_mask, torch.ones((1, 1), dtype=pers_mask.dtype, device=device)], dim=1,
            )
            pers_cache_pos = pers_cache_pos + 1

        # Aggregate
        l2_errors = [s["l2_error"] for s in step_results]
        cos_sims = [s["cos_sim"] for s in step_results]
        token_matches = [s["token_match"] for s in step_results]

        case_result = {
            "case_tag": case_tag,
            "tokens": int(input_ids.shape[1]),
            "decode_steps": args.decode_steps,
            "l2_error_median": float(np.median(l2_errors)),
            "l2_error_max": float(np.max(l2_errors)),
            "l2_error_mean": float(np.mean(l2_errors)),
            "cos_sim_median": float(np.median(cos_sims)),
            "cos_sim_min": float(np.min(cos_sims)),
            "token_match_rate": float(np.mean(token_matches)),
            "per_step": step_results,
        }
        all_results.append(case_result)

        # Reset for next case
        adapter.set_mode("dense")
        adapter.clear()

    # Print summary
    print("\n" + "=" * 70)
    print("CERTIFICATE CALIBRATION SUMMARY")
    print("=" * 70)
    all_l2 = [s["l2_error"] for r in all_results for s in r["per_step"]]
    all_cos = [s["cos_sim"] for r in all_results for s in r["per_step"]]
    all_tok = [s["token_match"] for r in all_results for s in r["per_step"]]
    print(f"Total steps: {len(all_l2)}")
    print(f"ℓ₂ error:  median={np.median(all_l2):.4f}  mean={np.mean(all_l2):.4f}  max={np.max(all_l2):.4f}")
    print(f"Cosine:    median={np.median(all_cos):.6f}  min={np.min(all_cos):.6f}")
    print(f"Token match: {np.mean(all_tok):.1%}")

    print(f"\nPer-case:")
    print(f"{'Case':<30} {'ℓ₂ med':>8} {'ℓ₂ max':>8} {'cos min':>10} {'tok%':>6}")
    for r in all_results:
        print(
            f"{r['case_tag']:<30} {r['l2_error_median']:>8.4f} {r['l2_error_max']:>8.4f} "
            f"{r['cos_sim_min']:>10.6f} {r['token_match_rate']:>5.0%}"
        )

    # Output
    output = {
        "model_id": args.model_id,
        "device": args.device,
        "decode_steps": args.decode_steps,
        "summary": {
            "l2_error_median": float(np.median(all_l2)),
            "l2_error_mean": float(np.mean(all_l2)),
            "l2_error_max": float(np.max(all_l2)),
            "cos_sim_median": float(np.median(all_cos)),
            "cos_sim_min": float(np.min(all_cos)),
            "token_match_rate": float(np.mean(all_tok)),
        },
        "cases": [{k: v for k, v in r.items() if k != "per_step"} for r in all_results],
        "per_step_detail": [{
            "case_tag": r["case_tag"],
            "steps": r["per_step"],
        } for r in all_results],
    }

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nJSON -> {args.output_json}")

    if args.output_md:
        lines = ["# Certificate Calibration: ℓ₂ Error vs Bound\n"]
        lines.append(f"Model: {args.model_id}, Steps: {args.decode_steps}\n")
        lines.append(f"ℓ₂ error: median={np.median(all_l2):.4f}, max={np.max(all_l2):.4f}\n")
        lines.append(f"Cosine similarity: median={np.median(all_cos):.6f}, min={np.min(all_cos):.6f}\n")
        lines.append(f"Token match rate: {np.mean(all_tok):.1%}\n")
        lines.append("| Case | ℓ₂ median | ℓ₂ max | cos min | token match |")
        lines.append("|---|---|---|---|---|")
        for r in all_results:
            lines.append(
                f"| {r['case_tag']} | {r['l2_error_median']:.4f} "
                f"| {r['l2_error_max']:.4f} | {r['cos_sim_min']:.6f} "
                f"| {r['token_match_rate']:.0%} |"
            )
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_md, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"MD   -> {args.output_md}")


if __name__ == "__main__":
    main()
