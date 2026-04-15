"""Needle-in-a-Haystack (NIAH) benchmark.

Plants a fact at varying (depth, context_length) positions, generates text,
checks retrieval accuracy. Produces the classic depth×length heatmap.

Run for both dense (FP16 attention, INT8 weights) and certified (full pipeline).
Any cell where dense succeeds but certified fails = critical bug.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Needles: distinct facts with unique answers
NEEDLES = [
    ("The secret project codename is Crimson Falcon.", "Crimson Falcon"),
    ("The emergency access code is 7429-DELTA.", "7429-DELTA"),
    ("The primary backup server is located in Zurich.", "Zurich"),
    ("The annual budget for the initiative is $4.7 million.", "4.7 million"),
    ("The lead researcher's employee ID is RX-9031.", "RX-9031"),
]

# Filler: repeated innocuous text (avoids any content that could match needles)
FILLER_BLOCK = (
    "The history of mathematics spans thousands of years and encompasses many "
    "different cultures and civilizations. From the earliest counting systems "
    "developed by ancient peoples to the sophisticated abstract algebras of the "
    "modern era, mathematical knowledge has grown through a process of discovery, "
    "invention, and refinement. The Babylonians developed a base-60 number system "
    "that still influences how we measure time and angles today. The ancient Greeks "
    "made foundational contributions to geometry, number theory, and logic. During "
    "the Islamic Golden Age, scholars preserved and extended Greek mathematics while "
    "making original advances in algebra and trigonometry. The Renaissance saw a "
    "flowering of mathematical activity in Europe, leading eventually to the "
    "development of calculus by Newton and Leibniz in the seventeenth century. "
    "In the nineteenth and twentieth centuries, mathematics became increasingly "
    "abstract and specialized, with new fields emerging at a rapid pace. Today, "
    "mathematics is essential to science, engineering, economics, and many other "
    "domains of human activity.\n\n"
)


def build_niah_prompt(needle_text: str, context_tokens: int, depth_fraction: float,
                      tokenizer) -> str:
    """Build a NIAH prompt with needle at specified depth.

    Args:
        needle_text: the fact to plant
        context_tokens: total context length in tokens
        depth_fraction: 0.0=start, 0.5=middle, 0.9=near end
        tokenizer: HF tokenizer for length estimation

    Returns:
        prompt string with needle embedded at depth
    """
    # Build filler to approximate target length
    # Estimate tokens per filler block
    filler_tokens = len(tokenizer.encode(FILLER_BLOCK, add_special_tokens=False))
    needle_tokens = len(tokenizer.encode(needle_text, add_special_tokens=False))

    # Reserve tokens for needle + question + some margin
    question = "\n\nBased on the above text, what is the answer to: What is the secret, code, location, budget, or ID mentioned in the special statement?\nAnswer:"
    question_tokens = len(tokenizer.encode(question, add_special_tokens=False))
    available = context_tokens - needle_tokens - question_tokens - 50  # margin

    num_blocks = max(available // filler_tokens, 2)
    needle_position = max(0, int(num_blocks * depth_fraction))

    parts = []
    for i in range(num_blocks):
        if i == needle_position:
            parts.append(f"\n[IMPORTANT] {needle_text}\n\n")
        parts.append(FILLER_BLOCK)

    prompt = "".join(parts) + question
    return prompt


def check_retrieval(generated_text: str, expected_answer: str) -> bool:
    """Check if the generated text contains the expected answer."""
    return expected_answer.lower() in generated_text.lower()


def run_niah_cell(
    model, tokenizer, adapter, mode: str,
    context_tokens: int, depth: float, needle_idx: int,
    calibrated_profile=None,
    max_new_tokens: int = 50,
    device: str = "cuda",
) -> dict:
    """Run one NIAH cell: plant needle, generate, check retrieval.

    Args:
        mode: 'dense' or 'certified'
    """
    needle_text, expected_answer = NEEDLES[needle_idx]
    prompt = build_niah_prompt(needle_text, context_tokens, depth, tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=context_tokens).to(device)
    seq_len = inputs["input_ids"].shape[1]

    if mode == "dense":
        # Dense: standard HF generation
        adapter.set_mode("dense")
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        generated_ids = outputs[0, seq_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    elif mode == "certified":
        # Certified: prefill dense, then decode certified
        from dotcache.integrations.llama import _ensure_certified_imports, CertifiedAttentionState
        from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model

        adapter.set_mode("dense")
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
        first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del outputs

        # Build tiered cache
        _ensure_certified_imports()
        layer_ids = list(range(model.config.num_hidden_layers))
        tiered_caches = create_tiered_cache_from_model(past_kv, layer_ids)
        del past_kv
        gc.collect()
        torch.cuda.empty_cache()

        # Get layer epsilons from calibrated profile or default
        if calibrated_profile is not None:
            layer_epsilons = calibrated_profile.get_layer_epsilons_min(seq_len)
        else:
            layer_epsilons = {}

        adapter.certified_state = CertifiedAttentionState(
            tiered_caches=tiered_caches,
            layer_epsilons=layer_epsilons,
            default_epsilon=1e-4,
            collect_stats=False,
            append_kv=True,
        )
        adapter.set_mode("certified")

        # Generate tokens
        cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
        current_input = first_token
        gen_ids = []

        for _ in range(max_new_tokens):
            with torch.inference_mode():
                out = model(
                    input_ids=current_input, use_cache=False,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                )
            tid = out.logits[:, -1, :].argmax(dim=-1).item()
            gen_ids.append(tid)
            if tid == tokenizer.eos_token_id:
                break
            current_input = torch.tensor([[tid]], dtype=torch.long, device=device)
            cache_position = cache_position + 1

        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        adapter.certified_state = None
        adapter.set_mode("dense")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    correct = check_retrieval(generated_text, expected_answer)

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "context_tokens": seq_len,
        "target_context": context_tokens,
        "depth": depth,
        "needle_idx": needle_idx,
        "mode": mode,
        "correct": correct,
        "expected": expected_answer,
        "generated": generated_text[:200],
    }


def run_niah_sweep(
    model, tokenizer, adapter,
    context_lengths: list[int],
    depths: list[float] = None,
    num_needles: int = 5,
    calibrated_profile=None,
    device: str = "cuda",
) -> dict:
    """Run full NIAH sweep across depths and context lengths."""
    if depths is None:
        depths = [i / 10 for i in range(10)]  # 0.0, 0.1, ..., 0.9

    results = {"dense": [], "certified": []}
    total = len(context_lengths) * len(depths) * num_needles * 2
    done = 0

    for ctx_len in context_lengths:
        for depth in depths:
            for needle_idx in range(num_needles):
                for mode in ["dense", "certified"]:
                    done += 1
                    r = run_niah_cell(
                        model, tokenizer, adapter, mode,
                        ctx_len, depth, needle_idx,
                        calibrated_profile=calibrated_profile,
                        device=device,
                    )
                    results[mode].append(r)

                    status = "OK" if r["correct"] else "FAIL"
                    print(f"  [{done}/{total}] {mode:>10} {ctx_len//1024}K d={depth:.1f} "
                          f"n={needle_idx} -> {status}")

                    if not r["correct"] and mode == "certified":
                        # Check if dense also failed
                        dense_r = results["dense"][-1] if results["dense"] else None
                        if dense_r and dense_r["correct"]:
                            print(f"    *** CRITICAL: dense OK but certified FAILED ***")

    # Compute accuracy heatmaps
    heatmaps = {}
    for mode in ["dense", "certified"]:
        heatmap = {}
        for r in results[mode]:
            key = (r["target_context"], r["depth"])
            if key not in heatmap:
                heatmap[key] = {"correct": 0, "total": 0}
            heatmap[key]["total"] += 1
            if r["correct"]:
                heatmap[key]["correct"] += 1
        heatmaps[mode] = {
            f"{k[0]//1024}K_d{k[1]:.1f}": v["correct"] / v["total"]
            for k, v in heatmap.items()
        }

    # Summary
    for mode in ["dense", "certified"]:
        total_correct = sum(1 for r in results[mode] if r["correct"])
        total_count = len(results[mode])
        print(f"\n{mode}: {total_correct}/{total_count} correct ({total_correct/total_count:.1%})")

    # Critical failures: certified fails where dense succeeds
    critical = []
    dense_map = {(r["target_context"], r["depth"], r["needle_idx"]): r
                 for r in results["dense"]}
    for r in results["certified"]:
        key = (r["target_context"], r["depth"], r["needle_idx"])
        if key in dense_map and dense_map[key]["correct"] and not r["correct"]:
            critical.append(r)

    if critical:
        print(f"\n*** {len(critical)} CRITICAL FAILURES: dense OK but certified FAILED ***")
        for c in critical[:5]:
            print(f"  {c['target_context']//1024}K d={c['depth']:.1f} n={c['needle_idx']}: "
                  f"expected '{c['expected']}', got '{c['generated'][:50]}'")

    return {
        "results": results,
        "heatmaps": heatmaps,
        "critical_failures": len(critical),
        "dense_accuracy": sum(1 for r in results["dense"] if r["correct"]) / max(len(results["dense"]), 1),
        "certified_accuracy": sum(1 for r in results["certified"] if r["correct"]) / max(len(results["certified"]), 1),
    }


def main():
    import argparse
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter
    from dotcache.config import DotCacheConfig
    from dotcache.calibration.calibrated_profile import CalibratedProfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--needles", type=int, default=3)
    parser.add_argument("--profile", default=None, help="Path to calibrated profile .npz")
    parser.add_argument("--output", default="benchmarks/results/niah.json")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading {args.model} (INT8)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant_config, device_map="auto", token=token,
    )
    model.eval()
    print(f"Model: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    config = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, config)

    profile = None
    if args.profile:
        profile = CalibratedProfile.load(args.profile)
        print(f"Loaded profile: {profile.summary()[:200]}")

    print(f"\nNIAH: contexts={[c//1024 for c in args.contexts]}K, needles={args.needles}")
    result = run_niah_sweep(
        model, tokenizer, adapter,
        context_lengths=args.contexts,
        num_needles=args.needles,
        calibrated_profile=profile,
    )

    print(f"\n{'='*50}")
    print(f"Dense accuracy:     {result['dense_accuracy']:.1%}")
    print(f"Certified accuracy: {result['certified_accuracy']:.1%}")
    print(f"Critical failures:  {result['critical_failures']}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "heatmaps": result["heatmaps"],
            "dense_accuracy": result["dense_accuracy"],
            "certified_accuracy": result["certified_accuracy"],
            "critical_failures": result["critical_failures"],
        }, f, indent=2)
    print(f"JSON -> {out_path}")


if __name__ == "__main__":
    main()
