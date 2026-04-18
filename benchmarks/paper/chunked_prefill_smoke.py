"""Validate chunked prefill vs one-shot prefill on a 4K prompt.

Expectation: chunked prefill is bit-identical or within documented BF16 drift
to one-shot. Any significant divergence invalidates using chunked prefill for
paper-grade runs.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotcache.config import DotCacheConfig  # noqa: E402
from dotcache.integrations.llama import LlamaDotCacheModelAdapter, _prefill_prompt  # noqa: E402


FILLER = (
    "The history of mathematics spans thousands of years. Ancient Babylonians "
    "developed a base-60 number system that still influences time measurement. "
    "The Greeks founded geometry, number theory, and logic. Islamic Golden Age "
    "scholars extended algebra and trigonometry. The Renaissance saw European "
    "mathematical activity blossom, leading to calculus by Newton and Leibniz. "
    "In the nineteenth and twentieth centuries, mathematics became increasingly "
    "abstract and specialized. Today, mathematics is essential to every field.\n\n"
)


def build_prompt(tokenizer, target_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    per_block = len(tokenizer.encode(FILLER, add_special_tokens=False))
    num_blocks = max(target_tokens // per_block, 2)
    text = FILLER * num_blocks + "What was the primary mathematical development in the Renaissance?\nAnswer:"
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=target_tokens)
    return enc["input_ids"], enc["attention_mask"]


def kv_max_diff(one_shot_layers, chunked_layers) -> tuple[float, float]:
    """Return (max_k_diff, max_v_diff) across all layers."""
    max_k = 0.0
    max_v = 0.0
    for (k1, v1), (k2, v2) in zip(one_shot_layers, chunked_layers):
        t1k = k1 if torch.is_tensor(k1) else torch.from_numpy(k1)
        t2k = k2 if torch.is_tensor(k2) else torch.from_numpy(k2)
        t1v = v1 if torch.is_tensor(v1) else torch.from_numpy(v1)
        t2v = v2 if torch.is_tensor(v2) else torch.from_numpy(v2)
        max_k = max(max_k, (t1k.float() - t2k.float()).abs().max().item())
        max_v = max(max_v, (t1v.float() - t2v.float()).abs().max().item())
    return max_k, max_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--target-tokens", type=int, default=4096)
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[512, 1024, 2048])
    ap.add_argument("--output", default="benchmarks/results/chunked_prefill_smoke.json")
    args = ap.parse_args()

    print(f"Loading {args.model} (INT8)...")
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cfg = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, cfg)

    input_ids, attention_mask = build_prompt(tokenizer, args.target_tokens)
    input_ids = input_ids.to("cuda:0")
    attention_mask = attention_mask.to("cuda:0")
    actual_len = int(input_ids.shape[1])
    print(f"Prompt length: {actual_len} tokens")

    # One-shot (reference)
    print("=== ONE-SHOT ===")
    torch.cuda.reset_peak_memory_stats()
    gc.collect(); torch.cuda.empty_cache()
    t0 = time.perf_counter()
    _, one_shot_layers, one_shot_first_tok = _prefill_prompt(model, adapter, input_ids, attention_mask)
    one_shot_ms = (time.perf_counter() - t0) * 1000.0
    one_shot_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"  ms={one_shot_ms:.1f} peak_mb={one_shot_peak_mb:.1f} first_tok={int(one_shot_first_tok.item())}")

    results = {
        "model": args.model,
        "seq_len": actual_len,
        "one_shot": {
            "ms": one_shot_ms,
            "peak_mb": one_shot_peak_mb,
            "first_token": int(one_shot_first_tok.item()),
        },
        "chunked": {},
    }

    for cs in args.chunk_sizes:
        print(f"=== CHUNK={cs} ===")
        torch.cuda.reset_peak_memory_stats()
        gc.collect(); torch.cuda.empty_cache()
        t0 = time.perf_counter()
        _, chunked_layers, chunked_first_tok = _prefill_prompt(
            model, adapter, input_ids, attention_mask, chunk_size=cs
        )
        chunked_ms = (time.perf_counter() - t0) * 1000.0
        chunked_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        first_tok = int(chunked_first_tok.item())
        max_k, max_v = kv_max_diff(one_shot_layers, chunked_layers)
        match = first_tok == int(one_shot_first_tok.item())
        print(f"  ms={chunked_ms:.1f} peak_mb={chunked_peak_mb:.1f} first_tok={first_tok} match={match}")
        print(f"  max_k_abs_diff={max_k:.6f} max_v_abs_diff={max_v:.6f}")
        results["chunked"][str(cs)] = {
            "ms": chunked_ms,
            "peak_mb": chunked_peak_mb,
            "first_token": first_tok,
            "first_token_match_one_shot": match,
            "max_k_abs_diff": max_k,
            "max_v_abs_diff": max_v,
            "peak_reduction_vs_one_shot_mb": one_shot_peak_mb - chunked_peak_mb,
            "peak_reduction_pct": (one_shot_peak_mb - chunked_peak_mb) / one_shot_peak_mb * 100.0,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nJSON -> {out_path}")

    print("\n## Summary")
    print(f"{'chunk':>6} | {'ms':>7} | {'peak_mb':>8} | {'Δpeak':>7} | first_tok match | max_k_diff | max_v_diff")
    print("-" * 92)
    print(f"{'(none)':>6} | {one_shot_ms:>7.1f} | {one_shot_peak_mb:>8.1f} | {0.0:>7.1f} | {'-':>14} | {'-':>10} | {'-':>10}")
    for cs in args.chunk_sizes:
        r = results["chunked"][str(cs)]
        print(f"{cs:>6} | {r['ms']:>7.1f} | {r['peak_mb']:>8.1f} | "
              f"{-r['peak_reduction_vs_one_shot_mb']:>+7.1f} | "
              f"{'YES' if r['first_token_match_one_shot'] else 'NO':>14} | "
              f"{r['max_k_abs_diff']:>10.6f} | {r['max_v_abs_diff']:>10.6f}")


if __name__ == "__main__":
    main()
