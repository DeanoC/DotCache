"""Sweep native Blackwell mixed-value split-K target without reloading model.

This is a profiling/planning harness, not a paper result generator. It keeps
the benchmark inputs fixed and varies only DOTCACHE_NATIVE_MIXEDV_BLOCKS_PER_SPLIT.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _provenance import (
    add_paper_cache_args,
    cache_config_dict,
    configure_paper_runtime_defaults,
)
from compare_decode_speed import measure_certified_decode, measure_dense_decode
from pg19_perplexity import load_pg19_chunks


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--eval-start", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--measure-steps", type=int, default=64)
    parser.add_argument(
        "--targets",
        default="32,48,64,96,128,160,192,256,384",
        help="Comma-separated DOTCACHE_NATIVE_MIXEDV_BLOCKS_PER_SPLIT values.",
    )
    parser.add_argument("--top-k-fp16", type=int, default=4)
    parser.add_argument("--tau-cov", type=float, default=0.995)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=128)
    parser.add_argument("--ranking-fallback", action="store_true")
    parser.add_argument("--ranking-r", type=int, default=1)
    parser.add_argument("--ranking-fallback-mode", default="full", choices=["full", "measure"])
    parser.add_argument("--eps-guard", type=float, default=0.01)
    parser.add_argument("--exploration-rate", type=float, default=0.02)
    parser.add_argument("--rung1-threshold", type=float, default=0.02)
    parser.add_argument("--rung1-multiplier", type=float, default=2.0)
    parser.add_argument("--phase-profile", action="store_true")
    parser.add_argument("--native-profile", action="store_true")
    parser.add_argument("--output", default="runs/native_splitk_sweep.json")
    add_paper_cache_args(parser)
    args = parser.parse_args()
    configure_paper_runtime_defaults()
    if args.native_profile:
        os.environ["DOTCACHE_NATIVE_PROFILE"] = "1"

    targets = [int(x.strip()) for x in args.targets.split(",") if x.strip()]
    prefix_len = int(args.context * args.eval_start)
    required = prefix_len + args.warmup_steps + args.measure_steps + 1
    if required > args.context:
        raise SystemExit(
            f"context too short for prefix+warmup+measure+target: need {required}, have {args.context}"
        )

    token = os.environ.get("HF_TOKEN") or None
    warnings.filterwarnings(
        "ignore",
        message=r"MatMul8bitLt: inputs will be cast from .* during quantization",
        category=UserWarning,
    )
    print(f"Loading {args.model} (INT8)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        dtype=torch.float16,
        token=token,
    )
    model.eval()

    from dotcache.config import DotCacheConfig
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    adapter = LlamaDotCacheModelAdapter(model, DotCacheConfig(head_dim=head_dim))

    chunks, book_indices = load_pg19_chunks(tokenizer, args.context, args.chunk_index + 1)
    input_ids = chunks[args.chunk_index].unsqueeze(0).to("cuda")
    print(
        f"Sweep context={args.context} prefix={prefix_len} warmup={args.warmup_steps} "
        f"measured={args.measure_steps} targets={targets}",
        flush=True,
    )

    dense = measure_dense_decode(
        model,
        input_ids,
        prefix_len=prefix_len,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )
    gc.collect()
    torch.cuda.empty_cache()
    _sync()

    results = []
    for target in targets:
        os.environ["DOTCACHE_NATIVE_MIXEDV_BLOCKS_PER_SPLIT"] = str(target)
        run_args = SimpleNamespace(**vars(args))
        certified = measure_certified_decode(
            model,
            adapter,
            input_ids,
            run_args,
            prefix_len=prefix_len,
            warmup_steps=args.warmup_steps,
            measure_steps=args.measure_steps,
        )
        result = {
            "target_blocks_per_split": int(target),
            "dense_tok_s": float(dense["decode_tok_s"]),
            "certified_tok_s": float(certified["decode_tok_s"]),
            "slowdown": float(dense["decode_tok_s"] / max(certified["decode_tok_s"], 1e-9)),
            "certified_vs_dense_ratio": float(certified["decode_tok_s"] / max(dense["decode_tok_s"], 1e-9)),
            "certified_ms_per_token": float(certified["decode_ms_per_token"]),
            "ppl_ratio": float(certified["perplexity"] / max(dense["perplexity"], 1e-9)),
            "certified": certified,
        }
        results.append(result)
        print(json.dumps({
            "target": target,
            "certified_tok_s": result["certified_tok_s"],
            "slowdown": result["slowdown"],
            "ppl_ratio": result["ppl_ratio"],
        }), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        _sync()

    best = max(results, key=lambda item: item["certified_tok_s"]) if results else None
    output = {
        "benchmark": "native_splitk_sweep",
        "model": args.model,
        "context_length": args.context,
        "chunk_index": args.chunk_index,
        "book_idx": book_indices[args.chunk_index] if book_indices else None,
        "prefix_len": prefix_len,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "cache_config": cache_config_dict(args),
        "dense": dense,
        "results": results,
        "best": best,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps({"best": best, "json": str(out_path)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
