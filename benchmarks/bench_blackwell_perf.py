"""Blackwell perf gate for native certified kernels.

This is intentionally synthetic: it isolates the mixed INT4/FP16 value
attention kernel that dominates long-context certified decode, avoiding model
reload time while preserving Llama-3.1-8B tensor shapes by default.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


def _time_ms(fn: Callable[[], torch.Tensor], *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(float(start.elapsed_time(end)))
    return out


def _build_mixed_inputs(
    *,
    n_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    group_size: int,
    topk: int,
    fallback_blocks: int,
    device: str,
    seed: int,
) -> dict:
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    gen = torch.Generator(device=device).manual_seed(seed)
    num_blocks = n_tokens // block_size
    n_tokens = num_blocks * block_size
    keys = torch.randn(num_kv_heads, n_tokens, head_dim, dtype=torch.float16, device=device, generator=gen)
    values = torch.randn(num_kv_heads, n_tokens, head_dim, dtype=torch.float16, device=device, generator=gen)
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys, values, block_size=block_size, group_size=group_size, max_new_tokens=0,
    )
    q = torch.randn(num_q_heads, head_dim, dtype=torch.float32, device=device, generator=gen)
    topk_mask = torch.zeros(num_q_heads, num_blocks, dtype=torch.int32, device=device)
    for h in range(num_q_heads):
        topk_mask[h, torch.randperm(num_blocks, generator=gen, device=device)[:topk]] = 1
    skip = torch.zeros_like(topk_mask)

    value_mask = torch.zeros_like(topk_mask)
    fallback = min(int(fallback_blocks), num_blocks)
    for h in range(num_q_heads):
        value_mask[h, torch.randperm(num_blocks, generator=gen, device=device)[:fallback]] = 1
    fallback_union = value_mask.any(dim=0).nonzero().flatten().tolist()
    slots = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
    scratch = torch.empty(
        num_kv_heads,
        max(len(fallback_union), 1) * block_size,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    for slot, bid in enumerate(fallback_union):
        slots[bid] = slot
        scratch[:, slot * block_size:(slot + 1) * block_size, :] = values[:, bid * block_size:(bid + 1) * block_size, :]

    return {
        "keys_int8": cache.keys_int8[:, :n_tokens, :],
        "keys_scale": cache.keys_scale[:, :num_blocks, :],
        "keys_zero_points": cache.keys_zero_points[:, :num_blocks, :],
        "keys_fp16": cache.keys_fp16_gpu[:, :n_tokens, :],
        "topk_mask": topk_mask,
        "values_int4_packed": cache.values_int4_packed[:, :n_tokens, :],
        "values_int4_scales": cache.values_int4_scales[:, :n_tokens, :],
        "values_int4_zeros": cache.values_int4_zeros[:, :n_tokens, :],
        "values_fp16_scratch": scratch,
        "value_fp16_mask": value_mask,
        "value_block_slots": slots,
        "q_all": q,
        "skip_mask_i32": skip,
        "gqa_group": num_q_heads // num_kv_heads,
        "block_size": block_size,
        "group_size": group_size,
        "q_scale": 1.0 / (head_dim ** 0.5),
        "num_blocks": num_blocks,
        "n_tokens": n_tokens,
        "fallback_union_blocks": len(fallback_union),
    }


def _summarize(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "mean_ms": float(statistics.mean(times)),
        "p50_ms": float(statistics.median(times)),
        "p95_ms": float(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768, 65536])
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--fallback-blocks", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    from dotcache.backends.certified_blackwell import hybrid_mixedv_split_k_cuda
    from dotcache.kernels.selective_attend_triton import selective_attend_multihead_hybrid_mixedv_split_k

    rows = []
    for ctx in args.contexts:
        inp = _build_mixed_inputs(
            n_tokens=ctx,
            num_q_heads=args.num_q_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            group_size=args.group_size,
            topk=args.topk,
            fallback_blocks=args.fallback_blocks,
            device="cuda",
            seed=ctx,
        )
        call_kwargs = {k: v for k, v in inp.items() if k not in {"num_blocks", "n_tokens", "fallback_union_blocks"}}

        def triton_call() -> torch.Tensor:
            return selective_attend_multihead_hybrid_mixedv_split_k(**call_kwargs)

        def native_call() -> torch.Tensor:
            return hybrid_mixedv_split_k_cuda(**call_kwargs)

        triton_out = triton_call()
        native_out = native_call()
        torch.cuda.synchronize()
        max_abs = float((triton_out - native_out).abs().max().item())
        mean_abs = float((triton_out - native_out).abs().mean().item())
        triton_ms = _time_ms(triton_call, warmup=args.warmup, iters=args.iters)
        native_ms = _time_ms(native_call, warmup=args.warmup, iters=args.iters)
        triton_summary = _summarize(triton_ms)
        native_summary = _summarize(native_ms)
        speedup = triton_summary["mean_ms"] / native_summary["mean_ms"]
        row = {
            "context": ctx,
            "num_blocks": inp["num_blocks"],
            "fallback_union_blocks": inp["fallback_union_blocks"],
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "triton": triton_summary,
            "native": native_summary,
            "native_speedup": float(speedup),
        }
        rows.append(row)
        print(
            f"{ctx//1024:>4}K blocks={inp['num_blocks']:<5} "
            f"triton={triton_summary['mean_ms']:.3f}ms "
            f"native={native_summary['mean_ms']:.3f}ms "
            f"speedup={speedup:.2f}x max_abs={max_abs:.2e}"
        )

    result = {
        "hardware": torch.cuda.get_device_name(0),
        "cuda_capability": torch.cuda.get_device_capability(0),
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
