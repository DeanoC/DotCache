"""Blackwell score/certify perf gate.

Isolates phase 1 of certified attention:
  INT8 asymmetric key scoring -> per-block m_b/S_b -> skip mask.

This is the first tensor-core backend target because its outputs are small and
can be compared directly against the current Triton implementation before any
attention/perplexity path is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Callable

import torch


def _time_ms(fn: Callable[[], object], *, warmup: int, iters: int) -> list[float]:
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


def _summarize(times: list[float]) -> dict[str, float]:
    ordered = sorted(times)
    return {
        "mean_ms": float(statistics.mean(times)),
        "p50_ms": float(statistics.median(times)),
        "p95_ms": float(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]),
    }


def _build_inputs(
    *,
    n_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    device: str,
    seed: int,
) -> dict:
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    gen = torch.Generator(device=device).manual_seed(seed)
    num_blocks = n_tokens // block_size
    n_tokens = num_blocks * block_size
    keys = torch.randn(num_kv_heads, n_tokens, head_dim, dtype=torch.float16, device=device, generator=gen)
    values = torch.randn(num_kv_heads, n_tokens, head_dim, dtype=torch.float16, device=device, generator=gen)
    cache = TieredKeyCacheLayer.from_fp16_cache(
        keys, values, block_size=block_size, max_new_tokens=0,
    )
    q = torch.randn(num_q_heads, head_dim, dtype=torch.float32, device=device, generator=gen)
    return {
        "K_int8_packed": cache.keys_int8[:, :n_tokens, :],
        "K_scale": cache.keys_scale[:, :num_blocks, :],
        "K_zero_points": cache.keys_zero_points[:, :num_blocks, :],
        "q_all": q,
        "correction": cache.correction[:, :num_blocks],
        "gqa_group": num_q_heads // num_kv_heads,
        "block_size": block_size,
        "q_scale": 1.0 / (head_dim ** 0.5),
        "block_epsilon": 0.0,
        "num_blocks": num_blocks,
        "n_tokens": n_tokens,
    }


def _score_call(inp: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from dotcache.kernels.fused_score_certify import fused_score_certify_multihead

    return fused_score_certify_multihead(
        K_int8_packed=inp["K_int8_packed"],
        K_scale=inp["K_scale"],
        K_zero_points=inp["K_zero_points"],
        q_all=inp["q_all"],
        correction=inp["correction"],
        gqa_group=inp["gqa_group"],
        block_size=inp["block_size"],
        q_scale=inp["q_scale"],
        block_epsilon=inp["block_epsilon"],
    )


def _dense_gemm_bound_prepare(inp: dict) -> dict:
    """Prepare an optimistic tensor-core bound for phase-1 scoring.

    This intentionally dequantizes keys to FP16 once outside the timed region,
    then uses batched matmul for the per-token scores. It is not a drop-in
    exact scorer; it answers whether the paper shape is tensor-core-friendly
    if dequant/layout costs were solved by a real fused CUTLASS kernel.
    """
    kv_heads = int(inp["K_int8_packed"].shape[0])
    n_tokens = int(inp["K_int8_packed"].shape[1])
    head_dim = int(inp["K_int8_packed"].shape[2])
    num_blocks = int(inp["num_blocks"])
    block_size = int(inp["block_size"])
    gqa_group = int(inp["gqa_group"])
    q_heads = int(inp["q_all"].shape[0])
    keys_i8 = inp["K_int8_packed"].to(torch.float32).reshape(
        kv_heads, num_blocks, block_size, head_dim,
    )
    deq = (
        keys_i8 * inp["K_scale"].unsqueeze(2)
        + inp["K_zero_points"].unsqueeze(2)
    ).reshape(kv_heads, n_tokens, head_dim).to(torch.float16)
    q_grouped = inp["q_all"].reshape(kv_heads, gqa_group, head_dim).to(torch.float16)
    return {
        "keys_t": deq.transpose(1, 2).contiguous(),  # [kv, hd, tokens]
        "q_grouped": q_grouped,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "gqa_group": gqa_group,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "q_scale": float(inp["q_scale"]),
        "correction": inp["correction"],
        "block_epsilon": float(inp["block_epsilon"]),
    }


def _native_dequant_gemm_prepare(inp: dict) -> dict:
    kv_heads = int(inp["K_int8_packed"].shape[0])
    head_dim = int(inp["K_int8_packed"].shape[2])
    gqa_group = int(inp["gqa_group"])
    return {
        "K_int8_packed": inp["K_int8_packed"],
        "K_scale": inp["K_scale"],
        "K_zero_points": inp["K_zero_points"],
        "q_grouped": inp["q_all"].reshape(kv_heads, gqa_group, head_dim).to(torch.float16),
        "q_heads": int(inp["q_all"].shape[0]),
        "kv_heads": kv_heads,
        "gqa_group": gqa_group,
        "num_blocks": int(inp["num_blocks"]),
        "block_size": int(inp["block_size"]),
        "q_scale": float(inp["q_scale"]),
        "correction": inp["correction"],
        "block_epsilon": float(inp["block_epsilon"]),
    }


def _native_dequant_keys_t(bound: dict) -> torch.Tensor:
    from dotcache.backends.cutlass_sm120 import dequant_keys_to_fp16_t

    return dequant_keys_to_fp16_t(
        bound["K_int8_packed"],
        bound["K_scale"],
        bound["K_zero_points"],
        block_size=bound["block_size"],
    )


def _dense_gemm_from_keys_t(bound: dict, keys_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.bmm(bound["q_grouped"], keys_t).float() * bound["q_scale"]
    scores = scores.reshape(
        bound["kv_heads"], bound["gqa_group"], bound["num_blocks"], bound["block_size"],
    ).reshape(bound["q_heads"], bound["num_blocks"], bound["block_size"])
    m_b = scores.amax(dim=2)
    s_b = torch.exp(scores - m_b.unsqueeze(2)).sum(dim=2)
    if bound["block_epsilon"] <= 0.0:
        skip = torch.zeros_like(m_b, dtype=torch.bool)
    else:
        correction = bound["correction"].repeat_interleave(bound["gqa_group"], dim=0)
        m_global = m_b.amax(dim=1, keepdim=True)
        mass = s_b * correction * torch.exp(m_b - m_global)
        total = mass.sum(dim=1, keepdim=True).clamp(min=1e-30)
        skip = (mass / total) < bound["block_epsilon"]
    return m_b, s_b, skip


def _native_dequant_gemm_call(bound: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys_t = _native_dequant_keys_t(bound)
    return _dense_gemm_from_keys_t(bound, keys_t)


def _dense_gemm_bound_call(bound: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _dense_gemm_from_keys_t(bound, bound["keys_t"])


def _compare(a: tuple[torch.Tensor, torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, float | int]:
    return {
        "m_b_max_abs": float((a[0] - b[0]).abs().max().item()),
        "s_b_max_abs": float((a[1] - b[1]).abs().max().item()),
        "skip_mismatches": int((a[2] != b[2]).sum().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768, 65536, 131072])
    parser.add_argument("--num-q-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"

    rows = []
    old_backend = os.environ.get("DOTCACHE_SCORE_BACKEND")
    old_enable = os.environ.get("DOTCACHE_CUTLASS_SM120_ENABLE_SCORE")
    try:
        for ctx in args.contexts:
            inp = _build_inputs(
                n_tokens=ctx,
                num_q_heads=args.num_q_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
                device="cuda",
                seed=ctx,
            )

            os.environ["DOTCACHE_SCORE_BACKEND"] = "triton"
            triton_out = _score_call(inp)
            triton_ms = _time_ms(lambda: _score_call(inp), warmup=args.warmup, iters=args.iters)

            dense_bound = _dense_gemm_bound_prepare(inp)
            dense_bound_out = _dense_gemm_bound_call(dense_bound)
            dense_bound_ms = _time_ms(
                lambda: _dense_gemm_bound_call(dense_bound),
                warmup=args.warmup,
                iters=args.iters,
            )
            native_dequant_bound = _native_dequant_gemm_prepare(inp)
            native_dequant_out = _native_dequant_gemm_call(native_dequant_bound)
            native_dequant_ms = _time_ms(
                lambda: (_native_dequant_keys_t(native_dequant_bound),)[0],
                warmup=args.warmup,
                iters=args.iters,
            )
            native_dequant_gemm_ms = _time_ms(
                lambda: _native_dequant_gemm_call(native_dequant_bound),
                warmup=args.warmup,
                iters=args.iters,
            )

            os.environ["DOTCACHE_SCORE_BACKEND"] = "cutlass_sm120"
            os.environ.setdefault("DOTCACHE_CUTLASS_SM120_ENABLE_SCORE", "0")
            cutlass_out = _score_call(inp)
            cutlass_ms = _time_ms(lambda: _score_call(inp), warmup=args.warmup, iters=args.iters)

            triton_summary = _summarize(triton_ms)
            cutlass_summary = _summarize(cutlass_ms)
            dense_bound_summary = _summarize(dense_bound_ms)
            native_dequant_summary = _summarize(native_dequant_ms)
            native_dequant_gemm_summary = _summarize(native_dequant_gemm_ms)
            speedup = triton_summary["mean_ms"] / max(cutlass_summary["mean_ms"], 1e-9)
            dense_bound_speedup = triton_summary["mean_ms"] / max(dense_bound_summary["mean_ms"], 1e-9)
            native_dequant_gemm_speedup = triton_summary["mean_ms"] / max(native_dequant_gemm_summary["mean_ms"], 1e-9)
            row = {
                "context": ctx,
                "num_blocks": inp["num_blocks"],
                "triton": triton_summary,
                "predequant_fp16_gemm_bound": dense_bound_summary,
                "predequant_fp16_gemm_bound_speedup": float(dense_bound_speedup),
                "predequant_fp16_gemm_bound_comparison": _compare(dense_bound_out, triton_out),
                "native_dequant_layout": native_dequant_summary,
                "native_dequant_fp16_gemm": native_dequant_gemm_summary,
                "native_dequant_fp16_gemm_speedup": float(native_dequant_gemm_speedup),
                "native_dequant_fp16_gemm_comparison": _compare(native_dequant_out, triton_out),
                "cutlass_backend": cutlass_summary,
                "cutlass_speedup": float(speedup),
                "comparison": _compare(cutlass_out, triton_out),
                "cutlass_enabled": os.environ.get("DOTCACHE_CUTLASS_SM120_ENABLE_SCORE", "0"),
            }
            rows.append(row)
            print(
                f"{ctx//1024:>4}K blocks={inp['num_blocks']:<5} "
                f"triton={triton_summary['mean_ms']:.3f}ms "
                f"fp16_gemm_bound={dense_bound_summary['mean_ms']:.3f}ms "
                f"dequant={native_dequant_summary['mean_ms']:.3f}ms "
                f"dequant+gemm={native_dequant_gemm_summary['mean_ms']:.3f}ms "
                f"cutlass_backend={cutlass_summary['mean_ms']:.3f}ms "
                f"speedup={speedup:.2f}x bound={dense_bound_speedup:.2f}x "
                f"deq_gemm={native_dequant_gemm_speedup:.2f}x "
                f"enabled={row['cutlass_enabled']}"
            )
    finally:
        if old_backend is None:
            os.environ.pop("DOTCACHE_SCORE_BACKEND", None)
        else:
            os.environ["DOTCACHE_SCORE_BACKEND"] = old_backend
        if old_enable is None:
            os.environ.pop("DOTCACHE_CUTLASS_SM120_ENABLE_SCORE", None)
        else:
            os.environ["DOTCACHE_CUTLASS_SM120_ENABLE_SCORE"] = old_enable

    result = {
        "hardware": torch.cuda.get_device_name(0),
        "cuda_capability": list(torch.cuda.get_device_capability(0)),
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
