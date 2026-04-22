"""Micro-bench isolating `selective_attend_multihead_hybrid` at 64K scale.

Lets us iterate on the kernel without the 3-min model reload. Builds
synthetic inputs shaped like Llama-3.1-8B (32 Q heads, 8 KV heads, head_dim
128, block_size 16) and the given context length, times the kernel
standalone, and compares against an SDPA dense-FP16 reference for
correctness and a per-layer throughput floor.

Usage:
  .venv/bin/python benchmarks/bench_hybrid_attend_kernel.py --n-tokens 65536
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F


def build_inputs(
    n_tokens: int,
    num_q_heads: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    block_size: int = 16,
    topk: int = 4,
    device: str = "cuda",
    seed: int = 0,
):
    g = torch.Generator(device=device).manual_seed(seed)
    num_blocks = n_tokens // block_size
    N = num_blocks * block_size

    keys_fp16 = torch.randn(num_kv_heads, N, head_dim, dtype=torch.float16, device=device, generator=g)
    values_fp16 = torch.randn(num_kv_heads, N, head_dim, dtype=torch.float16, device=device, generator=g)

    # Per-block INT8 symmetric quantisation of keys, channelwise scale.
    keys_f32 = keys_fp16.to(torch.float32).reshape(num_kv_heads, num_blocks, block_size, head_dim)
    # channel scale per (kv_head, block, channel) — matches the kernel layout.
    ch_max = keys_f32.abs().amax(dim=2).clamp(min=1e-8)  # [kv, blk, ch]
    keys_scale = ch_max / 127.0
    keys_int8 = (
        keys_f32 / keys_scale.unsqueeze(2)
    ).round().clamp(-127, 127).to(torch.int8).reshape(num_kv_heads, N, head_dim).contiguous()

    # Query — one vector per Q head.
    q_all = torch.randn(num_q_heads, head_dim, dtype=torch.float32, device=device, generator=g)

    # Top-K mask: mark `topk` random blocks per Q head as FP16.
    topk_mask = torch.zeros(num_q_heads, num_blocks, dtype=torch.int32, device=device)
    for h in range(num_q_heads):
        idx = torch.randperm(num_blocks, generator=g, device=device)[:topk]
        topk_mask[h, idx] = 1

    # Skip mask: zero everywhere (no skipping, full attend).
    skip_mask = torch.zeros(num_q_heads, num_blocks, dtype=torch.int32, device=device)

    return dict(
        keys_int8=keys_int8,
        keys_scale=keys_scale.to(torch.float32).contiguous(),  # [kv, blk, ch] float32
        keys_fp16=keys_fp16,
        values_fp16=values_fp16,
        q_all=q_all,
        skip_mask_i32=skip_mask,
        topk_mask=topk_mask,
        num_blocks=num_blocks,
        N=N,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
    )


def time_fn(fn, *, warmup: int = 10, iters: int = 50):
    # Warmup.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Timed.
    times_ms = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def sdpa_reference(inp, gqa_group: int, q_scale: float):
    """Dense FP16 attention via torch SDPA — the 'free' baseline."""
    # q_all: [num_q, head_dim] float32 → broadcast to [num_kv, gqa_group, head_dim]
    num_q = inp["num_q_heads"]
    num_kv = inp["num_kv_heads"]
    d = inp["head_dim"]
    N = inp["N"]
    q = inp["q_all"].reshape(num_kv, gqa_group, d).to(torch.float16)  # [kv, g, d]
    # SDPA wants [batch, heads, seq, d]. Treat kv as batch × g as heads.
    q_sdpa = q.reshape(1, num_q, 1, d)   # [1, H_q, 1, d]
    # Keys/values: expand kv to q heads via repeat_interleave along the head dim.
    k = inp["keys_fp16"].repeat_interleave(gqa_group, dim=0).unsqueeze(0)  # [1, H_q, N, d]
    v = inp["values_fp16"].repeat_interleave(gqa_group, dim=0).unsqueeze(0)
    scale = q_scale
    out = F.scaled_dot_product_attention(q_sdpa, k, v, scale=scale)  # [1, H_q, 1, d]
    return out.reshape(num_q, d).to(torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=65536)
    ap.add_argument("--num-q-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.init()

    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"Building inputs: n_tokens={args.n_tokens} num_q={args.num_q_heads} "
          f"num_kv={args.num_kv_heads} head_dim={args.head_dim} block={args.block_size} topk={args.topk}")

    inp = build_inputs(
        n_tokens=args.n_tokens,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        topk=args.topk,
    )

    gqa_group = args.num_q_heads // args.num_kv_heads
    q_scale = 1.0 / (args.head_dim ** 0.5)

    from dotcache.kernels.selective_attend_triton import (
        selective_attend_multihead_hybrid,
        selective_attend_multihead_hybrid_split_k,
    )

    def call_hybrid():
        return selective_attend_multihead_hybrid(
            keys_int8=inp["keys_int8"],
            keys_scale=inp["keys_scale"],
            keys_fp16=inp["keys_fp16"],
            topk_mask=inp["topk_mask"],
            values_fp16=inp["values_fp16"],
            q_all=inp["q_all"],
            skip_mask_i32=inp["skip_mask_i32"],
            gqa_group=gqa_group,
            block_size=args.block_size,
            q_scale=q_scale,
        )

    def call_split_k(num_splits=None):
        return selective_attend_multihead_hybrid_split_k(
            keys_int8=inp["keys_int8"],
            keys_scale=inp["keys_scale"],
            keys_fp16=inp["keys_fp16"],
            topk_mask=inp["topk_mask"],
            values_fp16=inp["values_fp16"],
            q_all=inp["q_all"],
            skip_mask_i32=inp["skip_mask_i32"],
            gqa_group=gqa_group,
            block_size=args.block_size,
            q_scale=q_scale,
            num_splits=num_splits,
        )

    # Correctness smoke check — kernel should produce finite output of right shape.
    out = call_hybrid()
    assert out.shape == (args.num_q_heads, args.head_dim), out.shape
    assert torch.isfinite(out).all(), "non-finite kernel output"

    # Compare against dense FP16 SDPA to sanity check ordering / scale.
    ref = sdpa_reference(inp, gqa_group, q_scale)
    assert ref.shape == out.shape
    # These won't be bitwise equal (int8 dequant on non-top-K blocks), but should
    # be correlated: top-K-FP16 matches dense exactly; rest differ by int8 error.
    err = (out - ref).abs()
    rel_err = err / (ref.abs().clamp(min=1e-6))
    print(f"  kernel vs SDPA dense: mean |Δ|={err.mean():.4f}  p50 |Δ|={err.median():.4f}  "
          f"p95 |Δ|={err.flatten().sort().values[int(0.95*err.numel())]:.4f}")
    print(f"  kernel range: [{out.min():.3f}, {out.max():.3f}]   SDPA range: [{ref.min():.3f}, {ref.max():.3f}]")

    # Time hybrid kernel.
    ms = time_fn(call_hybrid, warmup=args.warmup, iters=args.iters)
    print(f"\n=== Hybrid kernel (1 launch per layer) ===")
    print(f"  n_tokens={inp['N']}  num_blocks={inp['num_blocks']}  grid=(num_q_heads,)={args.num_q_heads}")
    print(f"  per-launch: mean {statistics.mean(ms):.3f} ms  p50 {statistics.median(ms):.3f} ms  "
          f"p95 {sorted(ms)[int(0.95*len(ms))]:.3f} ms")
    print(f"  per 32 layers (one decode step): {statistics.mean(ms)*32:.2f} ms → "
          f"~{1000/(statistics.mean(ms)*32):.2f} tok/s attend-only")

    # Time SDPA reference (dense FP16, should be ~optimal floor).
    def call_sdpa():
        return sdpa_reference(inp, gqa_group, q_scale)
    ms_sdpa = time_fn(call_sdpa, warmup=args.warmup, iters=args.iters)
    print(f"\n=== SDPA dense FP16 (reference floor) ===")
    print(f"  per-launch: mean {statistics.mean(ms_sdpa):.3f} ms  p50 {statistics.median(ms_sdpa):.3f} ms")
    print(f"  per 32 layers: {statistics.mean(ms_sdpa)*32:.2f} ms → "
          f"~{1000/(statistics.mean(ms_sdpa)*32):.2f} tok/s")

    # Correctness: split-K vs original hybrid.
    out_split = call_split_k()
    assert out_split.shape == out.shape
    err_split = (out_split - out).abs()
    rel_err_split = err_split / (out.abs().clamp(min=1e-6))
    print(f"\n=== Split-K vs original hybrid (correctness) ===")
    print(f"  mean |Δ|={err_split.mean():.6f}  p50 |Δ|={err_split.median():.6f}  "
          f"max |Δ|={err_split.max():.6f}")
    print(f"  mean rel={rel_err_split.mean():.6f}  max rel={rel_err_split.max():.6f}")

    # Sweep num_splits to find the sweet spot.
    print(f"\n=== Split-K sweep ===")
    best_ms = None; best_ns = None
    for ns in [1, 2, 4, 8, 16, 32, 64]:
        if ns > inp["num_blocks"]:
            continue
        ms_ns = time_fn(lambda ns=ns: call_split_k(ns), warmup=args.warmup, iters=args.iters)
        mean_ms = statistics.mean(ms_ns)
        programs = args.num_q_heads * ns
        blocks_per = (inp["num_blocks"] + ns - 1) // ns
        tag = ""
        if best_ms is None or mean_ms < best_ms:
            best_ms = mean_ms; best_ns = ns; tag = " ←best"
        print(f"  ns={ns:>3}  programs={programs:>5}  blocks/split={blocks_per:>4}  "
              f"mean {mean_ms:>6.3f} ms  p50 {statistics.median(ms_ns):>6.3f} ms{tag}")

    print(f"\n=== Gap summary ===")
    gap_orig = statistics.mean(ms) / statistics.mean(ms_sdpa)
    gap_best = best_ms / statistics.mean(ms_sdpa)
    speedup = statistics.mean(ms) / best_ms
    print(f"  original hybrid: {statistics.mean(ms):.3f} ms  ({gap_orig:.1f}× slower than SDPA)")
    print(f"  split-K best   : {best_ms:.3f} ms (ns={best_ns})  ({gap_best:.1f}× slower than SDPA)")
    print(f"  SDPA reference : {statistics.mean(ms_sdpa):.3f} ms")
    print(f"  split-K speedup vs original: {speedup:.2f}×")
    print(f"  projected per-step attend time: {statistics.mean(ms)*32:.1f} ms → {best_ms*32:.1f} ms")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
