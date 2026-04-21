"""Phase 2: Triton selective attention — single-head and multi-head versions.

Multi-head version processes all Q heads in one kernel launch, with each
program handling one Q head. Skip mask is per-head.

Two variants:
  - Float32 keys (original): takes pre-dequantised K
  - INT8 keys (new): takes K_int8 + per-block K_scale, dequantises in-register
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _multihead_selective_attend_kernel(
    # Data — KV heads packed
    K_ptr,           # [num_kv_heads * N, head_dim] float32 contiguous
    V_ptr,           # [num_kv_heads * N, d_v] float32 contiguous
    Q_ptr,           # [num_q_heads, head_dim] float32
    Skip_ptr,        # [num_q_heads, num_blocks] int32
    Out_ptr,         # [num_q_heads, d_v] float32
    # Layout
    N: tl.constexpr,
    stride_k: tl.constexpr,     # head_dim
    stride_v: tl.constexpr,     # d_v
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    d_v: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    # Tiles
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """One program per Q head. Iterates blocks, skips flagged ones."""
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        kvh = qh // gqa_group
        kv_base = kvh * N

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        # Online softmax state — FP64 per paper §9.9 (avoids catastrophic
        # cancellation when scale_old is very small). Intermediate compute
        # (scores, exp, weights) stays FP32; only the running state is FP64.
        m = tl.full((), float("-inf"), dtype=tl.float64)
        l = tl.full((), 0.0, dtype=tl.float64)
        acc = tl.zeros((TILE_V,), dtype=tl.float64)

        for bid in range(num_blocks):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:  # attend
                base_tok = kv_base + bid * block_size

                # Score: q · K for block_size tokens
                scores = tl.zeros((block_size,), dtype=tl.float32)
                row_ptrs = K_ptr + (base_tok + t_offs) * stride_k
                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    k_ptrs = row_ptrs[:, None] + d_off[None, :]
                    k_tile = tl.load(k_ptrs, mask=dm[None, :], other=0).to(tl.float32)
                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                # Online softmax update (FP64 state, FP32 intermediates)
                block_max = tl.max(scores)                          # fp32
                new_m = tl.maximum(m, block_max.to(tl.float64))     # fp64
                alpha = tl.exp(m - new_m)                           # fp64
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m.to(tl.float32))     # fp32 intermediate
                l += tl.sum(weights).to(tl.float64)

                # V accumulation
                v_row_ptrs = V_ptr + (base_tok + t_offs) * stride_v
                v_off = v_offs  # assumes TILE_V >= d_v
                vm = v_off < d_v
                v_ptrs = v_row_ptrs[:, None] + v_off[None, :]
                v_tile = tl.load(v_ptrs, mask=vm[None, :], other=0).to(tl.float32)
                w_v = tl.sum(weights[:, None] * v_tile, axis=0)     # fp32
                acc += w_v.to(tl.float64)
                m = new_m

        # Normalise in fp64; cast down at the store.
        safe_l = tl.where(l > 0.0, l, 1.0)
        output = (acc / safe_l).to(tl.float32)
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


# ─── INT8 variant: dequantise keys in-register ────────────────────────

@triton.jit
def _multihead_selective_attend_int8_kernel(
    # Data — INT8 keys + scale
    K_int8_ptr,      # [num_kv_heads * N, head_dim] int8 contiguous
    K_scale_ptr,     # [num_kv_heads, num_blocks, head_dim] float32
    V_ptr,           # [num_kv_heads * N, d_v] float16 contiguous
    Q_ptr,           # [num_q_heads, head_dim] float32
    Skip_ptr,        # [num_q_heads, num_blocks] int32
    Out_ptr,         # [num_q_heads, d_v] float32
    # Layout
    N: tl.constexpr,
    stride_k: tl.constexpr,     # head_dim
    stride_v: tl.constexpr,     # d_v
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    d_v: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    # Tiles
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """One program per Q head. Loads INT8 keys, dequantises in-register."""
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        kvh = qh // gqa_group
        kv_base = kvh * N

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        # Online softmax state — FP64 per paper §9.9.
        m = tl.full((), float("-inf"), dtype=tl.float64)
        l = tl.full((), 0.0, dtype=tl.float64)
        acc = tl.zeros((TILE_V,), dtype=tl.float64)

        for bid in range(num_blocks):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:  # attend this block
                base_tok = kv_base + bid * block_size
                scale_base = (kvh * num_blocks + bid) * head_dim

                # Score: q · (K_int8 * scale) for block_size tokens
                scores = tl.zeros((block_size,), dtype=tl.float32)
                row_ptrs = K_int8_ptr + (base_tok + t_offs) * stride_k
                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                    k_ptrs = row_ptrs[:, None] + d_off[None, :]
                    # Load INT8, cast to float32, multiply by per-channel scale — all in-register
                    k_int8_tile = tl.load(k_ptrs, mask=dm[None, :], other=0)
                    k_tile = k_int8_tile.to(tl.float32) * ch_scale[None, :]
                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                # Online softmax update (FP64 state, FP32 intermediates)
                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max.to(tl.float64))
                alpha = tl.exp(m - new_m)
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m.to(tl.float32))
                l += tl.sum(weights).to(tl.float64)

                # V accumulation (FP16 values → float32 → float64)
                v_row_ptrs = V_ptr + (base_tok + t_offs) * stride_v
                v_off = v_offs
                vm = v_off < d_v
                v_ptrs = v_row_ptrs[:, None] + v_off[None, :]
                v_tile = tl.load(v_ptrs, mask=vm[None, :], other=0).to(tl.float32)
                w_v = tl.sum(weights[:, None] * v_tile, axis=0)
                acc += w_v.to(tl.float64)
                m = new_m

        # Normalise in fp64; cast down at the store.
        safe_l = tl.where(l > 0.0, l, 1.0)
        output = (acc / safe_l).to(tl.float32)
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


def selective_attend_multihead_int8(
    keys_int8: torch.Tensor,      # [num_kv_heads, N, head_dim] int8
    keys_scale: torch.Tensor,     # [num_kv_heads, num_blocks, head_dim] float32
    values_fp16: torch.Tensor,    # [num_kv_heads, N, d_v] float16
    q_all: torch.Tensor,          # [num_q_heads, head_dim] float32
    skip_mask_i32: torch.Tensor,  # [num_q_heads, num_blocks] int32
    gqa_group: int,
    block_size: int = 16,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Multi-head selective attention with INT8 keys, dequantised in-register.

    Eliminates the need to materialise a float32 key tensor in VRAM.
    Keys are loaded as INT8 and dequantised per-block inside the kernel.
    Values are read as FP16 and cast to float32 in-register.

    Returns [num_q_heads, d_v] float32.
    """
    num_kv_heads, N, head_dim = keys_int8.shape
    d_v = values_fp16.shape[2]
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_int8.device

    K_flat = keys_int8.reshape(num_kv_heads * N, head_dim).contiguous()
    V_flat = values_fp16.reshape(num_kv_heads * N, d_v).contiguous()
    output = torch.empty(num_q_heads, d_v, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)

    _multihead_selective_attend_int8_kernel[(num_q_heads,)](
        K_flat, keys_scale.contiguous(), V_flat,
        q_all.contiguous(), skip_mask_i32.contiguous(), output,
        N=N,
        stride_k=head_dim,
        stride_v=d_v,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        d_v=d_v,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_group=gqa_group,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output


# ─── INT8 keys + INT4 per-group values variant ───────────────────────

@triton.jit
def _multihead_selective_attend_int8k_int4v_kernel(
    # INT8 keys
    K_int8_ptr,      # [num_kv_heads * N, head_dim] int8 contiguous
    K_scale_ptr,     # [num_kv_heads, num_blocks, head_dim] float32
    # INT4 packed values + per-group scales/zeros
    V_packed_ptr,    # [num_kv_heads * N, d_v // 2] uint8 contiguous
    V_scales_ptr,    # [num_kv_heads * N, num_groups] float16 contiguous
    V_zeros_ptr,     # [num_kv_heads * N, num_groups] float16 contiguous
    # Query + skip
    Q_ptr,           # [num_q_heads, head_dim] float32
    Skip_ptr,        # [num_q_heads, num_blocks] int32
    Out_ptr,         # [num_q_heads, d_v] float32
    # Layout
    N: tl.constexpr,
    stride_k: tl.constexpr,
    d_v: tl.constexpr,
    d_v_half: tl.constexpr,     # d_v // 2
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """INT8 keys + INT4 per-group values, both dequantised in-register."""
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        kvh = qh // gqa_group
        kv_base = kvh * N

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        # Online softmax state — FP64 per paper §9.9.
        m = tl.full((), float("-inf"), dtype=tl.float64)
        l = tl.full((), 0.0, dtype=tl.float64)
        acc = tl.zeros((TILE_V,), dtype=tl.float64)

        for bid in range(num_blocks):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:
                base_tok = kv_base + bid * block_size

                # ── K scoring (INT8 in-register per-channel dequant) ──
                scale_base = (kvh * num_blocks + bid) * head_dim
                scores = tl.zeros((block_size,), dtype=tl.float32)
                row_ptrs = K_int8_ptr + (base_tok + t_offs) * stride_k
                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)
                    k_ptrs = row_ptrs[:, None] + d_off[None, :]
                    k_int8_tile = tl.load(k_ptrs, mask=dm[None, :], other=0)
                    k_tile = k_int8_tile.to(tl.float32) * ch_scale[None, :]
                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                # ── Online softmax (FP64 state, FP32 intermediates) ──
                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max.to(tl.float64))
                alpha = tl.exp(m - new_m)
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m.to(tl.float32))
                l += tl.sum(weights).to(tl.float64)

                # ── V accumulation (INT4 per-group in-register dequant) ──
                # Load packed uint8 [block_size, d_v//2], unpack to [block_size, d_v]
                # Then apply per-group scale+zero
                # Process in TILE_V chunks over the d_v dimension
                for v_start in range(0, d_v, TILE_V):
                    v_off = v_start + v_offs
                    vm_local = v_off < d_v

                    # Packed index: each byte holds 2 values
                    # Even indices = low nibble, odd indices = high nibble
                    packed_idx = v_off // 2  # [TILE_V]
                    is_high = (v_off % 2)    # 0 for low nibble, 1 for high

                    # Load packed bytes for all tokens in block
                    v_packed_ptrs = V_packed_ptr + (base_tok + t_offs[:, None]) * d_v_half + packed_idx[None, :]
                    packed_bytes = tl.load(v_packed_ptrs, mask=vm_local[None, :], other=0)

                    # Unpack: select low or high nibble
                    low_nibble = packed_bytes & 0x0F
                    high_nibble = (packed_bytes >> 4) & 0x0F
                    unpacked = tl.where(is_high[None, :] == 1, high_nibble, low_nibble)
                    v_int4 = unpacked.to(tl.float32)

                    # Per-group dequant: which group does each v_off belong to?
                    group_idx = v_off // group_size  # [TILE_V]
                    # Load scales and zeros for all tokens × relevant groups
                    scale_ptrs = V_scales_ptr + (base_tok + t_offs[:, None]) * num_groups + group_idx[None, :]
                    zero_ptrs = V_zeros_ptr + (base_tok + t_offs[:, None]) * num_groups + group_idx[None, :]
                    v_scale = tl.load(scale_ptrs, mask=vm_local[None, :], other=0.0).to(tl.float32)
                    v_zero = tl.load(zero_ptrs, mask=vm_local[None, :], other=0.0).to(tl.float32)

                    # Dequantise: val = int4 * scale + zero
                    v_tile = v_int4 * v_scale + v_zero  # [block_size, TILE_V]

                    # Weighted accumulation (cast up to fp64 before add-in)
                    w_v = tl.sum(weights[:, None] * v_tile, axis=0)  # fp32 [TILE_V]
                    w_v64 = w_v.to(tl.float64)

                    # Add to accumulator at the right offset. TILE_V >= d_v in
                    # practice so this is the only iteration; the guard exists
                    # for future tiling.
                    if v_start > 0:
                        acc = tl.where(vm_local, acc + w_v64, acc)
                    else:
                        zeros64 = tl.zeros_like(w_v64)
                        acc = acc + tl.where(vm_local, w_v64, zeros64)

                m = new_m

        safe_l = tl.where(l > 0.0, l, 1.0)
        output = (acc / safe_l).to(tl.float32)
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


def selective_attend_multihead_int8k_int4v(
    keys_int8: torch.Tensor,          # [num_kv_heads, N, head_dim] int8
    keys_scale: torch.Tensor,         # [num_kv_heads, num_blocks, head_dim] float32
    values_int4_packed: torch.Tensor,  # [num_kv_heads, N, d_v//2] uint8
    values_int4_scales: torch.Tensor,  # [num_kv_heads, N, num_groups] float16
    values_int4_zeros: torch.Tensor,   # [num_kv_heads, N, num_groups] float16
    q_all: torch.Tensor,              # [num_q_heads, head_dim] float32
    skip_mask_i32: torch.Tensor,      # [num_q_heads, num_blocks] int32
    gqa_group: int,
    block_size: int = 16,
    group_size: int = 32,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """INT8 keys + INT4 per-group values, both dequantised in-register.

    Keys: INT8 per-block symmetric, dequant = int8 * k_scale
    Values: INT4 per-group asymmetric, dequant = int4 * v_scale + v_zero

    Returns [num_q_heads, d_v] float32.
    """
    num_kv_heads, N, head_dim = keys_int8.shape
    d_v = values_int4_packed.shape[2] * 2  # packed: d_v // 2
    num_groups = d_v // group_size
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_int8.device

    K_flat = keys_int8.reshape(num_kv_heads * N, head_dim).contiguous()
    V_packed_flat = values_int4_packed.reshape(num_kv_heads * N, d_v // 2).contiguous()
    V_scales_flat = values_int4_scales.reshape(num_kv_heads * N, num_groups).contiguous()
    V_zeros_flat = values_int4_zeros.reshape(num_kv_heads * N, num_groups).contiguous()

    output = torch.empty(num_q_heads, d_v, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)

    _multihead_selective_attend_int8k_int4v_kernel[(num_q_heads,)](
        K_flat, keys_scale.contiguous(),
        V_packed_flat, V_scales_flat, V_zeros_flat,
        q_all.contiguous(), skip_mask_i32.contiguous(), output,
        N=N,
        stride_k=head_dim,
        d_v=d_v,
        d_v_half=d_v // 2,
        num_groups=num_groups,
        group_size=group_size,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_group=gqa_group,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output


# ─── Hybrid INT8/FP16 keys variant (top-K FP16 fallback) ─────────────

@triton.jit
def _multihead_selective_attend_hybrid_kernel(
    # INT8 keys
    K_int8_ptr,      # [num_kv_heads * N, head_dim] int8
    K_scale_ptr,     # [num_kv_heads, num_blocks, head_dim] float32
    # FP16 keys (same layout, only top-K blocks read from here)
    K_fp16_ptr,      # [num_kv_heads * N, head_dim] float16
    # Per-head mask: 1 = use FP16 keys, 0 = use INT8 keys
    TopK_mask_ptr,   # [num_q_heads, num_blocks] int32
    # Values + query + skip
    V_ptr,           # [num_kv_heads * N, d_v] float16
    Q_ptr,           # [num_q_heads, head_dim] float32
    Skip_ptr,        # [num_q_heads, num_blocks] int32
    Out_ptr,         # [num_q_heads, d_v] float32
    # Layout
    N: tl.constexpr,
    stride_k: tl.constexpr,
    stride_v: tl.constexpr,
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    d_v: tl.constexpr,
    q_scale: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    gqa_group: tl.constexpr,
    # Valid token count within the LAST block (trailing partial block).
    # Positions t_offs >= last_block_valid inside the last block are masked
    # to -inf before the softmax so they contribute zero mass. Pass
    # block_size when the last block is full (no masking).
    last_block_valid: tl.constexpr,
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """Hybrid: INT8 keys for most blocks, FP16 keys for top-K blocks."""
    qh = tl.program_id(0)
    valid = qh < num_q_heads
    if valid:
        kvh = qh // gqa_group
        kv_base = kvh * N

        t_offs = tl.arange(0, block_size)
        d_offs = tl.arange(0, TILE_D)
        v_offs = tl.arange(0, TILE_V)
        v_mask = v_offs < d_v

        # Online softmax state — FP64 per paper §9.9.
        m = tl.full((), float("-inf"), dtype=tl.float64)
        l = tl.full((), 0.0, dtype=tl.float64)
        acc = tl.zeros((TILE_V,), dtype=tl.float64)

        for bid in range(num_blocks):
            skip_val = tl.load(Skip_ptr + qh * num_blocks + bid)
            if skip_val == 0:
                base_tok = kv_base + bid * block_size
                use_fp16 = tl.load(TopK_mask_ptr + qh * num_blocks + bid)

                # Score: always load INT8 + dequant. For top-K, also load FP16
                # and overwrite. INT8 load is 1 byte/elem (cheap); FP16 load
                # only happens for ~4 blocks out of hundreds (amortised zero cost).
                scores = tl.zeros((block_size,), dtype=tl.float32)
                scale_base = (kvh * num_blocks + bid) * head_dim
                int8_row_ptrs = K_int8_ptr + (base_tok + t_offs) * stride_k
                fp16_row_ptrs = K_fp16_ptr + (base_tok + t_offs) * stride_k

                for d_start in range(0, head_dim, TILE_D):
                    d_off = d_start + d_offs
                    dm = d_off < head_dim
                    q_tile = tl.load(Q_ptr + qh * head_dim + d_off, mask=dm, other=0.0).to(tl.float32)
                    ch_scale = tl.load(K_scale_ptr + scale_base + d_off, mask=dm, other=0.0).to(tl.float32)

                    # Always load INT8 (cheap — 1 byte per element)
                    k_int8 = tl.load(int8_row_ptrs[:, None] + d_off[None, :], mask=dm[None, :], other=0)
                    k_tile = k_int8.to(tl.float32) * ch_scale[None, :]

                    # Conditionally load FP16 for top-K blocks (mask-gated)
                    # For non-top-K blocks, the mask zeros out the FP16 contribution.
                    # The load still happens but data is discarded — the cost is the
                    # memory transaction, which hits cache for the zeros buffer.
                    k_fp16 = tl.load(fp16_row_ptrs[:, None] + d_off[None, :], mask=(dm[None, :] & (use_fp16 == 1)), other=0).to(tl.float32)
                    # If use_fp16, replace int8 result with fp16 result
                    k_tile = tl.where(use_fp16 == 1, k_fp16, k_tile)

                    scores += tl.sum(k_tile * q_tile[None, :], axis=1)
                scores = scores * q_scale

                # Mask out-of-range tokens in the trailing partial block. For
                # non-last blocks or a fully-valid last block (last_block_valid
                # == block_size) this is a no-op. The mask sets invalid
                # positions to -inf *before* block_max so they cannot win the
                # softmax argmax and cannot contribute to l or acc.
                if bid == num_blocks - 1:
                    valid_tok = t_offs < last_block_valid
                    scores = tl.where(valid_tok, scores, float("-inf"))

                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max.to(tl.float64))
                alpha = tl.exp(m - new_m)
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m.to(tl.float32))
                l += tl.sum(weights).to(tl.float64)

                v_row_ptrs = V_ptr + (base_tok + t_offs) * stride_v
                v_off = v_offs
                vm = v_off < d_v
                v_ptrs = v_row_ptrs[:, None] + v_off[None, :]
                v_tile = tl.load(v_ptrs, mask=vm[None, :], other=0).to(tl.float32)
                w_v = tl.sum(weights[:, None] * v_tile, axis=0)
                acc += w_v.to(tl.float64)
                m = new_m

        safe_l = tl.where(l > 0.0, l, 1.0)
        output = (acc / safe_l).to(tl.float32)
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


def selective_attend_multihead_hybrid(
    keys_int8: torch.Tensor,       # [num_kv_heads, N, head_dim] int8
    keys_scale: torch.Tensor,      # [num_kv_heads, num_blocks, head_dim] float32
    keys_fp16: torch.Tensor,       # [num_kv_heads, N, head_dim] float16
    topk_mask: torch.Tensor,       # [num_q_heads, num_blocks] int32 (1=fp16, 0=int8)
    values_fp16: torch.Tensor,     # [num_kv_heads, N, d_v] float16
    q_all: torch.Tensor,           # [num_q_heads, head_dim] float32
    skip_mask_i32: torch.Tensor,   # [num_q_heads, num_blocks] int32
    gqa_group: int,
    block_size: int = 16,
    q_scale: float = 1.0,
    last_block_valid: int | None = None,
) -> torch.Tensor:
    """Hybrid INT8/FP16 keys: reads INT8 for most blocks, FP16 for top-K.

    The top-K mask is per-head: different Q heads can use FP16 for different blocks.
    Only the top-K blocks are read from keys_fp16 — the rest from keys_int8.
    Zero Python-level materialisation of dequant tensors.
    """
    num_kv_heads, N, head_dim = keys_int8.shape
    d_v = values_fp16.shape[2]
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_int8.device

    K_int8_flat = keys_int8.reshape(num_kv_heads * N, head_dim).contiguous()
    K_fp16_flat = keys_fp16.reshape(num_kv_heads * N, head_dim).contiguous()
    V_flat = values_fp16.reshape(num_kv_heads * N, d_v).contiguous()
    output = torch.empty(num_q_heads, d_v, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)

    lbv = block_size if last_block_valid is None else int(last_block_valid)
    _multihead_selective_attend_hybrid_kernel[(num_q_heads,)](
        K_int8_flat, keys_scale.contiguous(),
        K_fp16_flat,
        topk_mask.contiguous(),
        V_flat,
        q_all.contiguous(), skip_mask_i32.contiguous(), output,
        N=N,
        stride_k=head_dim,
        stride_v=d_v,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        d_v=d_v,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        gqa_group=gqa_group,
        last_block_valid=lbv,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output


def selective_attend_multihead(
    keys_packed: torch.Tensor,    # [num_kv_heads, N, head_dim] float32
    values_packed: torch.Tensor,  # [num_kv_heads, N, d_v] float32
    q_all: torch.Tensor,          # [num_q_heads, head_dim] float32
    skip_mask_i32: torch.Tensor,  # [num_q_heads, num_blocks] int32
    gqa_group: int,
    block_size: int = 16,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Multi-head selective attention in one kernel launch.

    Returns [num_q_heads, d_v] output.
    """
    num_kv_heads, N, head_dim = keys_packed.shape
    d_v = values_packed.shape[2]
    num_q_heads = q_all.shape[0]
    num_blocks = N // block_size
    device = keys_packed.device

    K_flat = keys_packed.reshape(num_kv_heads * N, head_dim).contiguous()
    V_flat = values_packed.reshape(num_kv_heads * N, d_v).contiguous()
    output = torch.empty(num_q_heads, d_v, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)

    _multihead_selective_attend_kernel[(num_q_heads,)](
        K_flat, V_flat, q_all.contiguous(), skip_mask_i32.contiguous(), output,
        N=N,
        stride_k=head_dim,
        stride_v=d_v,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        d_v=d_v,
        q_scale=q_scale,
        num_q_heads=num_q_heads,
        gqa_group=gqa_group,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output


# Keep single-head version for compatibility
def selective_attend_triton(
    keys: torch.Tensor,
    values: torch.Tensor,
    q: torch.Tensor,
    skip_mask_i32: torch.Tensor,
    block_size: int = 16,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Single-head selective attention."""
    N, head_dim = keys.shape
    d_v = values.shape[1]
    # Wrap as multi-head with 1 head
    out = selective_attend_multihead(
        keys.unsqueeze(0), values.unsqueeze(0),
        q.unsqueeze(0), skip_mask_i32.unsqueeze(0),
        gqa_group=1, block_size=block_size, q_scale=q_scale,
    )
    return out.squeeze(0)
