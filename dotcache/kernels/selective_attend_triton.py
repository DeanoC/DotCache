"""Phase 2: Triton selective attention on non-skipped blocks.

Single kernel that:
  1. Reads the skip mask
  2. For non-skipped blocks: loads K and V, computes q·K scores, accumulates
     softmax-weighted V output using online softmax (FlashAttention-style)
  3. Writes final output vector

Eliminates PyTorch tensor indexing/gather overhead by doing block selection
inside the kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _selective_attend_kernel(
    # Data (all for one KV head)
    K_ptr,           # [N, head_dim] float32/16 contiguous
    V_ptr,           # [N, d_v] float32/16 contiguous
    Q_ptr,           # [head_dim] float32
    Skip_ptr,        # [num_blocks] int32 (1=skip, 0=attend)
    Out_ptr,         # [d_v] float32 output
    # Strides
    stride_k_n,      # head_dim
    stride_v_n,      # d_v
    # Dims
    num_blocks: tl.constexpr,
    block_size: tl.constexpr,
    head_dim: tl.constexpr,
    d_v: tl.constexpr,
    q_scale: tl.constexpr,
    # Tiles
    TILE_D: tl.constexpr,
    TILE_V: tl.constexpr,
):
    """Single program: iterate over all blocks, skip flagged ones,
    accumulate attention output with online softmax."""

    # Load query
    d_offs = tl.arange(0, TILE_D)
    d_mask = d_offs < head_dim
    q = tl.load(Q_ptr + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    # Online softmax state
    m = tl.full((), float("-inf"), dtype=tl.float32)  # running max
    l = tl.full((), 0.0, dtype=tl.float32)            # running sum of exp
    # Accumulator for weighted V
    v_offs = tl.arange(0, TILE_V)
    v_mask = v_offs < d_v
    acc = tl.zeros((TILE_V,), dtype=tl.float32)

    t_offs = tl.arange(0, block_size)

    for bid in range(num_blocks):
        skip_val = tl.load(Skip_ptr + bid)
        if skip_val == 0:  # attend this block
            base = bid * block_size

            # Score all tokens in block: [block_size] dot products
            scores = tl.zeros((block_size,), dtype=tl.float32)
            row_ptrs = K_ptr + (base + t_offs) * stride_k_n
            for d_start in range(0, head_dim, TILE_D):
                d_off = d_start + d_offs
                dm = d_off < head_dim
                q_tile = tl.load(Q_ptr + d_off, mask=dm, other=0.0).to(tl.float32)
                k_ptrs = row_ptrs[:, None] + d_off[None, :]
                k_tile = tl.load(k_ptrs, mask=dm[None, :], other=0).to(tl.float32)
                scores += tl.sum(k_tile * q_tile[None, :], axis=1)
            scores = scores * q_scale

            # Online softmax update for this block's tokens
            block_max = tl.max(scores)
            new_m = tl.maximum(m, block_max)

            # Rescale existing accumulator
            alpha = tl.exp(m - new_m)
            acc = acc * alpha
            l = l * alpha

            # Add this block's contribution
            weights = tl.exp(scores - new_m)  # [block_size]
            l += tl.sum(weights)

            # Weighted V accumulation: acc += Σ_t weights[t] * V[base+t, :]
            v_row_ptrs = V_ptr + (base + t_offs) * stride_v_n
            for v_start in range(0, d_v, TILE_V):
                v_off = v_start + v_offs
                vm = v_off < d_v
                v_ptrs = v_row_ptrs[:, None] + v_off[None, :]
                v_tile = tl.load(v_ptrs, mask=vm[None, :], other=0).to(tl.float32)
                # weights: [block_size], v_tile: [block_size, TILE_V]
                # weighted sum: [TILE_V]
                w_v = tl.sum(weights[:, None] * v_tile, axis=0)
                if v_start == 0:
                    acc += w_v
                else:
                    # For d_v > TILE_V: need separate accumulators
                    # Simplified: only works when TILE_V >= d_v
                    acc += w_v

            m = new_m

    # Normalise: output = acc / l
    output = acc / l
    tl.store(Out_ptr + v_offs, output, mask=v_mask)


def selective_attend_triton(
    keys: torch.Tensor,          # [N, head_dim] float32
    values: torch.Tensor,        # [N, d_v] float32
    q: torch.Tensor,             # [head_dim] float32
    skip_mask_i32: torch.Tensor, # [num_blocks] int32 (1=skip)
    block_size: int = 16,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Triton selective attention — no PyTorch indexing overhead."""
    N, head_dim = keys.shape
    d_v = values.shape[1]
    num_blocks = N // block_size
    device = keys.device

    output = torch.empty(d_v, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)
    TILE_V = triton.next_power_of_2(d_v)

    _selective_attend_kernel[(1,)](
        keys, values, q, skip_mask_i32, output,
        stride_k_n=head_dim,
        stride_v_n=d_v,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
        d_v=d_v,
        q_scale=q_scale,
        TILE_D=TILE_D,
        TILE_V=TILE_V,
    )
    return output
