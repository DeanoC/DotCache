"""Phase 2: Triton selective attention — single-head and multi-head versions.

Multi-head version processes all Q heads in one kernel launch, with each
program handling one Q head. Skip mask is per-head.
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

        # Online softmax state
        m = tl.full((), float("-inf"), dtype=tl.float32)
        l = tl.full((), 0.0, dtype=tl.float32)
        acc = tl.zeros((TILE_V,), dtype=tl.float32)

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

                # Online softmax update
                block_max = tl.max(scores)
                new_m = tl.maximum(m, block_max)
                alpha = tl.exp(m - new_m)
                acc = acc * alpha
                l = l * alpha
                weights = tl.exp(scores - new_m)
                l += tl.sum(weights)

                # V accumulation
                v_row_ptrs = V_ptr + (base_tok + t_offs) * stride_v
                v_off = v_offs  # assumes TILE_V >= d_v
                vm = v_off < d_v
                v_ptrs = v_row_ptrs[:, None] + v_off[None, :]
                v_tile = tl.load(v_ptrs, mask=vm[None, :], other=0).to(tl.float32)
                w_v = tl.sum(weights[:, None] * v_tile, axis=0)
                acc += w_v
                m = new_m

        # Normalise and store
        safe_l = tl.where(l > 0.0, l, 1.0)
        output = acc / safe_l
        tl.store(Out_ptr + qh * d_v + v_offs, output, mask=v_mask)


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
