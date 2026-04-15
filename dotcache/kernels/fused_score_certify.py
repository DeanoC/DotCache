"""Fused INT8 block scoring + certification kernel.

Single Triton kernel scores all blocks, two-kernel pipeline certifies.
The scoring kernel loads K_int8 as 2D [block_size, TILE_D] tiles and
computes all 16 per-token dot products in parallel via tl.dot or
vectorised element-wise multiply + reduce.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_score_reduce_kernel(
    K_int8_ptr,
    K_scale_ptr,
    Q_ptr,
    M_b_ptr,
    S_b_ptr,
    stride_k_n,       # K_int8 stride along token dim (= head_dim for contiguous)
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    q_scale: tl.constexpr,
    TILE_D: tl.constexpr,
):
    """One program per KV block. Vectorised across block_size tokens."""
    block_id = tl.program_id(0)
    k_scale = tl.load(K_scale_ptr + block_id).to(tl.float32)
    base_row = block_id * block_size

    # scores[t] accumulates the dot product for each of block_size tokens
    scores = tl.zeros((block_size,), dtype=tl.float32)
    t_offsets = tl.arange(0, block_size)  # [block_size]

    for d_start in range(0, head_dim, TILE_D):
        d_offsets = d_start + tl.arange(0, TILE_D)
        d_mask = d_offsets < head_dim

        # q_tile: [TILE_D]
        q_tile = tl.load(Q_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32)

        # K_tile: [block_size, TILE_D] — 2D load
        # k_ptrs[t, d] = K_int8_ptr + (base_row + t) * stride_k_n + d_offsets[d]
        k_ptrs = K_int8_ptr + (base_row + t_offsets[:, None]) * stride_k_n + d_offsets[None, :]
        k_mask = d_mask[None, :]  # broadcast: [1, TILE_D] → [block_size, TILE_D]
        k_tile = tl.load(k_ptrs, mask=k_mask, other=0).to(tl.float32)  # [block_size, TILE_D]

        # Dequantise
        k_fp = k_tile * k_scale  # [block_size, TILE_D]

        # Per-token dot product: scores[t] += sum_d(k_fp[t, d] * q_tile[d])
        partial = tl.sum(k_fp * q_tile[None, :], axis=1)  # [block_size]
        scores += partial

    scores = scores * q_scale

    # Reduce: m_b = max, S_b = sum(exp(scores - m_b))
    m_b = tl.max(scores)
    s_b = tl.sum(tl.exp(scores - m_b))

    tl.store(M_b_ptr + block_id, m_b)
    tl.store(S_b_ptr + block_id, s_b)


@triton.jit
def _certify_skip_kernel(
    M_b_ptr,
    S_b_ptr,
    Corr_ptr,
    Skip_ptr,
    TotalMass_ptr,
    MGlobal_ptr,
    num_blocks: tl.constexpr,
    block_epsilon: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Single program: global max → total mass → skip mask."""
    # Pass 1: global max
    m_global = tl.full((), float("-inf"), dtype=tl.float32)
    for start in range(0, num_blocks, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < num_blocks
        m_global = tl.maximum(m_global, tl.max(tl.load(M_b_ptr + offs, mask=mask, other=float("-inf"))))
    tl.store(MGlobal_ptr, m_global)

    # Pass 2: total mass
    total_mass = tl.full((), 0.0, dtype=tl.float32)
    for start in range(0, num_blocks, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < num_blocks
        s = tl.load(S_b_ptr + offs, mask=mask, other=0.0)
        m = tl.load(M_b_ptr + offs, mask=mask, other=float("-inf"))
        c = tl.load(Corr_ptr + offs, mask=mask, other=1.0)
        total_mass += tl.sum(tl.where(mask, s * c * tl.exp(m - m_global), 0.0))
    tl.store(TotalMass_ptr, total_mass)

    # Pass 3: skip mask
    for start in range(0, num_blocks, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < num_blocks
        s = tl.load(S_b_ptr + offs, mask=mask, other=0.0)
        m = tl.load(M_b_ptr + offs, mask=mask, other=float("-inf"))
        c = tl.load(Corr_ptr + offs, mask=mask, other=1.0)
        res = s * c * tl.exp(m - m_global)
        skip = (res / total_mass) < block_epsilon
        tl.store(Skip_ptr + offs, tl.where(mask, skip.to(tl.int32), 0), mask=mask)


def fused_score_certify(
    K_int8: torch.Tensor,
    K_scale: torch.Tensor,
    q: torch.Tensor,
    correction: torch.Tensor,
    block_size: int = 16,
    q_scale: float = 1.0,
    block_epsilon: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Fused INT8 scoring + certification.

    Returns (m_b, S_b, skip_mask, m_global, total_mass).
    """
    N, head_dim = K_int8.shape
    num_blocks = N // block_size
    device = K_int8.device

    m_b = torch.empty(num_blocks, dtype=torch.float32, device=device)
    S_b = torch.empty(num_blocks, dtype=torch.float32, device=device)

    TILE_D = min(triton.next_power_of_2(head_dim), 128)

    _fused_score_reduce_kernel[(num_blocks,)](
        K_int8, K_scale, q, m_b, S_b,
        stride_k_n=head_dim,
        head_dim=head_dim,
        block_size=block_size,
        q_scale=q_scale,
        TILE_D=TILE_D,
    )

    skip_i32 = torch.empty(num_blocks, dtype=torch.int32, device=device)
    tm = torch.empty(1, dtype=torch.float32, device=device)
    mg = torch.empty(1, dtype=torch.float32, device=device)
    BLOCK_N = min(triton.next_power_of_2(num_blocks), 1024)

    _certify_skip_kernel[(1,)](
        m_b, S_b, correction, skip_i32, tm, mg,
        num_blocks=num_blocks,
        block_epsilon=block_epsilon,
        BLOCK_N=BLOCK_N,
    )

    return m_b, S_b, skip_i32.bool(), mg.item(), tm.item()


def selective_attend(
    keys_fp: torch.Tensor,
    values_fp: torch.Tensor,
    q: torch.Tensor,
    skip_mask: torch.Tensor,
    block_size: int = 16,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Phase 2: attend non-skipped blocks."""
    num_blocks = skip_mask.shape[0]
    idx = (~skip_mask).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return torch.zeros(values_fp.shape[-1], dtype=torch.float32, device=q.device)
    k_bl = keys_fp.reshape(num_blocks, block_size, -1)
    v_bl = values_fp.reshape(num_blocks, block_size, -1)
    ak = k_bl[idx].reshape(-1, keys_fp.shape[-1]).to(torch.float32)
    av = v_bl[idx].reshape(-1, values_fp.shape[-1]).to(torch.float32)
    s = torch.matmul(ak, q.to(torch.float32)) * q_scale
    w = torch.softmax(s, dim=0)
    return w @ av
