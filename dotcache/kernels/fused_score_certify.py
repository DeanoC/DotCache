"""Fused INT8 score + block-reduce + certify: single kernel launch.

One Triton kernel that:
  1. Loads K_int8 tiles from global memory
  2. Dequantises in registers
  3. Computes q·K dot products via vectorised multiply-accumulate
  4. Reduces per-block: m_b = max(scores), S_b = sum(exp(scores - m_b))
  5. Writes m_b, S_b to global memory

Then a tiny certification kernel (single launch, O(num_blocks) scalar ops)
computes global max, total mass, and skip mask.

Total: 2 kernel launches for the entire Phase 1 pipeline.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Fused matmul + block reduce (one launch, one program per block)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_matmul_reduce_kernel(
    # Data
    K_int8_ptr,      # [N, head_dim] int8 contiguous
    K_scale_ptr,     # [num_blocks] float32
    Q_ptr,           # [head_dim] float32
    # Output
    M_b_ptr,         # [num_blocks] float32
    S_b_ptr,         # [num_blocks] float32
    # Layout
    stride_k_n: tl.constexpr,     # head_dim (K contiguous)
    head_dim: tl.constexpr,
    block_size: tl.constexpr,     # 16
    q_scale: tl.constexpr,
    num_blocks: tl.constexpr,
    # Tuning
    TILE_D: tl.constexpr,         # head_dim tile (power of 2, >= head_dim)
    BPP: tl.constexpr,            # blocks per program
):
    """Each program handles BPP blocks, amortising launch overhead."""
    pid = tl.program_id(0)
    t_offs = tl.arange(0, block_size)  # [BS]
    d_offs = tl.arange(0, TILE_D)     # [TD]
    num_tiles = (head_dim + TILE_D - 1) // TILE_D

    for local_b in range(BPP):
        bid = pid * BPP + local_b
        still_valid = bid < num_blocks
        if still_valid:
            base = bid * block_size
            k_scale = tl.load(K_scale_ptr + bid).to(tl.float32)
            scores = tl.zeros((block_size,), dtype=tl.float32)
            row_ptrs = K_int8_ptr + (base + t_offs) * stride_k_n

            for tile_idx in range(num_tiles):
                d_start = tile_idx * TILE_D
                d_off = d_start + d_offs
                d_mask = d_off < head_dim
                q_tile = tl.load(Q_ptr + d_off, mask=d_mask, other=0.0).to(tl.float32)
                k_ptrs = row_ptrs[:, None] + d_off[None, :]
                k_tile = tl.load(k_ptrs, mask=d_mask[None, :], other=0).to(tl.float32)
                k_fp = k_tile * k_scale
                scores += tl.sum(k_fp * q_tile[None, :], axis=1)

            scores = scores * q_scale
            m_b = tl.max(scores)
            s_b = tl.sum(tl.exp(scores - m_b))
            tl.store(M_b_ptr + bid, m_b)
            tl.store(S_b_ptr + bid, s_b)


# ---------------------------------------------------------------------------
# Kernel 2: Certify (single program, O(num_blocks) work, ~1-2μs)
# ---------------------------------------------------------------------------
@triton.jit
def _certify_kernel(
    M_b_ptr,          # [num_blocks] float32
    S_b_ptr,          # [num_blocks] float32
    Corr_ptr,         # [num_blocks] float32
    Skip_ptr,         # [num_blocks] int32 output
    num_blocks: tl.constexpr,
    block_epsilon: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # Pass 1: global max
    m_g = tl.full((), float("-inf"), dtype=tl.float32)
    for s in range(0, num_blocks, TILE_N):
        o = s + tl.arange(0, TILE_N)
        mk = o < num_blocks
        m_g = tl.maximum(m_g, tl.max(tl.load(M_b_ptr + o, mask=mk, other=float("-inf"))))

    # Pass 2: total mass
    total = tl.full((), 0.0, dtype=tl.float32)
    for s in range(0, num_blocks, TILE_N):
        o = s + tl.arange(0, TILE_N)
        mk = o < num_blocks
        sb = tl.load(S_b_ptr + o, mask=mk, other=0.0)
        mb = tl.load(M_b_ptr + o, mask=mk, other=float("-inf"))
        cr = tl.load(Corr_ptr + o, mask=mk, other=1.0)
        total += tl.sum(tl.where(mk, sb * cr * tl.exp(mb - m_g), 0.0))

    # Pass 3: skip mask
    for s in range(0, num_blocks, TILE_N):
        o = s + tl.arange(0, TILE_N)
        mk = o < num_blocks
        sb = tl.load(S_b_ptr + o, mask=mk, other=0.0)
        mb = tl.load(M_b_ptr + o, mask=mk, other=float("-inf"))
        cr = tl.load(Corr_ptr + o, mask=mk, other=1.0)
        res = sb * cr * tl.exp(mb - m_g)
        skip = (res / total) < block_epsilon
        tl.store(Skip_ptr + o, tl.where(mk, skip.to(tl.int32), 0), mask=mk)


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------
def fused_score_certify(
    K_int8: torch.Tensor,         # [N, head_dim] int8 contiguous
    K_scale: torch.Tensor,        # [num_blocks] float32
    q: torch.Tensor,              # [head_dim] float32
    correction: torch.Tensor,     # [num_blocks] float32
    block_size: int = 16,
    q_scale: float = 1.0,
    block_epsilon: float = 0.001,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused INT8 scoring + certification.

    Returns (m_b, S_b, skip_mask).
    """
    N, head_dim = K_int8.shape
    num_blocks = N // block_size
    device = K_int8.device

    m_b = torch.empty(num_blocks, dtype=torch.float32, device=device)
    S_b = torch.empty(num_blocks, dtype=torch.float32, device=device)

    TILE_D = triton.next_power_of_2(head_dim)

    # Fewer programs, more blocks per program → amortise launch overhead
    # Target ~32 programs (one per SM on typical GPUs)
    BPP = max(1, (num_blocks + 31) // 32)
    n_progs = (num_blocks + BPP - 1) // BPP

    _fused_matmul_reduce_kernel[(n_progs,)](
        K_int8, K_scale, q, m_b, S_b,
        stride_k_n=head_dim,
        head_dim=head_dim,
        block_size=block_size,
        q_scale=q_scale,
        num_blocks=num_blocks,
        TILE_D=TILE_D,
        BPP=BPP,
    )

    # Kernel 2: certify — single program
    skip_i32 = torch.empty(num_blocks, dtype=torch.int32, device=device)
    TILE_N = min(triton.next_power_of_2(num_blocks), 1024)
    _certify_kernel[(1,)](
        m_b, S_b, correction, skip_i32,
        num_blocks=num_blocks,
        block_epsilon=block_epsilon,
        TILE_N=TILE_N,
    )

    return m_b, S_b, skip_i32.bool()


def selective_attend(
    keys_fp: torch.Tensor,        # [N, head_dim]
    values_fp: torch.Tensor,      # [N, d_v]
    q: torch.Tensor,              # [head_dim]
    skip_mask: torch.Tensor,      # [num_blocks] bool
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
