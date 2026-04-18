"""Tiered KV cache: per-channel INT8 keys in VRAM, FP16 originals in pinned CPU RAM.

Keys are quantised per-channel: each of the head_dim channels gets its own
symmetric INT8 scale within each block of block_size tokens.  This preserves
resolution for low-magnitude channels that per-block quantisation crushes.

Quantisation is deferred: trailing partial blocks (< block_size tokens) stay
in FP16 in VRAM.  When a block fills to block_size tokens, all tokens are
quantised together with per-channel scales computed from the full block.
The hybrid attend kernel handles the FP16 trailing block.

Two value storage modes:
  - FP16 values in VRAM (original, higher quality)
  - INT4 per-group values in VRAM (v2, ~38% less VRAM, mass-weighted safety)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np


@dataclass
class TieredKeyCacheLayer:
    """Per-layer tiered key cache for one KV head group."""

    # VRAM (hot) — INT8 quantised keys (only complete blocks have valid data)
    keys_int8: torch.Tensor          # [kv_heads, N, head_dim] int8, device=cuda
    keys_scale: torch.Tensor          # [kv_heads, num_blocks, head_dim] float32 (per-channel)
    # INT8 certification correction per block
    correction: torch.Tensor          # [kv_heads, num_blocks] float32, device=cuda

    # VRAM — FP16 values (always resident, needed for attend)
    values_fp16: torch.Tensor         # [kv_heads, N, d_v] float16, device=cuda

    # CPU pinned RAM (cold) — FP16 original keys for fallback
    keys_fp16_cpu: torch.Tensor       # [kv_heads, N, head_dim] float16, pinned CPU

    # Layout
    kv_heads: int
    num_tokens: int
    head_dim: int
    d_v: int
    block_size: int
    num_blocks: int
    num_quantized_blocks: int = 0     # blocks with valid INT8 data + per-channel scales

    # Pre-allocated VRAM buffer for page-in (avoid allocation on critical path)
    _pagein_buffer: torch.Tensor | None = None  # [max_pagein_blocks * block_size, head_dim] fp16 cuda

    # Pre-computed dequantised keys and float32 values (avoid per-call allocation)
    # For quantised blocks: dequantised INT8.  For trailing partial block: exact FP16→f32.
    _keys_deq_f32: torch.Tensor | None = None   # [kv_heads, N, head_dim] float32, cuda
    _values_f32: torch.Tensor | None = None      # [kv_heads, N, d_v] float32, cuda

    # INT4 per-group quantised values (v2 — optional, replaces values_fp16 in VRAM)
    values_int4_packed: torch.Tensor | None = None   # [kv_heads, N, d_v//2] uint8, cuda
    values_int4_scales: torch.Tensor | None = None   # [kv_heads, N, num_groups] float16, cuda
    values_int4_zeros: torch.Tensor | None = None    # [kv_heads, N, num_groups] float16, cuda
    values_int4_errors: torch.Tensor | None = None   # [kv_heads, num_blocks] float32, cuda
    values_int4_group_size: int = 32

    # CPU warm tier — FP16 values for fallback when INT4 error too high
    values_fp16_cpu: torch.Tensor | None = None  # [kv_heads, N, d_v] float16, pinned CPU

    # VRAM mirror of keys_fp16_cpu — avoids CPU→GPU copy per layer per decode step
    keys_fp16_gpu: torch.Tensor | None = None    # [kv_heads, N, head_dim] model dtype, cuda

    @classmethod
    def from_fp16_cache(
        cls,
        keys_fp16: torch.Tensor,     # [kv_heads, N, head_dim] float16/32, device=cuda
        values_fp16: torch.Tensor,   # [kv_heads, N, d_v] float16/32, device=cuda
        block_size: int = 16,
        max_pagein_blocks: int = 64,
        max_new_tokens: int = 512,
    ) -> "TieredKeyCacheLayer":
        """Create tiered cache from existing FP16 KV tensors.

        Quantises complete blocks to per-channel INT8 on GPU, then moves
        FP16 originals to pinned CPU.  N must be block-aligned (caller
        should trim trailing tokens and append them separately).
        Pre-allocates extra slots for max_new_tokens decode steps.
        """
        kv_heads, N, head_dim = keys_fp16.shape
        d_v = values_fp16.shape[2]
        num_blocks = N // block_size
        device = keys_fp16.device
        capacity = N + max_new_tokens  # pre-allocate for decode

        # Reshape to blocks for per-channel quantisation
        keys_blocked = keys_fp16.reshape(kv_heads, num_blocks, block_size, head_dim).to(torch.float32)

        # Per-channel symmetric INT8 quantisation: each channel gets its own scale
        # k_max: [kv_heads, num_blocks, head_dim] — max(abs) across the block_size tokens
        k_max = keys_blocked.abs().amax(dim=2).clamp(min=1e-8)
        k_scale = k_max / 127.0  # [kv_heads, num_blocks, head_dim]
        keys_int8 = (
            (keys_blocked / k_scale[:, :, None, :])  # broadcast over token dim
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
            .reshape(kv_heads, N, head_dim)
            .contiguous()
        )

        # Correction factor uses L2 norm of per-channel scale vector.
        # Per-token reconstruction error: ||k - k'||_2 = (1/2) * ||k_scale||_2 / 127
        # Score error bound: delta ≈ ||k_scale||_2 / 127
        # Much tighter than per-block: sqrt(d) * max_scale / 127
        k_scale_l2 = k_scale.norm(dim=-1)  # [kv_heads, num_blocks]
        delta_per_block = k_scale_l2 / 127.0
        correction = torch.exp(2.0 * delta_per_block)

        # Preserve the model's native dtype (BF16 or FP16) for keys/values.
        # This ensures SDPA attend matches dense attention precision exactly.
        kv_dtype = keys_fp16.dtype  # typically bfloat16 from LLaMA

        # Move originals to pinned CPU memory (preserve dtype)
        keys_fp16_cpu = keys_fp16.to(dtype=kv_dtype).cpu().pin_memory()

        # Values stay in VRAM (preserve dtype)
        values_fp16_cuda = values_fp16.to(dtype=kv_dtype, device=device).contiguous()

        # Pre-allocate page-in buffer
        pagein_buffer = torch.empty(
            max_pagein_blocks * block_size, head_dim,
            dtype=kv_dtype, device=device,
        )

        # Pre-allocate decode buffers (zero-copy append)
        max_new_blocks = (max_new_tokens + block_size - 1) // block_size
        keys_int8_buf = torch.zeros(kv_heads, capacity, head_dim, dtype=torch.int8, device=device)
        keys_int8_buf[:, :N, :] = keys_int8
        values_fp16_buf = torch.zeros(kv_heads, capacity, d_v, dtype=kv_dtype, device=device)
        values_fp16_buf[:, :N, :] = values_fp16_cuda

        # Scale/correction buffers — per-channel: [kv_heads, max_blocks, head_dim]
        max_total_blocks = num_blocks + max_new_blocks
        scale_buf = torch.zeros(kv_heads, max_total_blocks, head_dim, dtype=torch.float32, device=device)
        scale_buf[:, :num_blocks, :] = k_scale.to(torch.float32)
        corr_buf = torch.ones(kv_heads, max_total_blocks, dtype=torch.float32, device=device)
        corr_buf[:, :num_blocks] = correction.to(torch.float32)

        # CPU buffer for keys (preserve model dtype)
        keys_fp16_cpu_buf = torch.zeros(kv_heads, capacity, head_dim, dtype=kv_dtype, pin_memory=True)
        keys_fp16_cpu_buf[:, :N, :] = keys_fp16_cpu

        # GPU mirror so decode-time attend avoids a CPU→GPU copy per step
        keys_fp16_gpu_buf = torch.zeros(kv_heads, capacity, head_dim, dtype=kv_dtype, device=device)
        keys_fp16_gpu_buf[:, :N, :] = keys_fp16.to(dtype=kv_dtype, device=device)

        # Pre-compute dequant into buffer
        # Per-channel: k_scale is [kv_heads, num_blocks, head_dim], broadcast over token dim
        keys_deq_buf = torch.zeros(kv_heads, capacity, head_dim, dtype=torch.float32, device=device)
        keys_deq_buf[:, :N, :] = (
            keys_int8.to(torch.float32).reshape(kv_heads, num_blocks, block_size, head_dim)
            * k_scale.to(torch.float32)[:, :, None, :]
        ).reshape(kv_heads, N, head_dim)

        result = cls(
            keys_int8=keys_int8_buf,
            keys_scale=scale_buf,
            correction=corr_buf,
            values_fp16=values_fp16_buf,
            keys_fp16_cpu=keys_fp16_cpu_buf,
            kv_heads=kv_heads,
            num_tokens=N,
            head_dim=head_dim,
            d_v=d_v,
            block_size=block_size,
            num_blocks=num_blocks,
            num_quantized_blocks=num_blocks,  # all prefill blocks are complete
            _pagein_buffer=pagein_buffer,
            _keys_deq_f32=keys_deq_buf,
            keys_fp16_gpu=keys_fp16_gpu_buf,
        )
        return result

    @classmethod
    def from_fp16_cache_int4v(
        cls,
        keys_fp16: torch.Tensor,     # [kv_heads, N, head_dim] float16/32, device=cuda
        values_fp16: torch.Tensor,   # [kv_heads, N, d_v] float16/32, device=cuda
        block_size: int = 16,
        group_size: int = 32,
        max_pagein_blocks: int = 64,
    ) -> "TieredKeyCacheLayer":
        """Create tiered cache with INT4 per-group values.

        Keys: per-channel INT8 in VRAM (same as v1)
        Values: INT4 per-group in VRAM (NEW — saves ~38% vs FP16)
        FP16 originals: pinned CPU (both K and V)
        """
        from dotcache.kernels.int4_group_quantise import quantise_int4_grouped_block

        # Build the base cache (INT8 keys, FP16 values)
        base = cls.from_fp16_cache(keys_fp16, values_fp16, block_size, max_pagein_blocks)

        # Quantise values to INT4 per-group
        int4_result = quantise_int4_grouped_block(
            values_fp16.to(torch.float16), block_size=block_size, group_size=group_size,
        )

        # Store INT4 values on the cache
        base.values_int4_packed = int4_result["data_packed"].contiguous()
        base.values_int4_scales = int4_result["scales"].contiguous()
        base.values_int4_zeros = int4_result["zeros"].contiguous()
        base.values_int4_errors = int4_result["error_bounds"].contiguous()
        base.values_int4_group_size = group_size

        # Move FP16 values to CPU pinned (they're currently in VRAM as values_fp16)
        base.values_fp16_cpu = base.values_fp16.cpu().pin_memory()

        # Free FP16 values from VRAM — INT4 replaces them
        base.values_fp16 = None

        return base

    def dequantise_int4_values(self) -> torch.Tensor:
        """Dequantise all INT4 values to float32 [kv_heads, N, d_v]."""
        from dotcache.kernels.int4_group_quantise import dequantise_int4_grouped

        kv_heads = self.values_int4_packed.shape[0]
        N = self.values_int4_packed.shape[1]
        results = []
        for h in range(kv_heads):
            deq = dequantise_int4_grouped(
                self.values_int4_packed[h],
                self.values_int4_scales[h],
                self.values_int4_zeros[h],
                self.values_int4_group_size,
            )
            results.append(deq)
        return torch.stack(results).to(torch.float32)

    def get_values_f32(self) -> torch.Tensor:
        """Get float32 values from whichever tier is available in VRAM."""
        if self.values_fp16 is not None:
            return self.values_fp16.to(torch.float32)
        if self.values_int4_packed is not None:
            return self.dequantise_int4_values()
        raise ValueError("No values available in VRAM")

    def _quantize_block(self, block_idx: int) -> None:
        """Quantize a completed block to per-channel INT8.

        Called when the block_idx-th block has exactly block_size tokens.
        Reads FP16 keys from CPU, computes per-channel absmax scales,
        writes INT8 + scales + correction to VRAM.
        """
        start = block_idx * self.block_size
        end = start + self.block_size
        device = self.keys_int8.device

        # Read FP16 keys for this block from CPU
        keys_block = self.keys_fp16_cpu[:, start:end, :].to(
            device=device, dtype=torch.float32
        )  # [kv_heads, block_size, head_dim]

        # Per-channel absmax scale
        k_max = keys_block.abs().amax(dim=1).clamp(min=1e-8)  # [kv_heads, head_dim]
        k_scale = k_max / 127.0  # [kv_heads, head_dim]

        # Quantize all block_size tokens at once
        k_int8 = (
            (keys_block / k_scale[:, None, :])
            .round().clamp(-127, 127).to(torch.int8)
        )  # [kv_heads, block_size, head_dim]

        # Write INT8 keys to VRAM
        self.keys_int8[:, start:end, :] = k_int8

        # Write per-channel scale
        self.keys_scale[:, block_idx, :] = k_scale

        # Correction factor: delta = ||k_scale||_2 / 127
        k_scale_l2 = k_scale.norm(dim=-1)  # [kv_heads]
        delta = k_scale_l2 / 127.0
        self.correction[:, block_idx] = torch.exp(2.0 * delta)

        # Update dequant buffer with INT8-dequantised values (replaces the exact FP16)
        if self._keys_deq_f32 is not None:
            self._keys_deq_f32[:, start:end, :] = (
                k_int8.to(torch.float32) * k_scale[:, None, :]
            )

        self.num_quantized_blocks = block_idx + 1

    def append_token(
        self,
        key_fp16: torch.Tensor,    # [kv_heads, 1, head_dim] float16/32
        value_fp16: torch.Tensor,  # [kv_heads, 1, d_v] float16/32
    ) -> None:
        """Append one token to the cache with deferred INT8 quantisation.

        Tokens are stored as FP16 in the trailing partial block.  When
        the block fills to block_size tokens, _quantize_block() is called
        to compute per-channel INT8 scales from all tokens at once.
        """
        pos = self.num_tokens
        device = self.keys_int8.device
        kv_dtype = self.values_fp16.dtype if self.values_fp16 is not None else torch.float16

        new_k = key_fp16.to(dtype=kv_dtype).squeeze(1)  # [kv_heads, head_dim]
        new_v = value_fp16.to(dtype=kv_dtype).squeeze(1)

        # Write value into pre-allocated VRAM buffer (preserves model dtype)
        if self.values_fp16 is not None:
            self.values_fp16[:, pos, :] = new_v.to(device=device, dtype=kv_dtype)

        # Write key into pre-allocated CPU buffer (preserves model dtype)
        self.keys_fp16_cpu[:, pos, :] = new_k.cpu()

        # Mirror into GPU key buffer so decode-time attend avoids a CPU→GPU copy
        if self.keys_fp16_gpu is not None:
            self.keys_fp16_gpu[:, pos, :] = new_k.to(device=device, dtype=kv_dtype)

        if self.values_fp16_cpu is not None:
            self.values_fp16_cpu[:, pos, :] = new_v.cpu()

        # Write exact key→float32 into dequant buffer (for hybrid attend path)
        new_k_f32 = new_k.to(device=device, dtype=torch.float32)
        if self._keys_deq_f32 is not None:
            self._keys_deq_f32[:, pos, :] = new_k_f32

        # Write key into hot cache if it exists (for hybrid kernel)
        if hasattr(self, '_fp16_hot_cache') and self._fp16_hot_cache is not None:
            self._fp16_hot_cache[:, pos, :] = new_k.to(device=device, dtype=kv_dtype)

        # Update counts — ceiling division so num_blocks includes partial trailing block
        self.num_tokens = pos + 1
        self.num_blocks = (self.num_tokens + self.block_size - 1) // self.block_size

        block_idx = pos // self.block_size
        pos_in_block = pos % self.block_size

        # Check if this token completes a block
        if pos_in_block == self.block_size - 1:
            # Block is full — compute per-channel scales and quantize all 16 tokens
            self._quantize_block(block_idx)

        # Poison unused positions in new blocks so the scoring kernel
        # gives them near-zero softmax weight
        if pos_in_block == 0 and self.num_tokens < self.aligned_tokens:
            # First token of a new block — poison padding positions
            aligned = self.num_blocks * self.block_size
            if aligned > self.num_tokens:
                self.keys_int8[:, self.num_tokens:aligned, :] = -127
                if self._keys_deq_f32 is not None:
                    # Poison dequant for unused slots (will be overwritten by
                    # subsequent appends or by _quantize_block when block fills)
                    self._keys_deq_f32[:, self.num_tokens:aligned, :] = 0.0

    @property
    def has_trailing_partial_block(self) -> bool:
        """True if the last block has fewer than block_size tokens."""
        return self.num_tokens > 0 and self.num_tokens % self.block_size != 0

    @property
    def trailing_block_idx(self) -> int | None:
        """Index of the partial trailing block, or None if all blocks are complete."""
        if self.has_trailing_partial_block:
            return self.num_tokens // self.block_size
        return None

    @property
    def active_tokens(self) -> int:
        """Number of active (written) tokens. May be less than buffer capacity."""
        return self.num_tokens

    @property
    def aligned_tokens(self) -> int:
        """Block-aligned token count (rounds UP to include partial trailing block)."""
        return ((self.num_tokens + self.block_size - 1) // self.block_size) * self.block_size

    def keys_int8_active(self) -> torch.Tensor:
        """INT8 keys for active tokens (rounded up to block boundary).
        Note: trailing partial block positions contain zeros/poison, not valid INT8."""
        n = self.aligned_tokens
        return self.keys_int8[:, :n, :]

    @property
    def active_blocks(self) -> int:
        """Number of blocks (including partial trailing block)."""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def keys_scale_active(self) -> torch.Tensor:
        """Per-channel scales for active blocks: [kv_heads, active_blocks, head_dim]."""
        return self.keys_scale[:, :self.active_blocks, :]

    def correction_active(self) -> torch.Tensor:
        return self.correction[:, :self.active_blocks]

    def values_fp16_active(self) -> torch.Tensor:
        """FP16 values for active tokens (rounded up to block boundary)."""
        if self.values_fp16 is None:
            return None
        return self.values_fp16[:, :self.aligned_tokens, :]

    def keys_fp16_cpu_active(self) -> torch.Tensor:
        return self.keys_fp16_cpu[:, :self.num_tokens, :]

    def precompute_dequant(self) -> None:
        """Pre-compute dequantised keys and float32 values to avoid per-call allocation."""
        device = self.keys_int8.device
        nq = self.num_quantized_blocks
        qt = nq * self.block_size
        self._keys_deq_f32 = torch.zeros(
            self.kv_heads, self.aligned_tokens, self.head_dim,
            dtype=torch.float32, device=device,
        )
        # Quantised blocks: dequant INT8 with per-channel scales
        if nq > 0:
            self._keys_deq_f32[:, :qt, :] = (
                self.keys_int8[:, :qt, :].to(torch.float32)
                .reshape(self.kv_heads, nq, self.block_size, self.head_dim)
                * self.keys_scale[:, :nq, None, :]
            ).reshape(self.kv_heads, qt, self.head_dim)
        # Trailing partial block: exact FP16
        if qt < self.num_tokens:
            self._keys_deq_f32[:, qt:self.num_tokens, :] = (
                self.keys_fp16_cpu[:, qt:self.num_tokens, :]
                .to(device=device, dtype=torch.float32)
            )
        if self.values_fp16 is not None:
            self._values_f32 = self.values_fp16.to(torch.float32).contiguous()

    def vram_bytes(self) -> int:
        """Total VRAM usage."""
        total = self.keys_int8.nelement() * 1      # INT8
        total += self.keys_scale.nelement() * 4     # float32 per-channel: [kv_heads, blocks, head_dim]
        total += self.correction.nelement() * 4     # float32
        if self.values_fp16 is not None:
            total += self.values_fp16.nelement() * 2    # float16
        if self._pagein_buffer is not None:
            total += self._pagein_buffer.nelement() * 2
        if self._keys_deq_f32 is not None:
            total += self._keys_deq_f32.nelement() * 4
        if self._values_f32 is not None:
            total += self._values_f32.nelement() * 4
        # INT4 value storage
        if self.values_int4_packed is not None:
            total += self.values_int4_packed.nelement() * 1   # uint8 (packed)
            total += self.values_int4_scales.nelement() * 2   # float16
            total += self.values_int4_zeros.nelement() * 2    # float16
            total += self.values_int4_errors.nelement() * 4   # float32
        return total

    def cpu_bytes(self) -> int:
        """Total CPU pinned RAM usage."""
        total = self.keys_fp16_cpu.nelement() * 2
        if self.values_fp16_cpu is not None:
            total += self.values_fp16_cpu.nelement() * 2
        return total

    def page_in_blocks(
        self,
        kv_head_idx: int,
        block_ids: torch.Tensor,  # [n] int64, block indices to page in
        stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Async page FP16 key blocks from CPU to VRAM.

        Returns [n * block_size, head_dim] float16 tensor on CUDA.
        """
        n = block_ids.shape[0]
        if n == 0:
            return torch.empty(0, self.head_dim, dtype=torch.float16, device=self.keys_int8.device)

        # Gather from CPU (this is the slow part — minimise it)
        block_ids_cpu = block_ids.cpu()
        out_buf = self._pagein_buffer[:n * self.block_size] if (
            self._pagein_buffer is not None and n * self.block_size <= self._pagein_buffer.shape[0]
        ) else torch.empty(n * self.block_size, self.head_dim, dtype=torch.float16, device=self.keys_int8.device)

        # Async copy with optional stream
        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            for i, bid in enumerate(block_ids_cpu.tolist()):
                start = int(bid) * self.block_size
                end = start + self.block_size
                src = self.keys_fp16_cpu[kv_head_idx, start:end, :]
                out_buf[i * self.block_size:(i + 1) * self.block_size].copy_(src, non_blocking=True)

        return out_buf[:n * self.block_size]


class nullcontext:
    """Minimal no-op context manager."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def create_tiered_cache_int4v_from_model(
    past_kv,
    layer_ids: list[int],
    block_size: int = 16,
    group_size: int = 32,
) -> dict[int, TieredKeyCacheLayer]:
    """Create tiered caches with INT4 per-group values from HF past_key_values."""
    caches = {}
    for layer_id in layer_ids:
        if hasattr(past_kv, "layers"):
            keys = past_kv.layers[layer_id].keys[0]
            values = past_kv.layers[layer_id].values[0]
        else:
            keys = past_kv[layer_id][0][0]
            values = past_kv[layer_id][1][0]

        seq_len = keys.shape[1]
        aligned_len = (seq_len // block_size) * block_size
        keys_aligned = keys[:, :aligned_len, :].contiguous()
        values_aligned = values[:, :aligned_len, :].contiguous()

        caches[layer_id] = TieredKeyCacheLayer.from_fp16_cache_int4v(
            keys, values, block_size=block_size, group_size=group_size,
        )
        caches[layer_id].num_tokens = seq_len
    return caches


def create_tiered_cache_from_model(
    past_kv,
    layer_ids: list[int],
    block_size: int = 16,
    max_new_tokens: int = 512,
) -> dict[int, TieredKeyCacheLayer]:
    """Create tiered caches from a HuggingFace model's past_key_values.

    Args:
        past_kv: DynamicCache or tuple of (K, V) per layer
        layer_ids: which layers to create tiered caches for
        block_size: tokens per block
        max_new_tokens: pre-allocate capacity for this many decode tokens

    Returns:
        dict mapping layer_id → TieredKeyCacheLayer
    """
    caches = {}
    for layer_id in layer_ids:
        if hasattr(past_kv, 'layers'):
            keys = past_kv.layers[layer_id].keys[0]   # [kv_heads, seq, hd]
            values = past_kv.layers[layer_id].values[0]
        else:
            keys = past_kv[layer_id][0][0]
            values = past_kv[layer_id][1][0]

        # Trim to block-aligned, build cache with per-channel INT8 for complete
        # blocks, then append trailing tokens (which stay FP16 until their
        # block fills).
        seq_len = keys.shape[1]
        aligned_len = (seq_len // block_size) * block_size
        keys_aligned = keys[:, :aligned_len, :].contiguous()
        values_aligned = values[:, :aligned_len, :].contiguous()

        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys_aligned, values_aligned, block_size=block_size,
            max_new_tokens=max_new_tokens + (seq_len - aligned_len),
        )

        # Append the trailing (non-block-aligned) tokens — they stay FP16
        for t in range(aligned_len, seq_len):
            cache.append_token(
                keys[:, t:t+1, :],
                values[:, t:t+1, :],
            )

        # Poison padding positions so they get ~zero softmax weight.
        at = cache.aligned_tokens
        nt = cache.num_tokens
        if at > nt:
            cache.keys_int8[:, nt:at, :] = -127
            if cache._keys_deq_f32 is not None:
                cache._keys_deq_f32[:, nt:at, :] = 0.0

        caches[layer_id] = cache
    return caches
