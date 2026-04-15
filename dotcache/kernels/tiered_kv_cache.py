"""Tiered KV cache: INT8 keys in VRAM, FP16 originals in pinned CPU RAM.

The INT8 keys are the hot path — used for scoring in the fused kernel.
The FP16 originals are cold storage — only paged in when a block's INT8
certification fails (measured at 0-3% of blocks).

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

    # VRAM (hot) — INT8 quantised keys
    keys_int8: torch.Tensor          # [kv_heads, N, head_dim] int8, device=cuda
    keys_scale: torch.Tensor          # [kv_heads, num_blocks] float32, device=cuda
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

    # Pre-allocated VRAM buffer for page-in (avoid allocation on critical path)
    _pagein_buffer: torch.Tensor | None = None  # [max_pagein_blocks * block_size, head_dim] fp16 cuda

    # Pre-computed dequantised keys and float32 values (avoid per-call allocation)
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

        Quantises keys to INT8 on GPU, then moves FP16 originals to pinned CPU.
        Pre-allocates extra slots for max_new_tokens decode steps (zero-copy append).
        """
        kv_heads, N, head_dim = keys_fp16.shape
        d_v = values_fp16.shape[2]
        num_blocks = N // block_size
        device = keys_fp16.device
        capacity = N + max_new_tokens  # pre-allocate for decode

        # Reshape to blocks for per-block quantisation
        keys_blocked = keys_fp16.reshape(kv_heads, num_blocks, block_size, head_dim).to(torch.float32)

        # Per-block symmetric INT8 quantisation
        k_max = keys_blocked.abs().amax(dim=(2, 3)).clamp(min=1e-8)
        k_scale = k_max / 127.0
        keys_int8 = (
            (keys_blocked / k_scale[:, :, None, None])
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
            .reshape(kv_heads, N, head_dim)
            .contiguous()
        )

        # Conservative correction
        q_norm_est = (head_dim ** 0.5)
        q_scale_est = 1.0 / (head_dim ** 0.5)
        delta_per_block = q_norm_est * (k_scale * 2 / 255) / 2 * q_scale_est
        correction = torch.exp(3 * delta_per_block)

        # Move FP16 originals to pinned CPU memory
        keys_fp16_cpu = keys_fp16.to(dtype=torch.float16).cpu().pin_memory()

        # Values stay in VRAM as FP16
        values_fp16_cuda = values_fp16.to(dtype=torch.float16, device=device).contiguous()

        # Pre-allocate page-in buffer
        pagein_buffer = torch.empty(
            max_pagein_blocks * block_size, head_dim,
            dtype=torch.float16, device=device,
        )

        # Pre-allocate decode buffers (zero-copy append)
        max_new_blocks = (max_new_tokens + block_size - 1) // block_size
        keys_int8_buf = torch.zeros(kv_heads, capacity, head_dim, dtype=torch.int8, device=device)
        keys_int8_buf[:, :N, :] = keys_int8
        values_fp16_buf = torch.zeros(kv_heads, capacity, d_v, dtype=torch.float16, device=device)
        values_fp16_buf[:, :N, :] = values_fp16_cuda

        # Scale/correction buffers
        max_total_blocks = num_blocks + max_new_blocks
        scale_buf = torch.zeros(kv_heads, max_total_blocks, dtype=torch.float32, device=device)
        scale_buf[:, :num_blocks] = k_scale.to(torch.float32)
        corr_buf = torch.ones(kv_heads, max_total_blocks, dtype=torch.float32, device=device)
        corr_buf[:, :num_blocks] = correction.to(torch.float32)

        # CPU FP16 buffer for keys
        keys_fp16_cpu_buf = torch.zeros(kv_heads, capacity, head_dim, dtype=torch.float16, pin_memory=True)
        keys_fp16_cpu_buf[:, :N, :] = keys_fp16_cpu

        # Pre-compute dequant into buffer (avoids per-call materialisation)
        keys_deq_buf = torch.zeros(kv_heads, capacity, head_dim, dtype=torch.float32, device=device)
        keys_deq_buf[:, :N, :] = (
            keys_int8.to(torch.float32).reshape(kv_heads, num_blocks, block_size, head_dim)
            * k_scale.to(torch.float32)[:, :, None, None]
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
            _pagein_buffer=pagein_buffer,
            _keys_deq_f32=keys_deq_buf,
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

        Keys: INT8 in VRAM (same as v1)
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

    def append_token(
        self,
        key_fp16: torch.Tensor,    # [kv_heads, 1, head_dim] float16/32
        value_fp16: torch.Tensor,  # [kv_heads, 1, d_v] float16/32
    ) -> None:
        """Append one token to the cache. Zero-copy into pre-allocated buffers.

        The buffers were sized at creation time (max_new_tokens). This method
        just writes into the next slot — no allocation, no torch.cat.
        """
        pos = self.num_tokens
        device = self.keys_int8.device

        new_k = key_fp16.to(dtype=torch.float16).squeeze(1)  # [kv_heads, head_dim]
        new_v = value_fp16.to(dtype=torch.float16).squeeze(1)

        # Write FP16 value into pre-allocated VRAM buffer
        if self.values_fp16 is not None:
            self.values_fp16[:, pos, :] = new_v.to(device=device, dtype=torch.float16)

        # Write FP16 key into pre-allocated CPU buffer
        self.keys_fp16_cpu[:, pos, :] = new_k.cpu()

        if self.values_fp16_cpu is not None:
            self.values_fp16_cpu[:, pos, :] = new_v.cpu()

        # Quantise key to INT8 and write into pre-allocated VRAM buffer
        new_k_f32 = new_k.to(device=device, dtype=torch.float32)
        block_idx = pos // self.block_size

        if block_idx >= self.num_blocks:
            # First token of a new block — write fresh scale + correction
            # (subsequent tokens in this block reuse this scale)
            k_max = new_k_f32.abs().amax(dim=-1).clamp(min=1e-8)
            k_scale_new = k_max / 127.0
            self.keys_scale[:, block_idx] = k_scale_new
            q_norm_est = self.head_dim ** 0.5
            q_scale_est = 1.0 / (self.head_dim ** 0.5)
            delta = q_norm_est * (k_scale_new * 2 / 255) / 2 * q_scale_est
            self.correction[:, block_idx] = torch.exp(3 * delta)
            k_int8 = (new_k_f32 / k_scale_new.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
        else:
            k_scale = self.keys_scale[:, block_idx]
            k_int8 = (new_k_f32 / k_scale.unsqueeze(-1).clamp(min=1e-8)).round().clamp(-127, 127).to(torch.int8)

        self.keys_int8[:, pos, :] = k_int8

        # Update dequant buffer if it exists
        if self._keys_deq_f32 is not None:
            k_scale_for_deq = self.keys_scale[:, block_idx]
            self._keys_deq_f32[:, pos, :] = k_int8.to(torch.float32) * k_scale_for_deq.unsqueeze(-1)

        # Update counts
        self.num_tokens = pos + 1
        self.num_blocks = self.num_tokens // self.block_size

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
        Trailing unused slots in the partial block are zeros from pre-allocation."""
        n = self.aligned_tokens
        return self.keys_int8[:, :n, :]

    @property
    def active_blocks(self) -> int:
        """Number of blocks (including partial trailing block)."""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def keys_scale_active(self) -> torch.Tensor:
        return self.keys_scale[:, :self.active_blocks]

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
        self._keys_deq_f32 = (
            self.keys_int8.to(torch.float32).reshape(
                self.kv_heads, self.num_blocks, self.block_size, self.head_dim
            ) * self.keys_scale[:, :, None, None]
        ).reshape(self.kv_heads, self.num_tokens, self.head_dim).contiguous()
        self._values_f32 = self.values_fp16.to(torch.float32).contiguous()

    def vram_bytes(self) -> int:
        """Total VRAM usage."""
        total = self.keys_int8.nelement() * 1      # INT8
        total += self.keys_scale.nelement() * 4     # float32
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
) -> dict[int, TieredKeyCacheLayer]:
    """Create tiered caches from a HuggingFace model's past_key_values.

    Args:
        past_kv: DynamicCache or tuple of (K, V) per layer
        layer_ids: which layers to create tiered caches for
        block_size: tokens per block

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

        # Trim to block-aligned, build cache, then append the trailing tokens.
        # This ensures the scoring kernel only sees full blocks, while the
        # trailing tokens are still in the cache for attend.
        seq_len = keys.shape[1]
        aligned_len = (seq_len // block_size) * block_size
        keys_aligned = keys[:, :aligned_len, :].contiguous()
        values_aligned = values[:, :aligned_len, :].contiguous()

        cache = TieredKeyCacheLayer.from_fp16_cache(
            keys_aligned, values_aligned, block_size=block_size,
            max_new_tokens=512 + (seq_len - aligned_len),
        )

        # Append the trailing (non-block-aligned) tokens
        for t in range(aligned_len, seq_len):
            cache.append_token(
                keys[:, t:t+1, :],
                values[:, t:t+1, :],
            )

        # Poison padding positions so they get ~zero softmax weight.
        # Positions num_tokens..aligned_tokens are zero from pre-allocation.
        # Set all key representations to large negative values.
        at = cache.aligned_tokens
        nt = cache.num_tokens
        if at > nt:
            cache.keys_int8[:, nt:at, :] = -127
            cache.keys_fp16_cpu[:, nt:at, :] = -100.0
            if cache._keys_deq_f32 is not None:
                cache._keys_deq_f32[:, nt:at, :] = -1e4

        caches[layer_id] = cache
    return caches
