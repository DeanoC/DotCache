"""Tiered KV cache: INT8 keys in VRAM, FP16 originals in pinned CPU RAM.

The INT8 keys are the hot path — used for scoring in the fused kernel.
The FP16 originals are cold storage — only paged in when a block's INT8
certification fails (measured at 0-3% of blocks).

Values stay in VRAM as FP16 (needed for attend, no quantised alternative yet).
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

    @classmethod
    def from_fp16_cache(
        cls,
        keys_fp16: torch.Tensor,     # [kv_heads, N, head_dim] float16/32, device=cuda
        values_fp16: torch.Tensor,   # [kv_heads, N, d_v] float16/32, device=cuda
        block_size: int = 16,
        max_pagein_blocks: int = 64,
    ) -> "TieredKeyCacheLayer":
        """Create tiered cache from existing FP16 KV tensors.

        Quantises keys to INT8 on GPU, then moves FP16 originals to pinned CPU.
        """
        kv_heads, N, head_dim = keys_fp16.shape
        d_v = values_fp16.shape[2]
        num_blocks = N // block_size
        device = keys_fp16.device

        # Reshape to blocks for per-block quantisation
        keys_blocked = keys_fp16.reshape(kv_heads, num_blocks, block_size, head_dim).to(torch.float32)

        # Per-block symmetric INT8 quantisation
        k_max = keys_blocked.abs().amax(dim=(2, 3)).clamp(min=1e-8)  # [kv_heads, num_blocks]
        k_scale = k_max / 127.0
        keys_int8 = (
            (keys_blocked / k_scale[:, :, None, None])
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
            .reshape(kv_heads, N, head_dim)
            .contiguous()
        )

        # Compute correction factor: conservative per-block bound
        # Use a reference query (mean of centroids) to estimate max score error
        # For production: compute tighter per-head corrections at query time
        keys_deq = keys_int8.to(torch.float32).reshape(kv_heads, num_blocks, block_size, head_dim)
        keys_deq = keys_deq * k_scale[:, :, None, None]

        # Conservative correction: use ||q_typical|| * delta_step / 2 * 3
        # With per-block scales, delta_step = 2 * k_scale / 255
        q_norm_est = (head_dim ** 0.5)  # typical query norm
        q_scale_est = 1.0 / (head_dim ** 0.5)
        delta_per_block = q_norm_est * (k_scale * 2 / 255) / 2 * q_scale_est
        correction = torch.exp(3 * delta_per_block)  # [kv_heads, num_blocks]

        # Move FP16 originals to pinned CPU memory
        keys_fp16_cpu = keys_fp16.to(dtype=torch.float16).cpu().pin_memory()

        # Values stay in VRAM as FP16
        values_fp16_cuda = values_fp16.to(dtype=torch.float16, device=device).contiguous()

        # Pre-allocate page-in buffer
        pagein_buffer = torch.empty(
            max_pagein_blocks * block_size, head_dim,
            dtype=torch.float16, device=device,
        )

        return cls(
            keys_int8=keys_int8,
            keys_scale=k_scale.to(torch.float32),
            correction=correction.to(torch.float32),
            values_fp16=values_fp16_cuda,
            keys_fp16_cpu=keys_fp16_cpu,
            kv_heads=kv_heads,
            num_tokens=N,
            head_dim=head_dim,
            d_v=d_v,
            block_size=block_size,
            num_blocks=num_blocks,
            _pagein_buffer=pagein_buffer,
        )

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
        total += self.values_fp16.nelement() * 2    # float16
        if self._pagein_buffer is not None:
            total += self._pagein_buffer.nelement() * 2
        if self._keys_deq_f32 is not None:
            total += self._keys_deq_f32.nelement() * 4
        if self._values_f32 is not None:
            total += self._values_f32.nelement() * 4
        return total

    def cpu_bytes(self) -> int:
        """Total CPU pinned RAM usage."""
        return self.keys_fp16_cpu.nelement() * 2

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

        # Trim to block-aligned length
        seq_len = keys.shape[1]
        aligned_len = (seq_len // block_size) * block_size
        keys = keys[:, :aligned_len, :].contiguous()
        values = values[:, :aligned_len, :].contiguous()

        caches[layer_id] = TieredKeyCacheLayer.from_fp16_cache(
            keys, values, block_size=block_size,
        )
    return caches
