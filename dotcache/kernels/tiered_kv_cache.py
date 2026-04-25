"""Tiered KV cache: per-channel INT8 keys in VRAM, FP16 originals in pinned CPU RAM.

Keys are quantised per-channel: each of the head_dim channels gets its own
asymmetric INT8 scale + zero point within each block of block_size tokens
(paper §2.3, range [-128, 127]). This preserves resolution for low-magnitude
channels that per-block quantisation crushes, AND captures the per-channel
mean offset that symmetric quantisation wastes a code on.

Quantisation is deferred: trailing partial blocks (< block_size tokens) stay
in FP16 in VRAM.  When a block fills to block_size tokens, all tokens are
quantised together with per-channel scales computed from the full block.
The hybrid attend kernel handles the FP16 trailing block.

Two value storage modes:
  - FP16 values in VRAM (original, higher quality)
  - INT4 per-group values in VRAM (v2, ~38% less VRAM, mass-weighted safety)
"""
from __future__ import annotations

from collections import OrderedDict
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
    # Per-channel zero points (paper §2.3 asymmetric quant). Same shape as
    # keys_scale; values typically lie in [-128, 127]. Dequant: x = (q - z) * s.
    keys_zero_points: torch.Tensor    # [kv_heads, num_blocks, head_dim] float32
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
    values_int4_errors: torch.Tensor | None = None   # [kv_heads, num_blocks] mean relative L2 error, cuda
    values_int4_error_sums: torch.Tensor | None = None  # [kv_heads, num_blocks] running sums for append
    values_int4_error_counts: torch.Tensor | None = None  # [kv_heads, num_blocks] running counts for append
    values_int4_group_size: int = 16  # paper §7
    # Per-block max value-vector ℓ₂ norm: ν_b = max_{t∈b} ‖V_t‖₂ (paper §2.3
    # last paragraph). The Theorem-1 key-error bound uses V_max = max_b ν_b.
    # One float per block per kv-head; written at quant time and updated
    # incrementally by append_token. Required by §4.5 E_key telemetry.
    values_norm_max_per_block: torch.Tensor | None = None  # [kv_heads, num_blocks] float32, cuda

    # CPU warm tier — FP16 values for fallback when INT4 error too high
    values_fp16_cpu: torch.Tensor | None = None  # [kv_heads, N, d_v] float16, pinned CPU
    # Optional VRAM mirror/cache for Rung-2 FP16 values. In the INT4 paper
    # path this is fallback-only: values_fp16 remains None, so the fast path
    # still consumes INT4 values unless Rung-2 explicitly selects a block.
    # Full-mirror mode shape: [kv_heads, N, d_v].
    # Bounded-cache mode shape: [kv_heads, fp16_value_cache_capacity * B, d_v].
    values_fp16_gpu: torch.Tensor | None = None
    fp16_value_cache_capacity: int | None = None  # None = full mirror; int = bounded cache (# blocks)
    _fp16_value_resident: "OrderedDict[int, int]" = field(default_factory=OrderedDict)
    _fp16_value_free_slots: list[int] = field(default_factory=list)
    _fp16_value_cache_hits: int = 0
    _fp16_value_cache_misses: int = 0
    _fp16_value_cache_h2d_bytes: int = 0
    _fp16_value_cache_evictions: int = 0

    # VRAM-side FP16 key buffer. Two modes:
    #   - Full mirror (keys_fp16_gpu non-None, fp16_key_cache_capacity is None):
    #     legacy behaviour — all blocks pre-populated, ensure_fp16_keys_resident
    #     is a no-op. Not paper-faithful.
    #   - Bounded page cache (keys_fp16_gpu non-None used as scratch, capacity set):
    #     only `fp16_key_cache_capacity` blocks are valid at any time; LRU
    #     eviction. This IS the paper's tiered architecture — FP16 lives on
    #     CPU pinned RAM (keys_fp16_cpu), the VRAM buffer is a transparent
    #     cache that reduces H2D via locality.
    keys_fp16_gpu: torch.Tensor | None = None    # [kv_heads, N, head_dim] model dtype, cuda
    fp16_key_cache_capacity: int | None = None   # None = full mirror; int = bounded cache (# blocks)
    # OrderedDict serves double duty as both the residency set (membership
    # test via `bid in od`) and the LRU order (tail = most recently used;
    # front = next LRU victim). All operations are O(1):
    #   MRU bump:    od.move_to_end(bid, last=True)
    #   insert MRU:  od[bid] = True
    #   evict LRU:   od.popitem(last=False)
    # The caller controls insertion order to implement priority-ordered
    # eviction: insert low-priority blocks first (they land toward the
    # LRU-front and get evicted first); insert high-priority blocks last
    # (they stay at the MRU-tail and survive longer).
    _fp16_key_resident: "OrderedDict[int, bool]" = field(default_factory=OrderedDict)
    _fp16_key_cache_hits: int = 0
    _fp16_key_cache_misses: int = 0
    _fp16_key_cache_h2d_bytes: int = 0
    _fp16_key_cache_evictions: int = 0

    @classmethod
    def from_fp16_cache(
        cls,
        keys_fp16: torch.Tensor,     # [kv_heads, N, head_dim] float16/32, device=cuda
        values_fp16: torch.Tensor,   # [kv_heads, N, d_v] float16/32, device=cuda
        block_size: int = 16,
        max_pagein_blocks: int = 64,
        max_new_tokens: int = 512,
        fp16_key_cache_capacity: int | None = None,
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

        # Per-channel ASYMMETRIC INT8 quantisation (paper §2.3 Eq. 1):
        #   q   = clamp(round((k - z) / s), -128, 127)
        #   k̂  = q · s + z
        # where z is a real-valued offset in fp space (NOT constrained to
        # the integer range — that's the standard "fp zero point" convention).
        # We pick z = midpoint of [k_min, k_max] to centre the quant grid on
        # the channel's data, and s = range / 255 to spend all 256 codes on
        # the actual span. For data shifted far from origin (positive-only
        # activations) this preserves precision that symmetric quant wastes.
        k_min = keys_blocked.amin(dim=2)
        k_max = keys_blocked.amax(dim=2)
        k_range = (k_max - k_min).clamp(min=1e-8)  # degenerate-channel guard
        k_scale = k_range / 255.0
        k_zero_points = (k_min + k_max) / 2.0  # fp-space midpoint
        keys_int8 = (
            ((keys_blocked - k_zero_points[:, :, None, :]) / k_scale[:, :, None, :])
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
            .reshape(kv_heads, N, head_dim)
            .contiguous()
        )

        # Correction factor uses L2 norm of per-channel scale vector.
        # Per-channel ULP = scale (one quant level). Per-element max error = scale/2.
        # Per-token L2 error bound = ||scale||_2 / 2 (worst-case all channels at half-ULP).
        # Asymmetric quant has tighter scales than symmetric for the same data
        # (255 levels vs 127), so the same formula yields a smaller delta.
        # The /127 divisor preserves the existing score-error proportionality
        # used by the certification kernel (see correction_active() consumers).
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

        # Scale/zero-point/correction buffers — per-channel: [kv_heads, max_blocks, head_dim]
        max_total_blocks = num_blocks + max_new_blocks
        scale_buf = torch.zeros(kv_heads, max_total_blocks, head_dim, dtype=torch.float32, device=device)
        scale_buf[:, :num_blocks, :] = k_scale.to(torch.float32)
        zp_buf = torch.zeros(kv_heads, max_total_blocks, head_dim, dtype=torch.float32, device=device)
        zp_buf[:, :num_blocks, :] = k_zero_points.to(torch.float32)
        corr_buf = torch.ones(kv_heads, max_total_blocks, dtype=torch.float32, device=device)
        corr_buf[:, :num_blocks] = correction.to(torch.float32)

        # Per-block max value-norm ν_b (paper §2.3) — required by Theorem-1
        # key-error bound and the §4.5 E_key telemetry contract.
        # values_fp16: [kv_heads, N, d_v]. Per-token ‖V_t‖₂, then per-block max.
        v_norm_per_token = values_fp16.to(torch.float32).norm(dim=-1)         # [kv_heads, N]
        v_norm_per_block = (
            v_norm_per_token.reshape(kv_heads, num_blocks, block_size).amax(dim=-1)
        )                                                                      # [kv_heads, num_blocks]
        v_norm_buf = torch.zeros(kv_heads, max_total_blocks, dtype=torch.float32, device=device)
        v_norm_buf[:, :num_blocks] = v_norm_per_block

        # CPU buffer for keys (preserve model dtype)
        keys_fp16_cpu_buf = torch.zeros(kv_heads, capacity, head_dim, dtype=kv_dtype, pin_memory=True)
        keys_fp16_cpu_buf[:, :N, :] = keys_fp16_cpu

        # VRAM scratch for the FP16 key cache. Two modes:
        #   - fp16_key_cache_capacity is None → legacy full mirror: pre-populated
        #     with every prefill block, zero H2D during decode.
        #   - fp16_key_cache_capacity = K → paper-matching bounded cache: the
        #     scratch is allocated but NOT pre-populated; ensure_fp16_keys_resident
        #     fetches blocks from keys_fp16_cpu on demand via H2D, evicting LRU
        #     once K blocks are resident.
        keys_fp16_gpu_buf = torch.zeros(kv_heads, capacity, head_dim, dtype=kv_dtype, device=device)
        if fp16_key_cache_capacity is None:
            keys_fp16_gpu_buf[:, :N, :] = keys_fp16.to(dtype=kv_dtype, device=device)

        # Pre-compute dequant into buffer (paper §2.3: k̂ = q · s + z)
        # Per-channel: k_scale and k_zero_points are [kv_heads, num_blocks, head_dim],
        # broadcast over token dim.
        keys_deq_buf = torch.zeros(kv_heads, capacity, head_dim, dtype=torch.float32, device=device)
        keys_deq_buf[:, :N, :] = (
            keys_int8.to(torch.float32).reshape(kv_heads, num_blocks, block_size, head_dim)
            * k_scale.to(torch.float32)[:, :, None, :]
            + k_zero_points.to(torch.float32)[:, :, None, :]
        ).reshape(kv_heads, N, head_dim)

        result = cls(
            keys_int8=keys_int8_buf,
            keys_scale=scale_buf,
            keys_zero_points=zp_buf,
            correction=corr_buf,
            values_norm_max_per_block=v_norm_buf,
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
            fp16_key_cache_capacity=fp16_key_cache_capacity,
        )
        # Legacy full-mirror mode: mark every prefill block as resident so
        # ensure_fp16_keys_resident short-circuits to all-hit.
        if fp16_key_cache_capacity is None:
            result._fp16_key_resident = OrderedDict((i, True) for i in range(num_blocks))
        return result

    @classmethod
    def from_fp16_cache_int4v(
        cls,
        keys_fp16: torch.Tensor,     # [kv_heads, N, head_dim] float16/32, device=cuda
        values_fp16: torch.Tensor,   # [kv_heads, N, d_v] float16/32, device=cuda
        block_size: int = 16,
        group_size: int = 16,  # paper §7
        max_pagein_blocks: int = 64,
        max_new_tokens: int = 512,
        fp16_key_cache_capacity: int | None = None,
        fp16_value_cache_capacity: int | None = None,
    ) -> "TieredKeyCacheLayer":
        """Create tiered cache with INT4 per-group values.

        Keys: per-channel INT8 in VRAM (same as v1)
        Values: INT4 per-group in VRAM (NEW — saves ~38% vs FP16)
        FP16 originals: pinned CPU (both K and V)

        max_new_tokens reserves growth room in the INT4 buffers so decode-time
        append_token() can quantise and write new tokens without reallocation.
        Without this the audit's "INT4 path has never run end-to-end" risk
        manifests as a shape mismatch in the kernel reshape.
        """
        from dotcache.kernels.int4_group_quantise import quantise_int4_grouped_block

        # Build the base cache (INT8 keys, FP16 values)
        base = cls.from_fp16_cache(
            keys_fp16, values_fp16, block_size, max_pagein_blocks,
            max_new_tokens=max_new_tokens,
            fp16_key_cache_capacity=fp16_key_cache_capacity,
        )

        # Quantise prefill values to INT4 per-group
        int4_result = quantise_int4_grouped_block(
            values_fp16.to(torch.float16), block_size=block_size, group_size=group_size,
        )

        kv_heads, N, d_v = values_fp16.shape
        capacity = base.keys_int8.shape[1]  # matches keys_int8 buffer capacity
        num_groups = d_v // group_size
        num_total_blocks = base.keys_scale.shape[1]
        device = values_fp16.device

        # Pre-allocate INT4 buffers sized to the same capacity as keys_int8 so
        # append_token can write new tokens at decode time.
        int4_packed_buf = torch.zeros(
            kv_heads, capacity, d_v // 2, dtype=torch.uint8, device=device,
        )
        int4_packed_buf[:, :N, :] = int4_result["data_packed"].contiguous()
        int4_scales_buf = torch.zeros(
            kv_heads, capacity, num_groups, dtype=torch.float16, device=device,
        )
        int4_scales_buf[:, :N, :] = int4_result["scales"].contiguous()
        int4_zeros_buf = torch.zeros(
            kv_heads, capacity, num_groups, dtype=torch.float16, device=device,
        )
        int4_zeros_buf[:, :N, :] = int4_result["zeros"].contiguous()
        int4_errors_buf = torch.zeros(
            kv_heads, num_total_blocks, dtype=torch.float32, device=device,
        )
        int4_errors_buf[:, :N // block_size] = int4_result["error_bounds"].contiguous()
        int4_error_sums_buf = torch.zeros(
            kv_heads, num_total_blocks, dtype=torch.float32, device=device,
        )
        int4_error_sums_buf[:, :N // block_size] = int4_result["error_sums"].contiguous()
        int4_error_counts_buf = torch.zeros(
            kv_heads, num_total_blocks, dtype=torch.int32, device=device,
        )
        int4_error_counts_buf[:, :N // block_size] = block_size

        base.values_int4_packed = int4_packed_buf
        base.values_int4_scales = int4_scales_buf
        base.values_int4_zeros = int4_zeros_buf
        base.values_int4_errors = int4_errors_buf
        base.values_int4_error_sums = int4_error_sums_buf
        base.values_int4_error_counts = int4_error_counts_buf
        base.values_int4_group_size = group_size

        # Pre-allocate CPU pinned FP16 buffer at capacity so append_token can
        # also write the FP16 ground truth (needed for Rung-2 fallback).
        values_fp16_cpu_buf = torch.zeros(
            kv_heads, capacity, d_v,
            dtype=base.values_fp16.dtype, pin_memory=True,
        )
        values_fp16_cpu_buf[:, :N, :] = base.values_fp16[:, :N, :].cpu()
        base.values_fp16_cpu = values_fp16_cpu_buf

        if fp16_value_cache_capacity is None:
            # Legacy/full-scratch mode: keep a fallback-only FP16 value mirror
            # in VRAM. This is not the normal value path (`values_fp16` is
            # cleared below); the mixed-value kernel reads it only for Rung-2
            # promoted blocks.
            base.values_fp16_gpu = base.values_fp16
        else:
            # Paper sweep mode: bounded compact VRAM scratch for Rung-2 FP16
            # values. Blocks are paged from pinned Tier-2 CPU RAM on demand
            # and retained with LRU replacement.
            cap = max(int(fp16_value_cache_capacity), 0)
            base.values_fp16_gpu = torch.empty(
                kv_heads,
                cap * block_size,
                d_v,
                dtype=base.values_fp16.dtype,
                device=device,
            )
            base.fp16_value_cache_capacity = cap
            base._fp16_value_free_slots = list(range(cap))

        # Clear the normal FP16 value path; INT4 replaces it for Phase 2.
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

        # In full-mirror mode the appended FP16 keys are already resident on
        # GPU, so quantise completed decode blocks from that mirror and avoid
        # forcing the hot path through pinned CPU. Bounded-cache mode still
        # uses the CPU tier because evicted blocks must be pageable later.
        if self.keys_fp16_gpu is not None and self.fp16_key_cache_capacity is None:
            keys_block = self.keys_fp16_gpu[:, start:end, :].to(dtype=torch.float32)
        else:
            keys_block = self.keys_fp16_cpu[:, start:end, :].to(
                device=device, dtype=torch.float32
            )

        # Per-channel ASYMMETRIC INT8 (paper §2.3 Eq. 1):
        #   q = clamp(round((k - z) / s), -128, 127),  k̂ = q · s + z
        # z is the fp-space midpoint of the channel's range.
        k_min = keys_block.amin(dim=1)  # [kv_heads, head_dim]
        k_max = keys_block.amax(dim=1)
        k_range = (k_max - k_min).clamp(min=1e-8)
        k_scale = k_range / 255.0
        k_zp = (k_min + k_max) / 2.0  # fp-space midpoint

        # Quantize all block_size tokens at once
        k_int8 = (
            ((keys_block - k_zp[:, None, :]) / k_scale[:, None, :])
            .round().clamp(-128, 127).to(torch.int8)
        )  # [kv_heads, block_size, head_dim]

        # Write INT8 keys + per-channel scale + zero point to VRAM
        self.keys_int8[:, start:end, :] = k_int8
        self.keys_scale[:, block_idx, :] = k_scale
        self.keys_zero_points[:, block_idx, :] = k_zp

        # Correction factor: delta = ||k_scale||_2 / 127 (proportionality preserved
        # from the symmetric-era formula; asymmetric scale is smaller for the same
        # data so delta is also smaller — tighter bound).
        k_scale_l2 = k_scale.norm(dim=-1)  # [kv_heads]
        delta = k_scale_l2 / 127.0
        self.correction[:, block_idx] = torch.exp(2.0 * delta)

        # Register the newly-completed block in the FP16 VRAM cache. Its
        # FP16 data was already written to keys_fp16_gpu at the right
        # offset during append_token; we just need to mark it resident and
        # (in bounded-capacity mode) evict an LRU victim to make room.
        if self.fp16_key_cache_capacity is not None and block_idx not in self._fp16_key_resident:
            if len(self._fp16_key_resident) >= int(self.fp16_key_cache_capacity):
                if self._fp16_key_resident:
                    self._fp16_key_resident.popitem(last=False)  # LRU = front
                    self._fp16_key_cache_evictions += 1
            # New block lands at MRU (tail). It's semantically "most recent"
            # since we just wrote its INT8+FP16 data.
            self._fp16_key_resident[int(block_idx)] = True

        # Update dequant buffer with INT8-dequantised values (replaces the exact FP16)
        # Paper §2.3: k̂ = q · s + z
        if self._keys_deq_f32 is not None:
            self._keys_deq_f32[:, start:end, :] = (
                k_int8.to(torch.float32) * k_scale[:, None, :] + k_zp[:, None, :]
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
        if self.values_fp16 is not None:
            kv_dtype = self.values_fp16.dtype
        elif self.values_fp16_cpu is not None:
            kv_dtype = self.values_fp16_cpu.dtype
        else:
            kv_dtype = torch.float16
        block_idx = pos // self.block_size
        pos_in_block = pos % self.block_size

        new_k = key_fp16.to(dtype=kv_dtype).squeeze(1)  # [kv_heads, head_dim]
        new_v = value_fp16.to(dtype=kv_dtype).squeeze(1)

        # Per-block max value-norm ν_b update (paper §2.3). Take max with
        # the existing block annotation so partial-block appends correctly
        # accumulate. Required by §4.5 E_key telemetry.
        if self.values_norm_max_per_block is not None:
            new_v_norm_per_head = new_v.to(device=device, dtype=torch.float32).norm(dim=-1)  # [kv_heads]
            self.values_norm_max_per_block[:, block_idx] = torch.maximum(
                self.values_norm_max_per_block[:, block_idx],
                new_v_norm_per_head,
            )

        # Write value into pre-allocated VRAM buffer (preserves model dtype)
        if self.values_fp16 is not None:
            self.values_fp16[:, pos, :] = new_v.to(device=device, dtype=kv_dtype)

        # INT4-values path: quantise the new token and write to the INT4 buffers.
        # Group quant is per-TOKEN (each token's d_v split into d_v/g groups,
        # each group's scale/zero from that token's own values) — no need to
        # wait for the block to fill, unlike keys (which need per-channel
        # absmax over the full block to compute per-channel scale).
        # The kernel reads N from keys_int8.shape[1], so values_int4_* must
        # grow in lockstep with keys (paper §3.1 / §7).
        # We also update the per-block error annotation η_b incrementally:
        # η_b = mean relative ℓ₂ reconstruction error across the block's
        # tokens. This is the dimensionless paper §7 value-tolerance scale
        # (~0.05 at g=16). Without this, decide_v_format_tight
        # silently under-estimates per-block error for blocks containing
        # appended tokens, biasing the Rung-2 decision toward INT4.
        if self.values_int4_packed is not None:
            from dotcache.kernels.int4_group_quantise import quantise_int4_grouped
            new_v_per_head = new_v.to(device=device)  # [kv_heads, d_v]
            r = quantise_int4_grouped(
                new_v_per_head,
                group_size=self.values_int4_group_size,
            )
            self.values_int4_packed[:, pos:pos+1, :] = r["data_packed"].unsqueeze(1)
            self.values_int4_scales[:, pos:pos+1, :] = r["scales"].unsqueeze(1)
            self.values_int4_zeros[:, pos:pos+1, :] = r["zeros"].unsqueeze(1)
            # Update η_b incrementally on device — keep mean over block tokens.
            err = r["per_token_error"].to(dtype=self.values_int4_errors.dtype)
            if self.values_int4_error_sums is not None and self.values_int4_error_counts is not None:
                self.values_int4_error_sums[:, block_idx] += err
                self.values_int4_error_counts[:, block_idx] += 1
                counts = self.values_int4_error_counts[:, block_idx].to(dtype=self.values_int4_errors.dtype)
                self.values_int4_errors[:, block_idx] = (
                    self.values_int4_error_sums[:, block_idx] / counts.clamp(min=1)
                )
            else:
                self.values_int4_errors[:, block_idx] = torch.maximum(
                    self.values_int4_errors[:, block_idx],
                    err,
                )

        # Mirror exact K/V into pinned Tier-2 buffers only when the bounded
        # cache may need CPU page-in later. Full-mirror quality runs keep the
        # authoritative decode-time FP16 copy in VRAM and quantise completed
        # blocks from that mirror.
        if not (
            self.keys_fp16_gpu is not None
            and self.fp16_key_cache_capacity is None
        ):
            self.keys_fp16_cpu[:, pos, :].copy_(new_k, non_blocking=True)

        # Mirror into GPU key buffer so decode-time attend avoids a CPU→GPU copy
        if self.keys_fp16_gpu is not None:
            self.keys_fp16_gpu[:, pos, :] = new_k.to(device=device, dtype=kv_dtype)

        if (
            self.values_fp16_cpu is not None
            and not (
                self.values_fp16_gpu is not None
                and self.fp16_value_cache_capacity is None
            )
        ):
            self.values_fp16_cpu[:, pos, :].copy_(new_v, non_blocking=True)
        if self.values_fp16_gpu is not None and self.fp16_value_cache_capacity is None:
            self.values_fp16_gpu[:, pos, :] = new_v.to(device=device, dtype=kv_dtype)
        elif (
            self.values_fp16_gpu is not None
            and self.fp16_value_cache_capacity is not None
            and block_idx in self._fp16_value_resident
        ):
            slot = self._fp16_value_resident[block_idx]
            dst = slot * self.block_size + pos_in_block
            self.values_fp16_gpu[:, dst, :] = new_v.to(device=device, dtype=kv_dtype)

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

    def keys_zero_points_active(self) -> torch.Tensor:
        """Per-channel zero points for active blocks: [kv_heads, active_blocks, head_dim].

        Paper §2.3 asymmetric INT8 quant. Dequant: x = (q - z) * s.
        """
        return self.keys_zero_points[:, :self.active_blocks, :]

    def v_max_global(self) -> float:
        """V_max = max_b ν_b across all active blocks and KV-heads (paper §2.3 / §4.5).

        Returns 0.0 if no per-block annotation exists (legacy caches that
        bypassed from_fp16_cache). Used by the §4.5 E_key telemetry.
        """
        if self.values_norm_max_per_block is None:
            return 0.0
        active = self.values_norm_max_per_block[:, :self.active_blocks]
        if active.numel() == 0:
            return 0.0
        return float(active.max().item())

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
        # Quantised blocks: dequant INT8 with per-channel scale + zero point
        # (paper §2.3 Eq. 1: k̂ = q · s + z)
        if nq > 0:
            self._keys_deq_f32[:, :qt, :] = (
                self.keys_int8[:, :qt, :].to(torch.float32)
                    .reshape(self.kv_heads, nq, self.block_size, self.head_dim)
                * self.keys_scale[:, :nq, None, :]
                + self.keys_zero_points[:, :nq, None, :]
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
        total += self.keys_zero_points.nelement() * 4  # float32 per-channel zero points
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
            if self.values_int4_error_sums is not None:
                total += self.values_int4_error_sums.nelement() * 4
            if self.values_int4_error_counts is not None:
                total += self.values_int4_error_counts.nelement() * 4
        if self.values_fp16_gpu is not None:
            total += self.values_fp16_gpu.nelement() * self.values_fp16_gpu.element_size()
        return total

    def cpu_bytes(self) -> int:
        """Total CPU pinned RAM usage."""
        total = self.keys_fp16_cpu.nelement() * 2
        if self.values_fp16_cpu is not None:
            total += self.values_fp16_cpu.nelement() * 2
        return total

    def ensure_fp16_keys_resident(
        self,
        block_ids,  # iterable of int block indices needing FP16 data
    ) -> tuple[int, int, int, int]:
        """Bring `block_ids` into the bounded FP16 VRAM key cache.

        No-op when `fp16_key_cache_capacity is None` (full mirror mode): all
        blocks are already in `keys_fp16_gpu`.

        Cache mode: per-block check residency.
          - hit:  the block's FP16 data is already at the correct offset in
                  keys_fp16_gpu; bump LRU.
          - miss: H2D copy from keys_fp16_cpu[:, b*bs : (b+1)*bs, :] into
                  keys_fp16_gpu at the same offset. Evict LRU victim if the
                  cache is at capacity. Record bytes transferred.

        Returns (hits, misses, bytes, evictions) for this call so the
        attention path can roll per-step telemetry.
        """
        if self.fp16_key_cache_capacity is None or self.keys_fp16_gpu is None:
            return (0, 0, 0, 0)

        bs = self.block_size
        el = self.keys_fp16_gpu.element_size()
        bytes_per_block = self.kv_heads * bs * self.head_dim * el
        device = self.keys_fp16_gpu.device

        hits = 0
        misses = 0
        h2d_bytes = 0
        evictions = 0

        # Dedup while preserving order of first occurrence.
        seen: set = set()
        ordered_ids: list = []
        for b in block_ids:
            bi = int(b)
            if bi in seen:
                continue
            seen.add(bi)
            ordered_ids.append(bi)

        capacity = int(self.fp16_key_cache_capacity)

        # Special case: capacity == 0 is the "no cache" floor used for the
        # capacity sweep. Every access is a miss, data is H2D'd into the
        # scratch at the right offset so the kernel reads correct bytes this
        # step, and nothing is retained — next step re-pages the same blocks.
        if capacity == 0:
            for bid in ordered_ids:
                start = bid * bs
                end = start + bs
                if end > self.num_tokens:
                    end = self.num_tokens
                if end > start and self.keys_fp16_cpu is not None:
                    src = self.keys_fp16_cpu[:, start:end, :]
                    self.keys_fp16_gpu[:, start:end, :] = src.to(
                        device=device, non_blocking=True
                    )
                    h2d_bytes += self.kv_heads * (end - start) * self.head_dim * el
                    misses += 1
            self._fp16_key_cache_hits += hits
            self._fp16_key_cache_misses += misses
            self._fp16_key_cache_h2d_bytes += h2d_bytes
            self._fp16_key_cache_evictions += evictions
            return (hits, misses, h2d_bytes, evictions)

        for bid in ordered_ids:
            if bid in self._fp16_key_resident:
                hits += 1
                # Bump to MRU (tail). O(1) with OrderedDict.
                self._fp16_key_resident.move_to_end(bid, last=True)
                continue

            # Miss — evict LRU victim if cache is full.
            if len(self._fp16_key_resident) >= capacity:
                if self._fp16_key_resident:
                    self._fp16_key_resident.popitem(last=False)  # LRU = front
                    evictions += 1

            # H2D copy: CPU pinned → GPU scratch at the block's offset. The
            # offset is (bid*bs : (bid+1)*bs) along the token dim, for all
            # kv_heads and head_dim channels.
            start = bid * bs
            end = start + bs
            if end > self.num_tokens:
                end = self.num_tokens  # trailing partial block guard
            if end > start and self.keys_fp16_cpu is not None:
                src = self.keys_fp16_cpu[:, start:end, :]
                self.keys_fp16_gpu[:, start:end, :] = src.to(
                    device=device, non_blocking=True
                )
                h2d_bytes += self.kv_heads * (end - start) * self.head_dim * el
                misses += 1
                # Newly loaded block lands at MRU. Caller-controlled ordering
                # means high-priority ids arrive last → end up at tail → survive
                # longer; low-priority ids arrive first → evicted sooner.
                self._fp16_key_resident[bid] = True

        self._fp16_key_cache_hits += hits
        self._fp16_key_cache_misses += misses
        self._fp16_key_cache_h2d_bytes += h2d_bytes
        self._fp16_key_cache_evictions += evictions
        return (hits, misses, h2d_bytes, evictions)

    def ensure_fp16_values_resident(
        self,
        block_ids,  # iterable of int block indices needing FP16 value data
    ) -> tuple[dict[int, int] | None, int, int, int, int]:
        """Bring value blocks into the bounded compact FP16 VRAM cache.

        Returns (block_to_slot, hits, misses, h2d_bytes, evictions). When the
        cache is disabled/full-mirror mode this returns a block->block mapping.
        When the bounded cache cannot hold the whole simultaneous working set,
        block_to_slot is None; callers must use the one-step compact page-in
        path for correctness.
        """
        if self.values_fp16_gpu is None:
            return (None, 0, 0, 0, 0)

        bs = self.block_size

        seen: set[int] = set()
        ordered_ids: list[int] = []
        for b in block_ids:
            bi = int(b)
            if bi in seen:
                continue
            seen.add(bi)
            ordered_ids.append(bi)

        if self.fp16_value_cache_capacity is None:
            return ({bid: bid for bid in ordered_ids}, 0, 0, 0, 0)

        capacity = int(self.fp16_value_cache_capacity)
        if capacity <= 0 or len(ordered_ids) > capacity:
            return (None, 0, 0, 0, 0)

        el = self.values_fp16_gpu.element_size()
        device = self.values_fp16_gpu.device
        hits = 0
        misses = 0
        h2d_bytes = 0
        evictions = 0

        for bid in ordered_ids:
            if bid in self._fp16_value_resident:
                hits += 1
                self._fp16_value_resident.move_to_end(bid, last=True)
                continue

            if self._fp16_value_free_slots:
                slot = self._fp16_value_free_slots.pop(0)
            else:
                _, slot = self._fp16_value_resident.popitem(last=False)
                evictions += 1

            start = bid * bs
            end = min(start + bs, self.num_tokens)
            if end > start and self.values_fp16_cpu is not None:
                dst_start = slot * bs
                dst_end = dst_start + (end - start)
                src = self.values_fp16_cpu[:, start:end, :]
                self.values_fp16_gpu[:, dst_start:dst_end, :] = src.to(
                    device=device,
                    non_blocking=True,
                )
                h2d_bytes += self.kv_heads * (end - start) * self.d_v * el
            misses += 1
            self._fp16_value_resident[bid] = slot

        self._fp16_value_cache_hits += hits
        self._fp16_value_cache_misses += misses
        self._fp16_value_cache_h2d_bytes += h2d_bytes
        self._fp16_value_cache_evictions += evictions
        return (dict(self._fp16_value_resident), hits, misses, h2d_bytes, evictions)

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
    group_size: int = 16,  # paper §7
    max_new_tokens: int = 512,
    fp16_key_cache_capacity: int | None = None,
    fp16_value_cache_capacity: int | None = None,
) -> dict[int, TieredKeyCacheLayer]:
    """Create tiered caches with INT4 per-group values from HF past_key_values.

    max_new_tokens reserves growth room in the INT4 buffers so decode-time
    append_token() can quantise additional tokens without overflowing the
    per-block annotation tensors (values_norm_max_per_block, values_int4_errors).
    Callers that decode more than the default 512 tokens MUST pass this
    explicitly, matching what they pass to create_tiered_cache_from_model().
    """
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

        # Constructor requires N % block_size == 0; pass the aligned slices.
        # (Prior code computed the aligned tensors then passed the unaligned
        # originals, which blew up on any seq_len not a multiple of block_size.)
        # The trailing (seq_len - aligned_len) FP16 tokens weren't quantised
        # into the INT4 prefill buffers, so they'll be re-appended via
        # append_token() — add their count to the reservation so the buffer
        # has room for them plus the max_new_tokens decode budget.
        caches[layer_id] = TieredKeyCacheLayer.from_fp16_cache_int4v(
            keys_aligned, values_aligned,
            block_size=block_size, group_size=group_size,
            max_new_tokens=max_new_tokens + (seq_len - aligned_len),
            fp16_key_cache_capacity=fp16_key_cache_capacity,
            fp16_value_cache_capacity=fp16_value_cache_capacity,
        )
        # num_tokens tracks what the INT4 packed tensor actually covers;
        # the trailing (seq_len - aligned_len) tokens weren't quantised and
        # don't exist in the INT4 buffers, so exposing num_tokens = seq_len
        # would tell downstream kernels to read past the end of the data.
        caches[layer_id].num_tokens = aligned_len
    return caches


def create_tiered_cache_from_model(
    past_kv,
    layer_ids: list[int],
    block_size: int = 16,
    max_new_tokens: int = 512,
    fp16_key_cache_capacity: int | None = None,
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
            fp16_key_cache_capacity=fp16_key_cache_capacity,
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
