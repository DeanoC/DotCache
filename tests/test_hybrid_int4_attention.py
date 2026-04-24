from __future__ import annotations

import pytest
import torch


CUDA = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_hybrid_int4_attention_matches_torch_reference():
    from dotcache.kernels.int4_group_quantise import dequantise_int4_grouped
    from dotcache.kernels.selective_attend_triton import (
        selective_attend_multihead_hybrid_int4v,
    )
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    torch.manual_seed(20260424)
    kv_heads, q_heads, gqa_group = 2, 4, 2
    n_blocks, block_size, head_dim, d_v = 4, 16, 32, 32
    n_tokens = n_blocks * block_size
    q_scale = 1.0 / (head_dim ** 0.5)

    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys, values, block_size=block_size, group_size=16, max_new_tokens=0,
    )
    q = torch.randn(q_heads, head_dim, dtype=torch.float32, device="cuda")

    # Mixed per-head promotion pattern: every head has at least one FP16-key
    # block and at least one INT8-tail block. This is the paper path that the
    # old INT4 branch did not exercise.
    topk_mask = torch.zeros(q_heads, n_blocks, dtype=torch.int32, device="cuda")
    topk_mask[0, [0, 2]] = 1
    topk_mask[1, [1]] = 1
    topk_mask[2, [2, 3]] = 1
    topk_mask[3, [0, 3]] = 1
    no_skip = torch.zeros_like(topk_mask)

    got = selective_attend_multihead_hybrid_int4v(
        keys_int8=cache.keys_int8[:, :n_tokens, :],
        keys_scale=cache.keys_scale[:, :n_blocks, :],
        keys_zero_points=cache.keys_zero_points[:, :n_blocks, :],
        keys_fp16=cache.keys_fp16_gpu[:, :n_tokens, :],
        topk_mask=topk_mask,
        values_int4_packed=cache.values_int4_packed[:, :n_tokens, :],
        values_int4_scales=cache.values_int4_scales[:, :n_tokens, :],
        values_int4_zeros=cache.values_int4_zeros[:, :n_tokens, :],
        q_all=q,
        skip_mask_i32=no_skip,
        gqa_group=gqa_group,
        block_size=block_size,
        group_size=16,
        q_scale=q_scale,
    )

    q_int8 = cache.keys_int8[:, :n_tokens, :].to(torch.float32).reshape(
        kv_heads, n_blocks, block_size, head_dim,
    )
    scales = cache.keys_scale[:, :n_blocks, :].unsqueeze(2)
    zero_points = cache.keys_zero_points[:, :n_blocks, :].unsqueeze(2)
    keys_deq = (q_int8 * scales + zero_points).reshape(kv_heads, n_tokens, head_dim)
    values_deq = torch.stack(
        [
            dequantise_int4_grouped(
                cache.values_int4_packed[h, :n_tokens, :],
                cache.values_int4_scales[h, :n_tokens, :],
                cache.values_int4_zeros[h, :n_tokens, :],
                16,
            )
            for h in range(kv_heads)
        ],
        dim=0,
    ).to(torch.float32)

    expected = torch.empty_like(got)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        k_blocks = []
        for bid in range(n_blocks):
            start = bid * block_size
            end = start + block_size
            if int(topk_mask[qh, bid].item()):
                k_blocks.append(keys[kvh, start:end, :].to(torch.float32))
            else:
                k_blocks.append(keys_deq[kvh, start:end, :])
        k_ref = torch.cat(k_blocks, dim=0)
        scores = (k_ref @ q[qh]) * q_scale
        weights = torch.softmax(scores, dim=0)
        expected[qh] = weights @ values_deq[kvh]

    torch.testing.assert_close(got, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_hybrid_mixed_value_attention_promotes_only_masked_blocks():
    from dotcache.kernels.int4_group_quantise import dequantise_int4_grouped
    from dotcache.kernels.selective_attend_triton import (
        selective_attend_multihead_hybrid_mixedv,
    )
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    torch.manual_seed(20260425)
    kv_heads, q_heads, gqa_group = 2, 4, 2
    n_blocks, block_size, head_dim, d_v = 4, 16, 32, 32
    n_tokens = n_blocks * block_size
    q_scale = 1.0 / (head_dim ** 0.5)

    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys, values, block_size=block_size, group_size=16, max_new_tokens=0,
    )
    q = torch.randn(q_heads, head_dim, dtype=torch.float32, device="cuda")

    topk_mask = torch.zeros(q_heads, n_blocks, dtype=torch.int32, device="cuda")
    topk_mask[0, [0, 2]] = 1
    topk_mask[1, [1]] = 1
    topk_mask[2, [2, 3]] = 1
    topk_mask[3, [0, 3]] = 1
    no_skip = torch.zeros_like(topk_mask)

    value_fp16_mask = torch.zeros(q_heads, n_blocks, dtype=torch.int32, device="cuda")
    value_fp16_mask[0, [1]] = 1
    value_fp16_mask[1, [1, 3]] = 1
    value_fp16_mask[2, [0]] = 1
    value_fp16_mask[3, [3]] = 1
    fallback_blocks = [0, 1, 3]
    value_block_slots = torch.full((n_blocks,), -1, dtype=torch.int32, device="cuda")
    for slot, bid in enumerate(fallback_blocks):
        value_block_slots[bid] = slot
    values_fp16_scratch = torch.empty(
        kv_heads, len(fallback_blocks) * block_size, d_v,
        dtype=torch.float16, device="cuda",
    )
    for slot, bid in enumerate(fallback_blocks):
        src = slice(bid * block_size, (bid + 1) * block_size)
        dst = slice(slot * block_size, (slot + 1) * block_size)
        values_fp16_scratch[:, dst, :] = values[:, src, :]

    got = selective_attend_multihead_hybrid_mixedv(
        keys_int8=cache.keys_int8[:, :n_tokens, :],
        keys_scale=cache.keys_scale[:, :n_blocks, :],
        keys_zero_points=cache.keys_zero_points[:, :n_blocks, :],
        keys_fp16=cache.keys_fp16_gpu[:, :n_tokens, :],
        topk_mask=topk_mask,
        values_int4_packed=cache.values_int4_packed[:, :n_tokens, :],
        values_int4_scales=cache.values_int4_scales[:, :n_tokens, :],
        values_int4_zeros=cache.values_int4_zeros[:, :n_tokens, :],
        values_fp16_scratch=values_fp16_scratch,
        value_fp16_mask=value_fp16_mask,
        value_block_slots=value_block_slots,
        q_all=q,
        skip_mask_i32=no_skip,
        gqa_group=gqa_group,
        block_size=block_size,
        group_size=16,
        q_scale=q_scale,
    )

    q_int8 = cache.keys_int8[:, :n_tokens, :].to(torch.float32).reshape(
        kv_heads, n_blocks, block_size, head_dim,
    )
    scales = cache.keys_scale[:, :n_blocks, :].unsqueeze(2)
    zero_points = cache.keys_zero_points[:, :n_blocks, :].unsqueeze(2)
    keys_deq = (q_int8 * scales + zero_points).reshape(kv_heads, n_tokens, head_dim)
    values_deq = torch.stack(
        [
            dequantise_int4_grouped(
                cache.values_int4_packed[h, :n_tokens, :],
                cache.values_int4_scales[h, :n_tokens, :],
                cache.values_int4_zeros[h, :n_tokens, :],
                16,
            )
            for h in range(kv_heads)
        ],
        dim=0,
    ).to(torch.float32)

    expected = torch.empty_like(got)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        k_blocks = []
        v_blocks = []
        for bid in range(n_blocks):
            start = bid * block_size
            end = start + block_size
            if int(topk_mask[qh, bid].item()):
                k_blocks.append(keys[kvh, start:end, :].to(torch.float32))
            else:
                k_blocks.append(keys_deq[kvh, start:end, :])
            if int(value_fp16_mask[qh, bid].item()):
                v_blocks.append(values[kvh, start:end, :].to(torch.float32))
            else:
                v_blocks.append(values_deq[kvh, start:end, :])
        scores = (torch.cat(k_blocks, dim=0) @ q[qh]) * q_scale
        weights = torch.softmax(scores, dim=0)
        expected[qh] = weights @ torch.cat(v_blocks, dim=0)

    torch.testing.assert_close(got, expected, atol=2e-3, rtol=2e-3)
