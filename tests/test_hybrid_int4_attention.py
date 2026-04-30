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
def test_hybrid_int4_attention_accepts_compact_key_slots():
    from dotcache.kernels.int4_group_quantise import dequantise_int4_grouped
    from dotcache.kernels.selective_attend_triton import (
        selective_attend_multihead_hybrid_int4v,
    )
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    torch.manual_seed(20260428)
    kv_heads, q_heads, gqa_group = 2, 4, 2
    n_blocks, block_size, head_dim, d_v = 4, 16, 32, 32
    n_tokens = n_blocks * block_size
    q_scale = 1.0 / (head_dim ** 0.5)

    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys,
        values,
        block_size=block_size,
        group_size=16,
        max_new_tokens=0,
        fp16_key_cache_capacity=2,
        fp16_value_cache_capacity=2,
    )
    assert cache.keys_fp16_gpu.shape[1] == 2 * block_size

    topk_mask = torch.zeros(q_heads, n_blocks, dtype=torch.int32, device="cuda")
    topk_mask[:, [1, 3]] = 1
    cache.ensure_fp16_keys_resident([1, 3])
    key_block_slots = torch.full((n_blocks,), 0, dtype=torch.int32, device="cuda")
    for bid, slot in cache._fp16_key_resident.items():
        key_block_slots[int(bid)] = int(slot)

    q = torch.randn(q_heads, head_dim, dtype=torch.float32, device="cuda")
    no_skip = torch.zeros_like(topk_mask)
    got = selective_attend_multihead_hybrid_int4v(
        keys_int8=cache.keys_int8[:, :n_tokens, :],
        keys_scale=cache.keys_scale[:, :n_blocks, :],
        keys_zero_points=cache.keys_zero_points[:, :n_blocks, :],
        keys_fp16=cache.keys_fp16_gpu,
        key_block_slots=key_block_slots,
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
    keys_deq = (
        q_int8
        * cache.keys_scale[:, :n_blocks, :].unsqueeze(2)
        + cache.keys_zero_points[:, :n_blocks, :].unsqueeze(2)
    ).reshape(kv_heads, n_tokens, head_dim)
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
        scores = (torch.cat(k_blocks, dim=0) @ q[qh]) * q_scale
        expected[qh] = torch.softmax(scores, dim=0) @ values_deq[kvh]

    torch.testing.assert_close(got, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_fp16_block_scores_accept_compact_key_slots():
    from dotcache.kernels.certified_attention import compute_fp16_block_scores
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    torch.manual_seed(20260429)
    kv_heads, q_heads, gqa_group = 2, 4, 2
    n_blocks, block_size, head_dim, d_v = 4, 16, 32, 32
    n_tokens = n_blocks * block_size
    q_scale = 1.0 / (head_dim ** 0.5)

    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys,
        values,
        block_size=block_size,
        group_size=16,
        max_new_tokens=0,
        fp16_key_cache_capacity=2,
        fp16_value_cache_capacity=2,
    )
    cache.ensure_fp16_keys_resident([1, 3])
    key_block_slots = torch.full((n_blocks,), -1, dtype=torch.int32, device="cuda")
    for bid, slot in cache._fp16_key_resident.items():
        key_block_slots[int(bid)] = int(slot)

    q = torch.randn(q_heads, head_dim, dtype=torch.float32, device="cuda")
    block_indices = torch.tensor([[1, 3], [3, 1], [1, 3], [3, 1]], device="cuda")
    got_scores, got_logmass = compute_fp16_block_scores(
        cache,
        q,
        block_indices,
        n_blocks,
        gqa_group,
        q_scale,
        return_log_mass=True,
        keys_fp16_override=cache.keys_fp16_gpu,
        key_block_slots=key_block_slots,
    )

    expected_scores = torch.empty_like(got_scores)
    expected_logmass = torch.empty_like(got_logmass)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        for kk, bid in enumerate(block_indices[qh].tolist()):
            start = int(bid) * block_size
            end = start + block_size
            logits = (keys[kvh, start:end, :].float() @ q[qh].float()) * q_scale
            expected_scores[qh, kk] = logits.max()
            expected_logmass[qh, kk] = torch.logsumexp(logits, dim=0)

    torch.testing.assert_close(got_scores, expected_scores, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(got_logmass, expected_logmass, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_bounded_fp16_slot_tables_track_residency_and_eviction():
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    torch.manual_seed(20260430)
    kv_heads, n_blocks, block_size, head_dim, d_v = 2, 4, 16, 32, 32
    n_tokens = n_blocks * block_size
    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys,
        values,
        block_size=block_size,
        group_size=16,
        max_new_tokens=0,
        fp16_key_cache_capacity=2,
        fp16_value_cache_capacity=2,
    )

    cache.ensure_fp16_keys_resident([0, 1])
    cache.ensure_fp16_values_resident([0, 1])
    assert cache.fp16_key_block_slots_gpu(n_blocks).cpu().tolist() == [0, 1, -1, -1]
    assert cache.fp16_value_block_slots_gpu(n_blocks).cpu().tolist() == [0, 1, -1, -1]

    cache.ensure_fp16_keys_resident([2])
    cache.ensure_fp16_values_resident([2])
    key_slots = cache.fp16_key_block_slots_gpu(n_blocks).cpu().tolist()
    value_slots = cache.fp16_value_block_slots_gpu(n_blocks).cpu().tolist()
    assert key_slots == [-1, 1, 0, -1]
    assert value_slots == [-1, 1, 0, -1]


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_bounded_fp16_key_slot_updates_on_append():
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    torch.manual_seed(20260501)
    kv_heads, n_blocks, block_size, head_dim, d_v = 2, 1, 16, 32, 32
    n_tokens = n_blocks * block_size
    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys,
        values,
        block_size=block_size,
        group_size=16,
        max_new_tokens=block_size,
        fp16_key_cache_capacity=2,
        fp16_value_cache_capacity=2,
    )

    first_key = torch.randn(kv_heads, 1, head_dim, dtype=torch.float16, device="cuda")
    first_value = torch.randn(kv_heads, 1, d_v, dtype=torch.float16, device="cuda")
    cache.append_token(first_key, first_value)
    cache.ensure_fp16_keys_resident([1])
    slot = cache._fp16_key_resident[1]

    second_key = torch.randn(kv_heads, 1, head_dim, dtype=torch.float16, device="cuda")
    second_value = torch.randn(kv_heads, 1, d_v, dtype=torch.float16, device="cuda")
    cache.append_token(second_key, second_value)

    dst = slot * block_size + 1
    torch.testing.assert_close(cache.keys_fp16_gpu[:, dst, :], second_key[:, 0, :])


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_static_resident_bounded_cache_identity_maps_future_blocks():
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    torch.manual_seed(20260502)
    kv_heads, n_blocks, block_size, head_dim, d_v = 2, 2, 16, 32, 32
    n_tokens = n_blocks * block_size
    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys,
        values,
        block_size=block_size,
        group_size=16,
        max_new_tokens=block_size,
        fp16_key_cache_capacity=3,
        fp16_value_cache_capacity=3,
    )

    assert cache.static_resident_key_cache is True
    assert cache.static_resident_value_cache is True
    assert cache.fp16_key_block_slots_gpu(3).detach().cpu().tolist() == [0, 1, 2]
    assert cache.fp16_value_block_slots_gpu(3).detach().cpu().tolist() == [0, 1, 2]
    torch.testing.assert_close(cache.keys_fp16_gpu[:, :n_tokens, :], keys)
    torch.testing.assert_close(cache.values_fp16_gpu[:, :n_tokens, :], values)

    key = torch.randn(kv_heads, 1, head_dim, dtype=torch.float16, device="cuda")
    value = torch.randn(kv_heads, 1, d_v, dtype=torch.float16, device="cuda")
    cache.append_token(key, value)
    dst = n_blocks * block_size
    torch.testing.assert_close(cache.keys_fp16_gpu[:, dst, :], key[:, 0, :])
    torch.testing.assert_close(cache.values_fp16_gpu[:, dst, :], value[:, 0, :])


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_static_resident_completed_key_block_quantizes_from_gpu_scratch():
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    torch.manual_seed(20260503)
    kv_heads, n_blocks, block_size, head_dim, d_v = 2, 1, 16, 32, 32
    n_tokens = n_blocks * block_size
    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys,
        values,
        block_size=block_size,
        group_size=16,
        max_new_tokens=block_size,
        fp16_key_cache_capacity=2,
        fp16_value_cache_capacity=2,
    )
    assert cache.static_resident_key_cache is True

    start = n_tokens
    end = start + block_size
    cache.keys_fp16_cpu[:, start:end, :].fill_(1234.0)
    appended = torch.randn(
        kv_heads, block_size, head_dim, dtype=torch.float16, device="cuda",
    )
    for pos in range(block_size):
        value = torch.randn(kv_heads, 1, d_v, dtype=torch.float16, device="cuda")
        cache.append_token(appended[:, pos:pos + 1, :], value)

    block_idx = n_blocks
    reconstructed = (
        cache.keys_int8[:, start:end, :].to(torch.float32)
        * cache.keys_scale[:, block_idx:block_idx + 1, :]
        + cache.keys_zero_points[:, block_idx:block_idx + 1, :]
    )
    torch.testing.assert_close(reconstructed, appended.to(torch.float32), atol=0.02, rtol=0.02)
    assert float(cache.keys_fp16_cpu[:, start:end, :].abs().mean()) > 1000.0


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_static_resident_completed_value_block_quantizes_from_gpu_scratch():
    from dotcache.kernels.int4_group_quantise import dequantise_int4_grouped
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    torch.manual_seed(20260504)
    kv_heads, n_blocks, block_size, head_dim, d_v = 2, 1, 16, 32, 32
    n_tokens = n_blocks * block_size
    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys,
        values,
        block_size=block_size,
        group_size=16,
        max_new_tokens=block_size,
        fp16_key_cache_capacity=2,
        fp16_value_cache_capacity=2,
    )
    assert cache.static_resident_value_cache is True
    assert cache.defer_int4_append_quantization is True

    start = n_tokens
    end = start + block_size
    cache.values_fp16_cpu[:, start:end, :].fill_(1234.0)
    appended_values = torch.randn(
        kv_heads, block_size, d_v, dtype=torch.float16, device="cuda",
    )
    for pos in range(block_size):
        key = torch.randn(kv_heads, 1, head_dim, dtype=torch.float16, device="cuda")
        cache.append_token(key, appended_values[:, pos:pos + 1, :])

    deq = dequantise_int4_grouped(
        cache.values_int4_packed[:, start:end, :].reshape(kv_heads * block_size, d_v // 2),
        cache.values_int4_scales[:, start:end, :].reshape(kv_heads * block_size, d_v // 16),
        cache.values_int4_zeros[:, start:end, :].reshape(kv_heads * block_size, d_v // 16),
        group_size=16,
    ).reshape(kv_heads, block_size, d_v)
    assert float((deq - appended_values.float()).abs().mean()) < 0.08
    assert float(cache.values_fp16_cpu[:, start:end, :].abs().mean()) > 1000.0


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_hybrid_mixed_value_attention_promotes_only_masked_blocks():
    from dotcache.kernels.int4_group_quantise import dequantise_int4_grouped
    from dotcache.kernels.selective_attend_triton import (
        selective_attend_multihead_hybrid_mixedv,
        selective_attend_multihead_hybrid_mixedv_split_k,
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
    got_split = selective_attend_multihead_hybrid_mixedv_split_k(
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
        num_splits=2,
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
    torch.testing.assert_close(got_split, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_hybrid_mixed_value_split_k_accepts_oversized_cache_slices():
    from dotcache.kernels.selective_attend_triton import (
        selective_attend_multihead_hybrid_mixedv_split_k,
    )
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    torch.manual_seed(20260427)
    kv_heads, q_heads, gqa_group = 2, 4, 2
    n_blocks, block_size, head_dim, d_v = 4, 16, 32, 32
    n_tokens = n_blocks * block_size
    q_scale = 1.0 / (head_dim ** 0.5)

    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys, values, block_size=block_size, group_size=16, max_new_tokens=32,
    )
    q = torch.randn(q_heads, head_dim, dtype=torch.float32, device="cuda")

    topk_mask = torch.zeros(q_heads, n_blocks, dtype=torch.int32, device="cuda")
    topk_mask[:, [0, 2]] = 1
    no_skip = torch.zeros_like(topk_mask)
    value_fp16_mask = torch.zeros(q_heads, n_blocks, dtype=torch.int32, device="cuda")
    value_fp16_mask[0, [1]] = 1
    value_fp16_mask[1, [3]] = 1
    value_fp16_mask[2, [1, 3]] = 1
    value_fp16_mask[3, [0]] = 1
    fallback_blocks = value_fp16_mask.any(dim=0).nonzero().flatten().tolist()
    value_block_slots = torch.full((n_blocks,), -1, dtype=torch.int32, device="cuda")
    scratch_full = torch.empty(
        kv_heads, (len(fallback_blocks) + 2) * block_size, d_v,
        dtype=torch.float16, device="cuda",
    )
    values_fp16_scratch = scratch_full[:, :len(fallback_blocks) * block_size, :]
    assert not values_fp16_scratch.is_contiguous()
    for slot, bid in enumerate(fallback_blocks):
        value_block_slots[bid] = slot
        src = slice(bid * block_size, (bid + 1) * block_size)
        dst = slice(slot * block_size, (slot + 1) * block_size)
        values_fp16_scratch[:, dst, :] = values[:, src, :]

    sliced_kwargs = dict(
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
        num_splits=2,
    )
    assert not sliced_kwargs["keys_int8"].is_contiguous()
    assert not sliced_kwargs["values_int4_packed"].is_contiguous()

    contiguous_kwargs = dict(sliced_kwargs)
    for name in (
        "keys_int8", "keys_scale", "keys_zero_points", "keys_fp16",
        "values_int4_packed", "values_int4_scales", "values_int4_zeros",
        "values_fp16_scratch",
    ):
        contiguous_kwargs[name] = contiguous_kwargs[name].contiguous()

    got = selective_attend_multihead_hybrid_mixedv_split_k(**sliced_kwargs)
    expected = selective_attend_multihead_hybrid_mixedv_split_k(**contiguous_kwargs)
    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_native_blackwell_mixed_value_attention_matches_triton_split_k():
    from dotcache.backends.certified_blackwell import (
        certified_blackwell_available,
        hybrid_mixedv_split_k_cuda,
    )
    from dotcache.kernels.selective_attend_triton import (
        selective_attend_multihead_hybrid_mixedv_split_k,
    )
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    if not certified_blackwell_available():
        pytest.skip("native Blackwell extension unavailable")

    torch.manual_seed(20260426)
    kv_heads, q_heads, gqa_group = 2, 4, 2
    n_blocks, block_size, head_dim, d_v = 8, 16, 32, 32
    n_tokens = n_blocks * block_size
    q_scale = 1.0 / (head_dim ** 0.5)

    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys, values, block_size=block_size, group_size=16, max_new_tokens=0,
    )
    q = torch.randn(q_heads, head_dim, dtype=torch.float32, device="cuda")

    topk_mask = torch.zeros(q_heads, n_blocks, dtype=torch.int32, device="cuda")
    topk_mask[:, [0, 3]] = 1
    no_skip = torch.zeros_like(topk_mask)
    value_fp16_mask = torch.zeros(q_heads, n_blocks, dtype=torch.int32, device="cuda")
    value_fp16_mask[0, [1, 5]] = 1
    value_fp16_mask[1, [2]] = 1
    value_fp16_mask[2, [4]] = 1
    value_fp16_mask[3, [7]] = 1

    fallback_blocks = value_fp16_mask.any(dim=0).nonzero().flatten().tolist()
    value_block_slots = torch.full((n_blocks,), -1, dtype=torch.int32, device="cuda")
    values_fp16_scratch = torch.empty(
        kv_heads, max(len(fallback_blocks), 1) * block_size, d_v,
        dtype=torch.float16, device="cuda",
    )
    for slot, bid in enumerate(fallback_blocks):
        value_block_slots[bid] = slot
        src = slice(bid * block_size, (bid + 1) * block_size)
        dst = slice(slot * block_size, (slot + 1) * block_size)
        values_fp16_scratch[:, dst, :] = values[:, src, :]

    kwargs = dict(
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
        num_splits=2,
    )
    expected = selective_attend_multihead_hybrid_mixedv_split_k(**kwargs)
    got = hybrid_mixedv_split_k_cuda(**kwargs)

    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_native_blackwell_mixed_value_attention_accepts_compact_key_slots():
    from dotcache.backends.certified_blackwell import (
        certified_blackwell_available,
        hybrid_mixedv_split_k_cuda,
    )
    from dotcache.kernels.selective_attend_triton import (
        selective_attend_multihead_hybrid_mixedv_split_k,
    )
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    if not certified_blackwell_available():
        pytest.skip("native Blackwell extension unavailable")

    torch.manual_seed(20260431)
    kv_heads, q_heads, gqa_group = 2, 4, 2
    n_blocks, block_size, head_dim, d_v = 8, 16, 32, 32
    n_tokens = n_blocks * block_size
    q_scale = 1.0 / (head_dim ** 0.5)

    keys = torch.randn(kv_heads, n_tokens, head_dim, dtype=torch.float16, device="cuda")
    values = torch.randn(kv_heads, n_tokens, d_v, dtype=torch.float16, device="cuda")
    cache = TieredKeyCacheLayer.from_fp16_cache_int4v(
        keys,
        values,
        block_size=block_size,
        group_size=16,
        max_new_tokens=0,
        fp16_key_cache_capacity=3,
        fp16_value_cache_capacity=3,
    )
    cache.ensure_fp16_keys_resident([1, 3, 6])
    key_block_slots = cache.fp16_key_block_slots_gpu(n_blocks)
    q = torch.randn(q_heads, head_dim, dtype=torch.float32, device="cuda")

    topk_mask = torch.zeros(q_heads, n_blocks, dtype=torch.int32, device="cuda")
    topk_mask[:, [1, 3, 6]] = 1
    no_skip = torch.zeros_like(topk_mask)
    value_fp16_mask = torch.zeros(q_heads, n_blocks, dtype=torch.int32, device="cuda")
    value_fp16_mask[0, [2]] = 1
    value_fp16_mask[1, [4]] = 1
    value_fp16_mask[2, [5]] = 1
    value_fp16_mask[3, [7]] = 1

    fallback_blocks = value_fp16_mask.any(dim=0).nonzero().flatten().tolist()
    value_block_slots = torch.full((n_blocks,), -1, dtype=torch.int32, device="cuda")
    values_fp16_scratch = torch.empty(
        kv_heads, max(len(fallback_blocks), 1) * block_size, d_v,
        dtype=torch.float16, device="cuda",
    )
    for slot, bid in enumerate(fallback_blocks):
        value_block_slots[bid] = slot
        src = slice(bid * block_size, (bid + 1) * block_size)
        dst = slice(slot * block_size, (slot + 1) * block_size)
        values_fp16_scratch[:, dst, :] = values[:, src, :]

    kwargs = dict(
        keys_int8=cache.keys_int8[:, :n_tokens, :],
        keys_scale=cache.keys_scale[:, :n_blocks, :],
        keys_zero_points=cache.keys_zero_points[:, :n_blocks, :],
        keys_fp16=cache.keys_fp16_gpu,
        key_block_slots=key_block_slots,
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
        num_splits=2,
    )
    expected = selective_attend_multihead_hybrid_mixedv_split_k(**kwargs)
    got = hybrid_mixedv_split_k_cuda(**kwargs)

    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)
