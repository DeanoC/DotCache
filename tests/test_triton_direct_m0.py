from __future__ import annotations

import pytest
import torch

from dotcache.backends.torch_mps import _score_m0_logits_packed32_grouped_torch, _unpack_metadata
from dotcache.backends.triton_direct_m0 import (
    fused_context_triton,
    fused_indexed_context_triton,
    fused_selected_blocks_context_triton,
    score_direct_m0_logits_triton,
    softmax_weights_triton,
    triton_direct_m0_available,
    triton_direct_m0_fused_available,
)
from dotcache.backends.native_direct_m0 import (
    fused_selected_blocks_context_cuda,
    fused_selected_blocks_stream_stats_cuda,
    native_direct_m0_available,
    native_direct_m0_final_mix_available,
    softmax_value_context_cuda,
    softmax_value_stream_stats_cuda,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_direct_m0_matches_grouped_packed_torch() -> None:
    if not triton_direct_m0_available():
        pytest.skip("Triton direct_m0 scorer not enabled")
    torch.manual_seed(0)
    device = "cuda"
    payload = torch.randint(0, 2**31 - 1, (1, 8, 5, 1, 8), dtype=torch.int32, device=device)
    queries = torch.randn(1, 3, 8, 32, dtype=torch.float32, device=device)
    scales = torch.randn(1, 8, 5, 1, dtype=torch.float32, device=device)
    bias = torch.randn(1, 8, 5, 1, dtype=torch.float32, device=device)
    query_group_sums = torch.randn(1, 3, 8, dtype=torch.float32, device=device)
    unpack_shifts, unpack_mask = _unpack_metadata(8, device_type="cuda")

    expected = _score_m0_logits_packed32_grouped_torch(
        payload,
        queries,
        scales,
        bias,
        query_group_sums,
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
    )
    actual = score_direct_m0_logits_triton(
        payload_words=payload[0, :, :, 0, :].contiguous(),
        queries=queries[0].contiguous(),
        scales=scales[0, :, :, 0].transpose(0, 1).contiguous(),
        bias=bias[0, :, :, 0].transpose(0, 1).contiguous(),
        query_group_sums=query_group_sums[0].contiguous(),
    )

    assert torch.allclose(actual, expected[0], atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_direct_m0_softmax_matches_torch() -> None:
    if not triton_direct_m0_available():
        pytest.skip("Triton direct_m0 scorer not enabled")
    torch.manual_seed(0)
    logits = torch.randn(3, 2048, dtype=torch.float32, device="cuda")
    expected = torch.softmax(logits * 0.125, dim=-1)
    actual = softmax_weights_triton(logits=logits, query_scale=0.125)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_direct_m0_fused_context_matches_torch() -> None:
    if not triton_direct_m0_fused_available():
        pytest.skip("Triton fused direct_m0 path not enabled")
    torch.manual_seed(0)
    device = "cuda"
    payload = torch.randint(0, 2**31 - 1, (1, 8, 5, 1, 8), dtype=torch.int32, device=device)
    queries = torch.randn(1, 3, 8, 32, dtype=torch.float32, device=device)
    scales = torch.randn(1, 8, 5, 1, dtype=torch.float32, device=device)
    bias = torch.randn(1, 8, 5, 1, dtype=torch.float32, device=device)
    query_group_sums = torch.randn(1, 3, 8, dtype=torch.float32, device=device)
    values = torch.randn(5, 128, dtype=torch.float32, device=device)
    unpack_shifts, unpack_mask = _unpack_metadata(8, device_type="cuda")

    logits = _score_m0_logits_packed32_grouped_torch(
        payload,
        queries,
        scales,
        bias,
        query_group_sums,
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
    )[0]
    weights = torch.softmax(logits * 0.125, dim=-1)
    expected = torch.matmul(weights, values)

    actual = fused_context_triton(
        payload_words=payload[0, :, :, 0, :].contiguous(),
        queries=queries[0].contiguous(),
        scales=scales[0, :, :, 0].transpose(0, 1).contiguous(),
        bias=bias[0, :, :, 0].transpose(0, 1).contiguous(),
        query_group_sums=query_group_sums[0].contiguous(),
        values=values,
        query_scale=0.125,
    )

    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_direct_m0_fused_indexed_context_matches_torch() -> None:
    if not triton_direct_m0_fused_available():
        pytest.skip("Triton fused direct_m0 path not enabled")
    torch.manual_seed(0)
    device = "cuda"
    payload = torch.randint(0, 2**31 - 1, (1, 8, 7, 1, 8), dtype=torch.int32, device=device)
    queries = torch.randn(1, 3, 8, 32, dtype=torch.float32, device=device)
    scales = torch.randn(1, 8, 7, 1, dtype=torch.float32, device=device)
    bias = torch.randn(1, 8, 7, 1, dtype=torch.float32, device=device)
    query_group_sums = torch.randn(1, 3, 8, dtype=torch.float32, device=device)
    values = torch.randn(7, 128, dtype=torch.float32, device=device)
    token_indices = torch.tensor([5, 1, 6, 2], dtype=torch.int64, device=device)
    unpack_shifts, unpack_mask = _unpack_metadata(8, device_type="cuda")

    gathered_payload = payload[:, :, token_indices, :, :]
    gathered_scales = scales[:, :, token_indices, :]
    gathered_bias = bias[:, :, token_indices, :]
    gathered_values = values.index_select(0, token_indices)
    logits = _score_m0_logits_packed32_grouped_torch(
        gathered_payload,
        queries,
        gathered_scales,
        gathered_bias,
        query_group_sums,
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
    )[0]
    weights = torch.softmax(logits * 0.125, dim=-1)
    expected = torch.matmul(weights, gathered_values)

    actual = fused_indexed_context_triton(
        payload_words=payload[0, :, :, 0, :].contiguous(),
        scales=scales[0, :, :, 0].transpose(0, 1).contiguous(),
        bias=bias[0, :, :, 0].transpose(0, 1).contiguous(),
        token_indices=token_indices,
        queries=queries[0].contiguous(),
        query_group_sums=query_group_sums[0].contiguous(),
        values=values,
        query_scale=0.125,
    )

    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_direct_m0_fused_selected_blocks_context_matches_torch() -> None:
    if not triton_direct_m0_fused_available():
        pytest.skip("Triton fused direct_m0 path not enabled")
    torch.manual_seed(0)
    device = "cuda"
    num_blocks = 6
    block_size = 16
    head_dim = 128
    payload = torch.randint(0, 2**31 - 1, (8, num_blocks, block_size, 8), dtype=torch.int32, device=device)
    scales = torch.randn(8, num_blocks, block_size, dtype=torch.float32, device=device)
    bias = torch.randn(8, num_blocks, block_size, dtype=torch.float32, device=device)
    valid_mask = torch.zeros(num_blocks, block_size, dtype=torch.bool, device=device)
    valid_mask[:, :13] = True
    queries = torch.randn(3, 8, 32, dtype=torch.float32, device=device)
    query_group_sums = torch.randn(3, 8, dtype=torch.float32, device=device)
    values = torch.randn(num_blocks, block_size, head_dim, dtype=torch.float32, device=device)
    selected_block_ids = torch.tensor([4, 1, 5], dtype=torch.int64, device=device)
    unpack_shifts, unpack_mask = _unpack_metadata(8, device_type="cuda")

    gathered_payload = payload.index_select(1, selected_block_ids).unsqueeze(0)
    gathered_scales = scales.index_select(1, selected_block_ids).unsqueeze(0)
    gathered_bias = bias.index_select(1, selected_block_ids).unsqueeze(0)
    logits = _score_m0_logits_packed32_grouped_torch(
        gathered_payload,
        queries.unsqueeze(0),
        gathered_scales,
        gathered_bias,
        query_group_sums.unsqueeze(0),
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
    )[0]
    flat_valid = valid_mask.index_select(0, selected_block_ids).reshape(-1)
    flat_values = values.index_select(0, selected_block_ids).reshape(-1, head_dim)
    masked_logits = logits.reshape(int(logits.shape[0]), -1).masked_fill(~flat_valid.unsqueeze(0), float("-inf"))
    weights = torch.softmax(masked_logits * 0.125, dim=-1)
    expected = torch.matmul(weights, flat_values)

    actual = fused_selected_blocks_context_triton(
        payload_words=payload,
        scales=scales,
        bias=bias,
        selected_block_ids=selected_block_ids,
        valid_mask=valid_mask,
        queries=queries,
        query_group_sums=query_group_sums,
        values=values,
        query_scale=0.125,
    )

    assert torch.allclose(actual, expected, atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_direct_m0_fused_selected_blocks_context_matches_torch() -> None:
    if not native_direct_m0_available():
        pytest.skip("native direct_m0 CUDA path not enabled")
    torch.manual_seed(0)
    device = "cuda"
    num_blocks = 6
    block_size = 16
    head_dim = 128
    payload = torch.randint(0, 2**31 - 1, (8, num_blocks, block_size, 8), dtype=torch.int32, device=device)
    scales = torch.randn(8, num_blocks, block_size, dtype=torch.float32, device=device)
    bias = torch.randn(8, num_blocks, block_size, dtype=torch.float32, device=device)
    valid_mask = torch.zeros(num_blocks, block_size, dtype=torch.bool, device=device)
    valid_mask[:, :13] = True
    queries = torch.randn(3, 8, 32, dtype=torch.float32, device=device)
    query_group_sums = torch.randn(3, 8, dtype=torch.float32, device=device)
    values = torch.randn(num_blocks, block_size, head_dim, dtype=torch.float32, device=device)
    selected_block_ids = torch.tensor([4, 1, 5], dtype=torch.int64, device=device)
    unpack_shifts, unpack_mask = _unpack_metadata(8, device_type="cuda")

    gathered_payload = payload.index_select(1, selected_block_ids).unsqueeze(0)
    gathered_scales = scales.index_select(1, selected_block_ids).unsqueeze(0)
    gathered_bias = bias.index_select(1, selected_block_ids).unsqueeze(0)
    logits = _score_m0_logits_packed32_grouped_torch(
        gathered_payload,
        queries.unsqueeze(0),
        gathered_scales,
        gathered_bias,
        query_group_sums.unsqueeze(0),
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
    )[0]
    flat_valid = valid_mask.index_select(0, selected_block_ids).reshape(-1)
    flat_values = values.index_select(0, selected_block_ids).reshape(-1, head_dim)
    masked_logits = logits.reshape(int(logits.shape[0]), -1).masked_fill(~flat_valid.unsqueeze(0), float("-inf"))
    weights = torch.softmax(masked_logits * 0.125, dim=-1)
    expected = torch.matmul(weights, flat_values)

    actual = fused_selected_blocks_context_cuda(
        payload_words=payload,
        scales=scales,
        bias=bias,
        selected_block_ids=selected_block_ids,
        valid_mask=valid_mask,
        queries=queries,
        query_group_sums=query_group_sums,
        values=values,
        query_scale=0.125,
    )

    assert torch.allclose(actual, expected, atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_direct_m0_fused_selected_blocks_stream_stats_match_torch() -> None:
    if not native_direct_m0_available():
        pytest.skip("native direct_m0 CUDA path not enabled")
    torch.manual_seed(0)
    device = "cuda"
    num_blocks = 6
    block_size = 16
    head_dim = 128
    payload = torch.randint(0, 2**31 - 1, (8, num_blocks, block_size, 8), dtype=torch.int32, device=device)
    scales = torch.randn(8, num_blocks, block_size, dtype=torch.float32, device=device)
    bias = torch.randn(8, num_blocks, block_size, dtype=torch.float32, device=device)
    valid_mask = torch.zeros(num_blocks, block_size, dtype=torch.bool, device=device)
    valid_mask[:, :13] = True
    queries = torch.randn(3, 8, 32, dtype=torch.float32, device=device)
    query_group_sums = torch.randn(3, 8, dtype=torch.float32, device=device)
    values = torch.randn(num_blocks, block_size, head_dim, dtype=torch.float32, device=device)
    selected_block_ids = torch.tensor([4, 1, 5], dtype=torch.int64, device=device)
    unpack_shifts, unpack_mask = _unpack_metadata(8, device_type="cuda")

    gathered_payload = payload.index_select(1, selected_block_ids).unsqueeze(0)
    gathered_scales = scales.index_select(1, selected_block_ids).unsqueeze(0)
    gathered_bias = bias.index_select(1, selected_block_ids).unsqueeze(0)
    logits = _score_m0_logits_packed32_grouped_torch(
        gathered_payload,
        queries.unsqueeze(0),
        gathered_scales,
        gathered_bias,
        query_group_sums.unsqueeze(0),
        unpack_shifts=unpack_shifts,
        unpack_mask=unpack_mask,
    )[0]
    flat_valid = valid_mask.index_select(0, selected_block_ids).reshape(-1)
    scaled_logits = logits.reshape(int(logits.shape[0]), -1).masked_fill(~flat_valid.unsqueeze(0), float("-inf")) * 0.125
    exp_scores = torch.exp(scaled_logits - scaled_logits.max(dim=-1, keepdim=True).values)
    expected_m = scaled_logits.max(dim=-1).values
    expected_l = exp_scores.sum(dim=-1)
    expected_h = torch.matmul(exp_scores, values.index_select(0, selected_block_ids).reshape(-1, head_dim))
    expected_block_max = scaled_logits.reshape(scaled_logits.shape[0], -1, block_size).max(dim=-1).values
    expected_block_mass = exp_scores.reshape(exp_scores.shape[0], -1, block_size).sum(dim=-1)

    actual_h, actual_m, actual_l, actual_block_max, actual_block_mass = fused_selected_blocks_stream_stats_cuda(
        payload_words=payload,
        scales=scales,
        bias=bias,
        selected_block_ids=selected_block_ids,
        valid_mask=valid_mask,
        queries=queries,
        query_group_sums=query_group_sums,
        values=values,
        query_scale=0.125,
    )

    assert torch.allclose(actual_h, expected_h, atol=3e-3, rtol=3e-3)
    assert torch.allclose(actual_m, expected_m, atol=3e-3, rtol=3e-3)
    assert torch.allclose(actual_l, expected_l, atol=3e-3, rtol=3e-3)
    assert torch.allclose(actual_block_max[:, : int(selected_block_ids.numel())], expected_block_max, atol=3e-3, rtol=3e-3)
    assert torch.allclose(actual_block_mass[:, : int(selected_block_ids.numel())], expected_block_mass, atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_direct_m0_softmax_value_context_matches_torch() -> None:
    if not native_direct_m0_final_mix_available():
        pytest.skip("native direct_m0 final_mix CUDA path not enabled")
    torch.manual_seed(0)
    device = "cuda"
    logits = torch.randn(4, 77, dtype=torch.float16, device=device)
    values = torch.randn(77, 128, dtype=torch.float16, device=device)
    expected = torch.matmul(
        torch.softmax(logits.to(dtype=torch.float32) * 0.125, dim=-1).to(dtype=torch.float16),
        values,
    ).to(dtype=torch.float32)

    actual = softmax_value_context_cuda(
        logits=logits.to(dtype=torch.float32),
        values=values,
        query_scale=0.125,
    )

    assert torch.allclose(actual, expected, atol=3e-3, rtol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_direct_m0_softmax_value_stream_stats_matches_torch() -> None:
    if not native_direct_m0_final_mix_available():
        pytest.skip("native direct_m0 final_mix CUDA path not enabled")
    torch.manual_seed(0)
    device = "cuda"
    logits = torch.randn(4, 77, dtype=torch.float32, device=device)
    token_block_ids = torch.randint(0, 6, (77,), dtype=torch.int64, device=device)
    values = torch.randn(77, 128, dtype=torch.float16, device=device)
    scaled_logits = logits * 0.125
    row_max = scaled_logits.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scaled_logits - row_max)
    expected_m = row_max.squeeze(-1)
    expected_l = exp_scores.sum(dim=-1)
    expected_h = torch.matmul(exp_scores.to(dtype=torch.float16), values).to(dtype=torch.float32)
    expected_block_max = torch.full((4, 6), float("-inf"), dtype=torch.float32, device=device)
    expected_block_mass = torch.zeros((4, 6), dtype=torch.float32, device=device)
    for block_id in range(6):
        mask = token_block_ids == block_id
        if bool(mask.any()):
            expected_block_max[:, block_id] = scaled_logits[:, mask].max(dim=-1).values
            expected_block_mass[:, block_id] = exp_scores[:, mask].sum(dim=-1)

    actual_h, actual_m, actual_l, actual_block_max, actual_block_mass = softmax_value_stream_stats_cuda(
        logits=logits,
        token_block_ids=token_block_ids,
        values=values,
        block_count=6,
        query_scale=0.125,
    )

    assert torch.allclose(actual_h, expected_h, atol=3e-3, rtol=3e-3)
    assert torch.allclose(actual_m, expected_m, atol=3e-3, rtol=3e-3)
    assert torch.allclose(actual_l, expected_l, atol=3e-3, rtol=3e-3)
    assert torch.allclose(actual_block_max, expected_block_max, atol=3e-3, rtol=3e-3)
    assert torch.allclose(actual_block_mass, expected_block_mass, atol=3e-3, rtol=3e-3)
