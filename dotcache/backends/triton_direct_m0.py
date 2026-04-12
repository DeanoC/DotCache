from __future__ import annotations

import os
from typing import Any


def _load_torch():
    import torch

    return torch


def _triton_enabled() -> bool:
    return os.environ.get("DOTCACHE_ENABLE_TRITON_DIRECT_M0", "0").strip() not in {"", "0", "false", "False"}


def _triton_fused_enabled() -> bool:
    return os.environ.get("DOTCACHE_ENABLE_TRITON_DIRECT_M0_FUSED", "0").strip() not in {"", "0", "false", "False"}


def triton_direct_m0_available() -> bool:
    if not _triton_enabled():
        return False
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except Exception:
        return False
    return True


def triton_direct_m0_fused_available() -> bool:
    return triton_direct_m0_available() and _triton_fused_enabled()


def _load_triton():
    import triton
    import triton.language as tl

    return triton, tl


def _validate_inputs(
    payload_words: Any,
    queries: Any,
    scales: Any,
    bias: Any,
    query_group_sums: Any,
) -> tuple[int, int, int, int, int]:
    torch = _load_torch()
    if not torch.is_tensor(payload_words) or not torch.is_tensor(queries):
        raise TypeError("payload_words and queries must be torch tensors")
    if payload_words.ndim != 3:
        raise ValueError("payload_words must have shape [num_groups, token_count, words_per_group]")
    if queries.ndim != 3:
        raise ValueError("queries must have shape [query_count, num_groups, group_size]")
    if scales.ndim != 2 or bias.ndim != 2:
        raise ValueError("scales and bias must have shape [token_count, num_groups]")
    if query_group_sums.ndim != 2:
        raise ValueError("query_group_sums must have shape [query_count, num_groups]")
    num_groups, token_count, words_per_group = map(int, payload_words.shape)
    query_count = int(queries.shape[0])
    group_size = int(queries.shape[-1])
    if int(queries.shape[1]) != num_groups:
        raise ValueError("queries must align with payload group count")
    if tuple(scales.shape) != (token_count, num_groups):
        raise ValueError("scales must align with payload shape")
    if tuple(bias.shape) != (token_count, num_groups):
        raise ValueError("bias must align with payload shape")
    if tuple(query_group_sums.shape) != (query_count, num_groups):
        raise ValueError("query_group_sums must align with queries")
    if payload_words.device.type != "cuda" or queries.device.type != "cuda":
        raise ValueError("Triton direct_m0 scorer requires CUDA tensors")
    if payload_words.dtype != torch.int32:
        raise ValueError("payload_words must be int32")
    if group_size != 32:
        raise ValueError("current Triton direct_m0 scorer expects group_size=32")
    if words_per_group != 8:
        raise ValueError("current Triton direct_m0 scorer expects 8-bit packed words_per_group=8")
    return num_groups, token_count, words_per_group, query_count, group_size


def score_direct_m0_logits_triton(
    *,
    payload_words: Any,
    queries: Any,
    scales: Any,
    bias: Any,
    query_group_sums: Any,
) -> Any:
    torch = _load_torch()
    triton, tl = _load_triton()
    num_groups, token_count, words_per_group, query_count, group_size = _validate_inputs(
        payload_words=payload_words,
        queries=queries,
        scales=scales,
        bias=bias,
        query_group_sums=query_group_sums,
    )
    if num_groups != 8:
        raise ValueError("current Triton direct_m0 scorer expects num_groups=8")

    @triton.jit
    def _kernel(
        payload_ptr,
        query_ptr,
        scale_ptr,
        bias_ptr,
        query_sum_ptr,
        output_ptr,
        query_count,
        token_count: tl.constexpr,
        num_groups: tl.constexpr,
        words_per_group: tl.constexpr,
        group_size: tl.constexpr,
        query_group_stride,
        query_token_stride,
        query_elem_stride,
        payload_group_stride,
        payload_token_stride,
        payload_word_stride,
        meta_token_group_stride,
        output_query_stride,
        output_token_stride,
        BLOCK_Q: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_t = tl.program_id(1)
        q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        q_mask = q_offsets < query_count
        t_mask = t_offsets < token_count
        acc = tl.zeros((BLOCK_Q, BLOCK_T), dtype=tl.float32)

        for group_idx in range(num_groups):
            q_sum = tl.load(query_sum_ptr + q_offsets * num_groups + group_idx, mask=q_mask, other=0.0)
            group_scale = tl.load(
                scale_ptr + t_offsets * meta_token_group_stride + group_idx,
                mask=t_mask,
                other=0.0,
            )
            group_bias = tl.load(
                bias_ptr + t_offsets * meta_token_group_stride + group_idx,
                mask=t_mask,
                other=0.0,
            )
            group_acc = tl.zeros((BLOCK_Q, BLOCK_T), dtype=tl.float32)
            for word_idx in range(words_per_group):
                packed = tl.load(
                    payload_ptr
                    + group_idx * payload_group_stride
                    + t_offsets * payload_token_stride
                    + word_idx * payload_word_stride,
                    mask=t_mask,
                    other=0,
                ).to(tl.int32)
                for byte_idx in range(4):
                    elem_idx = word_idx * 4 + byte_idx
                    shift = byte_idx * 8
                    code = ((packed >> shift) & 0xFF).to(tl.float32)
                    q_val = tl.load(
                        query_ptr
                        + q_offsets[:, None] * query_group_stride
                        + group_idx * query_token_stride
                        + elem_idx * query_elem_stride,
                        mask=q_mask[:, None],
                        other=0.0,
                    )
                    group_acc += q_val * code[None, :]
            acc += group_acc * group_scale[None, :] + q_sum[:, None] * group_bias[None, :]
        tl.store(
            output_ptr + q_offsets[:, None] * output_query_stride + t_offsets[None, :] * output_token_stride,
            acc,
            mask=q_mask[:, None] & t_mask[None, :],
        )

    output = torch.empty((query_count, token_count), dtype=torch.float32, device=queries.device)
    grid = (triton.cdiv(query_count, 4), triton.cdiv(token_count, 128))
    _kernel[grid](
        payload_words,
        queries,
        scales,
        bias,
        query_group_sums,
        output,
        query_count,
        token_count=token_count,
        num_groups=num_groups,
        words_per_group=words_per_group,
        group_size=group_size,
        query_group_stride=queries.stride(0),
        query_token_stride=queries.stride(1),
        query_elem_stride=queries.stride(2),
        payload_group_stride=payload_words.stride(0),
        payload_token_stride=payload_words.stride(1),
        payload_word_stride=payload_words.stride(2),
        meta_token_group_stride=scales.stride(0),
        output_query_stride=output.stride(0),
        output_token_stride=output.stride(1),
        BLOCK_Q=4,
        BLOCK_T=128,
    )
    return output


def direct_m0_softmax_available() -> bool:
    return triton_direct_m0_available()


def softmax_weights_triton(*, logits: Any, query_scale: float) -> Any:
    torch = _load_torch()
    triton, tl = _load_triton()
    if not torch.is_tensor(logits) or logits.ndim != 2:
        raise ValueError("logits must have shape [query_count, token_count]")
    if logits.device.type != "cuda":
        raise ValueError("Triton softmax requires CUDA logits")
    query_count, token_count = map(int, logits.shape)
    scaled_logits = logits.to(dtype=torch.float32) * float(query_scale)
    weights = torch.empty_like(scaled_logits)
    block_size = 1
    while block_size < token_count:
        block_size *= 2
    block_size = min(block_size, 65536)
    num_warps = 4 if block_size <= 2048 else 8

    @triton.jit
    def _softmax_kernel(
        input_ptr,
        output_ptr,
        input_row_stride,
        output_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

    _softmax_kernel[(query_count,)](
        scaled_logits,
        weights,
        scaled_logits.stride(0),
        weights.stride(0),
        token_count,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return weights


def fused_context_triton(
    *,
    payload_words: Any,
    queries: Any,
    scales: Any,
    bias: Any,
    query_group_sums: Any,
    values: Any,
    query_scale: float,
) -> Any:
    torch = _load_torch()
    triton, tl = _load_triton()
    num_groups, token_count, words_per_group, query_count, group_size = _validate_inputs(
        payload_words=payload_words,
        queries=queries,
        scales=scales,
        bias=bias,
        query_group_sums=query_group_sums,
    )
    if num_groups != 8:
        raise ValueError("current Triton fused direct_m0 path expects num_groups=8")
    if not torch.is_tensor(values) or values.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")
    if int(values.shape[0]) != token_count:
        raise ValueError("values token_count must align with payload_words")
    if values.device.type != "cuda":
        raise ValueError("values must be CUDA tensor")
    head_dim = int(values.shape[1])

    @triton.jit
    def _kernel(
        payload_ptr,
        query_ptr,
        scale_ptr,
        bias_ptr,
        query_sum_ptr,
        values_ptr,
        output_ptr,
        query_count,
        token_count,
        head_dim,
        query_scale,
        query_group_stride,
        query_token_stride,
        query_elem_stride,
        payload_group_stride,
        payload_token_stride,
        payload_word_stride,
        meta_token_group_stride,
        value_token_stride,
        value_dim_stride,
        output_query_stride,
        output_dim_stride,
        BLOCK_Q: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        WORDS_PER_GROUP: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_d = tl.program_id(1)
        q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        q_mask = q_offsets < query_count
        d_mask = d_offsets < head_dim
        running_m = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
        running_l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        for t_start in range(0, token_count, BLOCK_T):
            t_offsets = t_start + tl.arange(0, BLOCK_T)
            t_mask = t_offsets < token_count
            logits = tl.zeros((BLOCK_Q, BLOCK_T), dtype=tl.float32)
            for group_idx in range(NUM_GROUPS):
                q_sum = tl.load(query_sum_ptr + q_offsets * NUM_GROUPS + group_idx, mask=q_mask, other=0.0)
                group_scale = tl.load(
                    scale_ptr + t_offsets * meta_token_group_stride + group_idx,
                    mask=t_mask,
                    other=0.0,
                )
                group_bias = tl.load(
                    bias_ptr + t_offsets * meta_token_group_stride + group_idx,
                    mask=t_mask,
                    other=0.0,
                )
                group_logits = tl.zeros((BLOCK_Q, BLOCK_T), dtype=tl.float32)
                for word_idx in range(WORDS_PER_GROUP):
                    packed = tl.load(
                        payload_ptr
                        + group_idx * payload_group_stride
                        + t_offsets * payload_token_stride
                        + word_idx * payload_word_stride,
                        mask=t_mask,
                        other=0,
                    ).to(tl.int32)
                    for byte_idx in range(4):
                        elem_idx = word_idx * 4 + byte_idx
                        shift = byte_idx * 8
                        code = ((packed >> shift) & 0xFF).to(tl.float32)
                        q_val = tl.load(
                            query_ptr
                            + q_offsets[:, None] * query_group_stride
                            + group_idx * query_token_stride
                            + elem_idx * query_elem_stride,
                            mask=q_mask[:, None],
                            other=0.0,
                        )
                        group_logits += q_val * code[None, :]
                logits += group_logits * group_scale[None, :] + q_sum[:, None] * group_bias[None, :]
            logits *= query_scale
            block_m = tl.max(logits, axis=1)
            new_m = tl.maximum(running_m, block_m)
            alpha = tl.exp(running_m - new_m)
            probs = tl.exp(logits - new_m[:, None])
            running_l = running_l * alpha + tl.sum(probs, axis=1)
            acc = acc * alpha[:, None]
            value_block = tl.load(
                values_ptr
                + t_offsets[:, None] * value_token_stride
                + d_offsets[None, :] * value_dim_stride,
                mask=t_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(probs, value_block)
            running_m = new_m
        output = acc / tl.maximum(running_l[:, None], 1e-8)
        tl.store(
            output_ptr + q_offsets[:, None] * output_query_stride + d_offsets[None, :] * output_dim_stride,
            output,
            mask=q_mask[:, None] & d_mask[None, :],
        )

    output = torch.empty((query_count, head_dim), dtype=torch.float32, device=queries.device)
    grid = (triton.cdiv(query_count, 4), triton.cdiv(head_dim, 64))
    _kernel[grid](
        payload_words,
        queries,
        scales,
        bias,
        query_group_sums,
        values.to(dtype=torch.float32),
        output,
        query_count,
        token_count,
        head_dim,
        float(query_scale),
        queries.stride(0),
        queries.stride(1),
        queries.stride(2),
        payload_words.stride(0),
        payload_words.stride(1),
        payload_words.stride(2),
        scales.stride(0),
        values.stride(0),
        values.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_Q=4,
        BLOCK_T=32,
        BLOCK_D=64,
        NUM_GROUPS=num_groups,
        WORDS_PER_GROUP=words_per_group,
    )
    return output


def fused_indexed_context_triton(
    *,
    payload_words: Any,
    scales: Any,
    bias: Any,
    token_indices: Any,
    queries: Any,
    query_group_sums: Any,
    values: Any,
    query_scale: float,
) -> Any:
    torch = _load_torch()
    triton, tl = _load_triton()
    if not torch.is_tensor(token_indices) or token_indices.ndim != 1:
        raise ValueError("token_indices must have shape [selected_token_count]")
    if token_indices.device.type != "cuda":
        raise ValueError("token_indices must be CUDA tensor")
    num_groups, total_token_count, words_per_group, query_count, group_size = _validate_inputs(
        payload_words=payload_words,
        queries=queries,
        scales=scales,
        bias=bias,
        query_group_sums=query_group_sums,
    )
    if int(token_indices.numel()) <= 0:
        return torch.empty((int(queries.shape[0]), int(values.shape[1])), dtype=torch.float32, device=queries.device)
    if num_groups != 8:
        raise ValueError("current Triton fused indexed direct_m0 path expects num_groups=8")
    if not torch.is_tensor(values) or values.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")
    if int(values.shape[0]) != total_token_count:
        raise ValueError("values token_count must align with payload_words")
    if values.device.type != "cuda":
        raise ValueError("values must be CUDA tensor")
    head_dim = int(values.shape[1])
    selected_token_count = int(token_indices.shape[0])

    @triton.jit
    def _kernel(
        payload_ptr,
        scale_ptr,
        bias_ptr,
        token_index_ptr,
        query_ptr,
        query_sum_ptr,
        values_ptr,
        output_ptr,
        query_count,
        selected_token_count,
        head_dim,
        query_scale,
        payload_group_stride,
        payload_token_stride,
        payload_word_stride,
        meta_token_group_stride,
        query_group_stride,
        query_token_stride,
        query_elem_stride,
        value_token_stride,
        value_dim_stride,
        output_query_stride,
        output_dim_stride,
        BLOCK_Q: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        WORDS_PER_GROUP: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_d = tl.program_id(1)
        q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        q_mask = q_offsets < query_count
        d_mask = d_offsets < head_dim
        running_m = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
        running_l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        for t_start in range(0, selected_token_count, BLOCK_T):
            selection_offsets = t_start + tl.arange(0, BLOCK_T)
            selection_mask = selection_offsets < selected_token_count
            token_ids = tl.load(token_index_ptr + selection_offsets, mask=selection_mask, other=0).to(tl.int32)
            logits = tl.zeros((BLOCK_Q, BLOCK_T), dtype=tl.float32)
            for group_idx in range(NUM_GROUPS):
                q_sum = tl.load(query_sum_ptr + q_offsets * NUM_GROUPS + group_idx, mask=q_mask, other=0.0)
                group_scale = tl.load(
                    scale_ptr + token_ids * meta_token_group_stride + group_idx,
                    mask=selection_mask,
                    other=0.0,
                )
                group_bias = tl.load(
                    bias_ptr + token_ids * meta_token_group_stride + group_idx,
                    mask=selection_mask,
                    other=0.0,
                )
                group_logits = tl.zeros((BLOCK_Q, BLOCK_T), dtype=tl.float32)
                for word_idx in range(WORDS_PER_GROUP):
                    packed = tl.load(
                        payload_ptr
                        + group_idx * payload_group_stride
                        + token_ids * payload_token_stride
                        + word_idx * payload_word_stride,
                        mask=selection_mask,
                        other=0,
                    ).to(tl.int32)
                    for byte_idx in range(4):
                        elem_idx = word_idx * 4 + byte_idx
                        shift = byte_idx * 8
                        code = ((packed >> shift) & 0xFF).to(tl.float32)
                        q_val = tl.load(
                            query_ptr
                            + q_offsets[:, None] * query_group_stride
                            + group_idx * query_token_stride
                            + elem_idx * query_elem_stride,
                            mask=q_mask[:, None],
                            other=0.0,
                        )
                        group_logits += q_val * code[None, :]
                logits += group_logits * group_scale[None, :] + q_sum[:, None] * group_bias[None, :]
            logits *= query_scale
            block_m = tl.max(logits, axis=1)
            new_m = tl.maximum(running_m, block_m)
            alpha = tl.exp(running_m - new_m)
            probs = tl.exp(logits - new_m[:, None])
            running_l = running_l * alpha + tl.sum(probs, axis=1)
            acc = acc * alpha[:, None]
            value_block = tl.load(
                values_ptr
                + token_ids[:, None] * value_token_stride
                + d_offsets[None, :] * value_dim_stride,
                mask=selection_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.dot(probs, value_block)
            running_m = new_m
        output = acc / tl.maximum(running_l[:, None], 1e-8)
        tl.store(
            output_ptr + q_offsets[:, None] * output_query_stride + d_offsets[None, :] * output_dim_stride,
            output,
            mask=q_mask[:, None] & d_mask[None, :],
        )

    output = torch.empty((query_count, head_dim), dtype=torch.float32, device=queries.device)
    grid = (triton.cdiv(query_count, 4), triton.cdiv(head_dim, 64))
    _kernel[grid](
        payload_words,
        scales,
        bias,
        token_indices.to(dtype=torch.int32),
        queries,
        query_group_sums,
        values.to(dtype=torch.float32),
        output,
        query_count,
        selected_token_count,
        head_dim,
        float(query_scale),
        payload_words.stride(0),
        payload_words.stride(1),
        payload_words.stride(2),
        scales.stride(0),
        queries.stride(0),
        queries.stride(1),
        queries.stride(2),
        values.stride(0),
        values.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_Q=4,
        BLOCK_T=32,
        BLOCK_D=64,
        NUM_GROUPS=num_groups,
        WORDS_PER_GROUP=words_per_group,
    )
    return output


def fused_selected_blocks_context_triton(
    *,
    payload_words: Any,
    scales: Any,
    bias: Any,
    selected_block_ids: Any,
    valid_mask: Any,
    queries: Any,
    query_group_sums: Any,
    values: Any,
    query_scale: float,
) -> Any:
    torch = _load_torch()
    triton, tl = _load_triton()
    if not torch.is_tensor(selected_block_ids) or selected_block_ids.ndim != 1:
        raise ValueError("selected_block_ids must have shape [selected_block_count]")
    if selected_block_ids.device.type != "cuda":
        raise ValueError("selected_block_ids must be CUDA tensor")
    if not torch.is_tensor(valid_mask) or valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape [block_count, block_size]")
    if valid_mask.device.type != "cuda":
        raise ValueError("valid_mask must be CUDA tensor")
    if not torch.is_tensor(values) or values.ndim != 3:
        raise ValueError("values must have shape [block_count, block_size, head_dim]")
    if values.device.type != "cuda":
        raise ValueError("values must be CUDA tensor")
    if payload_words.ndim != 4:
        raise ValueError("payload_words must have shape [num_groups, block_count, block_size, words_per_group]")
    if scales.ndim != 3 or bias.ndim != 3:
        raise ValueError("scales and bias must have shape [num_groups, block_count, block_size]")
    if queries.ndim != 3:
        raise ValueError("queries must have shape [query_count, num_groups, group_size]")
    if query_group_sums.ndim != 2:
        raise ValueError("query_group_sums must have shape [query_count, num_groups]")
    num_groups, block_count, block_size, words_per_group = map(int, payload_words.shape)
    query_count = int(queries.shape[0])
    group_size = int(queries.shape[-1])
    selected_block_count = int(selected_block_ids.numel())
    if int(scales.shape[0]) != num_groups or int(scales.shape[1]) != block_count or int(scales.shape[2]) != block_size:
        raise ValueError("scales must align with payload_words")
    if int(bias.shape[0]) != num_groups or int(bias.shape[1]) != block_count or int(bias.shape[2]) != block_size:
        raise ValueError("bias must align with payload_words")
    if int(valid_mask.shape[0]) != block_count or int(valid_mask.shape[1]) != block_size:
        raise ValueError("valid_mask must align with payload_words")
    if int(values.shape[0]) != block_count or int(values.shape[1]) != block_size:
        raise ValueError("values must align with payload_words")
    if int(queries.shape[1]) != num_groups:
        raise ValueError("queries must align with payload_words group count")
    if tuple(query_group_sums.shape) != (query_count, num_groups):
        raise ValueError("query_group_sums must align with queries")
    if payload_words.device.type != "cuda" or queries.device.type != "cuda":
        raise ValueError("Triton fused selected-block direct_m0 path requires CUDA tensors")
    if payload_words.dtype != torch.int32:
        raise ValueError("payload_words must be int32")
    if group_size != 32:
        raise ValueError("current Triton direct_m0 scorer expects group_size=32")
    if words_per_group != 8:
        raise ValueError("current Triton direct_m0 scorer expects 8-bit packed words_per_group=8")
    if selected_block_count <= 0:
        return torch.empty((query_count, int(values.shape[-1])), dtype=torch.float32, device=queries.device)
    head_dim = int(values.shape[-1])

    @triton.jit
    def _kernel(
        payload_ptr,
        scale_ptr,
        bias_ptr,
        selected_block_ptr,
        valid_mask_ptr,
        query_ptr,
        query_sum_ptr,
        values_ptr,
        output_ptr,
        query_count,
        selected_block_count,
        head_dim,
        query_scale,
        payload_group_stride,
        payload_block_stride,
        payload_token_stride,
        payload_word_stride,
        meta_group_stride,
        meta_block_stride,
        meta_token_stride,
        valid_block_stride,
        query_group_stride,
        query_token_stride,
        query_elem_stride,
        value_block_stride,
        value_token_stride,
        value_dim_stride,
        output_query_stride,
        output_dim_stride,
        BLOCK_Q: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        WORDS_PER_GROUP: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_d = tl.program_id(1)
        q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        q_mask = q_offsets < query_count
        d_mask = d_offsets < head_dim
        running_m = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
        running_l = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        for b_start in range(0, selected_block_count, BLOCK_B):
            sel_offsets = b_start + tl.arange(0, BLOCK_B)
            sel_mask = sel_offsets < selected_block_count
            block_ids = tl.load(selected_block_ptr + sel_offsets, mask=sel_mask, other=0).to(tl.int32)
            for tok_idx in range(BLOCK_SIZE):
                token_valid = tl.load(
                    valid_mask_ptr + block_ids * valid_block_stride + tok_idx,
                    mask=sel_mask,
                    other=0,
                ).to(tl.int1)
                active_mask = sel_mask & token_valid
                logits = tl.zeros((BLOCK_Q, BLOCK_B), dtype=tl.float32)
                for group_idx in range(NUM_GROUPS):
                    q_sum = tl.load(query_sum_ptr + q_offsets * NUM_GROUPS + group_idx, mask=q_mask, other=0.0)
                    group_scale = tl.load(
                        scale_ptr
                        + group_idx * meta_group_stride
                        + block_ids * meta_block_stride
                        + tok_idx * meta_token_stride,
                        mask=active_mask,
                        other=0.0,
                    )
                    group_bias = tl.load(
                        bias_ptr
                        + group_idx * meta_group_stride
                        + block_ids * meta_block_stride
                        + tok_idx * meta_token_stride,
                        mask=active_mask,
                        other=0.0,
                    )
                    group_logits = tl.zeros((BLOCK_Q, BLOCK_B), dtype=tl.float32)
                    for word_idx in range(WORDS_PER_GROUP):
                        packed = tl.load(
                            payload_ptr
                            + group_idx * payload_group_stride
                            + block_ids * payload_block_stride
                            + tok_idx * payload_token_stride
                            + word_idx * payload_word_stride,
                            mask=active_mask,
                            other=0,
                        ).to(tl.int32)
                        for byte_idx in range(4):
                            elem_idx = word_idx * 4 + byte_idx
                            shift = byte_idx * 8
                            code = ((packed >> shift) & 0xFF).to(tl.float32)
                            q_val = tl.load(
                                query_ptr
                                + q_offsets[:, None] * query_group_stride
                                + group_idx * query_token_stride
                                + elem_idx * query_elem_stride,
                                mask=q_mask[:, None],
                                other=0.0,
                            )
                            group_logits += q_val * code[None, :]
                    logits += group_logits * group_scale[None, :] + q_sum[:, None] * group_bias[None, :]
                logits = tl.where(active_mask[None, :], logits, -float("inf"))
                logits *= query_scale
                block_m = tl.max(logits, axis=1)
                new_m = tl.maximum(running_m, block_m)
                alpha = tl.exp(running_m - new_m)
                probs = tl.exp(logits - new_m[:, None])
                probs = tl.where(active_mask[None, :], probs, 0.0)
                running_l = running_l * alpha + tl.sum(probs, axis=1)
                acc = acc * alpha[:, None]
                value_block = tl.load(
                    values_ptr
                    + block_ids[:, None] * value_block_stride
                    + tok_idx * value_token_stride
                    + d_offsets[None, :] * value_dim_stride,
                    mask=active_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                acc += tl.dot(probs, value_block)
                running_m = new_m
        output = acc / tl.maximum(running_l[:, None], 1e-8)
        tl.store(
            output_ptr + q_offsets[:, None] * output_query_stride + d_offsets[None, :] * output_dim_stride,
            output,
            mask=q_mask[:, None] & d_mask[None, :],
        )

    payload_words_kernel = payload_words.contiguous()
    scales_kernel = scales.contiguous()
    bias_kernel = bias.contiguous()
    selected_block_ids_kernel = selected_block_ids.to(dtype=torch.int32).contiguous()
    valid_mask_kernel = valid_mask.to(dtype=torch.int32).contiguous()
    queries_kernel = queries.contiguous()
    query_group_sums_kernel = query_group_sums.contiguous()
    values_kernel = values.to(dtype=torch.float32).contiguous()

    output = torch.empty((query_count, head_dim), dtype=torch.float32, device=queries.device)
    grid = (triton.cdiv(query_count, 4), triton.cdiv(head_dim, 64))
    _kernel[grid](
        payload_words_kernel,
        scales_kernel,
        bias_kernel,
        selected_block_ids_kernel,
        valid_mask_kernel,
        queries_kernel,
        query_group_sums_kernel,
        values_kernel,
        output,
        query_count,
        selected_block_count,
        head_dim,
        float(query_scale),
        payload_words_kernel.stride(0),
        payload_words_kernel.stride(1),
        payload_words_kernel.stride(2),
        payload_words_kernel.stride(3),
        scales_kernel.stride(0),
        scales_kernel.stride(1),
        scales_kernel.stride(2),
        valid_mask_kernel.stride(0),
        queries_kernel.stride(0),
        queries_kernel.stride(1),
        queries_kernel.stride(2),
        values_kernel.stride(0),
        values_kernel.stride(1),
        values_kernel.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_Q=4,
        BLOCK_B=16,
        BLOCK_D=64,
        BLOCK_SIZE=block_size,
        NUM_GROUPS=num_groups,
        WORDS_PER_GROUP=words_per_group,
    )
    return output
