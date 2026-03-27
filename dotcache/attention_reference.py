from __future__ import annotations

import numpy as np

from .decode_reference import decode_page
from .modes.m1_lut import dequantize_group_lut
from .page_format import load_group_words
from .packing import unpack_bits
from .types import EncodedPage


def _pad_query(query_slice: np.ndarray, padded_head_dim: int) -> np.ndarray:
    query = np.asarray(query_slice, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query_slice must have shape [head_dim]")
    if query.shape[0] > padded_head_dim:
        raise ValueError("query head_dim exceeds padded_head_dim")
    if query.shape[0] == padded_head_dim:
        return query
    return np.pad(query, (0, padded_head_dim - query.shape[0]), mode="constant")


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    return weights / np.sum(weights)


def score_page_ref(query_slice: np.ndarray, page: EncodedPage) -> np.ndarray:
    header = page.header
    query = _pad_query(query_slice, header.padded_head_dim)

    if header.mode_default == "M3":
        if page.escape_payload is None:
            raise ValueError("escape payload is missing")
        dense = page.escape_payload.astype(np.float32)[:, : header.head_dim]
        return dense @ query_slice.astype(np.float32)

    if header.mode_default == "M2":
        if page.m2_sketch is None or page.m2_basis is None:
            raise ValueError("M2 page is missing sketch payload")
        query_groups = query.reshape(header.num_groups, header.group_size)
        logits = np.zeros(header.token_count, dtype=np.float32)
        for group_index in range(header.num_groups):
            q_proj = page.m2_basis[group_index].astype(np.float32) @ query_groups[group_index]
            logits += page.m2_sketch[:, group_index, :].astype(np.float32) @ q_proj.astype(np.float32)
        return logits

    if page.payload is None:
        raise ValueError(f"{header.mode_default} page is missing payload")

    query_groups = query.reshape(header.num_groups, header.group_size)
    query_group_sums = query_groups.sum(axis=-1)
    logits = np.zeros(header.token_count, dtype=np.float32)

    for group_index in range(header.num_groups):
        words = load_group_words(page, group_index)
        codes_u8 = unpack_bits(words, header.bits, header.group_size)
        qg = query_groups[group_index]
        if header.mode_default == "M1":
            if page.codebooks is None:
                raise ValueError("M1 page is missing codebooks")
            group = dequantize_group_lut(codes_u8, codebook=np.asarray(page.codebooks[group_index], dtype=np.float32))
            logits += group @ qg
            continue

        if page.scales is None:
            raise ValueError("M0 page is missing scales")
        codes = codes_u8.astype(np.float32)
        scales = page.scales[:, group_index].astype(np.float32)

        if header.quant_scheme == "affine":
            if page.bias is None:
                raise ValueError("affine pages require bias metadata")
            int_dot = codes @ qg
            bias = page.bias[:, group_index].astype(np.float32)
            logits += scales * int_dot + bias * query_group_sums[group_index]
            continue

        zero_point = (1 << (header.bits - 1)) - 1
        logits += scales * ((codes - zero_point) @ qg)

    return logits


def mix_page_ref(attn_weights: np.ndarray, page: EncodedPage, out_acc: np.ndarray | None = None) -> np.ndarray:
    header = page.header
    weights = np.asarray(attn_weights, dtype=np.float32)
    if weights.shape != (header.token_count,):
        raise ValueError("attn_weights must have shape [token_count]")

    output = np.zeros(header.padded_head_dim, dtype=np.float32) if out_acc is None else np.asarray(out_acc, dtype=np.float32)
    if output.shape != (header.padded_head_dim,):
        raise ValueError("out_acc must have shape [padded_head_dim]")

    if header.mode_default == "M3":
        if page.escape_payload is None:
            raise ValueError("escape payload is missing")
        output[: header.head_dim] += weights @ page.escape_payload.astype(np.float32)[:, : header.head_dim]
        return output[: header.head_dim].copy()

    if header.mode_default == "M2":
        raise ValueError("M2 is only supported for key scoring in this phase")

    if page.payload is None:
        raise ValueError(f"{header.mode_default} page is missing payload")

    for group_index in range(header.num_groups):
        words = load_group_words(page, group_index)
        codes_u8 = unpack_bits(words, header.bits, header.group_size)

        if header.mode_default == "M1":
            if page.codebooks is None:
                raise ValueError("M1 page is missing codebooks")
            group = dequantize_group_lut(codes_u8, codebook=np.asarray(page.codebooks[group_index], dtype=np.float32))
        else:
            if page.scales is None:
                raise ValueError("M0 page is missing scales")
            codes = codes_u8.astype(np.float32)
            scales = page.scales[:, group_index].astype(np.float32)[:, None]

            if header.quant_scheme == "affine":
                if page.bias is None:
                    raise ValueError("affine pages require bias metadata")
                group = scales * codes + page.bias[:, group_index].astype(np.float32)[:, None]
            else:
                zero_point = (1 << (header.bits - 1)) - 1
                group = scales * (codes - zero_point)

        start = group_index * header.group_size
        end = start + header.group_size
        output[start:end] += weights @ group

    return output[: header.head_dim].copy()


def explicit_dequantized_score(query_slice: np.ndarray, page: EncodedPage) -> np.ndarray:
    dense = decode_page(page)
    query = np.asarray(query_slice, dtype=np.float32)
    return dense @ query


def explicit_dequantized_mix(attn_weights: np.ndarray, page: EncodedPage) -> np.ndarray:
    dense = decode_page(page)
    weights = np.asarray(attn_weights, dtype=np.float32)
    return weights @ dense


def run_attention_reference(query_slice: np.ndarray, key_page: EncodedPage, value_page: EncodedPage) -> tuple[np.ndarray, np.ndarray]:
    logits = score_page_ref(query_slice, key_page)
    weights = softmax(logits)
    output = mix_page_ref(weights, value_page)
    return logits, output


def explicit_dequantized_attention(query_slice: np.ndarray, key_page: EncodedPage, value_page: EncodedPage) -> tuple[np.ndarray, np.ndarray]:
    logits = explicit_dequantized_score(query_slice, key_page)
    weights = softmax(logits)
    output = explicit_dequantized_mix(weights, value_page)
    return logits, output
