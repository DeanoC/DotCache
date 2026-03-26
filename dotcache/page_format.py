from __future__ import annotations

from typing import cast

import numpy as np

from .packing import pack_bits, words_per_group
from .types import EncodedPage, Layout, PageHeader


def serialize_header(header: PageHeader) -> bytes:
    return header.to_json().encode("utf-8")


def deserialize_header(payload: bytes) -> PageHeader:
    return PageHeader.from_json(payload.decode("utf-8"))


def build_payload(codes: np.ndarray, bits: int, layout: Layout) -> np.ndarray:
    token_count, num_groups, group_size = codes.shape
    group_word_count = words_per_group(group_size, bits)

    if layout == "group_major":
        payload = np.zeros((num_groups, token_count, group_word_count), dtype=np.uint32)
        for group_index in range(num_groups):
            payload[group_index] = pack_bits(codes[:, group_index, :], bits)
        return payload

    payload = np.zeros((token_count, num_groups, group_word_count), dtype=np.uint32)
    for token_index in range(token_count):
        payload[token_index] = pack_bits(codes[token_index], bits)
    return payload


def load_group_words(page: EncodedPage, group_index: int) -> np.ndarray:
    if page.payload is None:
        raise ValueError("M3 pages do not have packed payload")
    if group_index < 0 or group_index >= page.header.num_groups:
        raise IndexError("group_index out of range")
    if page.header.layout == "group_major":
        return cast(np.ndarray, page.payload[group_index])
    return cast(np.ndarray, page.payload[:, group_index, :])

