from __future__ import annotations

import numpy as np


def words_per_group(group_size: int, bits: int) -> int:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if bits <= 0 or bits > 16:
        raise ValueError("bits must be between 1 and 16")
    return (group_size * bits + 31) // 32


def pack_bits(codes: np.ndarray, bits: int) -> np.ndarray:
    values = np.asarray(codes, dtype=np.uint32)
    if values.ndim == 0:
        raise ValueError("codes must have at least one dimension")

    mask = (1 << bits) - 1
    if np.any(values > mask):
        raise ValueError("codes contain values that do not fit in the requested bit width")

    symbol_count = values.shape[-1]
    word_count = words_per_group(symbol_count, bits)
    flat = values.reshape(-1, symbol_count)
    packed = np.zeros((flat.shape[0], word_count), dtype=np.uint32)

    for row_index, row in enumerate(flat):
        bit_offset = 0
        for raw_value in row:
            value = int(raw_value) & mask
            word_index = bit_offset // 32
            bit_index = bit_offset % 32
            packed[row_index, word_index] |= np.uint32(value << bit_index)
            spill = bit_index + bits - 32
            if spill > 0:
                packed[row_index, word_index + 1] |= np.uint32(value >> (bits - spill))
            bit_offset += bits

    return packed.reshape(*values.shape[:-1], word_count)


def unpack_bits(words: np.ndarray, bits: int, group_size: int) -> np.ndarray:
    packed = np.asarray(words, dtype=np.uint32)
    if packed.ndim == 0:
        raise ValueError("words must have at least one dimension")

    expected_words = words_per_group(group_size, bits)
    if packed.shape[-1] != expected_words:
        raise ValueError("word count does not match group_size and bits")

    flat = packed.reshape(-1, expected_words)
    unpacked = np.zeros((flat.shape[0], group_size), dtype=np.uint8)
    mask = (1 << bits) - 1

    for row_index, row in enumerate(flat):
        bit_offset = 0
        for symbol_index in range(group_size):
            word_index = bit_offset // 32
            bit_index = bit_offset % 32
            value = int(row[word_index] >> bit_index)
            spill = bit_index + bits - 32
            if spill > 0:
                value |= int(row[word_index + 1] & ((1 << spill) - 1)) << (bits - spill)
            unpacked[row_index, symbol_index] = value & mask
            bit_offset += bits

    return unpacked.reshape(*packed.shape[:-1], group_size)

