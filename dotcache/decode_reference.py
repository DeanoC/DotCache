from __future__ import annotations

import numpy as np

from .modes.m0_affine import dequantize_group
from .modes.m1_lut import dequantize_group_lut
from .modes.m3_escape import decode_escape_payload
from .page_format import load_group_words
from .packing import unpack_bits
from .types import EncodedPage


def decode_group_ref(page: EncodedPage, group_index: int) -> np.ndarray:
    page.record_group_decode()
    header = page.header

    if header.mode_default == "M3":
        if page.escape_payload is None:
            raise ValueError("escape payload is missing")
        start = group_index * header.group_size
        end = start + header.group_size
        return decode_escape_payload(page.escape_payload)[:, start:end]

    words = load_group_words(page, group_index)
    codes = unpack_bits(words, header.bits, header.group_size)
    if header.mode_default == "M1":
        if page.codebooks is None:
            raise ValueError("M1 page is missing codebooks")
        codebook = page.codebooks[group_index].astype(np.float32)
        return dequantize_group_lut(codes, codebook=codebook)
    if page.payload is None or page.scales is None:
        raise ValueError("M0 page is missing payload or scales")
    scales = page.scales[:, group_index].astype(np.float32)[:, None]
    bias = None
    if page.bias is not None:
        bias = page.bias[:, group_index].astype(np.float32)[:, None]
    return dequantize_group(
        codes,
        scales=scales,
        bias=bias,
        bits=header.bits,
        scheme=header.quant_scheme,
    )


def decode_page(page: EncodedPage) -> np.ndarray:
    page.record_full_decode()
    header = page.header

    if header.mode_default == "M3":
        if page.escape_payload is None:
            raise ValueError("escape payload is missing")
        return decode_escape_payload(page.escape_payload, head_dim=header.head_dim)

    groups = [decode_group_ref(page, group_index) for group_index in range(header.num_groups)]
    full = np.concatenate(groups, axis=-1)
    return full[:, : header.head_dim]
