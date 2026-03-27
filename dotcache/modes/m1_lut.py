from __future__ import annotations

from math import ceil

import numpy as np

from .m0_affine import pad_last_dim


def quantize_tensor_lut(
    values: np.ndarray,
    *,
    group_size: int,
    bits: int,
    refine_steps: int = 6,
    preconditioner: str = "none",
    precondition_strength: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")

    token_count, head_dim = array.shape
    num_groups = ceil(head_dim / group_size)
    padded_head_dim = num_groups * group_size
    padded = pad_last_dim(array, padded_head_dim)
    grouped = padded.reshape(token_count, num_groups, group_size)
    levels = 1 << bits

    codebooks = np.zeros((num_groups, levels), dtype=np.float32)
    codes = np.zeros((token_count, num_groups, group_size), dtype=np.uint8)

    quantile_positions = np.linspace(0.0, 1.0, num=levels, dtype=np.float32)

    for group_index in range(num_groups):
        group_values = grouped[:, group_index, :].reshape(-1)
        fit_values = group_values
        restore_mean = 0.0
        restore_scale = 1.0
        if preconditioner == "tanh":
            restore_mean = float(np.mean(group_values, dtype=np.float64))
            centered = group_values - restore_mean
            restore_scale = float(np.std(centered, dtype=np.float64))
            if restore_scale < 1e-6:
                restore_scale = 1.0
            fit_values = np.tanh(centered / (restore_scale * precondition_strength)).astype(np.float32)
        elif preconditioner != "none":
            raise ValueError("unsupported preconditioner")

        lut = np.quantile(fit_values, quantile_positions).astype(np.float32)
        if levels > 1:
            for _ in range(refine_steps):
                boundaries = (lut[:-1] + lut[1:]) * 0.5
                flat_codes = np.searchsorted(boundaries, fit_values, side="left").astype(np.int32)
                updated = lut.copy()
                for code_index in range(levels):
                    members = fit_values[flat_codes == code_index]
                    if members.size > 0:
                        updated[code_index] = float(np.mean(members, dtype=np.float64))
                if np.allclose(updated, lut, atol=1e-6, rtol=0.0):
                    lut = updated
                    break
                lut = updated
            boundaries = (lut[:-1] + lut[1:]) * 0.5
            source_values = grouped[:, group_index, :]
            if preconditioner == "tanh":
                source_values = np.tanh((source_values - restore_mean) / (restore_scale * precondition_strength)).astype(np.float32)
            group_codes = np.searchsorted(boundaries, source_values, side="left").astype(np.uint8)
        else:
            group_codes = np.zeros((token_count, group_size), dtype=np.uint8)
        if preconditioner == "tanh":
            lut = np.clip(lut, -0.999, 0.999)
            lut = np.arctanh(lut).astype(np.float32) * np.float32(restore_scale * precondition_strength) + np.float32(restore_mean)
        codebooks[group_index] = lut
        codes[:, group_index] = np.clip(group_codes, 0, levels - 1)

    return codes, codebooks, padded_head_dim


def dequantize_group_lut(codes: np.ndarray, *, codebook: np.ndarray) -> np.ndarray:
    code_array = np.asarray(codes, dtype=np.int64)
    lut = np.asarray(codebook, dtype=np.float32)
    if lut.ndim == 1:
        return lut[code_array]
    if lut.ndim == 2:
        if code_array.ndim == 1:
            return lut[np.arange(lut.shape[0]), code_array]
        if code_array.ndim == 2 and lut.shape[0] == code_array.shape[0]:
            token_indices = np.arange(code_array.shape[0])[:, None]
            return lut[token_indices, code_array]
    raise ValueError("unsupported codebook shape for LUT decode")
