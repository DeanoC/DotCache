from __future__ import annotations

from math import ceil

import numpy as np

from .m0_affine import pad_last_dim


def quantize_tensor_m2(
    values: np.ndarray,
    *,
    group_size: int,
    sketch_dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")

    token_count, head_dim = array.shape
    num_groups = ceil(head_dim / group_size)
    padded_head_dim = num_groups * group_size
    padded = pad_last_dim(array, padded_head_dim)
    grouped = padded.reshape(token_count, num_groups, group_size)

    rank = max(1, min(int(sketch_dim), group_size, token_count))
    coeffs = np.zeros((token_count, num_groups, rank), dtype=np.float32)
    basis = np.zeros((num_groups, rank, group_size), dtype=np.float32)

    for group_index in range(num_groups):
        group_values = grouped[:, group_index, :]
        u, s, vt = np.linalg.svd(group_values, full_matrices=False)
        coeffs[:, group_index, :] = (u[:, :rank] * s[:rank]).astype(np.float32, copy=False)
        basis[group_index, :, :] = vt[:rank, :].astype(np.float32, copy=False)

    return coeffs.astype(np.float16, copy=False), basis.astype(np.float16, copy=False), padded_head_dim


def reconstruct_group_m2(coefficients: np.ndarray, *, basis: np.ndarray) -> np.ndarray:
    coeff_array = np.asarray(coefficients, dtype=np.float32)
    basis_array = np.asarray(basis, dtype=np.float32)
    return coeff_array @ basis_array
