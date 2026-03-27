from __future__ import annotations

from math import ceil

import numpy as np

from .m0_affine import pad_last_dim


def quantize_tensor_m2(
    values: np.ndarray,
    *,
    group_size: int,
    sketch_dim: int,
    center: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
    mean = np.zeros((num_groups, group_size), dtype=np.float32)

    for group_index in range(num_groups):
        group_values = grouped[:, group_index, :]
        if center:
            group_mean = group_values.mean(axis=0, dtype=np.float32)
            residual = group_values - group_mean[None, :]
            mean[group_index, :] = group_mean
        else:
            residual = group_values
        u, s, vt = np.linalg.svd(residual, full_matrices=False)
        coeffs[:, group_index, :] = (u[:, :rank] * s[:rank]).astype(np.float32, copy=False)
        basis[group_index, :, :] = vt[:rank, :].astype(np.float32, copy=False)

    return (
        coeffs.astype(np.float16, copy=False),
        basis.astype(np.float16, copy=False),
        mean.astype(np.float16, copy=False),
        padded_head_dim,
    )


def reconstruct_group_m2(coefficients: np.ndarray, *, basis: np.ndarray, mean: np.ndarray | None = None) -> np.ndarray:
    coeff_array = np.asarray(coefficients, dtype=np.float32)
    basis_array = np.asarray(basis, dtype=np.float32)
    reconstructed = coeff_array @ basis_array
    if mean is not None:
        reconstructed = reconstructed + np.asarray(mean, dtype=np.float32)[None, :]
    return reconstructed
