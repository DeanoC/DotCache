from __future__ import annotations

from functools import lru_cache
from math import ceil

import numpy as np

from .m0_affine import pad_last_dim
from .turbo3 import fwht_last_dim


@lru_cache(maxsize=None)
def fixed_project_basis(group_size: int, rank: int) -> np.ndarray:
    if group_size <= 0 or (group_size & (group_size - 1)):
        raise ValueError("M4 fixed-project requires a power-of-two group_size")
    usable_rank = max(1, min(int(rank), group_size - 1))
    basis = fwht_last_dim(np.eye(group_size, dtype=np.float32))
    # Skip the DC row because the page mean already captures the constant offset.
    return np.asarray(basis[1 : 1 + usable_rank], dtype=np.float32)


def quantize_tensor_m4(
    values: np.ndarray,
    *,
    group_size: int,
    project_dim: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")

    token_count, head_dim = array.shape
    num_groups = ceil(head_dim / group_size)
    padded_head_dim = num_groups * group_size
    padded = pad_last_dim(array, padded_head_dim)
    grouped = padded.reshape(token_count, num_groups, group_size)

    basis = fixed_project_basis(group_size, project_dim)
    coeffs = np.zeros((token_count, num_groups, basis.shape[0]), dtype=np.float32)
    mean = np.zeros((num_groups, group_size), dtype=np.float32)

    for group_index in range(num_groups):
        group_values = grouped[:, group_index, :]
        group_mean = group_values.mean(axis=0, dtype=np.float32)
        centered = group_values - group_mean[None, :]
        coeffs[:, group_index, :] = centered @ basis.T
        mean[group_index, :] = group_mean

    return (
        coeffs.astype(np.float16, copy=False),
        mean.astype(np.float16, copy=False),
        padded_head_dim,
    )


def reconstruct_group_m4(coefficients: np.ndarray, *, mean: np.ndarray, group_size: int) -> np.ndarray:
    coeff_array = np.asarray(coefficients, dtype=np.float32)
    rank = int(coeff_array.shape[-1])
    basis = fixed_project_basis(int(group_size), rank)
    reconstructed = coeff_array @ basis
    return reconstructed + np.asarray(mean, dtype=np.float32)[None, :]
