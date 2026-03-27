from __future__ import annotations

from functools import lru_cache
from math import ceil

import numpy as np

from .m0_affine import pad_last_dim


@lru_cache(maxsize=128)
def projection_matrices(num_groups: int, group_size: int, sketch_dim: int) -> np.ndarray:
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if sketch_dim <= 0:
        raise ValueError("sketch_dim must be positive")
    rng = np.random.default_rng((num_groups * 73856093) ^ (group_size * 19349663) ^ (sketch_dim * 83492791))
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(num_groups, group_size, sketch_dim))
    return (signs / np.float32(np.sqrt(sketch_dim))).astype(np.float32, copy=False)


def quantize_tensor_m2(
    values: np.ndarray,
    *,
    group_size: int,
    sketch_dim: int,
) -> tuple[np.ndarray, int]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("values must have shape [token_count, head_dim]")

    token_count, head_dim = array.shape
    num_groups = ceil(head_dim / group_size)
    padded_head_dim = num_groups * group_size
    padded = pad_last_dim(array, padded_head_dim)
    grouped = padded.reshape(token_count, num_groups, group_size)
    projection = projection_matrices(num_groups, group_size, sketch_dim)
    sketches = np.einsum("tng,ngd->tnd", grouped, projection, optimize=True).astype(np.float16, copy=False)
    return sketches, padded_head_dim


def reconstruct_group_m2(sketch: np.ndarray, *, projection: np.ndarray) -> np.ndarray:
    sketch_array = np.asarray(sketch, dtype=np.float32)
    projection_array = np.asarray(projection, dtype=np.float32)
    return sketch_array @ projection_array.T
