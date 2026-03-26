from __future__ import annotations

import numpy as np


def encode_escape_payload(values: np.ndarray, dtype: str = "float16") -> np.ndarray:
    return np.asarray(values, dtype=np.dtype(dtype))


def decode_escape_payload(payload: np.ndarray, *, head_dim: int | None = None) -> np.ndarray:
    array = np.asarray(payload, dtype=np.float32)
    if head_dim is None:
        return array
    return array[:, :head_dim]

