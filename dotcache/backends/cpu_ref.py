from __future__ import annotations

import numpy as np

from ..attention_reference import mix_page_ref, score_page_ref
from ..types import EncodedPage


def score_page_cpu_ref(query_slice: np.ndarray, page: EncodedPage) -> np.ndarray:
    return score_page_ref(query_slice, page)


def mix_page_cpu_ref(attn_weights: np.ndarray, page: EncodedPage) -> np.ndarray:
    return mix_page_ref(attn_weights, page)

