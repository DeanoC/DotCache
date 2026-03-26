from __future__ import annotations

import numpy as np

from .attention_reference import mix_page_ref, score_page_ref, softmax
from .types import EncodedPage


def attention_step(query_slice: np.ndarray, key_page: EncodedPage, value_page: EncodedPage) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = score_page_ref(query_slice, key_page)
    weights = softmax(logits)
    output = mix_page_ref(weights, value_page)
    return logits, weights, output

