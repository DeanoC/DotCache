from __future__ import annotations

import numpy as np

from ..attention_reference import mix_page_ref, score_page_ref
from ..tracing import ExecutionTrace
from ..types import EncodedPage
from .torch_mps import PreparedPageMPS


def _source_page(page: EncodedPage | PreparedPageMPS) -> EncodedPage:
    if isinstance(page, PreparedPageMPS):
        return page.source_page
    return page


def _record_trace(page: EncodedPage, trace: ExecutionTrace | None) -> None:
    if trace is None:
        return
    trace.record_page_read(page.payload_nbytes, page.metadata_nbytes)
    if page.header.mode_default == "M0":
        trace.record_temporary(page.header.token_count * page.header.group_size * np.dtype(np.float32).itemsize)


def score_page_cpu_ref(
    query_slice: np.ndarray,
    page: EncodedPage | PreparedPageMPS,
    *,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    source_page = _source_page(page)
    _record_trace(source_page, trace)
    return score_page_ref(query_slice, source_page)


def mix_page_cpu_ref(
    attn_weights: np.ndarray,
    page: EncodedPage | PreparedPageMPS,
    *,
    out_acc: np.ndarray | None = None,
    trace: ExecutionTrace | None = None,
) -> np.ndarray:
    source_page = _source_page(page)
    _record_trace(source_page, trace)
    mixed = mix_page_ref(attn_weights, source_page)
    if out_acc is None:
        return mixed
    output = np.asarray(out_acc, dtype=np.float32)
    if output.shape != mixed.shape:
        raise ValueError("out_acc must have shape [head_dim]")
    return output + mixed
