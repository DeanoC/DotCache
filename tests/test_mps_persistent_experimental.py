from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotcache.attention_reference import mix_page_ref, score_page_ref, softmax
from dotcache.backends import mps_available
from dotcache.backends.mps_persistent_experimental import (
    PagedAttentionControllerConfig,
    build_synthetic_snapshot,
    decode_selected_pages_mps,
    load_paged_attention_snapshot,
    prepare_resident_layer_pages,
    run_paged_attention_step,
    run_reference_step,
    save_paged_attention_snapshot,
    score_pages_mps,
    score_pages_reference,
    select_pages_mps,
    select_pages_reference,
)
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page

requires_mps = pytest.mark.skipif(not mps_available(), reason="torch_mps is unavailable")


def _build_manual_snapshot():
    head_dim = 8
    tokens_per_page = 4
    num_pages = 5
    query = np.zeros(head_dim, dtype=np.float32)
    query[0] = 1.0
    k_pages = np.zeros((num_pages, tokens_per_page, head_dim), dtype=np.float32)
    v_pages = np.linspace(-1.0, 1.0, num_pages * tokens_per_page * head_dim, dtype=np.float32).reshape(
        num_pages,
        tokens_per_page,
        head_dim,
    )
    k_pages[2, :, 0] = 10.0
    k_pages[4, :2, 0] = 2.0
    page_token_counts = np.asarray([4, 4, 4, 4, 2], dtype=np.int64)
    page_token_starts = np.asarray([0, 4, 8, 12, 16], dtype=np.int64)
    prev_attn = np.asarray([0.4, 0.1, 0.6, 0.2, 0.9], dtype=np.float32)
    distance = np.asarray([0.0, 4.0, 8.0, 12.0, 16.0], dtype=np.float32)
    page_k_mean = np.stack(
        [k_pages[index, : int(page_token_counts[index]), :].mean(axis=0) for index in range(num_pages)],
        axis=0,
    ).astype(np.float32, copy=False)
    return {
        "query": query,
        "page_k_mean": page_k_mean,
        "prev_attn": prev_attn,
        "distance": distance,
        "k_pages": k_pages,
        "v_pages": v_pages,
        "page_token_counts": page_token_counts,
        "page_token_starts": page_token_starts,
    }


def _attention_reference_output(
    query: np.ndarray,
    *,
    k_pages: np.ndarray,
    v_pages: np.ndarray,
    selected_page_ids: list[int],
    page_token_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = DotCacheConfig(
        head_dim=int(query.shape[0]),
        group_size=int(query.shape[0]),
        bits_k=4,
        bits_v=4,
        tokens_per_page=int(k_pages.shape[1]),
        default_mode_k="M3",
        default_mode_v="M3",
        escape_dtype="float32",
    )
    key_pages = []
    value_pages = []
    logits_parts = []
    for page_id in selected_page_ids:
        token_count = int(page_token_counts[page_id])
        key_page = encode_page(k_pages[page_id, :token_count, :], config, kind="K", mode="M3")
        value_page = encode_page(v_pages[page_id, :token_count, :], config, kind="V", mode="M3")
        key_pages.append(key_page)
        value_pages.append(value_page)
        logits_parts.append(score_page_ref(query, key_page))
    logits = np.concatenate(logits_parts, axis=0).astype(np.float32, copy=False)
    weights = softmax(logits)
    output = np.zeros(query.shape[0], dtype=np.float32)
    offset = 0
    for value_page in value_pages:
        token_count = value_page.header.token_count
        output = mix_page_ref(weights[offset : offset + token_count], value_page, out_acc=output)
        offset += token_count
    return logits, weights.astype(np.float32, copy=False), output.astype(np.float32, copy=False)


def test_snapshot_round_trip(tmp_path: Path) -> None:
    snapshot = build_synthetic_snapshot(num_pages=3, tokens_per_page=4, head_dim=8, seed=7, partial_last_page_tokens=2)
    path = tmp_path / "paged_attention_snapshot.npz"

    save_paged_attention_snapshot(path, snapshot)
    loaded = load_paged_attention_snapshot(path)

    assert loaded.source == "mps_persistent_experimental"
    assert np.array_equal(loaded.page_token_counts, snapshot.page_token_counts)
    assert np.array_equal(loaded.page_token_starts, snapshot.page_token_starts)
    assert np.allclose(loaded.k_pages, snapshot.k_pages)
    assert np.allclose(loaded.v_pages, snapshot.v_pages)


def test_score_pages_matches_reference_on_cpu_resident() -> None:
    snapshot = build_synthetic_snapshot(num_pages=4, tokens_per_page=4, head_dim=8, seed=11, partial_last_page_tokens=3)
    resident = prepare_resident_layer_pages(
        page_k_mean=snapshot.page_k_mean,
        prev_attn=snapshot.prev_attn,
        distance=snapshot.distance,
        k_pages=snapshot.k_pages,
        v_pages=snapshot.v_pages,
        page_token_counts=snapshot.page_token_counts,
        page_token_starts=snapshot.page_token_starts,
        device="cpu",
    )

    actual = score_pages_mps(snapshot.query, resident).detach().cpu().numpy()
    expected = score_pages_reference(snapshot.query, snapshot.page_k_mean, snapshot.prev_attn, snapshot.distance)

    assert np.allclose(actual, expected, atol=1e-6)


def test_decode_selected_pages_matches_attention_reference_without_early_exit() -> None:
    snapshot = build_synthetic_snapshot(num_pages=3, tokens_per_page=4, head_dim=8, seed=19, partial_last_page_tokens=2)
    resident = prepare_resident_layer_pages(
        page_k_mean=snapshot.page_k_mean,
        prev_attn=snapshot.prev_attn,
        distance=snapshot.distance,
        k_pages=snapshot.k_pages,
        v_pages=snapshot.v_pages,
        page_token_counts=snapshot.page_token_counts,
        page_token_starts=snapshot.page_token_starts,
        device="cpu",
    )
    selected_page_ids = [0, 2]

    actual = decode_selected_pages_mps(
        snapshot.query,
        resident,
        selected_page_ids,
        page_chunk_size=1,
        early_exit=False,
        return_debug=True,
    )
    expected_logits, expected_weights, expected_output = _attention_reference_output(
        snapshot.query,
        k_pages=snapshot.k_pages,
        v_pages=snapshot.v_pages,
        selected_page_ids=selected_page_ids,
        page_token_counts=snapshot.page_token_counts,
    )

    assert actual.early_exit_triggered is False
    assert actual.pages_processed == len(selected_page_ids)
    assert np.allclose(actual.logits, expected_logits, atol=1e-5)
    assert np.allclose(actual.weights, expected_weights, atol=1e-5)
    assert np.allclose(actual.output, expected_output, atol=1e-5)


def test_controller_keeps_sink_and_recent_pages_and_reports_stable_counts() -> None:
    snapshot = _build_manual_snapshot()
    config = PagedAttentionControllerConfig(
        sink_window_tokens=4,
        recent_window_tokens=2,
        top_k=1,
        page_chunk_size=1,
        early_exit=False,
    )
    resident = prepare_resident_layer_pages(
        page_k_mean=snapshot["page_k_mean"],
        prev_attn=snapshot["prev_attn"],
        distance=snapshot["distance"],
        k_pages=snapshot["k_pages"],
        v_pages=snapshot["v_pages"],
        page_token_counts=snapshot["page_token_counts"],
        page_token_starts=snapshot["page_token_starts"],
        device="cpu",
    )

    selection_ref = select_pages_reference(
        snapshot["query"],
        page_k_mean=snapshot["page_k_mean"],
        prev_attn=snapshot["prev_attn"],
        distance=snapshot["distance"],
        page_token_starts=snapshot["page_token_starts"],
        page_token_counts=snapshot["page_token_counts"],
        config=config,
    )
    selection_torch = select_pages_mps(snapshot["query"], resident, config=config)
    result = run_paged_attention_step(snapshot["query"], resident, config=config, engine="mps_experimental")

    assert selection_ref.selected_page_ids == [0, 2, 4]
    assert selection_torch.selected_page_ids == [0, 2, 4]
    assert result.selected_page_ids == [0, 2, 4]
    assert result.selected_page_count == 3
    assert result.processed_page_count == 3
    assert result.tokens_processed == 10
    assert np.isfinite(result.output).all()


@requires_mps
def test_score_pages_mps_smoke_matches_reference() -> None:
    snapshot = build_synthetic_snapshot(num_pages=4, tokens_per_page=8, head_dim=16, seed=21, partial_last_page_tokens=5)
    resident = prepare_resident_layer_pages(
        page_k_mean=snapshot.page_k_mean,
        prev_attn=snapshot.prev_attn,
        distance=snapshot.distance,
        k_pages=snapshot.k_pages,
        v_pages=snapshot.v_pages,
        page_token_counts=snapshot.page_token_counts,
        page_token_starts=snapshot.page_token_starts,
        device="mps",
    )

    actual = score_pages_mps(snapshot.query, resident).detach().cpu().numpy()
    expected = score_pages_reference(snapshot.query, snapshot.page_k_mean, snapshot.prev_attn, snapshot.distance)

    assert np.allclose(actual, expected, atol=1e-4)


@requires_mps
def test_run_paged_attention_step_mps_smoke() -> None:
    snapshot = build_synthetic_snapshot(num_pages=6, tokens_per_page=8, head_dim=16, seed=33, partial_last_page_tokens=3)
    resident = prepare_resident_layer_pages(
        page_k_mean=snapshot.page_k_mean,
        prev_attn=snapshot.prev_attn,
        distance=snapshot.distance,
        k_pages=snapshot.k_pages,
        v_pages=snapshot.v_pages,
        page_token_counts=snapshot.page_token_counts,
        page_token_starts=snapshot.page_token_starts,
        device="mps",
    )
    config = PagedAttentionControllerConfig(
        sink_window_tokens=8,
        recent_window_tokens=8,
        top_k=2,
        page_chunk_size=2,
        early_exit=False,
    )

    result = run_paged_attention_step(snapshot.query, resident, config=config, engine="mps_experimental")
    reference = run_reference_step(snapshot, config=config)

    assert result.selected_page_count >= 2
    assert result.processed_page_count == result.selected_page_count
    assert result.tokens_processed > 0
    assert np.isfinite(result.output).all()
    assert np.allclose(result.output, reference.output, atol=5e-4, rtol=5e-4)
