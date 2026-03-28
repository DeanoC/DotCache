from __future__ import annotations

import numpy as np
from pathlib import Path

from dotcache.state_cache_sim import (
    StateTileSpec,
    load_captured_state_sample,
    simulate_state_codec,
    simulate_state_sequence,
)


def test_state_cache_codec_round_trip_shapes_and_bytes() -> None:
    tile = np.arange(32, dtype=np.float32).reshape(4, 8)
    decoded, payload_nbytes, metadata_nbytes = simulate_state_codec(
        tile,
        StateTileSpec(state_rows=4, state_cols=8, group_size=4, bits=4, mode="M0"),
    )
    assert decoded.shape == tile.shape
    assert payload_nbytes == 16
    assert metadata_nbytes == 64

    escape_decoded, escape_payload, escape_metadata = simulate_state_codec(
        tile,
        StateTileSpec(state_rows=4, state_cols=8, bits=32, mode="M3", escape_dtype="float32"),
    )
    assert np.allclose(escape_decoded, tile)
    assert escape_payload == tile.nbytes
    assert escape_metadata == 0


def test_state_cache_sequence_reports_deterministic_curves_and_renorm_changes_them() -> None:
    initial_state = np.linspace(-1.25, 1.75, 32, dtype=np.float32).reshape(4, 8)
    update_deltas = np.linspace(-0.15, 0.25, 96, dtype=np.float32).reshape(3, 4, 8)
    readout_projections = np.linspace(-0.4, 0.6, 48, dtype=np.float32).reshape(3, 8, 2)
    spec = StateTileSpec(state_rows=4, state_cols=8, group_size=4, bits=4, mode="M0")

    base = simulate_state_sequence(
        initial_state,
        update_deltas,
        readout_projections,
        spec=spec,
        renorm_interval=0,
    )
    renormed = simulate_state_sequence(
        initial_state,
        update_deltas,
        readout_projections,
        spec=spec,
        renorm_interval=2,
    )

    assert base.bytes_per_layer > 0
    assert base.bytes_per_token == base.bytes_per_layer * 2
    assert len(base.update_error_curve) == 3
    assert len(base.readout_error_curve) == 3
    assert base.to_dict()["mode"] == "M0"
    assert renormed.renorm_interval == 2
    assert renormed.update_error_curve != base.update_error_curve or renormed.readout_error_curve != base.readout_error_curve


def test_load_captured_state_sample_round_trips(tmp_path: Path) -> None:
    sample_path = tmp_path / "captured.npz"
    np.savez_compressed(
        sample_path,
        source=np.asarray("qwen35_deltanet_capture"),
        state_kind=np.asarray("recurrent"),
        layer_id=np.asarray(7, dtype=np.int64),
        prompt_length=np.asarray(32, dtype=np.int64),
        token_indices=np.asarray([32, 33], dtype=np.int64),
        initial_state=np.arange(24, dtype=np.float32).reshape(3, 8),
        update_deltas=np.ones((2, 3, 8), dtype=np.float32),
    )
    sample = load_captured_state_sample(sample_path)
    assert sample.source == "qwen35_deltanet_capture"
    assert sample.state_kind == "recurrent"
    assert sample.layer_id == 7
    assert sample.prompt_length == 32
    assert sample.token_indices == [32, 33]
    assert sample.initial_state.shape == (3, 8)
    assert sample.update_deltas.shape == (2, 3, 8)
