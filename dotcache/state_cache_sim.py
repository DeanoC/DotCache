from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

StateMode = Literal["M0", "M3"]


@dataclass(slots=True)
class StateTileSpec:
    state_rows: int
    state_cols: int
    group_size: int = 32
    bits: int = 8
    mode: StateMode = "M0"
    escape_dtype: str = "float32"

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(slots=True)
class StateLayerRecord:
    layer_id: int
    layer_type: str
    state_family: str
    conv_state_bytes: int
    recurrent_state_bytes: int
    layer_state_bytes: int
    state_shapes: dict[str, list[int]] = field(default_factory=dict)
    state_delta_norms: list[dict[str, float | int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StateAblationResult:
    stage_name: str
    bits: int | None
    max_abs_error: float
    max_rel_error: float
    output_max_abs_error: float
    error_grows_step_to_step: bool
    per_layer_max_abs_error: dict[str, float] = field(default_factory=dict)
    per_layer_max_rel_error: dict[str, float] = field(default_factory=dict)
    per_layer_output_max_abs_error: dict[str, float] = field(default_factory=dict)
    per_step_output_max_abs_error: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StateSimResult:
    mode: StateMode
    bits: int
    group_size: int
    renorm_interval: int
    bytes_per_token: int
    bytes_per_layer: int
    effective_compression_ratio: float
    update_error_curve: list[float] = field(default_factory=list)
    readout_error_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CapturedStateSample:
    source: str
    state_kind: str
    layer_id: int
    prompt_length: int
    token_indices: list[int]
    initial_state: np.ndarray
    update_deltas: np.ndarray

    @property
    def state_rows(self) -> int:
        return int(np.prod(self.initial_state.shape[:-1], dtype=np.int64)) if self.initial_state.ndim > 1 else 1

    @property
    def state_cols(self) -> int:
        return int(self.initial_state.shape[-1])

    @property
    def steps(self) -> int:
        return int(self.update_deltas.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "state_kind": self.state_kind,
            "layer_id": self.layer_id,
            "prompt_length": self.prompt_length,
            "token_indices": list(self.token_indices),
            "state_rows": self.state_rows,
            "state_cols": self.state_cols,
            "steps": self.steps,
        }


def _codec_dtype_bytes(escape_dtype: str) -> int:
    if escape_dtype == "float16":
        return np.dtype(np.float16).itemsize
    if escape_dtype == "float32":
        return np.dtype(np.float32).itemsize
    raise ValueError(f"unsupported StateCache escape_dtype {escape_dtype!r}")


def _renorm_rows(array: np.ndarray) -> np.ndarray:
    flat = array.reshape(-1, array.shape[-1]).astype(np.float32, copy=True)
    row_norms = np.linalg.norm(flat, axis=-1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-8)
    flat = flat / row_norms
    return flat.reshape(array.shape)


def simulate_state_codec(tile: np.ndarray, spec: StateTileSpec) -> tuple[np.ndarray, int, int]:
    array = np.asarray(tile, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError("state tiles must be at least 2D")
    if spec.mode == "M3":
        payload_nbytes = int(array.size * _codec_dtype_bytes(spec.escape_dtype))
        return array.astype(np.float32, copy=True), payload_nbytes, 0
    if spec.mode != "M0":
        raise ValueError(f"unsupported StateCache mode {spec.mode!r}")
    if spec.bits <= 0:
        raise ValueError("spec.bits must be positive")

    flat = array.reshape(-1, array.shape[-1]).astype(np.float32, copy=False)
    group_size = max(int(spec.group_size), 1)
    levels = max((1 << int(spec.bits)) - 1, 1)
    decoded = np.empty_like(flat)
    num_groups = 0

    for row_id in range(flat.shape[0]):
        row = flat[row_id]
        for start in range(0, row.shape[0], group_size):
            end = min(start + group_size, row.shape[0])
            group = row[start:end]
            num_groups += 1
            lo = float(group.min(initial=0.0))
            hi = float(group.max(initial=0.0))
            if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-8):
                decoded[row_id, start:end] = lo
                continue
            scale = (hi - lo) / float(levels)
            quantized = np.rint((group - lo) / scale).clip(0, levels)
            decoded[row_id, start:end] = quantized.astype(np.float32) * np.float32(scale) + np.float32(lo)

    payload_nbytes = int(math.ceil(array.size * int(spec.bits) / 8.0))
    metadata_nbytes = int(num_groups * 8)
    return decoded.reshape(array.shape), payload_nbytes, metadata_nbytes


def simulate_state_sequence(
    initial_state: np.ndarray,
    update_deltas: np.ndarray,
    readout_projections: np.ndarray,
    *,
    spec: StateTileSpec,
    renorm_interval: int = 0,
) -> StateSimResult:
    dense_state = np.asarray(initial_state, dtype=np.float32).copy()
    approx_state = dense_state.copy()
    deltas = np.asarray(update_deltas, dtype=np.float32)
    projections = np.asarray(readout_projections, dtype=np.float32)
    if deltas.ndim < 3:
        raise ValueError("update_deltas must have shape [steps, rows, cols]")
    if projections.ndim != 3:
        raise ValueError("readout_projections must have shape [steps, cols, output_dim]")
    if deltas.shape[0] != projections.shape[0]:
        raise ValueError("update_deltas and readout_projections must have the same step count")
    if tuple(deltas.shape[1:]) != tuple(dense_state.shape):
        raise ValueError("update_deltas shape must match initial_state")
    if projections.shape[1] != dense_state.shape[-1]:
        raise ValueError("readout projection input dimension must match state_cols")

    update_error_curve: list[float] = []
    readout_error_curve: list[float] = []
    stored_bytes = dense_state.nbytes
    compressed_bytes = stored_bytes

    for step_index in range(deltas.shape[0]):
        decoded_state, payload_nbytes, metadata_nbytes = simulate_state_codec(approx_state, spec)
        compressed_bytes = int(payload_nbytes + metadata_nbytes)
        dense_state = dense_state + deltas[step_index]
        approx_after = decoded_state + deltas[step_index]

        if renorm_interval > 0 and (step_index + 1) % int(renorm_interval) == 0:
            dense_state = _renorm_rows(dense_state)
            approx_after = _renorm_rows(approx_after)

        approx_state, _, _ = simulate_state_codec(approx_after, spec)

        dense_readout = dense_state @ projections[step_index]
        approx_readout = approx_state @ projections[step_index]
        update_error_curve.append(float(np.max(np.abs(approx_state - dense_state))))
        readout_error_curve.append(float(np.max(np.abs(approx_readout - dense_readout))))

    bytes_per_layer = int(compressed_bytes)
    bytes_per_token = int(bytes_per_layer * 2)
    effective_ratio = float(stored_bytes / max(bytes_per_layer, 1))
    return StateSimResult(
        mode=spec.mode,
        bits=int(spec.bits),
        group_size=int(spec.group_size),
        renorm_interval=int(renorm_interval),
        bytes_per_token=bytes_per_token,
        bytes_per_layer=bytes_per_layer,
        effective_compression_ratio=effective_ratio,
        update_error_curve=update_error_curve,
        readout_error_curve=readout_error_curve,
    )


def load_captured_state_sample(path: str | Path) -> CapturedStateSample:
    with np.load(Path(path), allow_pickle=False) as data:
        source = str(data["source"].item())
        state_kind = str(data["state_kind"].item())
        layer_id = int(data["layer_id"].item())
        prompt_length = int(data["prompt_length"].item())
        token_indices = [int(value) for value in np.asarray(data["token_indices"]).tolist()]
        initial_state = np.asarray(data["initial_state"], dtype=np.float32)
        update_deltas = np.asarray(data["update_deltas"], dtype=np.float32)
    if initial_state.ndim < 2:
        raise ValueError("captured initial_state must be at least 2D")
    if update_deltas.ndim < 3:
        raise ValueError("captured update_deltas must be at least 3D")
    if tuple(update_deltas.shape[1:]) != tuple(initial_state.shape):
        raise ValueError("captured update_deltas shape must match initial_state shape")
    return CapturedStateSample(
        source=source,
        state_kind=state_kind,
        layer_id=layer_id,
        prompt_length=prompt_length,
        token_indices=token_indices,
        initial_state=initial_state,
        update_deltas=update_deltas,
    )


__all__ = [
    "CapturedStateSample",
    "StateAblationResult",
    "StateLayerRecord",
    "StateSimResult",
    "StateTileSpec",
    "load_captured_state_sample",
    "simulate_state_codec",
    "simulate_state_sequence",
]
