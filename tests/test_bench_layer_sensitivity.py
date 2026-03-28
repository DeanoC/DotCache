from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_layer_sensitivity.py"
_SPEC = importlib.util.spec_from_file_location("bench_layer_sensitivity", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_select_probe_layers_uses_explicit_values() -> None:
    selected = _MODULE._select_probe_layers(10, explicit_layers=[7, 2, 2, 20, -1], max_layers=4)
    assert selected == [2, 7]


def test_select_probe_layers_spreads_evenly_when_sampling() -> None:
    selected = _MODULE._select_probe_layers(12, explicit_layers=[], max_layers=4)
    assert selected[0] == 0
    assert selected[-1] == 11
    assert len(selected) == 4
    assert selected == sorted(set(selected))
