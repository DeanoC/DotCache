from __future__ import annotations

from pathlib import Path

import pytest


def test_cutlass_submodule_version_header_is_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    version = root / "third_party" / "cutlass" / "include" / "cutlass" / "version.h"
    if not version.exists():
        pytest.skip("CUTLASS submodule is not initialized")
    text = version.read_text()
    assert "#define CUTLASS_MAJOR 4" in text
    assert "#define CUTLASS_MINOR 3" in text
    assert "#define CUTLASS_PATCH 1" in text


def test_cutlass_sm120_probe_if_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("SM120 GPU not available")

    from dotcache.backends.cutlass_sm120 import (
        cutlass_sm120_available,
        cutlass_sm120_metadata,
        cutlass_sm120_probe,
    )

    if not cutlass_sm120_available():
        pytest.skip("CUTLASS SM120 extension is not buildable in this environment")

    x = torch.arange(16, device="cuda", dtype=torch.float32)
    y = cutlass_sm120_probe(x)
    torch.cuda.synchronize()
    assert torch.equal(x, y)
    assert "cutlass=4.3.1" in cutlass_sm120_metadata()["metadata"]
