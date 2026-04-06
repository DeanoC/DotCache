from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_qwen35_cuda_needle_probe import _case_extra_args


def test_streaming_sink_recent_case_args() -> None:
    assert _case_extra_args("streaming_sink_recent") == [
        "--execution-recent-window",
        "1024",
        "--execution-sink-window",
        "256",
    ]
