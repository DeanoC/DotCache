from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_qwen35_cuda_longbench_qa_probe import _load_prompt_specs, _longbench_case_extra_args


def test_load_prompt_specs_reads_dataset_and_row_index(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(
        '[{"prompt_id":"hotpot_case_order","dataset":"hotpotqa","row_index":0}]',
        encoding="utf-8",
    )

    assert _load_prompt_specs(str(pack_path)) == [
        {"prompt_id": "hotpot_case_order", "dataset": "hotpotqa", "row_index": 0}
    ]


def test_longbench_case_extra_args_supports_topk8() -> None:
    assert _longbench_case_extra_args("shortlist_topk8") == [
        "--execution-recent-window",
        "1024",
        "--execution-sink-window",
        "256",
        "--execution-relevance-top-k",
        "8",
        "--execution-relevance-mode",
        "envelope",
    ]
