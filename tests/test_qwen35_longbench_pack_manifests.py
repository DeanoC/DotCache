from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_rows(name: str):
    path = REPO_ROOT / "configs" / "prompt_packs" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_longbench_pack_tiers_have_expected_sizes_and_unique_ids() -> None:
    mini = _load_rows("qwen35_cuda_longbench_qa_pack_v1.json")
    medium = _load_rows("qwen35_cuda_longbench_qa_pack_medium_v1.json")
    full = _load_rows("qwen35_cuda_longbench_qa_pack_full_v1.json")

    assert len(mini) == 4
    assert len(medium) == 12
    assert len(full) == 20

    assert len({row["prompt_id"] for row in mini}) == len(mini)
    assert len({row["prompt_id"] for row in medium}) == len(medium)
    assert len({row["prompt_id"] for row in full}) == len(full)


def test_longbench_pack_tiers_are_nested_by_dataset_and_row() -> None:
    mini = _load_rows("qwen35_cuda_longbench_qa_pack_v1.json")
    medium = _load_rows("qwen35_cuda_longbench_qa_pack_medium_v1.json")
    full = _load_rows("qwen35_cuda_longbench_qa_pack_full_v1.json")

    mini_pairs = {(row["dataset"], int(row["row_index"])) for row in mini}
    medium_pairs = {(row["dataset"], int(row["row_index"])) for row in medium}
    full_pairs = {(row["dataset"], int(row["row_index"])) for row in full}

    assert mini_pairs.issubset(medium_pairs)
    assert medium_pairs.issubset(full_pairs)
