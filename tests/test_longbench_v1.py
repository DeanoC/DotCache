from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "dotcache" / "longbench_v1.py"
SPEC = importlib.util.spec_from_file_location("longbench_v1", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_supported_datasets_include_original_longbench_v1_suite() -> None:
    datasets = MODULE.list_supported_datasets()
    assert "hotpotqa" in datasets
    assert "gov_report" in datasets
    assert "passage_retrieval_zh" in datasets
    assert "repobench-p" in datasets
    assert len(datasets) == 21


def test_dataset_spec_exposes_metric_and_task_family() -> None:
    spec = MODULE.get_dataset_spec("trec")
    assert spec.metric_name == "classification"
    assert spec.task_family == "classification"


def test_score_prediction_uses_official_metric_family() -> None:
    qa_score = MODULE.score_prediction("hotpotqa", "Gates v. Collier", ["Gates v. Collier"])
    assert qa_score["longbench_metric_name"] == "qa_f1"
    assert qa_score["longbench_official_score"] == 1.0

    retrieval_score = MODULE.score_prediction(
        "passage_retrieval_en",
        "Paragraph 7",
        ["Paragraph 7"],
    )
    assert retrieval_score["longbench_metric_name"] == "retrieval"
    assert retrieval_score["longbench_official_score"] == 1.0


def test_build_prompt_specs_from_zip_materializes_full_suite_rows(tmp_path: Path) -> None:
    archive_path = tmp_path / "longbench.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/hotpotqa.jsonl", "{}\n{}\n")
        archive.writestr("data/trec.jsonl", "{}\n")

    rows = MODULE.build_prompt_specs_from_zip(
        archive_path,
        datasets=["hotpotqa", "trec"],
    )

    assert rows == [
        {
            "prompt_id": "hotpotqa_row0",
            "dataset": "hotpotqa",
            "row_index": 0,
            "task_family": "qa",
            "metric_name": "qa_f1",
        },
        {
            "prompt_id": "hotpotqa_row1",
            "dataset": "hotpotqa",
            "row_index": 1,
            "task_family": "qa",
            "metric_name": "qa_f1",
        },
        {
            "prompt_id": "trec_row0",
            "dataset": "trec",
            "row_index": 0,
            "task_family": "classification",
            "metric_name": "classification",
        },
    ]


def test_build_prompt_specs_from_zip_supports_stratified_sampling(tmp_path: Path) -> None:
    archive_path = tmp_path / "longbench.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data/hotpotqa.jsonl", "\n".join("{}" for _ in range(10)) + "\n")

    rows = MODULE.build_prompt_specs_from_zip(
        archive_path,
        datasets=["hotpotqa"],
        stratified_limit_per_dataset=4,
    )

    assert [row["row_index"] for row in rows] == [0, 3, 6, 9]


def test_build_length_quartile_prompt_specs_from_zip_records_quartiles(tmp_path: Path) -> None:
    archive_path = tmp_path / "longbench.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        rows = []
        for length in range(16):
            rows.append(json.dumps({"length": length}))
        archive.writestr("data/hotpotqa.jsonl", "\n".join(rows) + "\n")

    prompt_specs = MODULE.build_length_quartile_prompt_specs_from_zip(
        archive_path,
        datasets=["hotpotqa"],
        rows_per_quartile=1,
        seed=7,
    )

    assert len(prompt_specs) == 4
    assert {item["length_quartile"] for item in prompt_specs} == {0, 1, 2, 3}
    assert all("row_length" in item for item in prompt_specs)


def test_qa_f1_score_matches_existing_behavior() -> None:
    assert MODULE.qa_f1_score("Vice Admiral", "Vice Admiral.") == 1.0
