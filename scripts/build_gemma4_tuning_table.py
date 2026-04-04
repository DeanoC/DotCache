#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "dotcache" / "integrations" / "data" / "gemma4_text_tuning_table.json"

PROFILE_ARTIFACTS = (
    REPO_ROOT / "benchmarks" / "results" / "gemma4_profile_sweep_20260404_matrix" / "gemma4_profile_sweep.jsonl",
    REPO_ROOT / "benchmarks" / "results" / "gemma4_profile_sweep_20260404_matrix_4096" / "gemma4_profile_sweep.jsonl",
    REPO_ROOT / "benchmarks" / "results" / "gemma4_profile_sweep_20260404_adaptive_confirm" / "gemma4_profile_sweep.jsonl",
)
KNOB_ARTIFACTS = (
    REPO_ROOT / "benchmarks" / "results" / "gemma4_profile_sweep_20260404_knob_grid_focus" / "gemma4_profile_sweep.jsonl",
)
VALUE_ARTIFACTS = (
    REPO_ROOT / "benchmarks" / "results" / "gemma4_profile_sweep_20260404_adaptive_values_final" / "gemma4_profile_sweep.jsonl",
)

RULE_SPECS = (
    {
        "min_prompt_length": 1,
        "max_prompt_length": 2047,
        "min_decode_budget": 1,
        "max_decode_budget": 23,
        "source_prompt_length": 1024,
        "source_decode_budget": 16,
    },
    {
        "min_prompt_length": 1,
        "max_prompt_length": 2047,
        "min_decode_budget": 24,
        "max_decode_budget": None,
        "source_prompt_length": 1024,
        "source_decode_budget": 24,
    },
    {
        "min_prompt_length": 2048,
        "max_prompt_length": 4095,
        "min_decode_budget": 1,
        "max_decode_budget": 23,
        "source_prompt_length": 2048,
        "source_decode_budget": 16,
    },
    {
        "min_prompt_length": 2048,
        "max_prompt_length": 4095,
        "min_decode_budget": 24,
        "max_decode_budget": None,
        "source_prompt_length": 2048,
        "source_decode_budget": 24,
    },
    {
        "min_prompt_length": 4096,
        "max_prompt_length": None,
        "min_decode_budget": 1,
        "max_decode_budget": 23,
        "source_prompt_length": 4096,
        "source_decode_budget": 16,
    },
    {
        "min_prompt_length": 4096,
        "max_prompt_length": None,
        "min_decode_budget": 24,
        "max_decode_budget": 31,
        "source_prompt_length": 4096,
        "source_decode_budget": 24,
    },
    {
        "min_prompt_length": 4096,
        "max_prompt_length": None,
        "min_decode_budget": 32,
        "max_decode_budget": None,
        "source_prompt_length": 4096,
        "source_decode_budget": 32,
    },
)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _score_row(row: dict[str, Any]) -> tuple[float, int]:
    return (
        float(row["teacher_forced_logit_max_abs_error"]),
        int(row["resident_bytes"]),
    )


def _best_profile_rows() -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for path in PROFILE_ARTIFACTS:
        for row in _read_jsonl_rows(path):
            if row.get("profile") == "adaptive":
                continue
            if float(row["greedy_token_agreement_rate"]) < 1.0:
                continue
            key = (int(row["prompt_length"]), int(row["max_new_tokens"]))
            current = rows.get(key)
            if current is None or _score_row(row) < _score_row(current):
                rows[key] = dict(row, _artifact_path=str(path.relative_to(REPO_ROOT)))
    return rows


def _best_knob_rows() -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for path in KNOB_ARTIFACTS:
        for row in _read_jsonl_rows(path):
            if float(row["greedy_token_agreement_rate"]) < 1.0:
                continue
            key = (int(row["prompt_length"]), int(row["max_new_tokens"]))
            current = rows.get(key)
            if current is None or _score_row(row) < _score_row(current):
                rows[key] = dict(row, _artifact_path=str(path.relative_to(REPO_ROOT)))
    return rows


def _value_layers_by_workload() -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for path in VALUE_ARTIFACTS:
        for row in _read_jsonl_rows(path):
            if float(row["greedy_token_agreement_rate"]) < 1.0:
                continue
            key = (int(row["prompt_length"]), int(row["max_new_tokens"]))
            overrides = tuple(
                sorted(
                    int(entry.split(":", 1)[1].split("=", 1)[0])
                    for entry in row.get("value_mode_overrides", ())
                    if str(entry).startswith("layer:")
                )
            )
            rows[key] = {
                "exact_value_layers": overrides or None,
                "artifact_path": str(path.relative_to(REPO_ROOT)),
                "teacher_forced_logit_max_abs_error": float(row["teacher_forced_logit_max_abs_error"]),
                "resident_bytes": int(row["resident_bytes"]),
            }
    return rows


def build_tuning_table() -> dict[str, Any]:
    profile_rows = _best_profile_rows()
    knob_rows = _best_knob_rows()
    value_rows = _value_layers_by_workload()

    rules: list[dict[str, Any]] = []
    for spec in RULE_SPECS:
        source_key = (int(spec["source_prompt_length"]), int(spec["source_decode_budget"]))
        profile_row = profile_rows[source_key]
        knob_row = knob_rows.get(source_key)
        value_row = value_rows.get(source_key)
        rules.append(
            {
                "min_prompt_length": int(spec["min_prompt_length"]),
                "max_prompt_length": None if spec["max_prompt_length"] is None else int(spec["max_prompt_length"]),
                "min_decode_budget": int(spec["min_decode_budget"]),
                "max_decode_budget": None if spec["max_decode_budget"] is None else int(spec["max_decode_budget"]),
                "source_prompt_length": source_key[0],
                "source_decode_budget": source_key[1],
                "profile_source_artifact": profile_row["_artifact_path"],
                "knob_source_artifact": None if knob_row is None else knob_row["_artifact_path"],
                "value_source_artifact": None if value_row is None else value_row["artifact_path"],
                "preset": {
                    "profile": str(profile_row["profile"]),
                    "bits_k": int(profile_row["bits_k"] if knob_row is None else knob_row["bits_k"]),
                    "group_size": int(profile_row["group_size"] if knob_row is None else knob_row["group_size"]),
                    "tokens_per_page": int(profile_row["tokens_per_page"] if knob_row is None else knob_row["tokens_per_page"]),
                    "exact_value_layers": None if value_row is None else value_row["exact_value_layers"],
                },
            }
        )

    return {
        "schema_version": 1,
        "model_family": "gemma4_text",
        "generated_from": {
            "profile_artifacts": [str(path.relative_to(REPO_ROOT)) for path in PROFILE_ARTIFACTS],
            "knob_artifacts": [str(path.relative_to(REPO_ROOT)) for path in KNOB_ARTIFACTS],
            "value_artifacts": [str(path.relative_to(REPO_ROOT)) for path in VALUE_ARTIFACTS],
        },
        "rules": rules,
    }


def main() -> None:
    payload = build_tuning_table()
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(DEFAULT_OUTPUT_PATH.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
