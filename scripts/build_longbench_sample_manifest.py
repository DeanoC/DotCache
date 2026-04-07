#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotcache.longbench_v1 import (
    build_length_quartile_prompt_specs_from_zip,
    list_supported_datasets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP_PATH = REPO_ROOT / "benchmarks" / "cache" / "longbench" / "data.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen LongBench sample manifest.")
    parser.add_argument("--zip-path", default=str(DEFAULT_ZIP_PATH))
    parser.add_argument("--manifest-name", default="LB21-16")
    parser.add_argument("--manifest-version", default="v1")
    parser.add_argument("--seed", type=int, default=20260406)
    parser.add_argument("--rows-per-quartile", type=int, default=4)
    parser.add_argument("--smoke-rows-per-dataset", type=int, default=2)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--smoke-output", required=True)
    parser.add_argument("--main-output", required=True)
    return parser.parse_args()


def _select_smoke_prompt_ids(
    prompt_specs: list[dict[str, Any]],
    *,
    smoke_rows_per_dataset: int,
) -> set[str]:
    prompt_ids: set[str] = set()
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prompt_specs:
        by_dataset[str(item["dataset"])].append(item)
    for dataset, rows in by_dataset.items():
        del dataset
        ordered = sorted(
            rows,
            key=lambda item: (
                99 if item.get("length_quartile") is None else int(item["length_quartile"]),
                int(item["row_length"]),
                int(item["row_index"]),
            ),
        )
        preferred: list[dict[str, Any]] = []
        for quartile in (0, 3):
            candidate = next((row for row in ordered if row.get("length_quartile") == quartile), None)
            if candidate is not None and candidate not in preferred:
                preferred.append(candidate)
        for row in ordered:
            if len(preferred) >= int(smoke_rows_per_dataset):
                break
            if row not in preferred:
                preferred.append(row)
        for row in preferred[: int(smoke_rows_per_dataset)]:
            prompt_ids.add(str(row["prompt_id"]))
    return prompt_ids


def _build_manifest_payload(
    *,
    manifest_name: str,
    manifest_version: str,
    seed: int,
    rows_per_quartile: int,
    smoke_rows_per_dataset: int,
    prompt_specs: list[dict[str, Any]],
    phase: str,
) -> dict[str, Any]:
    datasets = sorted({str(item["dataset"]) for item in prompt_specs})
    return {
        "manifest_name": manifest_name,
        "manifest_version": manifest_version,
        "phase": phase,
        "seed": int(seed),
        "length_field": "row.length",
        "selection_method": "length_quartile_seeded",
        "rows_per_quartile": int(rows_per_quartile),
        "rows_per_dataset": int(rows_per_quartile) * 4,
        "smoke_rows_per_dataset": int(smoke_rows_per_dataset),
        "datasets": datasets,
        "prompt_specs": prompt_specs,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip_path).expanduser().resolve()
    prompt_specs = build_length_quartile_prompt_specs_from_zip(
        zip_path,
        datasets=list_supported_datasets(),
        rows_per_quartile=int(args.rows_per_quartile),
        seed=int(args.seed),
    )
    smoke_prompt_ids = _select_smoke_prompt_ids(
        prompt_specs,
        smoke_rows_per_dataset=int(args.smoke_rows_per_dataset),
    )
    manifest_prompt_specs: list[dict[str, Any]] = []
    smoke_prompt_specs: list[dict[str, Any]] = []
    main_prompt_specs: list[dict[str, Any]] = []
    for item in prompt_specs:
        phase = "smoke" if str(item["prompt_id"]) in smoke_prompt_ids else "main"
        enriched = dict(item)
        enriched["phase"] = phase
        manifest_prompt_specs.append(enriched)
        if phase == "smoke":
            smoke_prompt_specs.append(enriched)
        else:
            main_prompt_specs.append(enriched)

    _write_json(
        Path(args.manifest_output),
        _build_manifest_payload(
            manifest_name=str(args.manifest_name),
            manifest_version=str(args.manifest_version),
            seed=int(args.seed),
            rows_per_quartile=int(args.rows_per_quartile),
            smoke_rows_per_dataset=int(args.smoke_rows_per_dataset),
            prompt_specs=manifest_prompt_specs,
            phase="all",
        ),
    )
    _write_json(
        Path(args.smoke_output),
        _build_manifest_payload(
            manifest_name=str(args.manifest_name),
            manifest_version=str(args.manifest_version),
            seed=int(args.seed),
            rows_per_quartile=int(args.rows_per_quartile),
            smoke_rows_per_dataset=int(args.smoke_rows_per_dataset),
            prompt_specs=smoke_prompt_specs,
            phase="smoke",
        ),
    )
    _write_json(
        Path(args.main_output),
        _build_manifest_payload(
            manifest_name=str(args.manifest_name),
            manifest_version=str(args.manifest_version),
            seed=int(args.seed),
            rows_per_quartile=int(args.rows_per_quartile),
            smoke_rows_per_dataset=int(args.smoke_rows_per_dataset),
            prompt_specs=main_prompt_specs,
            phase="main",
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
