#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dotcache.evaluation_protocol import EvaluationMetadata, build_evaluation_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wrap an existing benchmark JSON result in the standardized evaluation envelope.")
    parser.add_argument("--input", required=True, help="Path to a JSON or JSONL result file.")
    parser.add_argument("--output", required=True, help="Path to write the normalized JSON or JSONL output.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--torch-dtype", required=True)
    parser.add_argument("--split", choices=["calibration", "held_out"], required=True)
    parser.add_argument("--lane", choices=["systems", "quality", "diagnostic"], required=True)
    parser.add_argument(
        "--prompt-family",
        choices=["synthetic_exact_length", "held_out_natural_text", "standardized_long_context"],
        required=True,
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--prompt-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--truth-type", choices=["reference_trace", "paged_runtime"], required=True)
    parser.add_argument("--effective-budget-rule", required=True)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--decode-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--drop-source-result", action="store_true")
    return parser.parse_args()


def _load_records(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return records, True
    return [json.loads(path.read_text(encoding="utf-8"))], False


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata = EvaluationMetadata(
        model_id=args.model_id,
        model_family=args.model_family,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
        split=args.split,
        lane=args.lane,
        prompt_family=args.prompt_family,
        dataset_name=args.dataset_name,
        prompt_count=args.prompt_count,
        batch_size=args.batch_size,
        truth_type=args.truth_type,
        effective_budget_rule=args.effective_budget_rule,
        context_length=args.context_length,
        decode_steps=args.decode_steps,
        eval_steps=args.eval_steps,
    )
    records, is_jsonl = _load_records(input_path)
    normalized = [
        build_evaluation_record(
            metadata,
            record,
            include_source_result=not args.drop_source_result,
        ).to_dict()
        for record in records
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if is_jsonl:
        with output_path.open("w", encoding="utf-8") as handle:
            for record in normalized:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    else:
        output_path.write_text(json.dumps(normalized[0], sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
