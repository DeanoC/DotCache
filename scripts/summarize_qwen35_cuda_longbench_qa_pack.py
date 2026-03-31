#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Qwen3.5 CUDA LongBench QA pack artifacts.")
    parser.add_argument("input", help="LongBench QA pack JSONL artifact.")
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args()


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _render(rows: list[dict]) -> str:
    error_rows = [row for row in rows if "dotcache_decode_ms_per_step" not in row]
    success_rows = [row for row in rows if "dotcache_decode_ms_per_step" in row]
    if not success_rows:
        raise SystemExit("no successful rows found")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in success_rows:
        grouped[row["runner_case"]].append(row)

    lines = [
        "# Qwen3.5 CUDA LongBench QA Pack Summary",
        "",
        "| Case | n prompts | Mean QA F1 | Exact-match rate | Mean decode ms/step | Min | Max | Stddev | 95% CI | Mean prompt tokens | Mean selected pages | Datasets |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for case in sorted(grouped):
        sample_rows = grouped[case]
        decode_values = [float(row["dotcache_decode_ms_per_step"]) for row in sample_rows]
        qa_f1_values = [float(row.get("longbench_qa_f1_max", 0.0)) for row in sample_rows]
        exact_values = [1.0 if row.get("longbench_answer_exact_match") else 0.0 for row in sample_rows]
        prompt_lengths = [int(row.get("longbench_prompt_token_length", row.get("prompt_length", 0))) for row in sample_rows]
        selected_pages = [int(row.get("execution_shortlist_selected_pages", 0)) for row in sample_rows]
        datasets = ", ".join(sorted({str(row.get("longbench_dataset", "unknown")) for row in sample_rows}))
        std = stdev(decode_values) if len(decode_values) > 1 else 0.0
        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    str(len(sample_rows)),
                    _fmt(mean(qa_f1_values)),
                    _fmt(mean(exact_values)),
                    _fmt(mean(decode_values)),
                    _fmt(min(decode_values)),
                    _fmt(max(decode_values)),
                    _fmt(std),
                    _fmt(_ci95(decode_values)),
                    _fmt(mean(prompt_lengths)),
                    _fmt(mean(selected_pages)),
                    datasets,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Prompt Rows",
            "",
            "| Prompt | Dataset | Row | Case | QA F1 | Exact-match | Prompt tokens | Decode ms/step |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(success_rows, key=lambda item: (str(item["evaluation_prompt_id"]), str(item["runner_case"]))):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("evaluation_prompt_id", "unknown")),
                    str(row.get("longbench_dataset", "unknown")),
                    str(row.get("longbench_row_index", "unknown")),
                    str(row.get("runner_case", "unknown")),
                    _fmt(float(row.get("longbench_qa_f1_max", 0.0))),
                    _fmt(1.0 if row.get("longbench_answer_exact_match") else 0.0),
                    str(row.get("longbench_prompt_token_length", row.get("prompt_length", "unknown"))),
                    _fmt(float(row.get("dotcache_decode_ms_per_step", 0.0))),
                ]
            )
            + " |"
        )

    if error_rows:
        lines.extend(
            [
                "",
                "## Skipped Error Rows",
                "",
                "| Prompt | Dataset | Row | Case | Error type | Message |",
                "| --- | --- | ---: | --- | --- | --- |",
            ]
        )
        for row in error_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("evaluation_prompt_id", "unknown")),
                        str(row.get("longbench_dataset", "unknown")),
                        str(row.get("longbench_row_index", "unknown")),
                        str(row.get("runner_case", "unknown")),
                        str(row.get("error_type", row.get("status", "error"))),
                        str(row.get("error_message", "")).replace("\n", " "),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = _load_rows(Path(args.input))
    if not rows:
        raise SystemExit("no rows found")
    markdown = _render(rows)
    if args.markdown_output:
        out = Path(args.markdown_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
