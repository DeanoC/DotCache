#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Qwen3.5 CUDA passkey pack artifacts.")
    parser.add_argument("input", help="Passkey pack JSONL artifact.")
    parser.add_argument("--markdown-output", default=None, help="Optional markdown output path.")
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

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in success_rows:
        grouped[(row["runner_case"], int(row["prompt_length"]))].append(row)

    lines = [
        "# Qwen3.5 CUDA Passkey Pack Summary",
        "",
        "| Case | Context | n prompts | Retrieval accuracy | Exact-match rate | Mean decode ms/step | Min | Max | Stddev | 95% CI | Mean selected pages | Decode path |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key in sorted(grouped):
        sample_rows = grouped[key]
        decode_values = [float(row["dotcache_decode_ms_per_step"]) for row in sample_rows]
        selected_pages = [int(row.get("execution_shortlist_selected_pages", 0)) for row in sample_rows]
        retrieval = [1.0 if row.get("passkey_answer_correct") else 0.0 for row in sample_rows]
        exact = [1.0 if row.get("passkey_answer_exact_match") else 0.0 for row in sample_rows]
        path_counts = sorted({json.dumps(row.get("decode_path_counts", {}), sort_keys=True) for row in sample_rows})
        std = stdev(decode_values) if len(decode_values) > 1 else 0.0
        lines.append(
            "| "
            + " | ".join(
                [
                    key[0],
                    str(key[1]),
                    str(len(sample_rows)),
                    _fmt(mean(retrieval)),
                    _fmt(mean(exact)),
                    _fmt(mean(decode_values)),
                    _fmt(min(decode_values)),
                    _fmt(max(decode_values)),
                    _fmt(std),
                    _fmt(_ci95(decode_values)),
                    _fmt(mean(selected_pages)),
                    "; ".join(path_counts),
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
                "| Prompt | Case | Context | Error type | Message |",
                "| --- | --- | ---: | --- | --- |",
            ]
        )
        for row in error_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("evaluation_prompt_id", "unknown")),
                        str(row.get("runner_case", "unknown")),
                        str(row.get("prompt_length", "unknown")),
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
