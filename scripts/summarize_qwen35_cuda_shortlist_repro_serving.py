#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated Qwen3.5 shortlist serving JSONL artifacts.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="benchmarks/results/qwen35_cuda_shortlist_large_context_repro_serving",
        help="Directory containing default_repeat*.jsonl and forced_grouped_repeat*.jsonl artifacts.",
    )
    parser.add_argument("--markdown-output", default=None, help="Optional markdown summary output path.")
    return parser.parse_args()


def _load_rows(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_dir.glob("*.jsonl")):
        mode = "forced_grouped" if path.name.startswith("forced_grouped") else "default"
        repeat = int(path.stem.split("repeat")[-1])
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            record["summary_source_file"] = path.name
            record["summary_mode"] = mode
            record["summary_repeat"] = repeat
            rows.append(record)
    return rows


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def _ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def _render_markdown(rows: list[dict]) -> str:
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["summary_mode"], row["runner_case"], int(row["prompt_length"]))
        grouped[key].append(row)

    lines = [
        "# Qwen3.5 CUDA Shortlist Repro Serving Summary",
        "",
        "| Mode | Case | Context | n prompts | Mean decode ms/step | Min | Max | Stddev | 95% CI | Selected pages | Decode path |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for key in sorted(grouped):
        samples = grouped[key]
        decode_values = [float(sample["dotcache_decode_ms_per_step"]) for sample in samples]
        selected_pages = sorted({int(sample.get("execution_shortlist_selected_pages", 0)) for sample in samples})
        path_counts = sorted({json.dumps(sample.get("decode_path_counts", {}), sort_keys=True) for sample in samples})
        stddev_value = stdev(decode_values) if len(decode_values) > 1 else 0.0
        lines.append(
            "| "
            + " | ".join(
                [
                    key[0],
                    key[1],
                    str(key[2]),
                    str(len(samples)),
                    _fmt_float(mean(decode_values)),
                    _fmt_float(min(decode_values)),
                    _fmt_float(max(decode_values)),
                    _fmt_float(stddev_value),
                    _fmt_float(_ci95(decode_values)),
                    ", ".join(str(value) for value in selected_pages),
                    "; ".join(path_counts),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    rows = _load_rows(input_dir)
    if not rows:
        raise SystemExit(f"no JSONL rows found in {input_dir}")

    markdown = _render_markdown(rows)
    if args.markdown_output:
        output_path = Path(args.markdown_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
