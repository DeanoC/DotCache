from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


_DEFAULT_FRONTIER_INPUTS = [
    Path(__file__).resolve().parents[1]
    / "benchmarks/results/qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_external/qwen35_persistent_exact_key_frontier.json",
    Path(__file__).resolve().parents[1]
    / "benchmarks/results/qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_broad/qwen35_persistent_exact_key_frontier.json",
    Path(__file__).resolve().parents[1]
    / "benchmarks/results/qwen35_persistent_exact_key_frontier_20260412_repo_promptfiles_large/qwen35_persistent_exact_key_frontier.json",
]


def _load_cases(frontier_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frontier_path in frontier_paths:
        payload = json.loads(frontier_path.read_text(encoding="utf-8"))
        sweep_by_threshold = {
            float(sweep["threshold"]): {str(record["case_tag"]): record for record in sweep["records"]}
            for sweep in payload["sweeps"]
        }
        corpus_name = frontier_path.parent.name
        for baseline_record in payload["baseline"]["records"]:
            case_tag = str(baseline_record["case_tag"])
            rows.append(
                {
                    "corpus": corpus_name,
                    "case_tag": case_tag,
                    "prompt_length": int(baseline_record["prompt_length"]),
                    "baseline_ms_per_step": float(baseline_record["decode_ms_per_step"]),
                    "threshold_ms_per_step": {
                        float(threshold): float(records[case_tag]["decode_ms_per_step"])
                        for threshold, records in sweep_by_threshold.items()
                    },
                }
            )
    return rows


def _policy_baseline(_: dict[str, Any]) -> str:
    return "baseline"


def _policy_layer15_always_020(_: dict[str, Any]) -> str:
    return "0.20"


def _policy_layer15_always_024(_: dict[str, Any]) -> str:
    return "0.24"


def _policy_length_le_1536_else_024(row: dict[str, Any]) -> str:
    return "0.20" if int(row["prompt_length"]) <= 1536 else "0.24"


def _policy_length_le_2048_else_024(row: dict[str, Any]) -> str:
    return "0.20" if int(row["prompt_length"]) <= 2048 else "0.24"


POLICIES: list[tuple[str, str, Callable[[dict[str, Any]], str]]] = [
    ("baseline", "Current global policy with no layer-15 override.", _policy_baseline),
    ("layer15_always_020", "Always set layer 15 to 0.20.", _policy_layer15_always_020),
    ("layer15_always_024", "Always set layer 15 to 0.24.", _policy_layer15_always_024),
    (
        "layer15_len_le_1536_else_024",
        "Use 0.20 when prompt length is <= 1536, otherwise 0.24.",
        _policy_length_le_1536_else_024,
    ),
    (
        "layer15_len_le_2048_else_024",
        "Use 0.20 when prompt length is <= 2048, otherwise 0.24.",
        _policy_length_le_2048_else_024,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare cheap layer-15 policy variants against saved frontier studies.")
    parser.add_argument(
        "--frontier-inputs",
        nargs="*",
        default=[str(path) for path in _DEFAULT_FRONTIER_INPUTS],
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def _score_policy(rows: list[dict[str, Any]], *, name: str, description: str, chooser: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    corpus_totals: dict[str, list[float]] = {}
    chosen_counts: dict[str, int] = {}
    for row in rows:
        choice = str(chooser(row))
        chosen_counts[choice] = chosen_counts.get(choice, 0) + 1
        if choice == "baseline":
            value = float(row["baseline_ms_per_step"])
        else:
            value = float(row["threshold_ms_per_step"][float(choice)])
        corpus_totals.setdefault(str(row["corpus"]), []).append(value)
    corpus_summary = {
        corpus: float(sum(values) / len(values))
        for corpus, values in sorted(corpus_totals.items())
    }
    overall = float(sum(sum(values) for values in corpus_totals.values()) / sum(len(values) for values in corpus_totals.values()))
    return {
        "name": name,
        "description": description,
        "overall_avg_ms_per_step": overall,
        "per_corpus_avg_ms_per_step": corpus_summary,
        "chosen_count_by_policy_value": {key: int(value) for key, value in sorted(chosen_counts.items())},
    }


def _render_markdown(*, payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5 Exact-Key Policy Study",
        "",
        "This compares simple layer-15 policy choices against the checked-in exact-key frontier studies.",
        "",
        "## Ranked policies",
        "",
    ]
    for record in payload["policies"]:
        lines.extend(
            [
                f"- `{record['name']}`:",
                f"  - description: {record['description']}",
                f"  - overall avg ms/step: {float(record['overall_avg_ms_per_step']):.4f}",
                f"  - per-corpus avg ms/step: {json.dumps(record['per_corpus_avg_ms_per_step'], sort_keys=True)}",
                f"  - chosen counts: {json.dumps(record['chosen_count_by_policy_value'], sort_keys=True)}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    frontier_paths = [Path(value) for value in list(args.frontier_inputs)]
    rows = _load_cases(frontier_paths)
    policy_rows = sorted(
        (
            _score_policy(rows, name=name, description=description, chooser=chooser)
            for name, description, chooser in POLICIES
        ),
        key=lambda record: float(record["overall_avg_ms_per_step"]),
    )
    payload = {
        "frontier_inputs": [str(path) for path in frontier_paths],
        "case_count": int(len(rows)),
        "policies": policy_rows,
    }
    if args.output_json:
        output_json_path = Path(str(args.output_json))
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_md:
        output_md_path = Path(str(args.output_md))
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(_render_markdown(payload=payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
