from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GroupExample:
    source_file: str
    layer_id: int
    step_index: int
    kv_head_id: int
    prompt_length: int
    approx_top_budget: int
    approx_exact_top_recall: float
    approx_boundary_margin_normalized: float | None
    anchor_pages: int
    recent_old_pages: int
    recent_old_run_length: int
    leading_anchor_pages: int
    leading_recent_old_pages: int
    first_recent_old_rank: int
    category_run_count: int

    @property
    def anchor_fraction(self) -> float:
        return float(self.anchor_pages) / float(max(self.approx_top_budget, 1))

    @property
    def recent_old_fraction(self) -> float:
        return float(self.recent_old_pages) / float(max(self.approx_top_budget, 1))


@dataclass(frozen=True)
class Clause:
    feature_name: str
    comparator: str
    threshold: float

    def matches(self, example: GroupExample) -> bool:
        value = getattr(example, self.feature_name)
        if value is None:
            return False
        numeric_value = float(value)
        if self.comparator == ">=":
            return numeric_value >= self.threshold
        if self.comparator == "<=":
            return numeric_value <= self.threshold
        raise ValueError(f"unsupported comparator: {self.comparator}")

    def describe(self) -> str:
        return f"{self.feature_name} {self.comparator} {self.threshold:g}"


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _longest_contiguous_run(page_ranges: list[dict[str, int]]) -> int:
    if not page_ranges:
        return 0
    sorted_ranges = sorted(page_ranges, key=lambda item: (int(item["token_start"]), int(item["token_end"])))
    longest = 1
    current = 1
    previous = sorted_ranges[0]
    for current_range in sorted_ranges[1:]:
        if int(current_range["token_start"]) == int(previous["token_end"]):
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = current_range
    return int(longest)


def _page_category(
    page_range: dict[str, int],
    *,
    anchor_window: int,
    recent_start: int,
    recent_band_window: int,
) -> str:
    page_end = int(page_range["token_end"])
    if page_end <= int(anchor_window):
        return "anchor"
    if page_end <= int(recent_start) and page_end > int(recent_start - recent_band_window):
        return "recent_old"
    return "other"


def _leading_category_count(categories: list[str], target: str) -> int:
    count = 0
    for category in categories:
        if category != target:
            break
        count += 1
    return int(count)


def _first_rank(categories: list[str], target: str) -> int:
    for index, category in enumerate(categories, start=1):
        if category == target:
            return int(index)
    return int(len(categories) + 1)


def _category_run_count(categories: list[str]) -> int:
    if not categories:
        return 0
    runs = 1
    previous = categories[0]
    for category in categories[1:]:
        if category != previous:
            runs += 1
        previous = category
    return int(runs)


def _candidate_thresholds(values: list[float]) -> list[float]:
    unique_values = sorted({float(value) for value in values})
    if not unique_values:
        return []
    thresholds: list[float] = list(unique_values)
    for left, right in zip(unique_values, unique_values[1:]):
        midpoint = float((left + right) / 2.0)
        thresholds.append(midpoint)
    return sorted({float(value) for value in thresholds})


def _collect_examples(
    input_paths: list[Path],
    *,
    target_layer: int,
    anchor_window: int,
    recent_band_window: int,
) -> list[GroupExample]:
    examples: list[GroupExample] = []
    for input_path in input_paths:
        records = _load_jsonl_records(input_path)
        for record in records:
            for layer_record in record.get("scorer_layer_records", []):
                if int(layer_record.get("layer_id", -1)) != int(target_layer):
                    continue
                for group in layer_record.get("groups", []):
                    top_ranges = list(group.get("top_approx_page_ranges", []))
                    context_length = int(group.get("context_length", 0))
                    layer_recent_window = int(group.get("layer_recent_window", 0))
                    recent_start = int(context_length) - int(layer_recent_window)
                    categories = [
                        _page_category(
                            page,
                            anchor_window=int(anchor_window),
                            recent_start=int(recent_start),
                            recent_band_window=int(recent_band_window),
                        )
                        for page in top_ranges
                    ]
                    anchor_pages = sum(
                        1
                        for page in top_ranges
                        if int(page["token_end"]) <= int(anchor_window)
                    )
                    recent_old_ranges = [
                        page
                        for page in top_ranges
                        if (
                            int(page["token_end"]) <= int(recent_start)
                            and int(page["token_end"]) > int(recent_start - recent_band_window)
                        )
                    ]
                    examples.append(
                        GroupExample(
                            source_file=str(input_path),
                            layer_id=int(layer_record["layer_id"]),
                            step_index=int(layer_record["step_index"]),
                            kv_head_id=int(group["kv_head_id"]),
                            prompt_length=int(record.get("prompt_length", 0)),
                            approx_top_budget=int(group.get("approx_top_budget", 0)),
                            approx_exact_top_recall=float(group.get("approx_exact_top_recall", 0.0)),
                            approx_boundary_margin_normalized=(
                                None
                                if group.get("approx_boundary_margin_normalized") is None
                                else float(group["approx_boundary_margin_normalized"])
                            ),
                            anchor_pages=int(anchor_pages),
                            recent_old_pages=int(len(recent_old_ranges)),
                            recent_old_run_length=int(_longest_contiguous_run(recent_old_ranges)),
                            leading_anchor_pages=int(_leading_category_count(categories, "anchor")),
                            leading_recent_old_pages=int(_leading_category_count(categories, "recent_old")),
                            first_recent_old_rank=int(_first_rank(categories, "recent_old")),
                            category_run_count=int(_category_run_count(categories)),
                        )
                    )
    return examples


def _rule_metrics(examples: list[GroupExample], bad_recall_threshold: float, clauses: list[Clause]) -> dict[str, Any]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    positives: list[dict[str, Any]] = []
    for example in examples:
        label_bad = float(example.approx_exact_top_recall) <= float(bad_recall_threshold)
        predicted_bad = all(clause.matches(example) for clause in clauses)
        if predicted_bad and label_bad:
            true_positive += 1
        elif predicted_bad and not label_bad:
            false_positive += 1
        elif (not predicted_bad) and label_bad:
            false_negative += 1
        if predicted_bad:
            positives.append(
                {
                    "source_file": example.source_file,
                    "step_index": int(example.step_index),
                    "kv_head_id": int(example.kv_head_id),
                    "prompt_length": int(example.prompt_length),
                    "approx_exact_top_recall": float(example.approx_exact_top_recall),
                    "anchor_pages": int(example.anchor_pages),
                    "recent_old_pages": int(example.recent_old_pages),
                    "recent_old_run_length": int(example.recent_old_run_length),
                    "leading_anchor_pages": int(example.leading_anchor_pages),
                    "leading_recent_old_pages": int(example.leading_recent_old_pages),
                    "first_recent_old_rank": int(example.first_recent_old_rank),
                    "category_run_count": int(example.category_run_count),
                    "approx_boundary_margin_normalized": example.approx_boundary_margin_normalized,
                }
            )
    precision = float(true_positive / max(true_positive + false_positive, 1))
    recall = float(true_positive / max(true_positive + false_negative, 1))
    f1 = float((2.0 * precision * recall) / max(precision + recall, 1e-8))
    return {
        "clauses": [clause.describe() for clause in clauses],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "predicted_examples": positives,
    }


def _candidate_clauses(examples: list[GroupExample]) -> list[Clause]:
    clauses: list[Clause] = []
    integer_features = (
        "anchor_pages",
        "recent_old_pages",
        "recent_old_run_length",
        "leading_anchor_pages",
        "leading_recent_old_pages",
        "first_recent_old_rank",
        "category_run_count",
    )
    fractional_features = ("anchor_fraction", "recent_old_fraction")
    for feature_name in integer_features:
        thresholds = _candidate_thresholds([float(getattr(example, feature_name)) for example in examples])
        for threshold in thresholds:
            clauses.append(Clause(feature_name=feature_name, comparator=">=", threshold=float(threshold)))
            clauses.append(Clause(feature_name=feature_name, comparator="<=", threshold=float(threshold)))
    for feature_name in fractional_features:
        thresholds = _candidate_thresholds([float(getattr(example, feature_name)) for example in examples])
        for threshold in thresholds:
            clauses.append(Clause(feature_name=feature_name, comparator=">=", threshold=float(threshold)))
            clauses.append(Clause(feature_name=feature_name, comparator="<=", threshold=float(threshold)))
    margin_thresholds = _candidate_thresholds(
        [
            float(example.approx_boundary_margin_normalized)
            for example in examples
            if example.approx_boundary_margin_normalized is not None
        ]
    )
    for threshold in margin_thresholds:
        clauses.append(Clause(feature_name="approx_boundary_margin_normalized", comparator="<=", threshold=float(threshold)))
        clauses.append(Clause(feature_name="approx_boundary_margin_normalized", comparator=">=", threshold=float(threshold)))
    deduped: dict[tuple[str, str, float], Clause] = {}
    for clause in clauses:
        deduped[(clause.feature_name, clause.comparator, clause.threshold)] = clause
    return list(deduped.values())


def _search_rules(examples: list[GroupExample], *, bad_recall_threshold: float, max_clauses: int) -> list[dict[str, Any]]:
    candidate_clauses = _candidate_clauses(examples)
    scored_rules: list[dict[str, Any]] = []
    for clause in candidate_clauses:
        scored_rules.append(_rule_metrics(examples, bad_recall_threshold, [clause]))
    if max_clauses >= 2:
        for index, left in enumerate(candidate_clauses):
            for right in candidate_clauses[index + 1 :]:
                if left.feature_name == right.feature_name and left.comparator == right.comparator:
                    continue
                scored_rules.append(_rule_metrics(examples, bad_recall_threshold, [left, right]))
    scored_rules.sort(
        key=lambda item: (
            float(item["f1"]),
            float(item["precision"]),
            float(item["recall"]),
            -int(item["false_positive"]),
            -int(item["false_negative"]),
            -len(item["clauses"]),
        ),
        reverse=True,
    )
    return scored_rules


def _write_markdown_report(
    output_path: Path,
    *,
    examples: list[GroupExample],
    bad_recall_threshold: float,
    top_rules: list[dict[str, Any]],
) -> None:
    bad_examples = [example for example in examples if float(example.approx_exact_top_recall) <= float(bad_recall_threshold)]
    lines: list[str] = []
    lines.append("# Layer 23 Offline Distillation Report")
    lines.append("")
    lines.append(f"- examples: `{len(examples)}`")
    lines.append(f"- bad recall threshold: `{bad_recall_threshold}`")
    lines.append(f"- bad examples: `{len(bad_examples)}`")
    lines.append("")
    lines.append("## Top Rules")
    lines.append("")
    for index, rule in enumerate(top_rules[:5], start=1):
        lines.append(f"### Rule {index}")
        lines.append("")
        lines.append(f"- clauses: `{'; '.join(rule['clauses'])}`")
        lines.append(f"- precision: `{rule['precision']:.3f}`")
        lines.append(f"- recall: `{rule['recall']:.3f}`")
        lines.append(f"- f1: `{rule['f1']:.3f}`")
        lines.append(
            f"- confusion: `tp={rule['true_positive']} fp={rule['false_positive']} fn={rule['false_negative']}`"
        )
        if rule["predicted_examples"]:
            lines.append("- predicted groups:")
            for predicted in rule["predicted_examples"]:
                lines.append(
                    "  - "
                    f"`step={predicted['step_index']} kv={predicted['kv_head_id']} "
                    f"recall={predicted['approx_exact_top_recall']:.3f} "
                    f"anchor={predicted['anchor_pages']} recent_old={predicted['recent_old_pages']} "
                    f"run={predicted['recent_old_run_length']} "
                    f"margin={predicted['approx_boundary_margin_normalized']}`"
                )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline rule search for Qwen3.5 serving shortlist diagnostics.")
    parser.add_argument("input", nargs="+", help="One or more scorer-diagnostic JSONL artifacts.")
    parser.add_argument("--target-layer", type=int, default=23)
    parser.add_argument("--anchor-window", type=int, default=1024)
    parser.add_argument("--recent-band-window", type=int, default=1024)
    parser.add_argument("--bad-recall-threshold", type=float, default=0.4)
    parser.add_argument("--max-clauses", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = [Path(value) for value in args.input]
    examples = _collect_examples(
        input_paths,
        target_layer=int(args.target_layer),
        anchor_window=int(args.anchor_window),
        recent_band_window=int(args.recent_band_window),
    )
    if not examples:
        raise SystemExit("no matching layer examples found in input artifacts")
    ranked_rules = _search_rules(
        examples,
        bad_recall_threshold=float(args.bad_recall_threshold),
        max_clauses=int(args.max_clauses),
    )
    payload = {
        "input_files": [str(path) for path in input_paths],
        "target_layer": int(args.target_layer),
        "anchor_window": int(args.anchor_window),
        "recent_band_window": int(args.recent_band_window),
        "bad_recall_threshold": float(args.bad_recall_threshold),
        "example_count": int(len(examples)),
        "bad_example_count": int(
            sum(1 for example in examples if float(example.approx_exact_top_recall) <= float(args.bad_recall_threshold))
        ),
        "top_rules": ranked_rules[: int(args.top_k)],
        "examples": [
            {
                "source_file": example.source_file,
                "step_index": int(example.step_index),
                "kv_head_id": int(example.kv_head_id),
                "prompt_length": int(example.prompt_length),
                "approx_top_budget": int(example.approx_top_budget),
                "approx_exact_top_recall": float(example.approx_exact_top_recall),
                "approx_boundary_margin_normalized": example.approx_boundary_margin_normalized,
                "anchor_pages": int(example.anchor_pages),
                "recent_old_pages": int(example.recent_old_pages),
                "recent_old_run_length": int(example.recent_old_run_length),
                "leading_anchor_pages": int(example.leading_anchor_pages),
                "leading_recent_old_pages": int(example.leading_recent_old_pages),
                "first_recent_old_rank": int(example.first_recent_old_rank),
                "category_run_count": int(example.category_run_count),
                "anchor_fraction": float(example.anchor_fraction),
                "recent_old_fraction": float(example.recent_old_fraction),
            }
            for example in examples
        ],
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        _write_markdown_report(
            Path(args.output_md),
            examples=examples,
            bad_recall_threshold=float(args.bad_recall_threshold),
            top_rules=ranked_rules,
        )
    print(json.dumps(payload["top_rules"][:3], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
