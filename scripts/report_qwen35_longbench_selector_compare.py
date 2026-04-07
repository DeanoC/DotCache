#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


BASE_EXPECTED_CASES = ("exact", "systems", "streaming_sink_recent")
OPTIONAL_CASES = ("quality", "quest_like")
EXTERNAL_CASES = ("streaming_sink_recent", "quest_like")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Qwen LongBench selector comparison runs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--markdown-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--title", default="Qwen LongBench Selector Compare")
    return parser.parse_args()


def _load_rows(path: Path, *, measurement_kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if payload.get("measurement_kind") == measurement_kind:
            rows.append(payload)
    if measurement_kind == "aggregate" and not rows:
        raise SystemExit(f"no aggregate rows found in {path}")
    return rows


def _fmt_float(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _mean_optional(values: list[object]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(mean(present))


def _markdown_table(rows: list[list[str]]) -> str:
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _group_key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row["comparison_max_prompt_tokens"]), str(row["comparison_case"]))


def _family_group_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["comparison_max_prompt_tokens"]),
        str(row["comparison_case"]),
        str(row.get("longbench_task_family") or "unknown"),
    )


def _prompt_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["comparison_max_prompt_tokens"]),
        str(row["evaluation_prompt_id"]),
        str(row["comparison_case"]),
    )


def _dataset_group_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["comparison_max_prompt_tokens"]),
        str(row["comparison_case"]),
        str(row.get("longbench_dataset") or ""),
    )


def _expected_cases(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    observed = {str(row.get("comparison_case")) for row in rows}
    cases = list(BASE_EXPECTED_CASES)
    for case in OPTIONAL_CASES:
        if case in observed:
            cases.append(case)
    return tuple(cases)


def _validate_expected_cases(rows: list[dict[str, Any]]) -> None:
    expected_cases = _expected_cases(rows)
    observed: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        observed[int(row["comparison_max_prompt_tokens"])].add(str(row["comparison_case"]))
    missing_by_bucket: dict[int, list[str]] = {}
    for max_prompt_tokens, cases in sorted(observed.items()):
        missing = [case for case in expected_cases if case not in cases]
        if missing:
            missing_by_bucket[max_prompt_tokens] = missing
    if missing_by_bucket:
        details = ", ".join(
            f"{max_prompt_tokens}: {', '.join(missing)}"
            for max_prompt_tokens, missing in missing_by_bucket.items()
        )
        raise SystemExit(f"incomplete aggregate coverage in report input; missing cases by context: {details}")


def _effective_bytes_per_token(row: dict[str, Any]) -> float | None:
    direct = row.get("effective_bytes_per_token")
    if direct is not None:
        return float(direct)
    resident = row.get("resident_bytes")
    prompt_length = row.get("prompt_length")
    if resident is None or prompt_length in (None, 0):
        return None
    return float(resident) / float(prompt_length)


def _sample_output_table(rows: list[dict[str, Any]]) -> str:
    samples: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _prompt_key(row)
        current = samples.get(key)
        if current is None or int(row.get("measurement_index", 10**9)) < int(current.get("measurement_index", 10**9)):
            samples[key] = row

    ordered = sorted(
        samples.values(),
        key=lambda row: (
            int(row["comparison_max_prompt_tokens"]),
            str(row["evaluation_prompt_id"]),
            str(row["comparison_case"]),
        ),
    )
    table = [[
        "max_prompt_tokens",
        "prompt",
        "dataset",
        "task_family",
        "case",
        "official_score",
        "generated",
    ]]
    for row in ordered:
        table.append(
            [
                str(int(row["comparison_max_prompt_tokens"])),
                str(row.get("evaluation_prompt_id", "")),
                str(row.get("longbench_dataset", "")),
                str(row.get("longbench_task_family", "")),
                str(row.get("comparison_case", "")),
                _fmt_float(row.get("longbench_official_score")),
                str(row.get("longbench_generated_text_cleaned", row.get("longbench_prediction_scored", ""))).replace("\n", "\\n"),
            ]
        )
    return _markdown_table(table)


def _build_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (max_prompt_tokens, case), case_rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        exact_match = [
            (1.0 if row.get("longbench_answer_exact_match_cleaned") else 0.0)
            for row in case_rows
            if row.get("longbench_answer_exact_match_cleaned") is not None
        ]
        qa_f1 = [
            float(row["longbench_qa_f1_max_cleaned"])
            for row in case_rows
            if row.get("longbench_qa_f1_max_cleaned") is not None
        ]
        decode = [float(row.get("dotcache_decode_ms_per_step", 0.0) or 0.0) for row in case_rows]
        decode_p95 = [
            float(row.get("dotcache_decode_ms_per_step_p95", row.get("dotcache_decode_ms_per_step", 0.0)) or 0.0)
            for row in case_rows
        ]
        summary_rows.append(
            {
                "max_prompt_tokens": int(max_prompt_tokens),
                "comparison_case": case,
                "n_rows": len(case_rows),
                "mean_exact_match": _mean_optional(exact_match),
                "mean_qa_f1": _mean_optional(qa_f1),
                "mean_official_score": _mean_optional([row.get("longbench_official_score") for row in case_rows]),
                "mean_decode_ms_per_step": float(mean(decode)),
                "p95_decode_ms_per_step": float(mean(decode_p95)),
                "mean_effective_bytes_per_token": _mean_optional(
                    [_effective_bytes_per_token(row) for row in case_rows]
                ),
                "mean_teacher_forced_perplexity_ratio": _mean_optional(
                    [row.get("teacher_forced_perplexity_ratio") for row in case_rows]
                ),
                "mean_teacher_forced_logit_rmse": _mean_optional(
                    [row.get("teacher_forced_logit_rmse") for row in case_rows]
                ),
                "worst_dataset_official_score": _mean_optional([row.get("longbench_official_score") for row in case_rows]),
            }
        )
    return summary_rows


def _build_family_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_family_group_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for (max_prompt_tokens, case, task_family), family_rows in sorted(grouped.items()):
        output.append(
            {
                "max_prompt_tokens": max_prompt_tokens,
                "comparison_case": case,
                "task_family": task_family,
                "n_rows": len(family_rows),
                "mean_official_score": _mean_optional([row.get("longbench_official_score") for row in family_rows]),
                "mean_decode_ms_per_step": _mean_optional(
                    [row.get("dotcache_decode_ms_per_step") for row in family_rows]
                ),
            }
        )
    return output


def _apply_worst_dataset_scores(summary_rows: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_dataset_group_key(row)].append(row)
    dataset_mins: dict[tuple[int, str], float] = {}
    for (max_prompt_tokens, case, _dataset), dataset_rows in grouped.items():
        score = _mean_optional([row.get("longbench_official_score") for row in dataset_rows])
        if score is None:
            continue
        key = (max_prompt_tokens, case)
        dataset_mins[key] = min(dataset_mins.get(key, score), score)
    for row in summary_rows:
        row["worst_dataset_official_score"] = dataset_mins.get(
            (int(row["max_prompt_tokens"]), str(row["comparison_case"]))
        )


def _build_tradeoff_rows(summary_rows: list[dict[str, Any]]) -> list[list[str]]:
    table = [[
        "max_prompt_tokens",
        "quality_vs_exact_speedup",
        "systems_vs_exact_speedup",
        "systems_vs_quality_speedup",
        "streaming_vs_exact_speedup",
        "quest_vs_exact_speedup",
        "quality_minus_systems_official_score",
    ]]
    summary_by_key = {(row["max_prompt_tokens"], row["comparison_case"]): row for row in summary_rows}
    for max_prompt_tokens in sorted({row["max_prompt_tokens"] for row in summary_rows}):
        exact = summary_by_key.get((max_prompt_tokens, "exact"))
        quality = summary_by_key.get((max_prompt_tokens, "quality"))
        systems = summary_by_key.get((max_prompt_tokens, "systems"))
        streaming = summary_by_key.get((max_prompt_tokens, "streaming_sink_recent"))
        quest = summary_by_key.get((max_prompt_tokens, "quest_like"))
        table.append(
            [
                str(max_prompt_tokens),
                _fmt_float(
                    (exact["mean_decode_ms_per_step"] / quality["mean_decode_ms_per_step"])
                    if exact and quality and quality["mean_decode_ms_per_step"] > 0.0
                    else None
                ),
                _fmt_float(
                    (exact["mean_decode_ms_per_step"] / systems["mean_decode_ms_per_step"])
                    if exact and systems and systems["mean_decode_ms_per_step"] > 0.0
                    else None
                ),
                _fmt_float(
                    (quality["mean_decode_ms_per_step"] / systems["mean_decode_ms_per_step"])
                    if quality and systems and systems["mean_decode_ms_per_step"] > 0.0
                    else None
                ),
                _fmt_float(
                    (exact["mean_decode_ms_per_step"] / streaming["mean_decode_ms_per_step"])
                    if exact and streaming and streaming["mean_decode_ms_per_step"] > 0.0
                    else None
                ),
                _fmt_float(
                    (exact["mean_decode_ms_per_step"] / quest["mean_decode_ms_per_step"])
                    if exact and quest and quest["mean_decode_ms_per_step"] > 0.0
                    else None
                ),
                _fmt_float(
                    (quality["mean_official_score"] - systems["mean_official_score"])
                    if quality and systems and quality.get("mean_official_score") is not None and systems.get("mean_official_score") is not None
                    else None
                ),
            ]
        )
    return table


def _select_parity_row(
    systems_row: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    match_mode: str,
) -> dict[str, Any] | None:
    if not candidates:
        return None

    def sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, float]:
        official_gap = abs(
            float(candidate.get("mean_official_score") or 0.0) - float(systems_row.get("mean_official_score") or 0.0)
        )
        memory_gap = abs(
            float(candidate.get("mean_effective_bytes_per_token") or 0.0)
            - float(systems_row.get("mean_effective_bytes_per_token") or 0.0)
        )
        ppl_gap = abs(
            float(candidate.get("mean_teacher_forced_perplexity_ratio") or 0.0)
            - float(systems_row.get("mean_teacher_forced_perplexity_ratio") or 0.0)
        )
        rmse_gap = abs(
            float(candidate.get("mean_teacher_forced_logit_rmse") or 0.0)
            - float(systems_row.get("mean_teacher_forced_logit_rmse") or 0.0)
        )
        if match_mode == "matched_quality":
            return (official_gap, ppl_gap, rmse_gap, memory_gap)
        return (memory_gap, official_gap, ppl_gap, rmse_gap)

    return min(candidates, key=sort_key)


def _build_parity_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_by_context: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        summary_by_context[int(row["max_prompt_tokens"])].append(row)

    parity_rows: list[dict[str, Any]] = []
    for max_prompt_tokens, context_rows in sorted(summary_by_context.items()):
        systems_row = next((row for row in context_rows if row["comparison_case"] == "systems"), None)
        if systems_row is None:
            continue
        candidates = [row for row in context_rows if row["comparison_case"] in EXTERNAL_CASES]
        for match_mode in ("matched_quality", "matched_memory"):
            selected = _select_parity_row(systems_row, candidates, match_mode=match_mode)
            if selected is None:
                continue
            parity_rows.append(
                {
                    "max_prompt_tokens": max_prompt_tokens,
                    "match_mode": match_mode,
                    "comparison_case": selected["comparison_case"],
                    "systems_official_score": systems_row.get("mean_official_score"),
                    "external_official_score": selected.get("mean_official_score"),
                    "official_score_gap": (
                        None
                        if systems_row.get("mean_official_score") is None or selected.get("mean_official_score") is None
                        else float(selected["mean_official_score"]) - float(systems_row["mean_official_score"])
                    ),
                    "systems_effective_bytes_per_token": systems_row.get("mean_effective_bytes_per_token"),
                    "external_effective_bytes_per_token": selected.get("mean_effective_bytes_per_token"),
                    "effective_bytes_gap": (
                        None
                        if systems_row.get("mean_effective_bytes_per_token") is None
                        or selected.get("mean_effective_bytes_per_token") is None
                        else float(selected["mean_effective_bytes_per_token"])
                        - float(systems_row["mean_effective_bytes_per_token"])
                    ),
                    "systems_decode_ms_per_step": systems_row.get("mean_decode_ms_per_step"),
                    "external_decode_ms_per_step": selected.get("mean_decode_ms_per_step"),
                    "systems_vs_external_speedup": (
                        None
                        if selected.get("mean_decode_ms_per_step") in (None, 0.0) or systems_row.get("mean_decode_ms_per_step") is None
                        else float(selected["mean_decode_ms_per_step"]) / float(systems_row["mean_decode_ms_per_step"])
                    ),
                }
            )
    return parity_rows


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must be non-empty")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, float(fraction))) * float(len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - float(lower)
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _bootstrap_mean_ci(values: list[float], *, bootstrap_samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        value = float(values[0])
        return value, value
    rng = random.Random(int(seed))
    draws: list[float] = []
    for _ in range(int(bootstrap_samples)):
        sample = [float(values[rng.randrange(len(values))]) for _ in range(len(values))]
        draws.append(float(mean(sample)))
    draws.sort()
    return (_percentile(draws, 0.025), _percentile(draws, 0.975))


def _build_confidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paired_rows: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            int(row["comparison_max_prompt_tokens"]),
            str(row.get("evaluation_prompt_id") or ""),
            str(row.get("longbench_dataset") or ""),
            str(row["comparison_case"]),
        )
        paired_rows[key] = row

    contexts = sorted({int(row["comparison_max_prompt_tokens"]) for row in rows})
    observed_cases = sorted({str(row["comparison_case"]) for row in rows})
    comparison_targets = [case for case in observed_cases if case != "systems"]
    output: list[dict[str, Any]] = []
    for max_prompt_tokens in contexts:
        for comparison_case in comparison_targets:
            dataset_deltas: list[float] = []
            dataset_wins = 0
            dataset_losses = 0
            dataset_ties = 0
            for dataset in sorted({str(row.get("longbench_dataset") or "") for row in rows}):
                prompt_ids = sorted(
                    {
                        str(row.get("evaluation_prompt_id") or "")
                        for row in rows
                        if int(row["comparison_max_prompt_tokens"]) == max_prompt_tokens
                        and str(row.get("longbench_dataset") or "") == dataset
                    }
                )
                prompt_deltas: list[float] = []
                for prompt_id in prompt_ids:
                    systems_row = paired_rows.get((max_prompt_tokens, prompt_id, dataset, "systems"))
                    comparison_row = paired_rows.get((max_prompt_tokens, prompt_id, dataset, comparison_case))
                    if systems_row is None or comparison_row is None:
                        continue
                    systems_score = systems_row.get("longbench_official_score")
                    comparison_score = comparison_row.get("longbench_official_score")
                    if systems_score is None or comparison_score is None:
                        continue
                    prompt_deltas.append(float(systems_score) - float(comparison_score))
                if not prompt_deltas:
                    continue
                dataset_delta = float(mean(prompt_deltas))
                dataset_deltas.append(dataset_delta)
                if dataset_delta > 0.0:
                    dataset_wins += 1
                elif dataset_delta < 0.0:
                    dataset_losses += 1
                else:
                    dataset_ties += 1
            if not dataset_deltas:
                continue
            ci_low, ci_high = _bootstrap_mean_ci(dataset_deltas)
            output.append(
                {
                    "max_prompt_tokens": max_prompt_tokens,
                    "comparison_case": comparison_case,
                    "delta_metric": "systems_minus_external_dataset_macro_official_score",
                    "mean_delta": float(mean(dataset_deltas)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_datasets": len(dataset_deltas),
                    "win_datasets": dataset_wins,
                    "loss_datasets": dataset_losses,
                    "tie_datasets": dataset_ties,
                }
            )
    return output


def build_report(
    rows: list[dict[str, Any]],
    *,
    title: str,
    trial_rows: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    _validate_expected_cases(rows)
    summary_rows = _build_summary_rows(rows)
    _apply_worst_dataset_scores(summary_rows, rows)
    family_breakdown_rows = _build_family_breakdown(rows)
    parity_rows = _build_parity_rows(summary_rows)
    confidence_rows = _build_confidence_rows(rows)

    markdown_rows = [[
        "max_prompt_tokens",
        "case",
        "n_rows",
        "mean_official_score",
        "mean_exact_match",
        "mean_qa_f1",
        "mean_decode_ms",
        "p95_decode_ms",
        "mean_eff_bytes_per_tok",
        "mean_ppl_ratio",
        "mean_rmse",
        "worst_dataset_score",
    ]]
    for summary in summary_rows:
        markdown_rows.append(
            [
                str(summary["max_prompt_tokens"]),
                summary["comparison_case"],
                str(summary["n_rows"]),
                _fmt_float(summary["mean_official_score"]),
                _fmt_float(summary["mean_exact_match"]),
                _fmt_float(summary["mean_qa_f1"]),
                _fmt_float(summary["mean_decode_ms_per_step"]),
                _fmt_float(summary["p95_decode_ms_per_step"]),
                _fmt_float(summary["mean_effective_bytes_per_token"]),
                _fmt_float(summary["mean_teacher_forced_perplexity_ratio"]),
                _fmt_float(summary["mean_teacher_forced_logit_rmse"]),
                _fmt_float(summary["worst_dataset_official_score"]),
            ]
        )

    family_table = [[
        "max_prompt_tokens",
        "case",
        "task_family",
        "n_rows",
        "mean_official_score",
        "mean_decode_ms",
    ]]
    for row in family_breakdown_rows:
        family_table.append(
            [
                str(row["max_prompt_tokens"]),
                str(row["comparison_case"]),
                str(row["task_family"]),
                str(row["n_rows"]),
                _fmt_float(row["mean_official_score"]),
                _fmt_float(row["mean_decode_ms_per_step"]),
            ]
        )

    parity_table = [[
        "max_prompt_tokens",
        "match_mode",
        "external_case",
        "systems_official_score",
        "external_official_score",
        "official_gap",
        "systems_eff_bytes",
        "external_eff_bytes",
        "eff_bytes_gap",
        "systems_vs_external_speedup",
    ]]
    for row in parity_rows:
        parity_table.append(
            [
                str(row["max_prompt_tokens"]),
                str(row["match_mode"]),
                str(row["comparison_case"]),
                _fmt_float(row["systems_official_score"]),
                _fmt_float(row["external_official_score"]),
                _fmt_float(row["official_score_gap"]),
                _fmt_float(row["systems_effective_bytes_per_token"]),
                _fmt_float(row["external_effective_bytes_per_token"]),
                _fmt_float(row["effective_bytes_gap"]),
                _fmt_float(row["systems_vs_external_speedup"]),
            ]
        )

    confidence_table = [[
        "max_prompt_tokens",
        "comparison_case",
        "mean_delta",
        "ci_low",
        "ci_high",
        "n_datasets",
        "win_datasets",
        "loss_datasets",
        "tie_datasets",
    ]]
    for row in confidence_rows:
        confidence_table.append(
            [
                str(row["max_prompt_tokens"]),
                str(row["comparison_case"]),
                _fmt_float(row["mean_delta"]),
                _fmt_float(row["ci_low"]),
                _fmt_float(row["ci_high"]),
                str(row["n_datasets"]),
                str(row["win_datasets"]),
                str(row["loss_datasets"]),
                str(row["tie_datasets"]),
            ]
        )

    markdown_sections = [
        f"# {title}",
        "",
        _markdown_table(markdown_rows),
        "",
        "## Tradeoff",
        "",
        _markdown_table(_build_tradeoff_rows(summary_rows)),
        "",
        "## Task Family Breakdown",
        "",
        _markdown_table(family_table),
        "",
        "## Parity",
        "",
        _markdown_table(parity_table),
        "",
        "## Confidence",
        "",
        _markdown_table(confidence_table),
    ]
    if trial_rows:
        markdown_sections.extend(["", "## Sample Outputs", "", _sample_output_table(trial_rows)])

    payload = {
        "rows": summary_rows,
        "task_family_rows": family_breakdown_rows,
        "parity_rows": parity_rows,
        "confidence_rows": confidence_rows,
    }
    return payload, "\n".join(markdown_sections)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    rows = _load_rows(input_path, measurement_kind="aggregate")
    trial_rows = _load_rows(input_path, measurement_kind="trial")
    payload, markdown = build_report(rows, title=str(args.title), trial_rows=trial_rows)
    Path(args.json_output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
