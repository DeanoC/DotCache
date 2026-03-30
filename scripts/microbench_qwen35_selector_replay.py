#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class SelectorScenario:
    source: str
    label: str
    prompt_length: int
    step_index: int
    per_call_total_pages: int
    per_call_candidate_pages: int
    per_call_selected_pages: int
    selector_calls: int
    selector_mode: str


@dataclass(frozen=True)
class ReplayInputs:
    minima_matrix: np.ndarray
    maxima_matrix: np.ndarray
    query: np.ndarray
    positive_query: np.ndarray
    negative_query: np.ndarray
    candidate_indices: np.ndarray
    candidate_start: int
    candidate_stop: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay and microbenchmark Qwen3.5 builtin selector paths.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more JSONL benchmark artifacts with dotcache_step_runtime_breakdown records.",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="*",
        default=[],
        help="Optional prompt lengths to keep.",
    )
    parser.add_argument(
        "--label-filter",
        default="",
        help="Optional substring filter for scenario labels.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Synthetic selector head dimension to replay.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations per scenario/mode.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=80,
        help="Timed iterations per scenario/mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for synthetic selector inputs.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write machine-readable results.",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _scenario_label(path: Path, *, prompt_length: int, step_index: int, mode: str) -> str:
    return f"{path.parent.name}/{path.stem}:ctx{prompt_length}:step{step_index}:{mode}"


def _load_scenarios(paths: Sequence[Path]) -> list[SelectorScenario]:
    scenarios: list[SelectorScenario] = []
    for path in paths:
        for row in _load_jsonl(path):
            prompt_length = int(
                row.get("target_prompt_length")
                or row.get("prompt_length")
                or row.get("input_token_count")
                or 0
            )
            steps = row.get("dotcache_step_runtime_breakdown")
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                score_all_calls = int(step.get("decode_builtin_selector_score_all_pages_calls", 0) or 0)
                candidate_only_calls = int(step.get("decode_builtin_selector_candidate_only_calls", 0) or 0)
                total_pages = int(step.get("decode_builtin_selector_total_pages", 0) or 0)
                candidate_pages = int(step.get("decode_builtin_selector_candidate_pages", 0) or 0)
                if total_pages <= 0 or candidate_pages <= 0:
                    continue
                if score_all_calls > 0:
                    calls = score_all_calls
                    mode = "score_all_pages"
                elif candidate_only_calls > 0:
                    calls = candidate_only_calls
                    mode = "candidate_only"
                else:
                    continue
                if calls <= 0:
                    continue
                if total_pages % calls != 0 or candidate_pages % calls != 0:
                    continue
                per_call_total_pages = total_pages // calls
                per_call_candidate_pages = candidate_pages // calls
                per_call_selected_pages = per_call_total_pages - per_call_candidate_pages
                step_index = int(step.get("step_index", len(scenarios)))
                scenarios.append(
                    SelectorScenario(
                        source=str(path),
                        label=_scenario_label(path, prompt_length=prompt_length, step_index=step_index, mode=mode),
                        prompt_length=prompt_length,
                        step_index=step_index,
                        per_call_total_pages=per_call_total_pages,
                        per_call_candidate_pages=per_call_candidate_pages,
                        per_call_selected_pages=per_call_selected_pages,
                        selector_calls=calls,
                        selector_mode=mode,
                    )
                )
    return scenarios


def _build_replay_inputs(scenario: SelectorScenario, *, head_dim: int, seed: int) -> ReplayInputs:
    rng = np.random.default_rng(
        seed
        + scenario.prompt_length * 17
        + scenario.step_index * 97
        + scenario.per_call_total_pages * 7
    )
    direct_rows = max(scenario.per_call_total_pages - 1, scenario.per_call_candidate_pages)
    minima_matrix = rng.normal(size=(direct_rows, head_dim)).astype(np.float32)
    span_scale = np.abs(rng.normal(size=(direct_rows, head_dim)).astype(np.float32)) + 0.25
    maxima_matrix = minima_matrix + span_scale
    query = rng.normal(size=(head_dim,), loc=0.0, scale=0.5).astype(np.float32)
    positive_query = np.maximum(query, 0.0)
    negative_query = np.minimum(query, 0.0)

    max_start = max(direct_rows - scenario.per_call_candidate_pages, 0)
    if scenario.per_call_selected_pages <= 0:
        candidate_start = 0
    else:
        candidate_start = min(max_start // 2 + 1, max_start)
    candidate_stop = candidate_start + scenario.per_call_candidate_pages
    if candidate_stop > direct_rows:
        candidate_start = max(0, direct_rows - scenario.per_call_candidate_pages)
        candidate_stop = direct_rows
    candidate_indices = np.arange(candidate_start, candidate_stop, dtype=np.int64)
    return ReplayInputs(
        minima_matrix=minima_matrix,
        maxima_matrix=maxima_matrix,
        query=query,
        positive_query=positive_query,
        negative_query=negative_query,
        candidate_indices=candidate_indices,
        candidate_start=int(candidate_start),
        candidate_stop=int(candidate_stop),
    )


def _score_all_pages(inputs: ReplayInputs) -> tuple[float, float, float, bool]:
    compute_started = perf_counter()
    all_scores = (
        inputs.maxima_matrix @ inputs.positive_query + inputs.minima_matrix @ inputs.negative_query
    ).astype(np.float32, copy=False)
    scores = np.asarray(all_scores[inputs.candidate_indices], dtype=np.float32)
    compute_ms = (perf_counter() - compute_started) * 1000.0
    return 0.0, compute_ms, compute_ms, bool(scores.flags["C_CONTIGUOUS"])


def _candidate_take(inputs: ReplayInputs) -> tuple[float, float, float, bool]:
    stack_started = perf_counter()
    candidate_minima = np.take(inputs.minima_matrix, inputs.candidate_indices, axis=0).astype(np.float32, copy=False)
    candidate_maxima = np.take(inputs.maxima_matrix, inputs.candidate_indices, axis=0).astype(np.float32, copy=False)
    stack_ms = (perf_counter() - stack_started) * 1000.0
    compute_started = perf_counter()
    scores = (candidate_maxima @ inputs.positive_query + candidate_minima @ inputs.negative_query).astype(
        np.float32,
        copy=False,
    )
    compute_ms = (perf_counter() - compute_started) * 1000.0
    return stack_ms, compute_ms, stack_ms + compute_ms, bool(candidate_minima.flags["C_CONTIGUOUS"] and scores.flags["C_CONTIGUOUS"])


def _candidate_view(inputs: ReplayInputs) -> tuple[float, float, float, bool]:
    stack_started = perf_counter()
    candidate_minima = np.asarray(inputs.minima_matrix[inputs.candidate_start : inputs.candidate_stop], dtype=np.float32)
    candidate_maxima = np.asarray(inputs.maxima_matrix[inputs.candidate_start : inputs.candidate_stop], dtype=np.float32)
    stack_ms = (perf_counter() - stack_started) * 1000.0
    compute_started = perf_counter()
    scores = (candidate_maxima @ inputs.positive_query + candidate_minima @ inputs.negative_query).astype(
        np.float32,
        copy=False,
    )
    compute_ms = (perf_counter() - compute_started) * 1000.0
    return stack_ms, compute_ms, stack_ms + compute_ms, bool(candidate_minima.flags["C_CONTIGUOUS"] and scores.flags["C_CONTIGUOUS"])


def _candidate_packedspan(inputs: ReplayInputs) -> tuple[float, float, float, bool]:
    stack_started = perf_counter()
    candidate_minima = np.ascontiguousarray(
        inputs.minima_matrix[inputs.candidate_start : inputs.candidate_stop],
        dtype=np.float32,
    )
    candidate_maxima = np.ascontiguousarray(
        inputs.maxima_matrix[inputs.candidate_start : inputs.candidate_stop],
        dtype=np.float32,
    )
    stack_ms = (perf_counter() - stack_started) * 1000.0
    compute_started = perf_counter()
    scores = (candidate_maxima @ inputs.positive_query + candidate_minima @ inputs.negative_query).astype(
        np.float32,
        copy=False,
    )
    compute_ms = (perf_counter() - compute_started) * 1000.0
    return stack_ms, compute_ms, stack_ms + compute_ms, bool(candidate_minima.flags["C_CONTIGUOUS"] and scores.flags["C_CONTIGUOUS"])


MODES: dict[str, Callable[[ReplayInputs], tuple[float, float, float, bool]]] = {
    "score_all_pages": _score_all_pages,
    "candidate_take": _candidate_take,
    "candidate_view": _candidate_view,
    "candidate_packedspan": _candidate_packedspan,
}


def _summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(np.mean(arr)),
    }


def run_microbench(
    scenarios: Sequence[SelectorScenario],
    *,
    head_dim: int,
    warmup: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for scenario in scenarios:
        inputs = _build_replay_inputs(scenario, head_dim=head_dim, seed=seed)
        for mode_name, mode_fn in MODES.items():
            for _ in range(max(warmup, 0)):
                mode_fn(inputs)
            stack_values: list[float] = []
            compute_values: list[float] = []
            total_values: list[float] = []
            contig_values: list[bool] = []
            for _ in range(max(iterations, 1)):
                stack_ms, compute_ms, total_ms, contiguous = mode_fn(inputs)
                stack_values.append(float(stack_ms))
                compute_values.append(float(compute_ms))
                total_values.append(float(total_ms))
                contig_values.append(bool(contiguous))
            results.append(
                {
                    **asdict(scenario),
                    "mode": mode_name,
                    "head_dim": int(head_dim),
                    "candidate_start": int(inputs.candidate_start),
                    "candidate_stop": int(inputs.candidate_stop),
                    "candidate_fraction": float(
                        float(scenario.per_call_candidate_pages) / float(scenario.per_call_total_pages)
                    ),
                    "stack": _summarize(stack_values),
                    "compute": _summarize(compute_values),
                    "total": _summarize(total_values),
                    "result_contiguous": bool(all(contig_values)),
                    "minima_stride0_bytes": int(inputs.minima_matrix.strides[0]),
                    "maxima_stride0_bytes": int(inputs.maxima_matrix.strides[0]),
                    "env": {
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
                        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
                        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
                        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
                        "numpy_version": np.__version__,
                    },
                }
            )
    return {"results": results}


def _print_report(payload: dict[str, object]) -> None:
    print(
        "| Label | Mode | Ctx | Step | Pages | Candidates | Cand % | Stack med | Compute med | Total med | Total p95 | Contig |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in payload["results"]:
        print(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    str(row["mode"]),
                    str(row["prompt_length"]),
                    str(row["step_index"]),
                    str(row["per_call_total_pages"]),
                    str(row["per_call_candidate_pages"]),
                    f"{float(row['candidate_fraction']):.4f}",
                    f"{float(row['stack']['median_ms']):.3f}",
                    f"{float(row['compute']['median_ms']):.3f}",
                    f"{float(row['total']['median_ms']):.3f}",
                    f"{float(row['total']['p95_ms']):.3f}",
                    "yes" if bool(row["result_contiguous"]) else "no",
                ]
            )
            + " |"
        )


def main() -> None:
    args = parse_args()
    paths = [Path(raw) for raw in args.inputs]
    scenarios = _load_scenarios(paths)
    if args.prompt_lengths:
        allowed = set(int(value) for value in args.prompt_lengths)
        scenarios = [scenario for scenario in scenarios if scenario.prompt_length in allowed]
    if args.label_filter:
        scenarios = [scenario for scenario in scenarios if args.label_filter in scenario.label]
    if not scenarios:
        raise SystemExit("no selector scenarios matched the provided filters")
    payload = run_microbench(
        scenarios,
        head_dim=int(args.head_dim),
        warmup=int(args.warmup),
        iterations=int(args.iterations),
        seed=int(args.seed),
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _print_report(payload)


if __name__ == "__main__":
    main()
