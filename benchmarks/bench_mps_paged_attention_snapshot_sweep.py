from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_mps_paged_attention_experimental import run_paged_attention_benchmark
from dotcache.backends.mps_persistent_experimental import PagedAttentionControllerConfig, load_paged_attention_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep experimental MPS paged-attention settings over exported snapshots.")
    parser.add_argument("--snapshot-paths", nargs="*", default=[])
    parser.add_argument("--snapshot-glob", action="append", default=[])
    parser.add_argument("--engines", nargs="+", choices=["cpu_ref", "torch_mps_baseline", "mps_experimental"], default=["torch_mps_baseline", "mps_experimental"])
    parser.add_argument("--top-ks", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--recent-windows", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--sink-windows", nargs="+", type=int, default=[128, 256])
    parser.add_argument("--page-chunk-sizes", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--approximate-modes", nargs="+", choices=["off", "on"], default=["off"])
    parser.add_argument("--approximate-max-optional-blocks", nargs="+", type=int, default=[0])
    parser.add_argument("--early-exit-modes", nargs="+", choices=["off", "on"], default=["off"])
    parser.add_argument("--early-exit-epsilons", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--max-abs-error-threshold", type=float, default=1e-3)
    parser.add_argument("--max-rel-error-threshold", type=float, default=1e-3)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _resolve_snapshot_paths(snapshot_paths: list[str], snapshot_globs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    candidates = [Path(path) for path in snapshot_paths]
    for pattern in snapshot_globs:
        candidates.extend(Path(path) for path in sorted(glob.glob(pattern)))
    for path in candidates:
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        if not resolved_path.exists():
            raise FileNotFoundError(f"snapshot path does not exist: {resolved_path}")
        if resolved_path.suffix != ".npz":
            continue
        seen.add(resolved_path)
        resolved.append(resolved_path)
    if not resolved:
        raise ValueError("at least one .npz snapshot path is required")
    return resolved


def _config_records(args: argparse.Namespace) -> list[PagedAttentionControllerConfig]:
    configs: list[PagedAttentionControllerConfig] = []
    seen: set[tuple[int, int, int, int, bool, int, bool, float]] = set()
    for top_k in args.top_ks:
        for recent_window in args.recent_windows:
            for sink_window in args.sink_windows:
                for page_chunk_size in args.page_chunk_sizes:
                    for approximate_mode in args.approximate_modes:
                        for approximate_max_optional_blocks in args.approximate_max_optional_blocks:
                            if approximate_mode == "off" and int(approximate_max_optional_blocks) != 0:
                                continue
                            for early_exit_mode in args.early_exit_modes:
                                if early_exit_mode == "off":
                                    key = (
                                        int(top_k),
                                        int(recent_window),
                                        int(sink_window),
                                        int(page_chunk_size),
                                        approximate_mode == "on",
                                        int(approximate_max_optional_blocks),
                                        False,
                                        0.0,
                                    )
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    configs.append(
                                        PagedAttentionControllerConfig(
                                            top_k=int(top_k),
                                            recent_window_tokens=int(recent_window),
                                            sink_window_tokens=int(sink_window),
                                            page_chunk_size=int(page_chunk_size),
                                            approximate_mode=approximate_mode == "on",
                                            approximate_max_optional_blocks=int(approximate_max_optional_blocks),
                                            early_exit=False,
                                            early_exit_eps=1e-4,
                                            mass_eps=1e-4,
                                            value_eps=1e-4,
                                        )
                                    )
                                    continue
                                for early_exit_eps in args.early_exit_epsilons:
                                    key = (
                                        int(top_k),
                                        int(recent_window),
                                        int(sink_window),
                                        int(page_chunk_size),
                                        approximate_mode == "on",
                                        int(approximate_max_optional_blocks),
                                        True,
                                        float(early_exit_eps),
                                    )
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    configs.append(
                                        PagedAttentionControllerConfig(
                                            top_k=int(top_k),
                                            recent_window_tokens=int(recent_window),
                                            sink_window_tokens=int(sink_window),
                                            page_chunk_size=int(page_chunk_size),
                                            approximate_mode=approximate_mode == "on",
                                            approximate_max_optional_blocks=int(approximate_max_optional_blocks),
                                            early_exit=True,
                                            early_exit_eps=float(early_exit_eps),
                                            mass_eps=float(early_exit_eps),
                                            value_eps=float(early_exit_eps),
                                        )
                                    )
    return configs


def _backend_label(engine: str) -> str:
    labels = {
        "cpu_ref": "CPU Reference",
        "torch_mps_baseline": "Baseline Backend",
        "mps_experimental": "Experimental Backend",
    }
    return labels.get(engine, engine)


def _controller_metadata(config: PagedAttentionControllerConfig) -> dict[str, str]:
    if bool(config.approximate_mode):
        return {
            "controller_mode": "approx_budget",
            "controller_label": "Approx Budget",
        }
    if bool(config.early_exit):
        return {
            "controller_mode": "certified_early_exit",
            "controller_label": "Certified Early Exit",
        }
    return {
        "controller_mode": "robust_full_pass",
        "controller_label": "Robust Full Pass",
    }


def _candidate_record(snapshot_path: Path, engine: str, config: PagedAttentionControllerConfig, result: dict[str, object]) -> dict[str, object]:
    record = dict(result)
    controller = _controller_metadata(config)
    backend_label = _backend_label(engine)
    record.update(
        {
            "record_type": "candidate",
            "snapshot_path": str(snapshot_path),
            "snapshot_name": snapshot_path.name,
            "config_key": (
                f"topk={int(config.top_k)}"
                f"|recent={int(config.recent_window_tokens)}"
                f"|sink={int(config.sink_window_tokens)}"
                f"|chunk={int(config.page_chunk_size)}"
                f"|approx={int(config.approximate_mode)}"
                f"|approx_opt={int(config.approximate_max_optional_blocks)}"
                f"|early_exit={int(config.early_exit)}"
                f"|eps={float(config.early_exit_eps):.6g}"
            ),
            "engine": engine,
            "backend_label": backend_label,
            "controller_mode": controller["controller_mode"],
            "controller_label": controller["controller_label"],
            "display_label": f"{backend_label} / {controller['controller_label']}",
        }
    )
    return record


def _aggregate_records(
    candidate_records: list[dict[str, object]],
    *,
    max_abs_error_threshold: float,
    max_rel_error_threshold: float,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in candidate_records:
        key = (str(record["engine"]), str(record["config_key"]))
        grouped.setdefault(key, []).append(record)

    aggregate_records: list[dict[str, object]] = []
    for (engine, config_key), records in sorted(grouped.items()):
        first = records[0]
        pass_count = sum(
            1
            for record in records
            if float(record["max_abs_error"]) <= max_abs_error_threshold
            and float(record["max_rel_error"]) <= max_rel_error_threshold
        )
        count = len(records)
        aggregate_records.append(
            {
                "record_type": "aggregate",
                "engine": engine,
                "config_key": config_key,
                "backend_label": str(first.get("backend_label", _backend_label(engine))),
                "controller_mode": str(first.get("controller_mode", "robust_full_pass")),
                "controller_label": str(first.get("controller_label", "Robust Full Pass")),
                "display_label": str(first.get("display_label", f"{_backend_label(engine)} / Robust Full Pass")),
                "snapshot_count": count,
                "pass_count": pass_count,
                "pass_rate": float(pass_count / max(count, 1)),
                "avg_total_step_time_ms": float(sum(float(record["total_step_time_ms"]) for record in records) / max(count, 1)),
                "avg_score_time_ms": float(sum(float(record["score_time_ms"]) for record in records) / max(count, 1)),
                "avg_selection_time_ms": float(sum(float(record["selection_time_ms"]) for record in records) / max(count, 1)),
                "avg_attention_time_ms": float(sum(float(record["attention_time_ms"]) for record in records) / max(count, 1)),
                "avg_selected_page_count": float(sum(float(record["selected_page_count"]) for record in records) / max(count, 1)),
                "avg_processed_page_count": float(sum(float(record["processed_page_count"]) for record in records) / max(count, 1)),
                "avg_tokens_processed": float(sum(float(record["tokens_processed"]) for record in records) / max(count, 1)),
                "avg_early_exit_rate": float(sum(float(record["early_exit_rate"]) for record in records) / max(count, 1)),
                "max_abs_error": float(max(float(record["max_abs_error"]) for record in records)),
                "max_rel_error": float(max(float(record["max_rel_error"]) for record in records)),
            }
        )
    return aggregate_records


def _recommend_records(aggregate_records: list[dict[str, object]]) -> list[dict[str, object]]:
    recommendations: list[dict[str, object]] = []
    by_engine: dict[str, list[dict[str, object]]] = {}
    for record in aggregate_records:
        by_engine.setdefault(str(record["engine"]), []).append(record)

    for engine, records in sorted(by_engine.items()):
        best = min(
            records,
            key=lambda record: (
                -float(record["pass_rate"]),
                float(record["avg_total_step_time_ms"]),
                float(record["max_abs_error"]),
                float(record["max_rel_error"]),
            ),
        )
        recommendation = dict(best)
        recommendation["record_type"] = "recommendation"
        recommendation["recommendation_reason"] = (
            "fastest_full_pass"
            if float(best["pass_rate"]) >= 1.0
            else "best_pass_rate_then_fastest"
        )
        recommendations.append(recommendation)
    return recommendations


def main() -> None:
    args = parse_args()
    snapshot_paths = _resolve_snapshot_paths(args.snapshot_paths, args.snapshot_glob)
    configs = _config_records(args)

    candidate_records: list[dict[str, object]] = []
    for snapshot_path in snapshot_paths:
        snapshot = load_paged_attention_snapshot(snapshot_path)
        for engine in args.engines:
            for config in configs:
                result = run_paged_attention_benchmark(
                    snapshot,
                    config=config,
                    engine=engine,
                    device=args.device,
                    dtype=args.dtype,
                    warmup_runs=args.warmup_runs,
                    measured_runs=args.measured_runs,
                )
                record = _candidate_record(snapshot_path, engine, config, result)
                candidate_records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    aggregate_records = _aggregate_records(
        candidate_records,
        max_abs_error_threshold=args.max_abs_error_threshold,
        max_rel_error_threshold=args.max_rel_error_threshold,
    )
    for record in aggregate_records:
        print(json.dumps(record, sort_keys=True), flush=True)

    recommendation_records = _recommend_records(aggregate_records)
    for record in recommendation_records:
        print(json.dumps(record, sort_keys=True), flush=True)

    if args.output_json is not None:
        payload = {
            "candidate_records": candidate_records,
            "aggregate_records": aggregate_records,
            "recommendation_records": recommendation_records,
        }
        Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
