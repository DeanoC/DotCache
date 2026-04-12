from __future__ import annotations

import argparse
import json
from pathlib import Path


_DEFAULT_BENCHMARK_PATHS: tuple[tuple[str, str, str, str], ...] = (
    (
        "large",
        "mps",
        "real_mixed",
        "benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_large_mps_refreshed/qwen35_persistent_real_mixed_probe.json",
    ),
    (
        "large",
        "mps",
        "non_m0",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_20260412_repo_promptfiles_large_mps_stage9_non_m0_refreshed/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "large",
        "mps",
        "conservative",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_20260412_repo_promptfiles_large_mps_conservative_priority_value_hybrid_ci16_refreshed/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "large",
        "cuda",
        "real_mixed",
        "benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_large_cuda_frontier_batchedresidual_v18_clean/qwen35_persistent_real_mixed_probe.json",
    ),
    (
        "large",
        "cuda",
        "non_m0",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_large_stage9_non_m0_currenttree_v2/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "large",
        "cuda",
        "conservative",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_large_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "broad",
        "mps",
        "real_mixed",
        "benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_broad_mps_refreshed/qwen35_persistent_real_mixed_probe.json",
    ),
    (
        "broad",
        "mps",
        "non_m0",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_20260412_repo_promptfiles_broad_mps_stage9_non_m0_refreshed/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "broad",
        "mps",
        "conservative",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_20260412_repo_promptfiles_broad_mps_conservative_priority_value_hybrid_ci16_refreshed/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "broad",
        "cuda",
        "real_mixed",
        "benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_broad_cuda_frontier_batchedresidual_v18_clean/qwen35_persistent_real_mixed_probe.json",
    ),
    (
        "broad",
        "cuda",
        "non_m0",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_broad_stage9_non_m0_currenttree_v2/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "broad",
        "cuda",
        "conservative",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_broad_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "external",
        "mps",
        "real_mixed",
        "benchmarks/results/qwen35_persistent_real_mixed_probe_20260412_repo_promptfiles_external_mps_refreshed/qwen35_persistent_real_mixed_probe.json",
    ),
    (
        "external",
        "mps",
        "non_m0",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_20260412_repo_promptfiles_external_mps_stage9_non_m0_refreshed/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "external",
        "mps",
        "conservative",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_20260412_repo_promptfiles_external_mps_conservative_priority_value_hybrid_ci16_refreshed/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "external",
        "cuda",
        "real_mixed",
        "benchmarks/results/qwen35_persistent_real_mixed_probe_cuda_repo_external_cuda_frontier_batchedresidual_v17_clean/qwen35_persistent_real_mixed_probe.json",
    ),
    (
        "external",
        "cuda",
        "non_m0",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_external_stage9_non_m0_currenttree/qwen35_persistent_serving_policy_compare.json",
    ),
    (
        "external",
        "cuda",
        "conservative",
        "benchmarks/results/qwen35_persistent_serving_policy_compare_cuda_repo_external_conservative_priority_value_hybrid_ci16/qwen35_persistent_serving_policy_compare.json",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit a compact backend/corpus/policy Stage 9 comparison matrix from checked-in benchmark bundles."
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def _load_row(repo_root: Path, *, corpus: str, backend: str, policy: str, path_str: str) -> dict[str, object]:
    path = repo_root / path_str
    payload = json.loads(path.read_text())
    summary = payload["summary"]
    config = payload["config"]
    return {
        "corpus": corpus,
        "backend": backend,
        "policy": policy,
        "path": path_str,
        "device": config.get("device"),
        "backend_name": config.get("backend"),
        "bias_ms_per_step": float(summary["bias_avg_ms_per_step"]),
        "hand_ms_per_step": float(summary["hand_tuned_avg_ms_per_step"]),
        "bias_vs_hand_exact_match_rate": float(summary["bias_vs_hand_exact_match_rate"]),
        "bias_beats_hand_tuned_latency_rate": float(summary["bias_beats_hand_tuned_latency_rate"]),
    }


def _winner_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    winners: list[dict[str, object]] = []
    for corpus in ("large", "broad", "external"):
        for backend in ("mps", "cuda"):
            subset = [row for row in rows if row["corpus"] == corpus and row["backend"] == backend]
            best = min(subset, key=lambda row: float(row["bias_ms_per_step"]))
            winners.append(
                {
                    "corpus": corpus,
                    "backend": backend,
                    "winner_policy": best["policy"],
                    "winner_bias_ms_per_step": best["bias_ms_per_step"],
                }
            )
    return winners


def build_payload(repo_root: Path) -> dict[str, object]:
    rows = [
        _load_row(repo_root, corpus=corpus, backend=backend, policy=policy, path_str=path_str)
        for corpus, backend, policy, path_str in _DEFAULT_BENCHMARK_PATHS
    ]
    return {
        "rows": rows,
        "winner_rows": _winner_rows(rows),
    }


def _render_markdown(payload: dict[str, object]) -> str:
    rows = payload["rows"]
    winner_rows = payload["winner_rows"]
    lines = [
        "# Qwen3.5 Stage 9 Backend Matrix",
        "",
        "## Winners",
        "",
        "| Corpus | MPS winner | CUDA winner |",
        "| --- | --- | --- |",
    ]
    for corpus in ("large", "broad", "external"):
        mps = next(row for row in winner_rows if row["corpus"] == corpus and row["backend"] == "mps")
        cuda = next(row for row in winner_rows if row["corpus"] == corpus and row["backend"] == "cuda")
        lines.append(
            f"| {corpus} | {mps['winner_policy']} `{mps['winner_bias_ms_per_step']:.2f}` | "
            f"{cuda['winner_policy']} `{cuda['winner_bias_ms_per_step']:.2f}` |"
        )
    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "| Corpus | Backend | Policy | Bias ms/step | Hand ms/step | Exact-match | Bias beats hand |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['corpus']} | {row['backend']} | {row['policy']} | "
            f"`{float(row['bias_ms_per_step']):.2f}` | `{float(row['hand_ms_per_step']):.2f}` | "
            f"`{float(row['bias_vs_hand_exact_match_rate']):.3f}` | "
            f"`{float(row['bias_beats_hand_tuned_latency_rate']):.3f}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_payload(repo_root)
    rendered_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    rendered_md = _render_markdown(payload)
    if args.output_json:
        Path(args.output_json).write_text(rendered_json)
    else:
        print(rendered_json, end="")
    if args.output_md:
        Path(args.output_md).write_text(rendered_md)
    elif not args.output_json:
        print(rendered_md)


if __name__ == "__main__":
    main()
