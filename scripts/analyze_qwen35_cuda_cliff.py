from __future__ import annotations

import json
from pathlib import Path


RESULTS_DIR = Path("benchmarks/results")
BASELINE_QUALITY = RESULTS_DIR / "qwen35_cuda_qualitycheck_16384_32768.jsonl"
BASELINE_SCORER = RESULTS_DIR / "qwen35_cuda_scorer_diagnostic_16384_32768.jsonl"
UNIONWIDE_QUALITY = RESULTS_DIR / "qwen35_cuda_highmargin_promote_quality_16384_32768_t0695_maxctx16384_unionwide_confirm.jsonl"
UNIONWIDE_SCORER = RESULTS_DIR / "qwen35_cuda_highmargin_promote_scorer_16384_32768_t0695_maxctx16384_unionwide_confirm.jsonl"
OUTPUT_JSON = RESULTS_DIR / "qwen35_cuda_cliff_analysis_20260329.json"
OUTPUT_MD = RESULTS_DIR / "qwen35_cuda_cliff_analysis_20260329.md"


TRACE_KEYS = [
    "score_ms_total",
    "mix_ms_total",
    "softmax_ms_total",
    "chunk_assembly_ms_total",
    "unpack_ms_total",
    "score_calls",
    "mix_calls",
    "payload_bytes_read",
    "metadata_bytes_read",
]

MEMORY_KEYS = [
    "resident_bytes",
    "kv_resident_bytes",
    "direct_page_resident_bytes",
    "prepared_chunk_resident_bytes",
    "dotcache_prefill_cuda_peak_memory_allocated_bytes",
    "dotcache_prefill_cuda_peak_memory_reserved_bytes",
    "dotcache_decode_cuda_peak_memory_allocated_bytes",
    "dotcache_decode_cuda_peak_memory_reserved_bytes",
]


def _load_rows(path: Path) -> dict[int, dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result: dict[int, dict[str, object]] = {}
    for row in rows:
        prompt_length = int(
            row.get("input_token_count")
            or row.get("prompt_length")
            or row.get("target_prompt_length")
            or 0
        )
        result[prompt_length] = row
    return result


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _layer23_group_summary(row: dict[str, object]) -> list[dict[str, object]]:
    for record in row.get("scorer_layer_records", []):
        if int(record.get("layer_id", -1)) != 23:
            continue
        return [
            {
                "kv_head_id": int(group.get("kv_head_id", -1)),
                "union_exact_promote_rescue_applied": bool(group.get("union_exact_promote_rescue_applied", False)),
                "union_exact_promote_rescue_selected_novel_count": int(
                    group.get("union_exact_promote_rescue_selected_novel_count", 0)
                ),
                "selected_old_pages": int(group.get("selected_old_pages", 0)),
                "selected_pages": int(group.get("selected_pages", 0)),
                "union_added_pages": int(group.get("union_added_pages", 0)),
                "union_added_exact_top_hits": int(group.get("union_added_exact_top_hits", 0)),
                "union_added_mean_exact_rank": group.get("union_added_mean_exact_rank"),
                "union_exact_promote_rescue_selected_novel_page_ranges": list(
                    group.get("union_exact_promote_rescue_selected_novel_page_ranges", [])
                ),
            }
            for group in record.get("groups", [])
        ]
    return []


def build_report() -> dict[str, object]:
    baseline_quality = _load_rows(BASELINE_QUALITY)
    baseline_scorer = _load_rows(BASELINE_SCORER)
    unionwide_quality = _load_rows(UNIONWIDE_QUALITY)
    unionwide_scorer = _load_rows(UNIONWIDE_SCORER)

    contexts = [16384, 32768]
    baseline_contexts: dict[str, object] = {}
    unionwide_contexts: dict[str, object] = {}
    for context in contexts:
        quality_row = baseline_quality[context]
        scorer_row = baseline_scorer[context]
        baseline_contexts[str(context)] = {
            "dotcache_decode_ms_per_step": float(quality_row["dotcache_decode_ms_per_step"]),
            "teacher_forced_logit_mean_abs_error": float(quality_row["teacher_forced_logit_mean_abs_error"]),
            "teacher_forced_logit_rmse": float(quality_row["teacher_forced_logit_rmse"]),
            "execution_shortlist_selected_pages": int(quality_row["execution_shortlist_selected_pages"]),
            "execution_shortlist_selected_pages_by_layer_23": int(
                quality_row["execution_shortlist_selected_pages_by_layer"]["23"]
            ),
            "trace": {key: scorer_row["decode_backend_trace"].get(key) for key in TRACE_KEYS},
            "memory": {key: quality_row.get(key) for key in MEMORY_KEYS},
        }

        quality_row = unionwide_quality[context]
        scorer_row = unionwide_scorer[context]
        unionwide_contexts[str(context)] = {
            "dotcache_decode_ms_per_step": float(quality_row["dotcache_decode_ms_per_step"]),
            "teacher_forced_logit_mean_abs_error": float(quality_row["teacher_forced_logit_mean_abs_error"]),
            "teacher_forced_logit_rmse": float(quality_row["teacher_forced_logit_rmse"]),
            "execution_shortlist_selected_pages": int(quality_row["execution_shortlist_selected_pages"]),
            "execution_shortlist_selected_pages_by_layer_23": int(
                quality_row["execution_shortlist_selected_pages_by_layer"]["23"]
            ),
            "trace": {key: scorer_row["decode_backend_trace"].get(key) for key in TRACE_KEYS},
            "memory": {key: quality_row.get(key) for key in MEMORY_KEYS},
            "layer23_groups": _layer23_group_summary(scorer_row),
        }

    baseline_16384 = baseline_contexts["16384"]
    baseline_32768 = baseline_contexts["32768"]
    baseline_trace_16 = baseline_16384["trace"]
    baseline_trace_32 = baseline_32768["trace"]
    baseline_memory_16 = baseline_16384["memory"]
    baseline_memory_32 = baseline_32768["memory"]

    return {
        "baseline": baseline_contexts,
        "unionwide_confirm": unionwide_contexts,
        "derived": {
            "baseline_32768_vs_16384": {
                "decode_ms_ratio": _ratio(
                    baseline_32768["dotcache_decode_ms_per_step"],
                    baseline_16384["dotcache_decode_ms_per_step"],
                ),
                "score_ms_ratio": _ratio(baseline_trace_32["score_ms_total"], baseline_trace_16["score_ms_total"]),
                "mix_ms_ratio": _ratio(baseline_trace_32["mix_ms_total"], baseline_trace_16["mix_ms_total"]),
                "softmax_ms_ratio": _ratio(
                    baseline_trace_32["softmax_ms_total"],
                    baseline_trace_16["softmax_ms_total"],
                ),
                "payload_bytes_ratio": _ratio(
                    baseline_trace_32["payload_bytes_read"],
                    baseline_trace_16["payload_bytes_read"],
                ),
                "metadata_bytes_ratio": _ratio(
                    baseline_trace_32["metadata_bytes_read"],
                    baseline_trace_16["metadata_bytes_read"],
                ),
                "resident_bytes_ratio": _ratio(
                    baseline_memory_32["resident_bytes"],
                    baseline_memory_16["resident_bytes"],
                ),
                "prepared_chunk_resident_bytes_ratio": _ratio(
                    baseline_memory_32["prepared_chunk_resident_bytes"],
                    baseline_memory_16["prepared_chunk_resident_bytes"],
                ),
                "decode_reserved_bytes_ratio": _ratio(
                    baseline_memory_32["dotcache_decode_cuda_peak_memory_reserved_bytes"],
                    baseline_memory_16["dotcache_decode_cuda_peak_memory_reserved_bytes"],
                ),
            }
        },
    }


def write_report(report: dict[str, object]) -> None:
    OUTPUT_JSON.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    derived = report["derived"]["baseline_32768_vs_16384"]
    baseline = report["baseline"]
    unionwide = report["unionwide_confirm"]
    lines = [
        "# Qwen3.5 CUDA 16K vs 32K Cliff Analysis",
        "",
        "## Summary",
        "",
        f"- Baseline decode grows from `{baseline['16384']['dotcache_decode_ms_per_step']:.2f} ms/step` at `16384` to `{baseline['32768']['dotcache_decode_ms_per_step']:.2f} ms/step` at `32768` (`{derived['decode_ms_ratio']:.2f}x`).",
        f"- The shortlist stays flat: `{baseline['16384']['execution_shortlist_selected_pages']}` selected pages at both lengths, with layer `23` fixed at `{baseline['16384']['execution_shortlist_selected_pages_by_layer_23']}`.",
        f"- Backend trace work terms do not grow in bytes or calls, but `score` and `mix` time do: `score {derived['score_ms_ratio']:.2f}x`, `mix {derived['mix_ms_ratio']:.2f}x`, `payload_bytes {derived['payload_bytes_ratio']:.2f}x`.",
        f"- Resident/runtime memory does grow materially: `resident_bytes {derived['resident_bytes_ratio']:.2f}x`, `prepared_chunk_resident_bytes {derived['prepared_chunk_resident_bytes_ratio']:.2f}x`, `decode_reserved_bytes {derived['decode_reserved_bytes_ratio']:.2f}x`.",
        "",
        "## Interpretation",
        "",
        "- The 32K cliff is unlikely to be caused by attending more shortlisted pages.",
        "- The strongest current explanation is worse memory locality or kernel efficiency under the larger resident/prepared working set.",
        "- The union-wide layer-23 rescue now activates correctly at 16K, but it still does not move quality enough to explain the 32K behavior.",
        "",
        "## Union-Wide Rescue Check",
        "",
    ]
    for group in unionwide["16384"]["layer23_groups"]:
        lines.append(
            f"- KV `{group['kv_head_id']}`: rescue_applied=`{group['union_exact_promote_rescue_applied']}`, "
            f"selected_novel=`{group['union_exact_promote_rescue_selected_novel_count']}`, "
            f"selected_old_pages=`{group['selected_old_pages']}`, union_added_pages=`{group['union_added_pages']}`, "
            f"union_added_mean_exact_rank=`{group['union_added_mean_exact_rank']}`."
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "- Investigate the 32K cliff as a locality/working-set issue in the grouped decode backend, not as a shortlist-size problem.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n")


def main() -> None:
    report = build_report()
    write_report(report)
    print(f"wrote {OUTPUT_JSON}")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
