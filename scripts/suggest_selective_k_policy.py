#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from typing import Any

from probe_attention_score_fidelity import (
    _compose_rescue_rows,
    _summary_from_rows,
    probe_attention_score_fidelity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest a selective exact-K policy from offline DotCache attention fidelity probes."
    )
    parser.add_argument("--family", choices=["llama", "qwen2"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--quant-scheme-k", choices=["affine", "lut", "sketch", "turbo3"], default="affine")
    parser.add_argument("--quant-scheme-v", choices=["affine", "lut", "turbo3"], default="affine")
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--candidate-layers", type=int, default=6)
    parser.add_argument("--max-kv-groups-per-layer", type=int, default=2)
    parser.add_argument("--max-combo-size", type=int, default=2)
    parser.add_argument("--target-recovery", type=float, default=0.65)
    parser.add_argument(
        "--budget-recovery-margin",
        type=float,
        default=0.9,
        help="Recommend the smallest selective policy whose composite recovery is within this fraction of the best selective policy.",
    )
    parser.add_argument("--top-policies", type=int, default=8)
    parser.add_argument(
        "--validate-top-policies",
        type=int,
        default=0,
        help="Optionally run end-to-end compare benchmarks for the top-N selective candidates.",
    )
    parser.add_argument(
        "--validation-max-new-tokens",
        type=int,
        default=4,
        help="Greedy decode length used by optional end-to-end candidate validation.",
    )
    parser.add_argument(
        "--validation-agreement-threshold",
        type=float,
        default=0.99,
        help="Agreement threshold for considering a validated policy successful.",
    )
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    return parser.parse_args()


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 1.0 if numerator >= 0.0 else 0.0
    return numerator / denominator


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _recovery_metrics(summary: dict[str, Any], *, baseline: dict[str, Any], exact: dict[str, Any]) -> dict[str, float]:
    top1 = _clamp01(
        _safe_ratio(
            float(summary["score_top1_match_rate"]) - float(baseline["score_top1_match_rate"]),
            float(exact["score_top1_match_rate"]) - float(baseline["score_top1_match_rate"]),
        )
    )
    topk = _clamp01(
        _safe_ratio(
            float(summary["score_topk_overlap_mean"]) - float(baseline["score_topk_overlap_mean"]),
            float(exact["score_topk_overlap_mean"]) - float(baseline["score_topk_overlap_mean"]),
        )
    )
    rank = _clamp01(
        _safe_ratio(
            float(summary["score_rank_corr_mean"]) - float(baseline["score_rank_corr_mean"]),
            float(exact["score_rank_corr_mean"]) - float(baseline["score_rank_corr_mean"]),
        )
    )
    kl = _clamp01(
        _safe_ratio(
            float(baseline["attn_kl_mean"]) - float(summary["attn_kl_mean"]),
            float(baseline["attn_kl_mean"]) - float(exact["attn_kl_mean"]),
        )
    )
    rmse = _clamp01(
        _safe_ratio(
            float(baseline["output_rmse_mean"]) - float(summary["output_rmse_mean"]),
            float(baseline["output_rmse_mean"]) - float(exact["output_rmse_mean"]),
        )
    )
    composite = (0.35 * top1) + (0.2 * topk) + (0.15 * rank) + (0.2 * kl) + (0.1 * rmse)
    return {
        "top1_recovery": top1,
        "topk_recovery": topk,
        "rank_recovery": rank,
        "kl_recovery": kl,
        "rmse_recovery": rmse,
        "composite_recovery": composite,
    }


def _normalize_targets(targets: set[tuple[int, int | None]]) -> tuple[tuple[int, int | None], ...]:
    full_layers = {layer_id for layer_id, kv_head_id in targets if kv_head_id is None}
    normalized = {(layer_id, None) for layer_id in full_layers}
    normalized.update(
        (layer_id, kv_head_id)
        for layer_id, kv_head_id in targets
        if kv_head_id is not None and layer_id not in full_layers
    )
    return tuple(sorted(normalized, key=lambda item: (item[0], -1 if item[1] is None else item[1])))


def _estimate_exact_fraction(targets: tuple[tuple[int, int | None], ...], *, num_layers: int, num_kv_heads: int) -> float:
    if num_layers <= 0 or num_kv_heads <= 0:
        return 0.0
    exact_units = 0
    for _, kv_head_id in targets:
        exact_units += num_kv_heads if kv_head_id is None else 1
    return float(exact_units / (num_layers * num_kv_heads))


def _target_label(targets: tuple[tuple[int, int | None], ...]) -> str:
    if not targets:
        return "all_m0"
    return ",".join(
        f"layer:{layer_id}=M3" if kv_head_id is None else f"layer:{layer_id}:kv:{kv_head_id}=M3"
        for layer_id, kv_head_id in targets
    )


def _override_flags(targets: tuple[tuple[int, int | None], ...]) -> list[str]:
    flags: list[str] = []
    for layer_id, kv_head_id in targets:
        if kv_head_id is None:
            flags.extend(["--key-mode-override", f"layer:{layer_id}=M3"])
        else:
            flags.extend(["--key-mode-override", f"layer:{layer_id}:kv:{kv_head_id}=M3"])
    return flags


def _candidate_rows(
    d5_rows: list[dict[str, Any]],
    d6_rows: list[dict[str, Any]],
    targets: tuple[tuple[int, int | None], ...],
) -> list[dict[str, Any]]:
    return _compose_rescue_rows(
        d5_rows=d5_rows,
        d6_rows=d6_rows,
        selector=lambda row, targets=set(targets): (
            (int(row["layer_id"]), None) in targets
            or (int(row["layer_id"]), int(row["kv_head_id"])) in targets
        ),
    )


def _benchmark_command_args(
    args: argparse.Namespace,
    *,
    label: str,
    targets: tuple[tuple[int, int | None], ...],
) -> list[str]:
    script_name = "bench_llama_compare.py" if args.family == "llama" else "bench_qwen2_compare.py"
    command = [
        ".venv/bin/python",
        f"benchmarks/{script_name}",
        "--model-id",
        args.model_id,
        "--backend",
        args.backend,
        "--target-prompt-lengths",
        str(args.prompt_length),
        "--max-new-tokens",
        str(args.validation_max_new_tokens),
        "--repeat-counts",
        "--continue-on-error",
    ]
    if args.device is not None:
        command.extend(["--device", args.device])
    if label == "exact_k":
        command.extend(["--default-mode-k", "M3", "--default-mode-v", "M0"])
    else:
        command.extend(["--default-mode-k", "M0", "--default-mode-v", "M0"])
        command.extend(_override_flags(targets))
    return command


def _benchmark_command(
    args: argparse.Namespace,
    *,
    label: str,
    targets: tuple[tuple[int, int | None], ...],
) -> str:
    return " ".join(_benchmark_command_args(args, label=label, targets=targets))


def _parse_json_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _policy_target_count(policy: dict[str, Any]) -> int:
    return len(tuple(policy.get("targets") or ()))


def _matching_validation_record(records: list[dict[str, Any]], *, prompt_length: int) -> dict[str, Any] | None:
    for record in records:
        if int(record.get("prompt_length") or -1) != int(prompt_length):
            continue
        prompt_mode = record.get("prompt_mode")
        if prompt_mode not in {None, "exact_length"}:
            continue
        return record
    return None


def _validate_candidate_policy(
    args: argparse.Namespace,
    *,
    label: str,
    targets: tuple[tuple[int, int | None], ...],
) -> dict[str, Any]:
    command = _benchmark_command_args(args, label=label, targets=targets)
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    records = _parse_json_records(completed.stdout)
    record = _matching_validation_record(records, prompt_length=args.prompt_length)
    validation: dict[str, Any] = {
        "label": label,
        "targets": [
            {"layer_id": int(layer_id), "kv_head_id": None if kv_head_id is None else int(kv_head_id)}
            for layer_id, kv_head_id in targets
        ],
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "record_found": record is not None,
    }
    if record is None:
        validation.update(
            {
                "status": "error",
                "agreement": None,
                "decode_ms_per_step": None,
                "kv_vs_dense": None,
                "kv_resident_bytes": None,
                "k_total_static_pages": None,
                "k_m3_pages": None,
                "k_exact_fraction": None,
                "error_type": "MissingRecord",
                "error_message": "benchmark output did not contain an exact-length prompt record",
            }
        )
        return validation
    status = str(record.get("status") or "ok")
    validation.update(
        {
            "status": status,
            "agreement": None if status == "error" else float(record.get("greedy_token_agreement_rate") or 0.0),
            "decode_ms_per_step": None if status == "error" else float(record.get("decode_ms_per_step") or 0.0),
            "kv_vs_dense": None
            if status == "error" or record.get("dotcache_vs_dense_kv_bytes_ratio") is None
            else float(record.get("dotcache_vs_dense_kv_bytes_ratio") or 0.0),
            "kv_resident_bytes": None
            if status == "error" or record.get("kv_resident_bytes") is None
            else int(record.get("kv_resident_bytes") or 0),
            "k_total_static_pages": None
            if status == "error" or record.get("k_total_static_pages") is None
            else int(record.get("k_total_static_pages") or 0),
            "k_m3_pages": None
            if status == "error" or record.get("k_m3_pages") is None
            else int(record.get("k_m3_pages") or 0),
            "k_exact_fraction": None,
            "error_type": record.get("error_type"),
            "error_message": record.get("error_message"),
        }
    )
    total_pages = validation.get("k_total_static_pages")
    exact_pages = validation.get("k_m3_pages")
    if status != "error" and isinstance(total_pages, int) and total_pages > 0 and isinstance(exact_pages, int):
        validation["k_exact_fraction"] = float(exact_pages / total_pages)
    return validation


def build_policy_suggestions(args: argparse.Namespace) -> dict[str, Any]:
    probe_args = argparse.Namespace(
        family=args.family,
        model_id=args.model_id,
        device=args.device,
        backend=args.backend,
        torch_dtype=args.torch_dtype,
        group_size=args.group_size,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        quant_scheme_k=args.quant_scheme_k,
        quant_scheme_v=args.quant_scheme_v,
        tokens_per_page=args.tokens_per_page,
        decode_steps=args.decode_steps,
        top_k=args.top_k,
        prompt_unit=args.prompt_unit,
        prompt_length=args.prompt_length,
        layer_rescue_sweep=True,
        target_layers=[],
        kv_group_rescue_sweep=False,
        combo_rescue=[],
        format="json",
        include_rows=True,
    )
    result = probe_attention_score_fidelity(probe_args)
    rows_by_variant = result["rows_by_variant"]
    d5_rows = rows_by_variant["D5"]
    d6_rows = rows_by_variant["D6"]
    baseline = next(summary for summary in result["variant_summaries"] if summary["variant_id"] == "D5")
    exact = next(summary for summary in result["variant_summaries"] if summary["variant_id"] == "D6")

    num_layers = len({int(row["layer_id"]) for row in d5_rows})
    num_kv_heads = len({int(row["kv_head_id"]) for row in d5_rows})
    hot_layers = [
        int(summary["rescue_layer_id"])
        for summary in result.get("layer_rescue_summaries", [])[: max(args.candidate_layers, 0)]
    ]

    candidate_atoms: list[tuple[tuple[int, int | None], ...]] = []
    seen_atoms: set[tuple[tuple[int, int | None], ...]] = set()
    for layer_id in hot_layers:
        atom = ((layer_id, None),)
        if atom not in seen_atoms:
            candidate_atoms.append(atom)
            seen_atoms.add(atom)
        kv_summaries: list[dict[str, Any]] = []
        kv_head_ids = sorted({int(row["kv_head_id"]) for row in d5_rows if int(row["layer_id"]) == layer_id})
        for kv_head_id in kv_head_ids:
            kv_rows = _compose_rescue_rows(
                d5_rows=d5_rows,
                d6_rows=d6_rows,
                selector=lambda row, layer_id=layer_id, kv_head_id=kv_head_id: (
                    int(row["layer_id"]) == layer_id and int(row["kv_head_id"]) == kv_head_id
                ),
            )
            kv_summary = _summary_from_rows(kv_rows, variant_id=f"candidate_L{layer_id}_KV{kv_head_id}")
            kv_summary["rescue_layer_id"] = layer_id
            kv_summary["rescue_kv_head_id"] = kv_head_id
            kv_summaries.append(kv_summary)
        kv_summaries.sort(
            key=lambda summary: (
                -float(summary["score_top1_match_rate"]),
                -float(summary["score_topk_overlap_mean"]),
                -float(summary["score_rank_corr_mean"]),
                float(summary["attn_kl_mean"]),
            )
        )
        for kv_summary in kv_summaries[: max(args.max_kv_groups_per_layer, 0)]:
            atom = ((int(kv_summary["rescue_layer_id"]), int(kv_summary["rescue_kv_head_id"])),)
            if atom not in seen_atoms:
                candidate_atoms.append(atom)
                seen_atoms.add(atom)

    policies: list[dict[str, Any]] = []
    policies.append(
        {
            "targets": (),
            "label": "all_m0",
            "summary": baseline,
        }
    )
    policies.append(
        {
            "targets": tuple((layer_id, None) for layer_id in range(num_layers)),
            "label": "exact_k",
            "summary": exact,
        }
    )
    for combo_size in range(1, max(args.max_combo_size, 1) + 1):
        for atoms in itertools.combinations(candidate_atoms, combo_size):
            targets: set[tuple[int, int | None]] = set()
            for atom in atoms:
                targets.update(atom)
            normalized_targets = _normalize_targets(targets)
            label = _target_label(normalized_targets)
            if any(policy["label"] == label for policy in policies):
                continue
            combo_rows = _candidate_rows(d5_rows, d6_rows, normalized_targets)
            policies.append(
                {
                    "targets": normalized_targets,
                    "label": label,
                    "summary": _summary_from_rows(combo_rows, variant_id=f"suggest_{label}"),
                }
            )

    enriched_policies: list[dict[str, Any]] = []
    for policy in policies:
        recovery = _recovery_metrics(policy["summary"], baseline=baseline, exact=exact)
        targets = tuple(policy["targets"])
        enriched_policies.append(
                {
                    "label": policy["label"],
                    "targets": targets,
                "estimated_k_exact_fraction": _estimate_exact_fraction(
                    targets,
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    ),
                    "summary": policy["summary"],
                    "recovery": recovery,
                    "benchmark_command": _benchmark_command(args, label=policy["label"], targets=targets),
                }
            )

    enriched_policies.sort(
        key=lambda policy: (
            -float(policy["recovery"]["composite_recovery"]),
            -float(policy["summary"]["score_top1_match_rate"]),
            float(policy["estimated_k_exact_fraction"]),
            float(policy["summary"]["attn_kl_mean"]),
        )
    )

    recommended = None
    best_selective = None
    selective_policies = [policy for policy in enriched_policies if policy["label"] not in {"all_m0", "exact_k"}]
    if selective_policies:
        best_selective = selective_policies[0]
        near_best = [
            policy
            for policy in selective_policies
            if float(policy["recovery"]["composite_recovery"])
            >= float(best_selective["recovery"]["composite_recovery"]) * float(args.budget_recovery_margin)
        ]
        if near_best:
            recommended = min(
                near_best,
                key=lambda policy: (
                    float(policy["estimated_k_exact_fraction"]),
                    -float(policy["recovery"]["composite_recovery"]),
                    float(policy["summary"]["attn_kl_mean"]),
                ),
            )

    target_policies = [
        policy
        for policy in enriched_policies
        if float(policy["recovery"]["composite_recovery"]) >= float(args.target_recovery)
        and policy["label"] not in {"all_m0", "exact_k"}
    ]
    if recommended is None and target_policies:
        recommended = min(
            target_policies,
            key=lambda policy: (
                float(policy["estimated_k_exact_fraction"]),
                -float(policy["recovery"]["composite_recovery"]),
                float(policy["summary"]["attn_kl_mean"]),
            ),
        )
    elif recommended is None and selective_policies:
        recommended = selective_policies[0]

    validated_policies: list[dict[str, Any]] = []
    validated_recommended = None
    if args.validate_top_policies > 0:
        selective_candidates = [
            policy
            for policy in enriched_policies
            if policy["label"] not in {"all_m0", "exact_k"}
        ]
        selected: list[dict[str, Any]] = selective_candidates[: args.validate_top_policies]
        extras: list[dict[str, Any]] = []
        whole_layer_candidates = [
            policy
            for policy in selective_candidates
            if all(kv_head_id is None for _, kv_head_id in policy["targets"])
        ]
        if whole_layer_candidates:
            extras.append(
                min(
                    whole_layer_candidates,
                    key=lambda policy: (
                        float(policy["estimated_k_exact_fraction"]),
                        _policy_target_count(policy),
                        -float(policy["recovery"]["composite_recovery"]),
                    ),
                )
            )
        if selective_candidates:
            extras.append(
                min(
                    selective_candidates,
                    key=lambda policy: (
                        float(policy["estimated_k_exact_fraction"]),
                        _policy_target_count(policy),
                        -float(policy["recovery"]["composite_recovery"]),
                    ),
                )
            )
        seen_labels: set[str] = set()
        policies_to_validate: list[dict[str, Any]] = []
        for policy in selected + extras:
            label = str(policy["label"])
            if label in seen_labels:
                continue
            seen_labels.add(label)
            policies_to_validate.append(policy)
        for policy in policies_to_validate:
            validation = _validate_candidate_policy(args, label=policy["label"], targets=policy["targets"])
            policy_copy = dict(policy)
            policy_copy["validation"] = validation
            validated_policies.append(policy_copy)
        successful = [
            policy
            for policy in validated_policies
            if policy["validation"]["status"] != "error"
            and float(policy["validation"]["agreement"] or 0.0) >= float(args.validation_agreement_threshold)
        ]
        if successful:
            validated_recommended = min(
                successful,
                key=lambda policy: (
                    float(policy["estimated_k_exact_fraction"]),
                    float(policy["validation"]["decode_ms_per_step"] or float("inf")),
                    -float(policy["recovery"]["composite_recovery"]),
                ),
            )

    return {
        "family": args.family,
        "model_id": args.model_id,
        "backend": args.backend,
        "device": args.device,
        "prompt_length": args.prompt_length,
        "decode_steps": args.decode_steps,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "baseline_summary": baseline,
        "exact_summary": exact,
        "best_selective_policy": best_selective,
        "recommended_policy": recommended,
        "validated_candidate_policies": validated_policies,
        "validated_recommended_policy": validated_recommended,
        "candidate_policies": enriched_policies[: max(args.top_policies, 0)],
        "hot_layers_considered": hot_layers,
        "target_recovery": args.target_recovery,
        "budget_recovery_margin": args.budget_recovery_margin,
        "validation_agreement_threshold": args.validation_agreement_threshold,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Selective K Policy Suggestion: {result['model_id']}",
        "",
        f"- Family: `{result['family']}`",
        f"- Backend: `{result['backend']}` on `{result['device'] or result['backend']}`",
        f"- Prompt length: `{result['prompt_length']}`",
        f"- Decode steps: `{result['decode_steps']}`",
        f"- Layers / KV heads: `{result['num_layers']}` / `{result['num_kv_heads']}`",
        f"- Hot layers considered: `{', '.join(str(layer_id) for layer_id in result['hot_layers_considered']) or '-'}`",
        "",
        "## Baselines",
        f"- `D5` all-M0: top1 `{result['baseline_summary']['score_top1_match_rate']:.6f}`, top8 `{result['baseline_summary']['score_topk_overlap_mean']:.6f}`, rank `{result['baseline_summary']['score_rank_corr_mean']:.6f}`, KL `{result['baseline_summary']['attn_kl_mean']:.6f}`",
        f"- `D6` exact-K: top1 `{result['exact_summary']['score_top1_match_rate']:.6f}`, top8 `{result['exact_summary']['score_topk_overlap_mean']:.6f}`, rank `{result['exact_summary']['score_rank_corr_mean']:.6f}`, KL `{result['exact_summary']['attn_kl_mean']:.6f}`",
    ]
    recommended = result.get("recommended_policy")
    best_selective = result.get("best_selective_policy")
    if best_selective is not None:
        lines.extend(
            [
                "",
                "## Best Selective Policy",
                f"- Overrides: `{best_selective['label']}`",
                f"- Estimated exact K fraction: `{best_selective['estimated_k_exact_fraction'] * 100:.2f}%`",
                f"- Composite recovery: `{best_selective['recovery']['composite_recovery']:.3f}`",
                f"- Summary: top1 `{best_selective['summary']['score_top1_match_rate']:.6f}`, top8 `{best_selective['summary']['score_topk_overlap_mean']:.6f}`, rank `{best_selective['summary']['score_rank_corr_mean']:.6f}`, KL `{best_selective['summary']['attn_kl_mean']:.6f}`",
            ]
        )
    if recommended is not None:
        lines.extend(
            [
                "",
                "## Recommended Policy",
                f"- Overrides: `{recommended['label']}`",
                f"- Estimated exact K fraction: `{recommended['estimated_k_exact_fraction'] * 100:.2f}%`",
                f"- Composite recovery: `{recommended['recovery']['composite_recovery']:.3f}`",
                f"- Top1 recovery: `{recommended['recovery']['top1_recovery']:.3f}`",
                f"- KL recovery: `{recommended['recovery']['kl_recovery']:.3f}`",
                f"- Summary: top1 `{recommended['summary']['score_top1_match_rate']:.6f}`, top8 `{recommended['summary']['score_topk_overlap_mean']:.6f}`, rank `{recommended['summary']['score_rank_corr_mean']:.6f}`, KL `{recommended['summary']['attn_kl_mean']:.6f}`",
                f"- Benchmark command: `{recommended['benchmark_command']}`",
            ]
        )
    if result.get("validated_candidate_policies"):
        lines.extend(
            [
                "",
                "## Validated Candidates",
                "| Policy | % K Exact | Agreement | Decode ms/step | Status |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for policy in result["validated_candidate_policies"]:
            validation = policy["validation"]
            agreement = validation["agreement"]
            decode_ms = validation["decode_ms_per_step"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(policy["label"]),
                        f"{policy['estimated_k_exact_fraction'] * 100:.2f}%",
                        "-" if agreement is None else f"{float(agreement):.3f}",
                        "-" if decode_ms is None else f"{float(decode_ms):.2f}",
                        str(validation["status"]),
                    ]
                )
                + " |"
            )
    validated_recommended = result.get("validated_recommended_policy")
    if validated_recommended is not None:
        lines.extend(
            [
                "",
                "## Validated Recommendation",
                f"- Overrides: `{validated_recommended['label']}`",
                f"- Estimated exact K fraction: `{validated_recommended['estimated_k_exact_fraction'] * 100:.2f}%`",
                f"- Agreement: `{validated_recommended['validation']['agreement']:.3f}`",
                f"- Decode ms/step: `{validated_recommended['validation']['decode_ms_per_step']:.2f}`",
                f"- Benchmark command: `{validated_recommended['validation']['command']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Candidate Policies",
            "| Policy | % K Exact | Composite Recovery | Top1 Recovery | KL Recovery | Top1 | Top8 | Rank | KL |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for policy in result["candidate_policies"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(policy["label"]),
                    f"{policy['estimated_k_exact_fraction'] * 100:.2f}%",
                    f"{policy['recovery']['composite_recovery']:.3f}",
                    f"{policy['recovery']['top1_recovery']:.3f}",
                    f"{policy['recovery']['kl_recovery']:.3f}",
                    f"{policy['summary']['score_top1_match_rate']:.6f}",
                    f"{policy['summary']['score_topk_overlap_mean']:.6f}",
                    f"{policy['summary']['score_rank_corr_mean']:.6f}",
                    f"{policy['summary']['attn_kl_mean']:.6f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = build_policy_suggestions(args)
    if args.format == "markdown":
        print(render_markdown(result))
        return
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
