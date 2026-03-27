#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from typing import Any

import suggest_selective_k_policy as suggest


@dataclass(frozen=True)
class CaseSpec:
    family: str
    model_id: str
    prompt_length: int


PRESET_SPECS: dict[str, list[CaseSpec]] = {
    "public-cuda-2048": [
        CaseSpec("qwen2", "Qwen/Qwen2.5-7B-Instruct", 2048),
        CaseSpec("qwen2", "Qwen/Qwen2.5-3B-Instruct", 2048),
        CaseSpec("qwen2", "Qwen/Qwen2.5-1.5B-Instruct", 2048),
        CaseSpec("llama", "HuggingFaceTB/SmolLM2-1.7B-Instruct", 2048),
        CaseSpec("llama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2048),
        CaseSpec("llama", "HuggingFaceTB/SmolLM2-360M-Instruct", 2048),
    ]
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run validated selective-K policy search across a public model set and emit a compact "
            "compressibility map."
        )
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_SPECS),
        default="public-cuda-2048",
        help="Built-in batch of model/prompt cases to evaluate.",
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=None,
        help="Optional case spec in the form family|model_id|prompt_length. Replaces --preset when provided.",
    )
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="torch_cuda")
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--candidate-layers", type=int, default=6)
    parser.add_argument("--max-kv-groups-per-layer", type=int, default=2)
    parser.add_argument("--max-combo-size", type=int, default=2)
    parser.add_argument("--target-recovery", type=float, default=0.65)
    parser.add_argument("--budget-recovery-margin", type=float, default=0.9)
    parser.add_argument("--top-policies", type=int, default=8)
    parser.add_argument("--validate-top-policies", type=int, default=3)
    parser.add_argument("--validation-max-new-tokens", type=int, default=4)
    parser.add_argument("--validation-agreement-threshold", type=float, default=0.99)
    parser.add_argument(
        "--validate-exact-k",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run an end-to-end exact-K benchmark so the report includes KV-vs-exact ratios.",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    return parser.parse_args(argv)


def _parse_spec(raw_spec: str) -> CaseSpec:
    parts = raw_spec.split("|", 2)
    if len(parts) != 3:
        raise ValueError(f"invalid spec {raw_spec!r}; expected family|model_id|prompt_length")
    family, model_id, prompt_length_text = parts
    prompt_length = int(prompt_length_text)
    if family not in {"llama", "qwen2"}:
        raise ValueError(f"unsupported family in spec {raw_spec!r}")
    return CaseSpec(family=family, model_id=model_id, prompt_length=prompt_length)


def _case_specs(args: argparse.Namespace) -> list[CaseSpec]:
    if args.spec:
        return [_parse_spec(raw_spec) for raw_spec in args.spec]
    return list(PRESET_SPECS[args.preset])


def _case_args(args: argparse.Namespace, spec: CaseSpec) -> argparse.Namespace:
    return argparse.Namespace(
        family=spec.family,
        model_id=spec.model_id,
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
        prompt_length=spec.prompt_length,
        candidate_layers=args.candidate_layers,
        max_kv_groups_per_layer=args.max_kv_groups_per_layer,
        max_combo_size=args.max_combo_size,
        target_recovery=args.target_recovery,
        budget_recovery_margin=args.budget_recovery_margin,
        top_policies=args.top_policies,
        validate_top_policies=args.validate_top_policies,
        validation_max_new_tokens=args.validation_max_new_tokens,
        validation_agreement_threshold=args.validation_agreement_threshold,
        format="json",
    )


def _model_label(model_id: str) -> str:
    return model_id.split("/", 1)[-1]


def _cleanup_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            pass


def _tok_per_sec(decode_ms_per_step: float | None) -> float | None:
    if decode_ms_per_step in (None, 0.0):
        return None
    return float(1000.0 / decode_ms_per_step)


def _fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def _kv_vs_exact(best_validation: dict[str, Any] | None, exact_validation: dict[str, Any] | None) -> float | None:
    if best_validation is None or exact_validation is None:
        return None
    best_kv = best_validation.get("kv_vs_dense")
    exact_kv = exact_validation.get("kv_vs_dense")
    if best_kv is None or exact_kv in (None, 0.0):
        return None
    return float(best_kv / exact_kv)


def _policy_display_name(row: dict[str, Any]) -> str:
    label = row.get("label")
    if label == "all_m0":
        return "all M0"
    if label == "exact_k":
        return "exact K"
    return str(label)


def _successful(validation: dict[str, Any] | None, threshold: float) -> bool:
    if validation is None or validation.get("status") == "error":
        return False
    agreement = validation.get("agreement")
    if agreement is None:
        return False
    return float(agreement) >= float(threshold)


def _classify_case(
    *,
    threshold: float,
    baseline_validation: dict[str, Any] | None,
    recommended_validation: dict[str, Any] | None,
    exact_validation: dict[str, Any] | None,
) -> tuple[str, str]:
    if _successful(baseline_validation, threshold):
        return "tolerates all-M0", "all_m0"
    if _successful(recommended_validation, threshold):
        return "benefits from selective exact K", "validated_selective"
    if _successful(exact_validation, threshold):
        return "needs global exact K", "exact_k"
    return "unknown", "unknown"


def _find_policy_by_label(result: dict[str, Any], label: str) -> dict[str, Any] | None:
    for policy in result.get("candidate_policies") or []:
        if str(policy.get("label")) == label:
            return policy
    return None


def _synthetic_policy(case_args: argparse.Namespace, *, label: str, exact_fraction: float) -> dict[str, Any]:
    return {
        "label": label,
        "estimated_k_exact_fraction": exact_fraction,
        "benchmark_command": suggest._benchmark_command(case_args, label=label, targets=()),
    }


def _preferred_command(selected_policy: dict[str, Any] | None, selected_validation: dict[str, Any] | None) -> str | None:
    policy_command = None if selected_policy is None else selected_policy.get("benchmark_command")
    validation_command = None if selected_validation is None else selected_validation.get("command")
    validation_label = None if selected_validation is None else selected_validation.get("label")
    policy_label = None if selected_policy is None else selected_policy.get("label")
    if policy_command and policy_label is not None and policy_label != validation_label:
        return str(policy_command)
    if validation_command:
        return str(validation_command)
    if policy_command:
        return str(policy_command)
    return None


def build_compressibility_map(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    for spec in _case_specs(args):
        case_args = _case_args(args, spec)
        _cleanup_accelerator_memory()
        try:
            result = suggest.build_policy_suggestions(case_args)
            baseline_validation = suggest._validate_candidate_policy(case_args, label="all_m0", targets=())
            exact_validation = None
            if args.validate_exact_k:
                exact_validation = suggest._validate_candidate_policy(case_args, label="exact_k", targets=())
            validated_recommended = result.get("validated_recommended_policy")
            recommended_validation = None if validated_recommended is None else validated_recommended.get("validation")
            classification, selected_key = _classify_case(
                threshold=args.validation_agreement_threshold,
                baseline_validation=baseline_validation,
                recommended_validation=recommended_validation,
                exact_validation=exact_validation,
            )

            if selected_key == "all_m0":
                selected_policy = _find_policy_by_label(result, "all_m0") or _synthetic_policy(
                    case_args,
                    label="all_m0",
                    exact_fraction=0.0,
                )
                selected_validation = baseline_validation
            elif selected_key == "validated_selective":
                selected_policy = validated_recommended
                selected_validation = recommended_validation
            elif selected_key == "exact_k":
                selected_policy = _find_policy_by_label(result, "exact_k") or _synthetic_policy(
                    case_args,
                    label="exact_k",
                    exact_fraction=1.0,
                )
                selected_validation = exact_validation
            else:
                selected_policy = (
                    validated_recommended
                    or result.get("recommended_policy")
                    or _find_policy_by_label(result, "all_m0")
                    or _synthetic_policy(case_args, label="all_m0", exact_fraction=0.0)
                )
                selected_validation = recommended_validation or baseline_validation or exact_validation

            k_exact_fraction = None
            if selected_validation is not None and selected_validation.get("k_exact_fraction") is not None:
                k_exact_fraction = float(selected_validation["k_exact_fraction"])
            elif selected_policy is not None and selected_policy.get("estimated_k_exact_fraction") is not None:
                k_exact_fraction = float(selected_policy["estimated_k_exact_fraction"])

            row = {
                "family": spec.family,
                "model_id": spec.model_id,
                "model": _model_label(spec.model_id),
                "prompt_length": spec.prompt_length,
                "classification": classification,
                "baseline_agreement": baseline_validation.get("agreement"),
                "baseline_status": baseline_validation.get("status"),
                "selected_policy": None if selected_policy is None else _policy_display_name(selected_policy),
                "selected_policy_label": None if selected_policy is None else selected_policy.get("label"),
                "k_exact_fraction": k_exact_fraction,
                "agreement": None if selected_validation is None else selected_validation.get("agreement"),
                "kv_vs_exact_k": _kv_vs_exact(selected_validation, exact_validation),
                "kv_vs_dense": None if selected_validation is None else selected_validation.get("kv_vs_dense"),
                "decode_ms_per_step": None if selected_validation is None else selected_validation.get("decode_ms_per_step"),
                "decode_tok_per_s": _tok_per_sec(
                    None if selected_validation is None else selected_validation.get("decode_ms_per_step")
                ),
                "status": None if selected_validation is None else selected_validation.get("status"),
                "recommended_command": _preferred_command(selected_policy, selected_validation),
                "error_type": None,
                "error_message": None,
            }
            case_results.append(
                {
                    "spec": {
                        "family": spec.family,
                        "model_id": spec.model_id,
                        "prompt_length": spec.prompt_length,
                    },
                    "classification": classification,
                    "baseline_validation": baseline_validation,
                    "exact_validation": exact_validation,
                    "suggestion_result": result,
                    "selected_policy_row": row,
                }
            )
        except Exception as exc:  # noqa: BLE001
            row = {
                "family": spec.family,
                "model_id": spec.model_id,
                "model": _model_label(spec.model_id),
                "prompt_length": spec.prompt_length,
                "classification": "error",
                "baseline_agreement": None,
                "baseline_status": "error",
                "selected_policy": "-",
                "selected_policy_label": None,
                "k_exact_fraction": None,
                "agreement": None,
                "kv_vs_exact_k": None,
                "kv_vs_dense": None,
                "decode_ms_per_step": None,
                "decode_tok_per_s": None,
                "status": "error",
                "recommended_command": None,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            case_results.append(
                {
                    "spec": {
                        "family": spec.family,
                        "model_id": spec.model_id,
                        "prompt_length": spec.prompt_length,
                    },
                    "classification": "error",
                    "baseline_validation": None,
                    "exact_validation": None,
                    "suggestion_result": None,
                    "selected_policy_row": row,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        finally:
            _cleanup_accelerator_memory()
        rows.append(row)

    return {
        "backend": args.backend,
        "device": args.device,
        "validation_agreement_threshold": args.validation_agreement_threshold,
        "rows": rows,
        "cases": case_results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Validated Compressibility Map ({report['backend']})",
        "",
        "| Model | Prompt | Classification | Baseline M0 | Policy | % K Exact | KV vs Exact-K | KV vs Dense | Decode tok/s | Status |",
        "|---|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["prompt_length"]),
                    str(row["classification"]),
                    _fmt_num(row["baseline_agreement"], digits=3),
                    str(row["selected_policy"] or "-"),
                    _fmt_pct(row["k_exact_fraction"]),
                    _fmt_num(row["kv_vs_exact_k"], digits=3) + "x" if row["kv_vs_exact_k"] is not None else "-",
                    _fmt_num(row["kv_vs_dense"], digits=3) + "x" if row["kv_vs_dense"] is not None else "-",
                    _fmt_num(row["decode_tok_per_s"], digits=2),
                    str(row["status"] or "-"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Commands",
            "| Model | Policy | Command |",
            "|---|---|---|",
        ]
    )
    for row in report["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["selected_policy"] or "-"),
                    f"`{row['recommended_command']}`" if row["recommended_command"] else "-",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report = build_compressibility_map(args)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
