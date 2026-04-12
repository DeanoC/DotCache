from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_exact_key_frontier import _run_bias_case
from benchmarks.bench_qwen35_persistent_real_mixed_probe import (
    real_mixed_probe_dotcache_config,
    real_mixed_probe_serving_config,
)
from benchmarks.bench_qwen35_persistent_serving_policy_compare import (
    _DEFAULT_POLICY_PATH,
    _resolve_prompt_records,
)
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheModelAdapter,
    load_qwen35_text_only_from_pretrained,
    transformers_available,
)


_DEFAULT_MANIFESTS = (
    Path(__file__).resolve().parents[1] / "benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_external_20260412.json",
    Path(__file__).resolve().parents[1] / "benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_broad_20260412.json",
    Path(__file__).resolve().parents[1] / "benchmarks/manifests/qwen35_real_mixed_repo_promptfiles_large_20260412.json",
    Path(__file__).resolve().parents[1] / "benchmarks/manifests/qwen35_stage9_repo_public_validation_20260412.json",
)


def _policy_baseline(_: dict[str, Any]) -> dict[int, float] | None:
    return None


def _policy_layer15_always_024(_: dict[str, Any]) -> dict[int, float] | None:
    return {15: 0.24}


def _policy_layer15_len_ge_1800_024(record: dict[str, Any]) -> dict[int, float] | None:
    return {15: 0.24} if int(record.get("prompt_length", 0)) >= 1800 else None


def _policy_layer15_code_or_len_ge_1800_024(record: dict[str, Any]) -> dict[int, float] | None:
    path = Path(str(record.get("prompt_file_path", "")))
    if path.suffix == ".py" or int(record.get("prompt_length", 0)) >= 1800:
        return {15: 0.24}
    return None


POLICIES: tuple[tuple[str, str, Callable[[dict[str, Any]], dict[int, float] | None]], ...] = (
    ("baseline", "Current global policy with no layer-15 override.", _policy_baseline),
    ("layer15_always_024", "Always set layer 15 to 0.24.", _policy_layer15_always_024),
    (
        "layer15_len_ge_1800_024",
        "Use layer 15 -> 0.24 only when prompt length is at least 1800 tokens.",
        _policy_layer15_len_ge_1800_024,
    ),
    (
        "layer15_code_or_len_ge_1800_024",
        "Use layer 15 -> 0.24 for code files or prompts of at least 1800 tokens.",
        _policy_layer15_code_or_len_ge_1800_024,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare simple cost-aware exact-key live policies on the real mixed Stage 9 lane.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="torch_mps")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--manifest-paths", nargs="*", default=[str(path) for path in _DEFAULT_MANIFESTS])
    parser.add_argument("--warmup-runs-per-manifest", type=int, default=1)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def _run_policy_records(
    *,
    manifest_path: Path,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    decode_steps: int,
    chooser: Callable[[dict[str, Any]], dict[int, float] | None],
) -> list[dict[str, Any]]:
    prompt_records = _resolve_prompt_records(
        manifest_path=str(manifest_path),
        prompt_files=[],
        prompt_file_target_length=0,
    )
    rows: list[dict[str, Any]] = []
    for prompt_record in prompt_records:
        override = chooser(prompt_record)
        result = _run_bias_case(
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            prompt_record=prompt_record,
            decode_steps=int(decode_steps),
            max_k_comp_error_by_layer=override,
        )
        rows.append(
            {
                "manifest_path": str(manifest_path),
                "case_tag": str(prompt_record["case_tag"]),
                "prompt_file_path": str(prompt_record["prompt_file_path"]),
                "prompt_length": int(prompt_record.get("prompt_length", 0)),
                "max_k_comp_error_by_layer": dict(override or {}),
                "decode_ms_per_step": float(result["decode_ms_per_step"]),
                "generated_ids": list(result["generated_ids"]),
                "executed_exact_key_m3_by_layer": dict(result["executed_exact_key_m3_by_layer"]),
            }
        )
    return rows


def _summarize_policy(
    *,
    name: str,
    description: str,
    rows: list[dict[str, Any]],
    baseline_by_manifest_case: dict[tuple[str, str], list[int]],
) -> dict[str, Any]:
    per_manifest: dict[str, list[float]] = {}
    exact_matches = 0
    for row in rows:
        per_manifest.setdefault(str(row["manifest_path"]), []).append(float(row["decode_ms_per_step"]))
        key = (str(row["manifest_path"]), str(row["case_tag"]))
        exact_matches += int(list(row["generated_ids"]) == list(baseline_by_manifest_case[key]))
    return {
        "name": name,
        "description": description,
        "overall_avg_ms_per_step": float(sum(float(row["decode_ms_per_step"]) for row in rows) / len(rows)),
        "exact_match_rate_vs_baseline": float(exact_matches / len(rows)),
        "per_manifest_avg_ms_per_step": {
            manifest_path: float(sum(values) / len(values)) for manifest_path, values in sorted(per_manifest.items())
        },
        "rows": rows,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5 Exact-Key Live Policy Compare",
        "",
        "This compares simple layer-15 policies in the live real-mixed Stage 9 runtime.",
        "",
        "## Ranked policies",
        "",
    ]
    for policy in payload["policies"]:
        lines.extend(
            [
                f"- `{policy['name']}`:",
                f"  - description: {policy['description']}",
                f"  - overall avg ms/step: {float(policy['overall_avg_ms_per_step']):.4f}",
                f"  - exact-match vs baseline: {float(policy['exact_match_rate_vs_baseline']):.3f}",
                f"  - per-manifest avg ms/step: {json.dumps(policy['per_manifest_avg_ms_per_step'], sort_keys=True)}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise RuntimeError("transformers is required for the exact-key live policy compare")

    model, tokenizer = load_qwen35_text_only_from_pretrained(
        str(args.model_id),
        device=str(args.device) if args.device else None,
        torch_dtype=str(args.torch_dtype),
        weight_quantization=str(args.weight_quantization),
    )
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=real_mixed_probe_dotcache_config(),
        persistent_serving_config=real_mixed_probe_serving_config(policy_path=str(_DEFAULT_POLICY_PATH)),
        backend=str(args.backend),
    )

    manifest_paths = [Path(value).resolve() for value in list(args.manifest_paths)]
    warmup_runs = max(int(args.warmup_runs_per_manifest), 0)
    for manifest_path in manifest_paths:
        prompt_records = _resolve_prompt_records(
            manifest_path=str(manifest_path),
            prompt_files=[],
            prompt_file_target_length=0,
        )
        for _ in range(warmup_runs):
            for prompt_record in prompt_records:
                _run_bias_case(
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    prompt_record=prompt_record,
                    decode_steps=int(args.decode_steps),
                    max_k_comp_error_by_layer=None,
                )

    policy_summaries: list[dict[str, Any]] = []
    baseline_by_manifest_case: dict[tuple[str, str], list[int]] = {}
    baseline_rows: list[dict[str, Any]] | None = None

    for name, description, chooser in POLICIES:
        rows: list[dict[str, Any]] = []
        for manifest_path in manifest_paths:
            rows.extend(
                _run_policy_records(
                    manifest_path=manifest_path,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    decode_steps=int(args.decode_steps),
                    chooser=chooser,
                )
            )
        if baseline_rows is None:
            baseline_rows = rows
            baseline_by_manifest_case = {
                (str(row["manifest_path"]), str(row["case_tag"])): list(row["generated_ids"]) for row in rows
            }
        policy_summaries.append(
            _summarize_policy(
                name=name,
                description=description,
                rows=rows,
                baseline_by_manifest_case=baseline_by_manifest_case,
            )
        )

    payload = {
        "manifest_paths": [str(path) for path in manifest_paths],
        "policies": sorted(policy_summaries, key=lambda record: float(record["overall_avg_ms_per_step"])),
    }
    rendered = json.dumps(payload, indent=2)
    if args.output_json:
        output_json_path = Path(str(args.output_json))
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(rendered, encoding="utf-8")
    if args.output_md:
        output_md_path = Path(str(args.output_md))
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
