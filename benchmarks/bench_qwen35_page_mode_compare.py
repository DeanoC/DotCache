from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_BENCH_CONTEXTS = (2048, 8192)
DEFAULT_WORKLOAD_CONTEXTS = (2048,)
DEFAULT_PAGED_BUDGETS = (32, 128)
DEFAULT_PAGE_MODES = ("exact", "M3/affine/4/int8")
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "f16"
DEFAULT_PROMPT = "Cache locality matters for fast decoding."
DEFAULT_TOTAL_SESSIONS = 4
DEFAULT_WAVE_SIZE = 2
DEFAULT_DECODE_ROUNDS_PER_WAVE = 1
DEFAULT_MAX_NEW_TOKENS = 4
DEFAULT_WARMUP_RUNS = 0
DEFAULT_FAMILY = "qwen35"
DEFAULT_RUNTIME = "paged_control"
DEFAULT_ATTENTION_PATH = "paged"
DEFAULT_CARGO_FEATURES = "candle-cuda"


@dataclass(frozen=True)
class WorkloadShape:
    total_sessions: int
    wave_size: int
    decode_rounds_per_wave: int
    max_new_tokens: int


@dataclass(frozen=True)
class PageModeVariant:
    label: str
    key_page_mode: str
    value_page_mode: str


@dataclass(frozen=True)
class RunSpec:
    model_id: str
    kind: str
    prompt_token_count: int
    resident_page_budget: int
    dtype: str
    device: str
    warmup_runs: int
    max_new_tokens: int
    prompt_text: str
    page_mode_variant: PageModeVariant
    workload_shape: WorkloadShape | None = None

    def run_slug(self) -> str:
        model_slug = _slugify(self.model_id.split("/")[-1])
        parts = [
            model_slug,
            self.kind,
            f"ctx{self.prompt_token_count}",
            f"budget{self.resident_page_budget}",
            self.page_mode_variant.label,
            self.dtype,
        ]
        if self.workload_shape is not None:
            parts.extend(
                [
                    f"sessions{self.workload_shape.total_sessions}",
                    f"wave{self.workload_shape.wave_size}",
                    f"rounds{self.workload_shape.decode_rounds_per_wave}",
                    f"decode{self.workload_shape.max_new_tokens}",
                ]
            )
        return "-".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact-vs-page-mode Qwen3.5 paged benchmark comparisons."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--bench-contexts", nargs="*", type=int, default=list(DEFAULT_BENCH_CONTEXTS))
    parser.add_argument(
        "--workload-contexts", nargs="*", type=int, default=list(DEFAULT_WORKLOAD_CONTEXTS)
    )
    parser.add_argument("--paged-budgets", nargs="*", type=int, default=list(DEFAULT_PAGED_BUDGETS))
    parser.add_argument("--page-modes", nargs="*", default=list(DEFAULT_PAGE_MODES))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--total-sessions", type=int, default=DEFAULT_TOTAL_SESSIONS)
    parser.add_argument("--wave-size", type=int, default=DEFAULT_WAVE_SIZE)
    parser.add_argument("--decode-rounds-per-wave", type=int, default=DEFAULT_DECODE_ROUNDS_PER_WAVE)
    parser.add_argument("--cargo-features", default=DEFAULT_CARGO_FEATURES)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-tag", default=datetime.now(UTC).strftime("%Y%m%d"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _runtime_dir() -> Path:
    return _repo_root() / "rust" / "paged-runtime"


def _default_output_root() -> Path:
    return _repo_root() / "benchmarks" / "results"


def _slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_page_mode_variant(raw: str) -> PageModeVariant:
    normalized = raw.strip()
    if not normalized:
        raise ValueError("page mode variant cannot be empty")
    if normalized == "exact":
        return PageModeVariant(label="exact", key_page_mode="exact", value_page_mode="exact")
    if "=" in normalized and "|" in normalized:
        label, specs = normalized.split("=", 1)
        key_mode, value_mode = specs.split("|", 1)
        return PageModeVariant(
            label=_slugify(label),
            key_page_mode=key_mode,
            value_page_mode=value_mode,
        )
    return PageModeVariant(
        label=_slugify(normalized),
        key_page_mode=normalized,
        value_page_mode=normalized,
    )


def build_matrix(
    *,
    model_id: str = DEFAULT_MODEL,
    dtype: str = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    prompt_text: str = DEFAULT_PROMPT,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    bench_contexts: tuple[int, ...] = DEFAULT_BENCH_CONTEXTS,
    workload_contexts: tuple[int, ...] = DEFAULT_WORKLOAD_CONTEXTS,
    paged_budgets: tuple[int, ...] = DEFAULT_PAGED_BUDGETS,
    page_modes: tuple[str, ...] = DEFAULT_PAGE_MODES,
    workload_shape: WorkloadShape = WorkloadShape(
        total_sessions=DEFAULT_TOTAL_SESSIONS,
        wave_size=DEFAULT_WAVE_SIZE,
        decode_rounds_per_wave=DEFAULT_DECODE_ROUNDS_PER_WAVE,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    ),
) -> list[RunSpec]:
    variants = [parse_page_mode_variant(raw) for raw in page_modes]
    specs: list[RunSpec] = []
    for context in bench_contexts:
        for budget in paged_budgets:
            for variant in variants:
                specs.append(
                    RunSpec(
                        model_id=model_id,
                        kind="bench",
                        prompt_token_count=context,
                        resident_page_budget=budget,
                        dtype=dtype,
                        device=device,
                        warmup_runs=warmup_runs,
                        max_new_tokens=max_new_tokens,
                        prompt_text=prompt_text,
                        page_mode_variant=variant,
                    )
                )
    for context in workload_contexts:
        for budget in paged_budgets:
            for variant in variants:
                specs.append(
                    RunSpec(
                        model_id=model_id,
                        kind="workload",
                        prompt_token_count=context,
                        resident_page_budget=budget,
                        dtype=dtype,
                        device=device,
                        warmup_runs=warmup_runs,
                        max_new_tokens=max_new_tokens,
                        prompt_text=prompt_text,
                        page_mode_variant=variant,
                        workload_shape=workload_shape,
                    )
                )
    return specs


def command_for_run(spec: RunSpec, *, out_prefix: Path, cargo_features: str) -> list[str]:
    example = "hf_bench" if spec.kind == "bench" else "hf_workload_bench"
    command = [
        "cargo",
        "run",
        "--features",
        cargo_features,
        "--example",
        example,
        "--",
        DEFAULT_FAMILY,
        spec.model_id,
        spec.prompt_text,
        str(out_prefix),
        "--device",
        spec.device,
        "--dtype",
        spec.dtype,
        "--runtime-mode",
        DEFAULT_RUNTIME,
        "--attention-path",
        DEFAULT_ATTENTION_PATH,
        "--resident-page-budget",
        str(spec.resident_page_budget),
        "--warmup-runs",
        str(spec.warmup_runs),
        "--max-new-tokens",
        str(spec.max_new_tokens),
        "--default-key-page-mode",
        spec.page_mode_variant.key_page_mode,
        "--default-value-page-mode",
        spec.page_mode_variant.value_page_mode,
        "--sync-stage-profile",
    ]
    if spec.kind == "bench":
        command.extend(["--prompt-token-target", str(spec.prompt_token_count)])
    else:
        assert spec.workload_shape is not None
        command.extend(
            [
                "--shared-prompt-token-target",
                str(spec.prompt_token_count),
                "--total-sessions",
                str(spec.workload_shape.total_sessions),
                "--wave-size",
                str(spec.workload_shape.wave_size),
                "--decode-rounds-per-wave",
                str(spec.workload_shape.decode_rounds_per_wave),
            ]
        )
    return command


def _summary_paths(out_prefix: Path) -> tuple[Path, Path]:
    return out_prefix.with_suffix(".summary.json"), out_prefix.with_suffix(".trace.jsonl")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_config_signature(spec: RunSpec, *, cargo_features: str) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "kind": spec.kind,
        "prompt_token_count": spec.prompt_token_count,
        "resident_page_budget": spec.resident_page_budget,
        "dtype": spec.dtype,
        "device": spec.device,
        "warmup_runs": spec.warmup_runs,
        "max_new_tokens": spec.max_new_tokens,
        "prompt_text_sha256": hashlib.sha256(spec.prompt_text.encode("utf-8")).hexdigest(),
        "page_mode_variant": asdict(spec.page_mode_variant),
        "workload_shape": asdict(spec.workload_shape) if spec.workload_shape is not None else None,
        "cargo_features": cargo_features,
    }


def _can_resume_existing_run(config_path: Path, expected_signature: dict[str, Any]) -> bool:
    if not config_path.exists():
        return False
    try:
        return _load_json(config_path) == expected_signature
    except (OSError, json.JSONDecodeError):
        return False


def _extract_summary_metrics(summary: dict[str, Any], *, kind: str) -> dict[str, Any]:
    prompt_key = "prompt_token_count" if kind == "bench" else "shared_prompt_token_count"
    prompt_target_key = "prompt_token_target" if kind == "bench" else "shared_prompt_token_target"
    return {
        "prompt_token_count": summary.get(prompt_key),
        "prompt_token_target": summary.get(prompt_target_key),
        "total_millis": summary.get("total_millis"),
        "total_tokens_per_second": summary.get("total_tokens_per_second"),
        "prefill_millis": summary.get(
            "prefill_millis",
            (
                (summary.get("cold_prefix_prefill_millis") or 0.0)
                + (summary.get("seed_suffix_prefill_millis") or 0.0)
                + (summary.get("attached_suffix_prefill_millis") or 0.0)
            ),
        ),
        "decode_millis": summary.get("decode_millis"),
        "spilled_bytes": summary.get("spilled_bytes"),
        "restored_bytes": summary.get("restored_bytes"),
        "stage_page_spill_millis": summary.get("stage_page_spill_millis"),
        "stage_page_restore_millis": summary.get("stage_page_restore_millis"),
    }


def execute_run(
    spec: RunSpec,
    *,
    output_dir: Path,
    cargo_features: str,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    run_dir = output_dir / spec.run_slug()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = run_dir / "run"
    summary_path, trace_path = _summary_paths(out_prefix)
    config_path = run_dir / "run_config.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command = command_for_run(spec, out_prefix=out_prefix, cargo_features=cargo_features)
    signature = _run_config_signature(spec, cargo_features=cargo_features)
    record: dict[str, Any] = {
        "run_id": spec.run_slug(),
        "model_id": spec.model_id,
        "kind": spec.kind,
        "prompt_token_count": spec.prompt_token_count,
        "resident_page_budget": spec.resident_page_budget,
        "dtype": spec.dtype,
        "device": spec.device,
        "runtime_mode": DEFAULT_RUNTIME,
        "attention_path": DEFAULT_ATTENTION_PATH,
        "warmup_runs": spec.warmup_runs,
        "max_new_tokens": spec.max_new_tokens,
        "page_mode_label": spec.page_mode_variant.label,
        "default_key_page_mode": spec.page_mode_variant.key_page_mode,
        "default_value_page_mode": spec.page_mode_variant.value_page_mode,
        "workload_shape": asdict(spec.workload_shape) if spec.workload_shape is not None else None,
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "summary_path": str(summary_path),
        "trace_jsonl_path": str(trace_path),
        "run_config_path": str(config_path),
        "started_at": _utc_now(),
    }
    if dry_run:
        print(f"[planned] {record['run_id']}", flush=True)
        record["status"] = "planned"
        record["exit_status"] = None
        record["completed_at"] = _utc_now()
        return record

    if resume and summary_path.exists() and _can_resume_existing_run(config_path, signature):
        print(f"[resume] {record['run_id']}", flush=True)
        summary = _load_json(summary_path)
        record["status"] = "reused_existing"
        record["exit_status"] = 0
        record["summary_metrics"] = _extract_summary_metrics(summary, kind=spec.kind)
        record["completed_at"] = _utc_now()
        return record

    print(f"[run] {record['run_id']}", flush=True)
    _write_json(config_path, signature)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=_runtime_dir(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )

    record["exit_status"] = completed.returncode
    record["completed_at"] = _utc_now()
    if completed.returncode == 0 and summary_path.exists():
        print(f"[done] {record['run_id']}", flush=True)
        summary = _load_json(summary_path)
        record["status"] = "completed"
        record["summary_metrics"] = _extract_summary_metrics(summary, kind=spec.kind)
        return record
    print(f"[failed] {record['run_id']}", flush=True)
    record["status"] = "failed"
    stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
    stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    record["error_message"] = (stderr_text or stdout_text).strip()[-4000:]
    return record


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        if record["status"] not in {"completed", "reused_existing", "failed"}:
            continue
        shape = record.get("workload_shape")
        shape_key = None if shape is None else (
            shape["total_sessions"],
            shape["wave_size"],
            shape["decode_rounds_per_wave"],
            shape["max_new_tokens"],
        )
        key = (
            record["model_id"],
            record["kind"],
            record["prompt_token_count"],
            record["resident_page_budget"],
            shape_key,
        )
        grouped.setdefault(key, []).append(record)

    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        model_id, kind, prompt_token_count, budget, shape_key = key
        entries = grouped[key]
        exact = next((entry for entry in entries if entry["page_mode_label"] == "exact"), None)
        exact_metrics = exact.get("summary_metrics") if exact is not None else None
        variants = [entry for entry in entries if entry["page_mode_label"] != "exact"]
        rows: list[dict[str, Any]] = []
        for entry in sorted(variants, key=lambda item: item["page_mode_label"]):
            row = {
                "run_id": entry["run_id"],
                "page_mode_label": entry["page_mode_label"],
                "default_key_page_mode": entry["default_key_page_mode"],
                "default_value_page_mode": entry["default_value_page_mode"],
                "status": entry["status"],
                "total_millis": None,
                "total_tokens_per_second": None,
                "spilled_bytes": None,
                "restored_bytes": None,
                "delta_total_millis_vs_exact": None,
                "total_millis_ratio_vs_exact": None,
                "delta_total_tokens_per_second_vs_exact": None,
                "total_tokens_per_second_ratio_vs_exact": None,
                "delta_spilled_bytes_vs_exact": None,
                "spilled_bytes_ratio_vs_exact": None,
            }
            metrics = entry.get("summary_metrics")
            if metrics is not None:
                row["total_millis"] = metrics.get("total_millis")
                row["total_tokens_per_second"] = metrics.get("total_tokens_per_second")
                row["spilled_bytes"] = metrics.get("spilled_bytes")
                row["restored_bytes"] = metrics.get("restored_bytes")
            if exact_metrics is not None and metrics is not None:
                exact_total = exact_metrics.get("total_millis")
                exact_tps = exact_metrics.get("total_tokens_per_second")
                exact_spilled = exact_metrics.get("spilled_bytes")
                total = metrics.get("total_millis")
                tps = metrics.get("total_tokens_per_second")
                spilled = metrics.get("spilled_bytes")
                if exact_total not in (None, 0) and total is not None:
                    row["delta_total_millis_vs_exact"] = total - exact_total
                    row["total_millis_ratio_vs_exact"] = total / exact_total
                if exact_tps not in (None, 0) and tps is not None:
                    row["delta_total_tokens_per_second_vs_exact"] = tps - exact_tps
                    row["total_tokens_per_second_ratio_vs_exact"] = tps / exact_tps
                if exact_spilled not in (None, 0) and spilled is not None:
                    row["delta_spilled_bytes_vs_exact"] = spilled - exact_spilled
                    row["spilled_bytes_ratio_vs_exact"] = spilled / exact_spilled
            rows.append(row)

        successful = [row for row in rows if row["total_millis"] is not None]
        groups.append(
            {
                "model_id": model_id,
                "kind": kind,
                "prompt_token_count": prompt_token_count,
                "resident_page_budget": budget,
                "workload_shape": None
                if shape_key is None
                else {
                    "total_sessions": shape_key[0],
                    "wave_size": shape_key[1],
                    "decode_rounds_per_wave": shape_key[2],
                    "max_new_tokens": shape_key[3],
                },
                "exact_baseline": None
                if exact is None
                else {
                    "run_id": exact["run_id"],
                    "status": exact["status"],
                    "summary_metrics": exact_metrics,
                },
                "variants": rows,
                "best_variant_by_total_millis": min(successful, key=lambda row: row["total_millis"])
                if successful
                else None,
                "best_variant_by_total_tokens_per_second": max(
                    [row for row in successful if row["total_tokens_per_second"] is not None],
                    key=lambda row: row["total_tokens_per_second"],
                )
                if successful
                else None,
                "best_variant_by_spilled_bytes": min(
                    [row for row in successful if row["spilled_bytes"] is not None],
                    key=lambda row: row["spilled_bytes"],
                )
                if successful
                else None,
            }
        )
    return {"generated_at": _utc_now(), "group_count": len(groups), "groups": groups}


def _fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def render_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5 Page Mode Compare",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
    ]
    for group in report["groups"]:
        title = (
            f"{group['model_id']} | {group['kind']} | ctx={group['prompt_token_count']} "
            f"| budget={group['resident_page_budget']}"
        )
        if group["workload_shape"] is not None:
            shape = group["workload_shape"]
            title += (
                f" | sessions={shape['total_sessions']}"
                f" wave={shape['wave_size']}"
                f" rounds={shape['decode_rounds_per_wave']}"
                f" decode={shape['max_new_tokens']}"
            )
        lines.extend([f"## {title}", ""])
        lines.append("| Variant | Status | Total ms | Tok/s | Spilled bytes | Delta ms vs exact | Ratio vs exact | Spill ratio vs exact |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        baseline = group["exact_baseline"]
        if baseline is None:
            lines.append("| exact | missing | - | - | - | - | - | - |")
        else:
            metrics = baseline["summary_metrics"] or {}
            lines.append(
                "| exact | {status} | {total} | {tps} | {spilled} | - | - | - |".format(
                    status=baseline["status"],
                    total=_fmt_float(metrics.get("total_millis")),
                    tps=_fmt_float(metrics.get("total_tokens_per_second")),
                    spilled=metrics.get("spilled_bytes", "-"),
                )
            )
        for row in group["variants"]:
            lines.append(
                "| {label} | {status} | {total} | {tps} | {spilled} | {delta} | {ratio} | {spill_ratio} |".format(
                    label=row["page_mode_label"],
                    status=row["status"],
                    total=_fmt_float(row["total_millis"]),
                    tps=_fmt_float(row["total_tokens_per_second"]),
                    spilled=row["spilled_bytes"] if row["spilled_bytes"] is not None else "-",
                    delta=_fmt_float(row["delta_total_millis_vs_exact"]),
                    ratio=_fmt_float(row["total_millis_ratio_vs_exact"]),
                    spill_ratio=_fmt_float(row["spilled_bytes_ratio_vs_exact"]),
                )
            )
        lines.append("")
    return "\n".join(lines)


def persist_outputs(output_dir: Path, manifest: dict[str, Any]) -> None:
    report = build_report(manifest["records"])
    report["output_dir"] = str(output_dir)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(render_report_markdown(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root is not None else _default_output_root()
    output_dir = output_root / f"qwen35_page_mode_compare_{args.run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    workload_shape = WorkloadShape(
        total_sessions=args.total_sessions,
        wave_size=args.wave_size,
        decode_rounds_per_wave=args.decode_rounds_per_wave,
        max_new_tokens=args.max_new_tokens,
    )
    specs = build_matrix(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        prompt_text=args.prompt_text,
        warmup_runs=args.warmup_runs,
        max_new_tokens=args.max_new_tokens,
        bench_contexts=tuple(args.bench_contexts),
        workload_contexts=tuple(args.workload_contexts),
        paged_budgets=tuple(args.paged_budgets),
        page_modes=tuple(args.page_modes),
        workload_shape=workload_shape,
    )
    manifest: dict[str, Any] = {
        "generated_at": _utc_now(),
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "dtype": args.dtype,
        "device": args.device,
        "bench_contexts": list(args.bench_contexts),
        "workload_contexts": list(args.workload_contexts),
        "paged_budgets": list(args.paged_budgets),
        "page_modes": list(args.page_modes),
        "workload_shape": asdict(workload_shape),
        "cargo_features": args.cargo_features,
        "records": [],
    }
    persist_outputs(output_dir, manifest)
    for spec in specs:
        record = execute_run(
            spec,
            output_dir=output_dir,
            cargo_features=args.cargo_features,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        manifest["records"].append(record)
        persist_outputs(output_dir, manifest)


if __name__ == "__main__":
    main()
