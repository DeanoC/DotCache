from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B")
DEFAULT_SINGLE_CONTEXTS = (8192, 32768)
DEFAULT_WORKLOAD_CONTEXTS = (8192,)
DEFAULT_PAGED_BUDGETS = (32, 128, 512)
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "bf16"
DEFAULT_FALLBACK_DTYPE = "f16"
DEFAULT_PROMPT = "Cache locality matters for fast decoding."
DEFAULT_TOTAL_SESSIONS = 4
DEFAULT_WAVE_SIZE = 2
DEFAULT_DECODE_ROUNDS_PER_WAVE = 1
DEFAULT_MAX_NEW_TOKENS = 4
DEFAULT_WARMUP_RUNS = 0
DEFAULT_FAMILY = "qwen35"
DEFAULT_RUNTIME_DENSE = "dense_control"
DEFAULT_RUNTIME_PAGED = "paged_control"
DEFAULT_ATTENTION_PATH = "fused"


@dataclass(frozen=True)
class WorkloadShape:
    total_sessions: int
    wave_size: int
    decode_rounds_per_wave: int
    max_new_tokens: int


@dataclass(frozen=True)
class RunSpec:
    model_id: str
    kind: str
    prompt_token_count: int
    runtime_mode: str
    attention_path: str | None
    resident_page_budget: int | None
    dtype: str
    device: str
    warmup_runs: int
    max_new_tokens: int
    prompt_text: str
    workload_shape: WorkloadShape | None = None

    def is_dense(self) -> bool:
        return self.runtime_mode == DEFAULT_RUNTIME_DENSE

    def run_slug(self) -> str:
        model_slug = _slugify(self.model_id.split("/")[-1])
        base = [
            model_slug,
            self.kind,
            f"ctx{self.prompt_token_count}",
            self.runtime_mode,
        ]
        if self.attention_path is not None:
            base.append(self.attention_path)
        if self.resident_page_budget is not None:
            base.append(f"budget{self.resident_page_budget}")
        if self.workload_shape is not None:
            base.extend(
                [
                    f"sessions{self.workload_shape.total_sessions}",
                    f"wave{self.workload_shape.wave_size}",
                    f"rounds{self.workload_shape.decode_rounds_per_wave}",
                    f"decode{self.workload_shape.max_new_tokens}",
                ]
            )
        base.append(self.dtype)
        return "-".join(base)

    def group_key(self) -> tuple[Any, ...]:
        shape_key: tuple[int, int, int, int] | None = None
        if self.workload_shape is not None:
            shape_key = (
                self.workload_shape.total_sessions,
                self.workload_shape.wave_size,
                self.workload_shape.decode_rounds_per_wave,
                self.workload_shape.max_new_tokens,
            )
        return (self.model_id, self.kind, self.prompt_token_count, shape_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Qwen3.5 large-model paged-vs-dense CUDA benchmark matrix."
    )
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--single-contexts", type=int, nargs="*", default=list(DEFAULT_SINGLE_CONTEXTS))
    parser.add_argument("--workload-contexts", type=int, nargs="*", default=list(DEFAULT_WORKLOAD_CONTEXTS))
    parser.add_argument("--paged-budgets", type=int, nargs="*", default=list(DEFAULT_PAGED_BUDGETS))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--fallback-dtype", default=DEFAULT_FALLBACK_DTYPE)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--total-sessions", type=int, default=DEFAULT_TOTAL_SESSIONS)
    parser.add_argument("--wave-size", type=int, default=DEFAULT_WAVE_SIZE)
    parser.add_argument("--decode-rounds-per-wave", type=int, default=DEFAULT_DECODE_ROUNDS_PER_WAVE)
    parser.add_argument("--cargo-features", default="candle-cuda")
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
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def build_model_specs(
    *,
    model_id: str,
    dtype: str,
    device: str,
    prompt_text: str,
    warmup_runs: int,
    max_new_tokens: int,
    single_contexts: tuple[int, ...],
    workload_contexts: tuple[int, ...],
    paged_budgets: tuple[int, ...],
    workload_shape: WorkloadShape,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for context in single_contexts:
        specs.append(
            RunSpec(
                model_id=model_id,
                kind="bench",
                prompt_token_count=context,
                runtime_mode=DEFAULT_RUNTIME_DENSE,
                attention_path=None,
                resident_page_budget=None,
                dtype=dtype,
                device=device,
                warmup_runs=warmup_runs,
                max_new_tokens=max_new_tokens,
                prompt_text=prompt_text,
            )
        )
        for budget in paged_budgets:
            specs.append(
                RunSpec(
                    model_id=model_id,
                    kind="bench",
                    prompt_token_count=context,
                    runtime_mode=DEFAULT_RUNTIME_PAGED,
                    attention_path=DEFAULT_ATTENTION_PATH,
                    resident_page_budget=budget,
                    dtype=dtype,
                    device=device,
                    warmup_runs=warmup_runs,
                    max_new_tokens=max_new_tokens,
                    prompt_text=prompt_text,
                )
            )
    for context in workload_contexts:
        specs.append(
            RunSpec(
                model_id=model_id,
                kind="workload",
                prompt_token_count=context,
                runtime_mode=DEFAULT_RUNTIME_DENSE,
                attention_path=None,
                resident_page_budget=None,
                dtype=dtype,
                device=device,
                warmup_runs=warmup_runs,
                max_new_tokens=max_new_tokens,
                prompt_text=prompt_text,
                workload_shape=workload_shape,
            )
        )
        for budget in paged_budgets:
            specs.append(
                RunSpec(
                    model_id=model_id,
                    kind="workload",
                    prompt_token_count=context,
                    runtime_mode=DEFAULT_RUNTIME_PAGED,
                    attention_path=DEFAULT_ATTENTION_PATH,
                    resident_page_budget=budget,
                    dtype=dtype,
                    device=device,
                    warmup_runs=warmup_runs,
                    max_new_tokens=max_new_tokens,
                    prompt_text=prompt_text,
                    workload_shape=workload_shape,
                )
            )
    return specs


def build_matrix(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    dtype: str = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
    prompt_text: str = DEFAULT_PROMPT,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    single_contexts: tuple[int, ...] = DEFAULT_SINGLE_CONTEXTS,
    workload_contexts: tuple[int, ...] = DEFAULT_WORKLOAD_CONTEXTS,
    paged_budgets: tuple[int, ...] = DEFAULT_PAGED_BUDGETS,
    workload_shape: WorkloadShape = WorkloadShape(
        total_sessions=DEFAULT_TOTAL_SESSIONS,
        wave_size=DEFAULT_WAVE_SIZE,
        decode_rounds_per_wave=DEFAULT_DECODE_ROUNDS_PER_WAVE,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    ),
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for model_id in models:
        specs.extend(
            build_model_specs(
                model_id=model_id,
                dtype=dtype,
                device=device,
                prompt_text=prompt_text,
                warmup_runs=warmup_runs,
                max_new_tokens=max_new_tokens,
                single_contexts=single_contexts,
                workload_contexts=workload_contexts,
                paged_budgets=paged_budgets,
                workload_shape=workload_shape,
            )
        )
    return specs


def command_for_run(
    spec: RunSpec,
    *,
    out_prefix: Path,
    cargo_features: str,
) -> list[str]:
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
        spec.runtime_mode,
        "--warmup-runs",
        str(spec.warmup_runs),
        "--max-new-tokens",
        str(spec.max_new_tokens),
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
    if spec.attention_path is not None:
        command.extend(["--attention-path", spec.attention_path])
    if spec.resident_page_budget is not None:
        command.extend(["--resident-page-budget", str(spec.resident_page_budget)])
    return command


def _summary_paths(out_prefix: Path) -> tuple[Path, Path]:
    return out_prefix.with_suffix(".summary.json"), out_prefix.with_suffix(".trace.jsonl")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _detect_gpu_snapshot() -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 4:
        return {"raw": lines[0]}
    return {
        "name": parts[0],
        "memory_total_mib": int(parts[1]),
        "memory_used_mib": int(parts[2]),
        "memory_free_mib": int(parts[3]),
    }


def _error_suggests_dtype_fallback(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    fallback_markers = (
        "bf16",
        "bfloat16",
        "unsupported dtype",
        "unsupported scalar type",
        "does not support bfloat16",
        "not implemented for 'bfloat16'",
    )
    return any(marker in combined for marker in fallback_markers)


def _error_looks_host_unsupported(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    markers = (
        "out of memory",
        "cuda error: out of memory",
        "cuda out of memory",
        "not enough memory",
        "resource exhausted",
    )
    return any(marker in combined for marker in markers)


def _extract_summary_metrics(summary: dict[str, Any], *, kind: str) -> dict[str, Any]:
    prompt_key = "prompt_token_count" if kind == "bench" else "shared_prompt_token_count"
    prompt_target_key = "prompt_token_target" if kind == "bench" else "shared_prompt_token_target"
    return {
        "prompt_token_count": summary.get(prompt_key),
        "prompt_token_target": summary.get(prompt_target_key),
        "total_millis": summary.get("total_millis"),
        "total_tokens_per_second": summary.get("total_tokens_per_second"),
        "prefill_millis": summary.get("prefill_millis"),
        "decode_millis": summary.get("decode_millis"),
        "stage_page_spill_millis": summary.get("stage_page_spill_millis"),
        "stage_page_restore_millis": summary.get("stage_page_restore_millis"),
        "generated_token_count": summary.get("generated_token_count", summary.get("total_generated_token_count")),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command = command_for_run(spec, out_prefix=out_prefix, cargo_features=cargo_features)
    record: dict[str, Any] = {
        "run_id": spec.run_slug(),
        "model_id": spec.model_id,
        "kind": spec.kind,
        "prompt_token_count": spec.prompt_token_count,
        "runtime_mode": spec.runtime_mode,
        "attention_path": spec.attention_path,
        "resident_page_budget": spec.resident_page_budget,
        "dtype": spec.dtype,
        "device": spec.device,
        "warmup_runs": spec.warmup_runs,
        "max_new_tokens": spec.max_new_tokens,
        "workload_shape": asdict(spec.workload_shape) if spec.workload_shape is not None else None,
        "command": command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "summary_path": str(summary_path),
        "trace_jsonl_path": str(trace_path),
        "started_at": _utc_now(),
    }

    if dry_run:
        print(f"[planned] {record['run_id']}", flush=True)
        record["status"] = "planned"
        record["exit_status"] = None
        record["completed_at"] = _utc_now()
        return record

    if resume and summary_path.exists():
        print(f"[resume] {record['run_id']}", flush=True)
        summary = _load_json(summary_path)
        record["status"] = "reused_existing"
        record["exit_status"] = 0
        record["summary_metrics"] = _extract_summary_metrics(summary, kind=spec.kind)
        record["completed_at"] = _utc_now()
        return record

    print(f"[run] {record['run_id']}", flush=True)
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

    stdout_text = _read_text(stdout_path)
    stderr_text = _read_text(stderr_path)
    record["exit_status"] = completed.returncode
    record["completed_at"] = _utc_now()

    if completed.returncode == 0 and summary_path.exists():
        print(f"[done] {record['run_id']}", flush=True)
        summary = _load_json(summary_path)
        record["status"] = "completed"
        record["summary_metrics"] = _extract_summary_metrics(summary, kind=spec.kind)
        return record

    if completed.returncode == 0:
        print(f"[missing-summary] {record['run_id']}", flush=True)
        record["status"] = "failed_missing_summary"
        record["error_message"] = "command exited successfully but summary file was not written"
        return record

    print(f"[failed] {record['run_id']}", flush=True)
    record["status"] = "unsupported_on_host" if _error_looks_host_unsupported(stdout_text, stderr_text) else "failed"
    record["error_message"] = (stderr_text or stdout_text).strip()[-4000:]
    return record


def resolve_model_dtype(
    *,
    model_id: str,
    requested_dtype: str,
    fallback_dtype: str,
    device: str,
    prompt_text: str,
    warmup_runs: int,
    max_new_tokens: int,
    single_contexts: tuple[int, ...],
    workload_contexts: tuple[int, ...],
    paged_budgets: tuple[int, ...],
    workload_shape: WorkloadShape,
    output_dir: Path,
    cargo_features: str,
    resume: bool,
    dry_run: bool,
) -> tuple[str, list[dict[str, Any]], list[RunSpec]]:
    initial_specs = build_model_specs(
        model_id=model_id,
        dtype=requested_dtype,
        device=device,
        prompt_text=prompt_text,
        warmup_runs=warmup_runs,
        max_new_tokens=max_new_tokens,
        single_contexts=single_contexts,
        workload_contexts=workload_contexts,
        paged_budgets=paged_budgets,
        workload_shape=workload_shape,
    )
    if not initial_specs:
        return requested_dtype, [], []

    probe_record = execute_run(
        initial_specs[0],
        output_dir=output_dir,
        cargo_features=cargo_features,
        resume=resume,
        dry_run=dry_run,
    )
    records = [probe_record]
    if dry_run or probe_record["status"] in {"completed", "reused_existing", "planned"}:
        return requested_dtype, records, initial_specs[1:]

    stdout_text = _read_text(Path(probe_record["stdout_path"]))
    stderr_text = _read_text(Path(probe_record["stderr_path"]))
    if requested_dtype != fallback_dtype and _error_suggests_dtype_fallback(stdout_text, stderr_text):
        fallback_specs = build_model_specs(
            model_id=model_id,
            dtype=fallback_dtype,
            device=device,
            prompt_text=prompt_text,
            warmup_runs=warmup_runs,
            max_new_tokens=max_new_tokens,
            single_contexts=single_contexts,
            workload_contexts=workload_contexts,
            paged_budgets=paged_budgets,
            workload_shape=workload_shape,
        )
        retry_record = execute_run(
            fallback_specs[0],
            output_dir=output_dir,
            cargo_features=cargo_features,
            resume=resume,
            dry_run=dry_run,
        )
        retry_record["dtype_fallback_from"] = requested_dtype
        records.append(retry_record)
        return fallback_dtype, records, fallback_specs[1:]
    return requested_dtype, records, initial_specs[1:]


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        if record["status"] not in {"completed", "reused_existing", "unsupported_on_host", "failed"}:
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
            shape_key,
        )
        grouped.setdefault(key, []).append(record)

    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        model_id, kind, prompt_token_count, shape_key = key
        entries = grouped[key]
        dense = next((entry for entry in entries if entry["runtime_mode"] == DEFAULT_RUNTIME_DENSE), None)
        paged = [entry for entry in entries if entry["runtime_mode"] == DEFAULT_RUNTIME_PAGED]
        dense_metrics = dense.get("summary_metrics") if dense is not None else None

        paged_rows: list[dict[str, Any]] = []
        for entry in sorted(paged, key=lambda item: item.get("resident_page_budget") or 0):
            row = {
                "run_id": entry["run_id"],
                "resident_page_budget": entry.get("resident_page_budget"),
                "dtype": entry["dtype"],
                "status": entry["status"],
                "total_millis": None,
                "total_tokens_per_second": None,
                "delta_total_millis_vs_dense": None,
                "total_millis_ratio_vs_dense": None,
                "delta_total_tokens_per_second_vs_dense": None,
                "total_tokens_per_second_ratio_vs_dense": None,
            }
            metrics = entry.get("summary_metrics")
            if metrics is not None:
                row["total_millis"] = metrics.get("total_millis")
                row["total_tokens_per_second"] = metrics.get("total_tokens_per_second")
            if dense_metrics is not None and metrics is not None:
                dense_total = dense_metrics.get("total_millis")
                dense_tps = dense_metrics.get("total_tokens_per_second")
                total = metrics.get("total_millis")
                tps = metrics.get("total_tokens_per_second")
                if dense_total not in (None, 0) and total is not None:
                    row["delta_total_millis_vs_dense"] = total - dense_total
                    row["total_millis_ratio_vs_dense"] = total / dense_total
                if dense_tps not in (None, 0) and tps is not None:
                    row["delta_total_tokens_per_second_vs_dense"] = tps - dense_tps
                    row["total_tokens_per_second_ratio_vs_dense"] = tps / dense_tps
            paged_rows.append(row)

        successful_paged = [row for row in paged_rows if row["total_millis"] is not None]
        best_latency = min(successful_paged, key=lambda row: row["total_millis"]) if successful_paged else None
        best_throughput = (
            max(
                [row for row in successful_paged if row["total_tokens_per_second"] is not None],
                key=lambda row: row["total_tokens_per_second"],
            )
            if successful_paged
            else None
        )
        groups.append(
            {
                "model_id": model_id,
                "kind": kind,
                "prompt_token_count": prompt_token_count,
                "workload_shape": None
                if shape_key is None
                else {
                    "total_sessions": shape_key[0],
                    "wave_size": shape_key[1],
                    "decode_rounds_per_wave": shape_key[2],
                    "max_new_tokens": shape_key[3],
                },
                "dense_baseline": None
                if dense is None
                else {
                    "run_id": dense["run_id"],
                    "dtype": dense["dtype"],
                    "status": dense["status"],
                    "summary_metrics": dense_metrics,
                },
                "paged_variants": paged_rows,
                "best_paged_by_total_millis": best_latency,
                "best_paged_by_total_tokens_per_second": best_throughput,
            }
        )
    return {
        "generated_at": _utc_now(),
        "group_count": len(groups),
        "groups": groups,
    }


def render_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5 Large-Model Paged vs Dense Matrix",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
    ]
    for group in report["groups"]:
        title = f"{group['model_id']} | {group['kind']} | ctx={group['prompt_token_count']}"
        if group["workload_shape"] is not None:
            shape = group["workload_shape"]
            title += (
                f" | sessions={shape['total_sessions']}"
                f" wave={shape['wave_size']}"
                f" rounds={shape['decode_rounds_per_wave']}"
                f" decode={shape['max_new_tokens']}"
            )
        lines.extend([f"## {title}", ""])
        lines.append("| Variant | Budget | DType | Status | Total ms | Tok/s | Delta ms vs dense | Ratio vs dense |")
        lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |")
        dense = group["dense_baseline"]
        if dense is None:
            lines.append("| dense | - | - | missing | - | - | - | - |")
        else:
            metrics = dense.get("summary_metrics") or {}
            lines.append(
                "| dense | - | {dtype} | {status} | {total} | {tps} | - | - |".format(
                    dtype=dense["dtype"],
                    status=dense["status"],
                    total=_fmt_float(metrics.get("total_millis")),
                    tps=_fmt_float(metrics.get("total_tokens_per_second")),
                )
            )
        for row in group["paged_variants"]:
            lines.append(
                "| paged:fused | {budget} | {dtype} | {status} | {total} | {tps} | {delta} | {ratio} |".format(
                    budget=row["resident_page_budget"],
                    dtype=row["dtype"],
                    status=row["status"],
                    total=_fmt_float(row["total_millis"]),
                    tps=_fmt_float(row["total_tokens_per_second"]),
                    delta=_fmt_float(row["delta_total_millis_vs_dense"]),
                    ratio=_fmt_float(row["total_millis_ratio_vs_dense"]),
                )
            )
        best_latency = group["best_paged_by_total_millis"]
        best_tps = group["best_paged_by_total_tokens_per_second"]
        if best_latency is not None:
            lines.append("")
            lines.append(
                f"Best paged latency budget: `{best_latency['resident_page_budget']}` "
                f"at `{_fmt_float(best_latency['total_millis'])} ms`."
            )
        if best_tps is not None:
            lines.append(
                f"Best paged throughput budget: `{best_tps['resident_page_budget']}` "
                f"at `{_fmt_float(best_tps['total_tokens_per_second'])} tok/s`."
            )
        lines.append("")
    return "\n".join(lines)


def _fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def persist_outputs(output_dir: Path, manifest: dict[str, Any]) -> None:
    report = build_report(manifest["records"])
    report["output_dir"] = str(output_dir)
    report_markdown = render_report_markdown(report)
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "report.json", report)
    (output_dir / "report.md").write_text(report_markdown + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root) if args.output_root is not None else _default_output_root()
    output_dir = output_root / f"qwen35_paged_dense_large_{args.run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    workload_shape = WorkloadShape(
        total_sessions=args.total_sessions,
        wave_size=args.wave_size,
        decode_rounds_per_wave=args.decode_rounds_per_wave,
        max_new_tokens=args.max_new_tokens,
    )
    gpu_snapshot = _detect_gpu_snapshot()
    if gpu_snapshot is not None and gpu_snapshot.get("memory_free_mib", 0) < 80_000:
        print(
            "warning: GPU free memory appears low for the full large-model matrix: "
            f"{gpu_snapshot['memory_free_mib']} MiB free on {gpu_snapshot['name']}",
            file=sys.stderr,
        )

    manifest: dict[str, Any] = {
        "generated_at": _utc_now(),
        "output_dir": str(output_dir),
        "device": args.device,
        "requested_dtype": args.dtype,
        "fallback_dtype": args.fallback_dtype,
        "models": list(args.models),
        "single_contexts": list(args.single_contexts),
        "workload_contexts": list(args.workload_contexts),
        "paged_budgets": list(args.paged_budgets),
        "workload_shape": asdict(workload_shape),
        "cargo_features": args.cargo_features,
        "gpu_snapshot": gpu_snapshot,
        "records": [],
        "resolved_model_dtypes": {},
    }
    persist_outputs(output_dir, manifest)

    try:
        for model_id in args.models:
            print(f"[model] {model_id}", flush=True)
            resolved_dtype, probe_records, remaining_specs = resolve_model_dtype(
                model_id=model_id,
                requested_dtype=args.dtype,
                fallback_dtype=args.fallback_dtype,
                device=args.device,
                prompt_text=args.prompt_text,
                warmup_runs=args.warmup_runs,
                max_new_tokens=args.max_new_tokens,
                single_contexts=tuple(args.single_contexts),
                workload_contexts=tuple(args.workload_contexts),
                paged_budgets=tuple(args.paged_budgets),
                workload_shape=workload_shape,
                output_dir=output_dir,
                cargo_features=args.cargo_features,
                resume=args.resume,
                dry_run=args.dry_run,
            )
            manifest["resolved_model_dtypes"][model_id] = resolved_dtype
            manifest["records"].extend(probe_records)
            persist_outputs(output_dir, manifest)
            for spec in remaining_specs:
                adjusted_spec = RunSpec(
                    model_id=spec.model_id,
                    kind=spec.kind,
                    prompt_token_count=spec.prompt_token_count,
                    runtime_mode=spec.runtime_mode,
                    attention_path=spec.attention_path,
                    resident_page_budget=spec.resident_page_budget,
                    dtype=resolved_dtype,
                    device=spec.device,
                    warmup_runs=spec.warmup_runs,
                    max_new_tokens=spec.max_new_tokens,
                    prompt_text=spec.prompt_text,
                    workload_shape=spec.workload_shape,
                )
                manifest["records"].append(
                    execute_run(
                        adjusted_spec,
                        output_dir=output_dir,
                        cargo_features=args.cargo_features,
                        resume=args.resume,
                        dry_run=args.dry_run,
                    )
                )
                persist_outputs(output_dir, manifest)
    except KeyboardInterrupt:
        persist_outputs(output_dir, manifest)
        raise

    print(str(output_dir))


if __name__ == "__main__":
    main()
