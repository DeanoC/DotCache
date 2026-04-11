from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_CONTEXTS = (2048, 8192)
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "f16"
DEFAULT_PROMPT = "Cache locality matters for fast decoding."
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_WARMUP_RUNS = 0
DEFAULT_FAMILY = "qwen35"
DEFAULT_MAIN_RUNTIME = "dense_control"
DEFAULT_MAIN_CARGO_FEATURES = "candle-cuda"
DEFAULT_MINIMAL_CARGO_FEATURES = "qwen35-minimal-cuda"
DEFAULT_LANES = ("main_dense_control", "minimal_control", "minimal_megakernel")


@dataclass(frozen=True)
class RunSpec:
    model_id: str
    prompt_token_count: int
    lane: str
    device: str
    warmup_runs: int
    max_new_tokens: int
    prompt_text: str
    dtype: str = DEFAULT_DTYPE

    def run_slug(self) -> str:
        model_slug = _slugify(self.model_id.split("/")[-1])
        return "-".join(
            [
                model_slug,
                self.lane,
                f"ctx{self.prompt_token_count}",
                self.device.replace(":", "-"),
                self.dtype,
                f"decode{self.max_new_tokens}",
            ]
        )

    def group_key(self) -> tuple[str, int]:
        return (self.model_id, self.prompt_token_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the Qwen3.5 minimal control lane against the main Rust/Candle runtime."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--contexts", nargs="*", type=int, default=list(DEFAULT_CONTEXTS))
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup-runs", type=int, default=DEFAULT_WARMUP_RUNS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--main-cargo-features", default=DEFAULT_MAIN_CARGO_FEATURES)
    parser.add_argument("--minimal-cargo-features", default=DEFAULT_MINIMAL_CARGO_FEATURES)
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


def _cargo_bin() -> str:
    cargo = shutil.which("cargo")
    if cargo is not None:
        return cargo
    fallback = Path.home() / ".cargo" / "bin" / "cargo"
    return str(fallback)


def build_matrix(
    *,
    model_id: str = DEFAULT_MODEL,
    contexts: tuple[int, ...] = DEFAULT_CONTEXTS,
    device: str = DEFAULT_DEVICE,
    dtype: str = DEFAULT_DTYPE,
    prompt_text: str = DEFAULT_PROMPT,
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for context in contexts:
        for lane in DEFAULT_LANES:
            specs.append(
                RunSpec(
                    model_id=model_id,
                    prompt_token_count=context,
                    lane=lane,
                    device=device,
                    warmup_runs=warmup_runs,
                    max_new_tokens=max_new_tokens,
                    prompt_text=prompt_text,
                    dtype=dtype,
                )
            )
    return specs


def env_for_run(spec: RunSpec) -> dict[str, str]:
    if spec.lane != "minimal_megakernel":
        return {}
    env = {"CANDLE_QWEN35_FULL_PREFILL_MEGAKERNEL": "1"}
    if spec.device.lower().startswith("hip"):
        env["CANDLE_QWEN35_HIP_PERSISTENT_FULL_PREFILL"] = "1"
    return env


def command_for_run(
    spec: RunSpec,
    *,
    out_prefix: Path,
    minimal_cargo_features: str,
    main_cargo_features: str,
) -> list[str]:
    if spec.lane in {"minimal_control", "minimal_megakernel"}:
        return [
            _cargo_bin(),
            "run",
            "--features",
            minimal_cargo_features,
            "--example",
            "hf_qwen35_minimal_bench",
            "--",
            spec.model_id,
            spec.prompt_text,
            str(out_prefix),
            "--device",
            spec.device,
            "--prompt-token-target",
            str(spec.prompt_token_count),
            "--warmup-runs",
            str(spec.warmup_runs),
            "--max-new-tokens",
            str(spec.max_new_tokens),
        ]
    if spec.lane == "main_dense_control":
        return [
            _cargo_bin(),
            "run",
            "--features",
            main_cargo_features,
            "--example",
            "hf_bench",
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
            DEFAULT_MAIN_RUNTIME,
            "--warmup-runs",
            str(spec.warmup_runs),
            "--max-new-tokens",
            str(spec.max_new_tokens),
            "--prompt-token-target",
            str(spec.prompt_token_count),
            "--sync-stage-profile",
        ]
    raise ValueError(f"unknown lane {spec.lane}")


def _summary_path(out_prefix: Path) -> Path:
    return out_prefix.with_suffix(".summary.json")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_config_signature(
    spec: RunSpec,
    *,
    minimal_cargo_features: str,
    main_cargo_features: str,
) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "prompt_token_count": spec.prompt_token_count,
        "lane": spec.lane,
        "device": spec.device,
        "dtype": spec.dtype,
        "warmup_runs": spec.warmup_runs,
        "max_new_tokens": spec.max_new_tokens,
        "prompt_text_sha256": hashlib.sha256(spec.prompt_text.encode("utf-8")).hexdigest(),
        "minimal_cargo_features": minimal_cargo_features,
        "main_cargo_features": main_cargo_features,
        "env_overrides": env_for_run(spec),
    }


def _can_resume_existing_run(config_path: Path, expected_signature: dict[str, Any]) -> bool:
    if not config_path.exists():
        return False
    try:
        return _load_json(config_path) == expected_signature
    except (OSError, json.JSONDecodeError):
        return False


def _extract_summary_metrics(summary: dict[str, Any], *, lane: str) -> dict[str, Any]:
    if lane in {"minimal_control", "minimal_megakernel"}:
        return {
            "prompt_token_count": summary.get("prompt_token_count"),
            "generated_token_count": summary.get("generated_token_count"),
            "prefill_millis": summary.get("prefill_millis"),
            "decode_millis": summary.get("decode_millis"),
            "total_millis": summary.get("total_millis"),
            "prefill_tokens_per_second": summary.get("prefill_tokens_per_second"),
            "decode_tokens_per_second": summary.get("decode_tokens_per_second"),
            "total_tokens_per_second": summary.get("total_tokens_per_second"),
            "full_prefill_megakernel_requested": summary.get(
                "full_prefill_megakernel_requested"
            ),
            "hip_persistent_full_prefill_requested": summary.get(
                "hip_persistent_full_prefill_requested"
            ),
        }
    return {
        "prompt_token_count": summary.get("prompt_token_count"),
        "generated_token_count": summary.get("generated_token_count"),
        "prefill_millis": summary.get("prefill_millis"),
        "decode_millis": summary.get("decode_millis"),
        "total_millis": summary.get("total_millis"),
        "prefill_tokens_per_second": summary.get("prefill_tokens_per_second"),
        "decode_tokens_per_second": summary.get("decode_tokens_per_second"),
        "total_tokens_per_second": summary.get("total_tokens_per_second"),
    }


def execute_run(
    spec: RunSpec,
    *,
    output_dir: Path,
    minimal_cargo_features: str,
    main_cargo_features: str,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    run_dir = output_dir / spec.run_slug()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = run_dir / "run"
    summary_path = _summary_path(out_prefix)
    config_path = run_dir / "run_config.json"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command = command_for_run(
        spec,
        out_prefix=out_prefix,
        minimal_cargo_features=minimal_cargo_features,
        main_cargo_features=main_cargo_features,
    )
    signature = _run_config_signature(
        spec,
        minimal_cargo_features=minimal_cargo_features,
        main_cargo_features=main_cargo_features,
    )
    record: dict[str, Any] = {
        "run_id": spec.run_slug(),
        "model_id": spec.model_id,
        "prompt_token_count": spec.prompt_token_count,
        "lane": spec.lane,
        "device": spec.device,
        "dtype": spec.dtype,
        "warmup_runs": spec.warmup_runs,
        "max_new_tokens": spec.max_new_tokens,
        "command": command,
        "env_overrides": env_for_run(spec),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "summary_path": str(summary_path),
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
        record["summary_metrics"] = _extract_summary_metrics(summary, lane=spec.lane)
        record["completed_at"] = _utc_now()
        return record

    print(f"[run] {record['run_id']}", flush=True)
    _write_json(config_path, signature)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        child_env = dict(os.environ)
        child_env.update(record["env_overrides"])
        completed = subprocess.run(
            command,
            cwd=_runtime_dir(),
            env=child_env,
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
        record["summary_metrics"] = _extract_summary_metrics(summary, lane=spec.lane)
        return record

    print(f"[failed] {record['run_id']}", flush=True)
    record["status"] = "failed"
    stderr_text = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
    stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
    record["error_message"] = (stderr_text or stdout_text).strip()[-4000:]
    return record


def _comparison_payload(
    baseline_metrics: dict[str, Any] | None, candidate_metrics: dict[str, Any] | None
) -> dict[str, Any]:
    comparison = {
        "delta_total_millis": None,
        "total_millis_ratio": None,
        "delta_prefill_millis": None,
        "prefill_millis_ratio": None,
        "delta_decode_millis": None,
        "decode_millis_ratio": None,
        "delta_total_tokens_per_second": None,
        "total_tokens_per_second_ratio": None,
    }
    if baseline_metrics is None or candidate_metrics is None:
        return comparison
    for metric in ("total_millis", "prefill_millis", "decode_millis", "total_tokens_per_second"):
        baseline_value = baseline_metrics.get(metric)
        candidate_value = candidate_metrics.get(metric)
        if baseline_value is None or candidate_value is None or baseline_value == 0:
            continue
        if metric == "total_millis":
            comparison["delta_total_millis"] = candidate_value - baseline_value
            comparison["total_millis_ratio"] = candidate_value / baseline_value
        elif metric == "prefill_millis":
            comparison["delta_prefill_millis"] = candidate_value - baseline_value
            comparison["prefill_millis_ratio"] = candidate_value / baseline_value
        elif metric == "decode_millis":
            comparison["delta_decode_millis"] = candidate_value - baseline_value
            comparison["decode_millis_ratio"] = candidate_value / baseline_value
        elif metric == "total_tokens_per_second":
            comparison["delta_total_tokens_per_second"] = candidate_value - baseline_value
            comparison["total_tokens_per_second_ratio"] = candidate_value / baseline_value
    return comparison


def build_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        if record["status"] not in {"completed", "reused_existing", "failed"}:
            continue
        key = (record["model_id"], record["prompt_token_count"])
        grouped.setdefault(key, []).append(record)

    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        model_id, prompt_token_count = key
        entries = grouped[key]
        lane_entries = {
            lane: next((entry for entry in entries if entry["lane"] == lane), None)
            for lane in DEFAULT_LANES
        }
        lane_metrics = {
            lane: None if entry is None else entry.get("summary_metrics")
            for lane, entry in lane_entries.items()
        }
        groups.append(
            {
                "model_id": model_id,
                "prompt_token_count": prompt_token_count,
                **{
                    lane: None
                    if lane_entries[lane] is None
                    else {
                        "run_id": lane_entries[lane]["run_id"],
                        "status": lane_entries[lane]["status"],
                        "summary_metrics": lane_metrics[lane],
                    }
                    for lane in DEFAULT_LANES
                },
                "comparisons": {
                    "minimal_control_vs_main": _comparison_payload(
                        lane_metrics["main_dense_control"],
                        lane_metrics["minimal_control"],
                    ),
                    "minimal_megakernel_vs_main": _comparison_payload(
                        lane_metrics["main_dense_control"],
                        lane_metrics["minimal_megakernel"],
                    ),
                    "minimal_megakernel_vs_minimal_control": _comparison_payload(
                        lane_metrics["minimal_control"],
                        lane_metrics["minimal_megakernel"],
                    ),
                },
            }
        )
    return {"generated_at": _utc_now(), "group_count": len(groups), "groups": groups}


def _fmt_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def render_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5 Minimal Control Compare",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
    ]
    for group in report["groups"]:
        lines.extend(
            [
                f"## {group['model_id']} | ctx={group['prompt_token_count']}",
                "",
                "| Lane | Status | Prefill ms | Decode ms | Total ms | Total tok/s | Megakernel env |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for lane in DEFAULT_LANES:
            entry = group[lane]
            if entry is None:
                lines.append(f"| {lane} | missing | - | - | - | - | - |")
                continue
            metrics = entry["summary_metrics"] or {}
            lines.append(
                "| {lane} | {status} | {prefill} | {decode} | {total} | {tps} | {megakernel} |".format(
                    lane=lane,
                    status=entry["status"],
                    prefill=_fmt_float(metrics.get("prefill_millis")),
                    decode=_fmt_float(metrics.get("decode_millis")),
                    total=_fmt_float(metrics.get("total_millis")),
                    tps=_fmt_float(metrics.get("total_tokens_per_second")),
                    megakernel=metrics.get("full_prefill_megakernel_requested", "-"),
                )
            )
        for name, comparison in group["comparisons"].items():
            lines.extend(
                [
                    "",
                    f"### {name}",
                    "",
                    f"- `delta_total_millis`: {_fmt_float(comparison['delta_total_millis'])}",
                    f"- `total_millis_ratio`: {_fmt_float(comparison['total_millis_ratio'])}",
                    f"- `delta_prefill_millis`: {_fmt_float(comparison['delta_prefill_millis'])}",
                    f"- `prefill_millis_ratio`: {_fmt_float(comparison['prefill_millis_ratio'])}",
                    f"- `delta_decode_millis`: {_fmt_float(comparison['delta_decode_millis'])}",
                    f"- `decode_millis_ratio`: {_fmt_float(comparison['decode_millis_ratio'])}",
                    f"- `delta_total_tokens_per_second`: {_fmt_float(comparison['delta_total_tokens_per_second'])}",
                    f"- `total_tokens_per_second_ratio`: {_fmt_float(comparison['total_tokens_per_second_ratio'])}",
                ]
            )
        lines.extend(
            [
                "",
            ]
        )
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
    output_dir = output_root / f"qwen35_minimal_control_compare_{args.run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = build_matrix(
        model_id=args.model_id,
        contexts=tuple(args.contexts),
        device=args.device,
        dtype=args.dtype,
        prompt_text=args.prompt_text,
        warmup_runs=args.warmup_runs,
        max_new_tokens=args.max_new_tokens,
    )
    manifest: dict[str, Any] = {
        "generated_at": _utc_now(),
        "output_dir": str(output_dir),
        "model_id": args.model_id,
        "contexts": args.contexts,
        "device": args.device,
        "dtype": args.dtype,
        "prompt_text_sha256": hashlib.sha256(args.prompt_text.encode("utf-8")).hexdigest(),
        "warmup_runs": args.warmup_runs,
        "max_new_tokens": args.max_new_tokens,
        "minimal_cargo_features": args.minimal_cargo_features,
        "main_cargo_features": args.main_cargo_features,
        "records": [],
    }
    persist_outputs(output_dir, manifest)
    for spec in specs:
        record = execute_run(
            spec,
            output_dir=output_dir,
            minimal_cargo_features=args.minimal_cargo_features,
            main_cargo_features=args.main_cargo_features,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        manifest["records"].append(record)
        persist_outputs(output_dir, manifest)


if __name__ == "__main__":
    main()
