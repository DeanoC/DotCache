from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mount a Hugging Face repo with hf-mount and run an existing compare harness against the mounted path."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--benchmark-kind", choices=["llama_compare", "qwen2_compare"], required=True)
    parser.add_argument("--mount-point", default=None)
    parser.add_argument("--hf-mount-bin", default=os.environ.get("HF_MOUNT_BIN", "hf-mount"))
    parser.add_argument("--mount-timeout-secs", type=float, default=30.0)
    parser.add_argument("--keep-mounted", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--target-prompt-lengths", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _default_mount_point(repo_id: str) -> Path:
    safe_name = repo_id.replace("/", "__")
    return Path(".cache") / "hf-mount" / safe_name


def _probe_hf_mount(binary_name: str) -> dict[str, object]:
    executable = shutil.which(binary_name)
    if executable is None:
        return {
            "hf_mount": binary_name,
            "hf_mount_found": False,
        }
    completed = subprocess.run(
        [executable, "--help"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    help_text = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return {
        "hf_mount": executable,
        "hf_mount_found": True,
        "hf_mount_help_rc": completed.returncode,
        "hf_mount_supports_repo_start": "start repo" in help_text or ("start" in help_text and "repo" in help_text),
        "hf_mount_help_excerpt": help_text[:1000],
    }


def _wait_for_mount_ready(mount_point: Path, *, timeout_secs: float) -> bool:
    deadline = time.perf_counter() + timeout_secs
    while time.perf_counter() < deadline:
        if mount_point.exists():
            try:
                next(mount_point.iterdir())
                return True
            except StopIteration:
                pass
            except OSError:
                pass
        time.sleep(0.25)
    return mount_point.exists()


def _benchmark_command(args: argparse.Namespace, mount_point: Path) -> list[str]:
    root = Path(__file__).resolve().parent.parent
    script_name = "bench_llama_compare.py" if args.benchmark_kind == "llama_compare" else "bench_qwen2_compare.py"
    command = [
        str(root / ".venv" / "bin" / "python"),
        str(root / "benchmarks" / script_name),
        "--model-id",
        str(mount_point.resolve()),
        "--backend",
        args.backend,
        "--torch-dtype",
        args.torch_dtype,
        "--tokens-per-page",
        str(args.tokens_per_page),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--target-prompt-lengths",
        *[str(length) for length in args.target_prompt_lengths],
    ]
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.device is not None:
        command.extend(["--device", args.device])
    return command


def _emit_record(record: dict[str, object]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    probe = _probe_hf_mount(args.hf_mount_bin)
    mount_point = Path(args.mount_point) if args.mount_point is not None else _default_mount_point(args.repo_id)
    record: dict[str, object] = {
        "benchmark": "hf_mount_compare",
        "repo_id": args.repo_id,
        "benchmark_kind": args.benchmark_kind,
        "mount_point": str(mount_point),
        "backend": args.backend,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "tokens_per_page": args.tokens_per_page,
        "max_new_tokens": args.max_new_tokens,
        "target_prompt_lengths": list(args.target_prompt_lengths),
        "keep_mounted": bool(args.keep_mounted),
    }
    record.update(probe)

    if not probe.get("hf_mount_found", False):
        record.update(
            {
                "status": "error",
                "error_type": "MissingExecutable",
                "error_message": f"hf-mount executable not found: {args.hf_mount_bin}",
            }
        )
        _emit_record(record)
        if not args.continue_on_error:
            raise SystemExit(record["error_message"])
        return

    mounted_by_runner = False
    mount_start_ms = 0.0
    if not mount_point.exists():
        mount_point.parent.mkdir(parents=True, exist_ok=True)
        mounted_by_runner = True
        start_command = [str(probe["hf_mount"]), "start", "repo", args.repo_id, str(mount_point)]
        start_time = time.perf_counter()
        started = subprocess.run(
            start_command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        mount_start_ms = (time.perf_counter() - start_time) * 1000.0
        record["hf_mount_start_command"] = start_command
        record["hf_mount_start_rc"] = started.returncode
        record["hf_mount_start_stdout_tail"] = started.stdout[-1000:]
        record["hf_mount_start_stderr_tail"] = started.stderr[-1000:]
        if started.returncode != 0 or not _wait_for_mount_ready(mount_point, timeout_secs=args.mount_timeout_secs):
            record.update(
                {
                    "status": "error",
                    "error_type": "MountFailed",
                    "error_message": f"hf-mount could not mount {args.repo_id} at {mount_point}",
                    "hf_mount_start_ms": mount_start_ms,
                }
            )
            _emit_record(record)
            if not args.continue_on_error:
                raise SystemExit(1)
            return

    command = _benchmark_command(args, mount_point)
    started_at = time.perf_counter()
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    wall_ms = (time.perf_counter() - started_at) * 1000.0

    record.update(
        {
            "status": "ok" if completed.returncode == 0 else "error",
            "hf_mount_start_ms": mount_start_ms,
            "mounted_by_runner": mounted_by_runner,
            "command": command,
            "wall_ms": wall_ms,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-2000:],
        }
    )

    if mounted_by_runner and not args.keep_mounted:
        stop_completed = subprocess.run(
            [str(probe["hf_mount"]), "stop", str(mount_point)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        record["hf_mount_stop_rc"] = stop_completed.returncode
        record["hf_mount_stop_stdout_tail"] = stop_completed.stdout[-1000:]
        record["hf_mount_stop_stderr_tail"] = stop_completed.stderr[-1000:]

    _emit_record(record)
    if completed.returncode != 0 and not args.continue_on_error:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
