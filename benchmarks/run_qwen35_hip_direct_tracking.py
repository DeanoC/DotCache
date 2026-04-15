#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "benchmarks" / "results" / "qwen35_hip_direct_tracking_20260413"
RENDER_SCRIPT = ROOT / "benchmarks" / "render_qwen35_hip_direct_tracking.py"
HISTORY_JSONL = TRACKING_DIR / "history.jsonl"


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def append_history(label: str, json_path: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    entry = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label,
        "model_id": payload["model_id"],
        "device": payload["device"],
        "load_mode": payload["load_mode"],
        "oracle": payload["oracle"],
        "prompt_token_count": payload["prompt_token_count"],
        "generated_token_count": payload["generated_token_count"],
        "max_new_tokens": payload["max_new_tokens"],
        "device_load_ms": payload["device_load_ms"],
        "device_prefill_ms": payload["device_prefill_ms"],
        "device_decode_ms": payload["device_decode_ms"],
        "prefill_max_delta": payload["prefill_max_delta"],
        "prefill_cache_max_delta": payload.get("prefill_cache_max_delta"),
        "decode_max_delta": payload["decode_max_delta"],
        "decode_input_hidden_max_delta": payload.get("decode_input_hidden_max_delta"),
        "decode_step_cache_max_delta": payload.get("decode_step_cache_max_delta"),
        "generated_text": payload["generated_text"],
        "source_json": json_path.name,
    }
    with HISTORY_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run canonical Qwen35 HIP-direct checkpoints and refresh the tracking report."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="hip:0")
    parser.add_argument("--features", default="qwen35-minimal,qwen35-minimal-hip")
    parser.add_argument("--short-prompt", default="Hello from DotCache")
    parser.add_argument(
        "--long-prompt",
        default="The direct HIP lane should stay correct while we keep specializing the decode path.",
    )
    parser.add_argument("--short-new-tokens", type=int, default=1)
    parser.add_argument("--long-new-tokens", type=int, default=8)
    args = parser.parse_args()

    TRACKING_DIR.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        "cargo",
        "run",
        "--manifest-path",
        "rust/Cargo.toml",
        "-p",
        "dotcache-paged-runtime",
        "--example",
        "hf_qwen35_minimal",
        "--features",
        args.features,
        "--",
        args.model_id,
    ]

    env = os.environ.copy()

    run(
        base_cmd
        + [
            args.short_prompt,
            str(args.short_new_tokens),
            "--device",
            args.device,
            "--load-mode",
            "hip-direct",
            "--record-json",
            str(TRACKING_DIR / "hip_direct_short.json"),
        ],
        env,
    )
    append_history("short_native_device", TRACKING_DIR / "hip_direct_short.json")

    run(
        base_cmd
        + [
            args.long_prompt,
            str(args.long_new_tokens),
            "--device",
            args.device,
            "--load-mode",
            "hip-direct",
            "--record-json",
            str(TRACKING_DIR / "hip_direct_longer.json"),
        ],
        env,
    )
    append_history("longer_native_device", TRACKING_DIR / "hip_direct_longer.json")

    run(
        base_cmd
        + [
            args.short_prompt,
            str(args.short_new_tokens),
            "--device",
            args.device,
            "--load-mode",
            "hip-direct",
            "--oracle",
            "cpu",
            "--record-json",
            str(TRACKING_DIR / "hip_direct_short_cpu_oracle.json"),
        ],
        env,
    )
    append_history("short_cpu_oracle", TRACKING_DIR / "hip_direct_short_cpu_oracle.json")

    run([sys.executable, str(RENDER_SCRIPT)], env)


if __name__ == "__main__":
    main()
