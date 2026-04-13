#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "benchmarks" / "results" / "qwen35_hip_direct_tracking_20260413"
RENDER_SCRIPT = ROOT / "benchmarks" / "render_qwen35_hip_direct_tracking.py"


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


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

    run([sys.executable, str(RENDER_SCRIPT)], env)


if __name__ == "__main__":
    main()
