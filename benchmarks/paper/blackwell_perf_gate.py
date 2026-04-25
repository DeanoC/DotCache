"""Blackwell certified-attention performance gate.

This script makes the long-run decision mechanical:
  1. verify the vendored CUTLASS/SM120 toolchain,
  2. run isolated mixed-value attention microbenchmarks,
  3. optionally run short PG-19 phase profiles with the exact paper flags.

It writes a JSON report so the paper runner can decide whether one GPU is
enough or whether the workload should be split across machines.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PAPER_FLAGS = [
    "--model", "NousResearch/Meta-Llama-3.1-8B",
    "--v-tolerance", "0.05",
    "--use-int4-values",
    "--group-size", "16",
    "--tau-cov", "0.995",
    "--k-min", "2",
    "--k-max", "128",
    "--ranking-fallback",
    "--ranking-r", "1",
    "--ranking-fallback-mode", "full",
    "--score-consistency-check",
    "--eps-guard", "0.01",
    "--exploration-rate", "0.02",
    "--rung1-threshold", "0.02",
    "--rung1-multiplier", "2.0",
    "--telemetry-mode", "off",
    "--certified-warmup-steps", "128",
]


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": time.perf_counter() - started,
        "stdout_tail": proc.stdout[-8000:],
    }


def _device_info() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        }
    except Exception as exc:
        return {"cuda_available": False, "error": repr(exc)}


def _cutlass_probe() -> dict[str, Any]:
    try:
        import torch
        from dotcache.backends.cutlass_sm120 import (
            cutlass_root,
            cutlass_sm120_available,
            cutlass_sm120_metadata,
            cutlass_sm120_probe,
        )

        root = cutlass_root()
        version_file = root / "include" / "cutlass" / "version.h"
        result: dict[str, Any] = {
            "root": str(root),
            "version_header_exists": version_file.exists(),
            "available": cutlass_sm120_available(),
        }
        if result["available"]:
            x = torch.arange(8, device="cuda", dtype=torch.float32)
            y = cutlass_sm120_probe(x)
            torch.cuda.synchronize()
            result["metadata"] = cutlass_sm120_metadata()
            result["probe_max_abs"] = float((x - y).abs().max().item())
        return result
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/paper_v2_20260425/blackwell_perf_gate.json"))
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--pg19-contexts", type=int, nargs="+", default=[65536])
    parser.add_argument("--pg19-steps", type=int, default=128)
    parser.add_argument("--skip-pg19", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.setdefault("DOTCACHE_FP16_BLOCK_SCORE_TRITON", "1")

    report: dict[str, Any] = {
        "device": _device_info(),
        "cutlass_sm120": _cutlass_probe(),
        "synthetic": None,
        "pg19": [],
        "gates": {
            "target_pg19_64k_tok_s": 20.0,
            "target_final_pg19_64k_tok_s": 40.0,
            "requires_cutlass_available": True,
        },
    }

    bench_cmd = [
        args.python,
        "benchmarks/bench_blackwell_perf.py",
        "--contexts",
        *[str(c) for c in args.contexts],
        "--warmup",
        "5",
        "--iters",
        "20",
        "--output",
        str(args.output.with_suffix(".synthetic.json")),
    ]
    report["synthetic"] = _run(bench_cmd, env=env)

    if not args.skip_pg19:
        for ctx in args.pg19_contexts:
            pg19_out = args.output.with_name(f"{args.output.stem}_pg19_{ctx}.json")
            pg19_cmd = [
                args.python,
                "benchmarks/paper/pg19_perplexity.py",
                "--context",
                str(ctx),
                "--num-chunks",
                "1",
                "--max-certified-steps",
                str(args.pg19_steps),
                "--output",
                str(pg19_out),
                *PAPER_FLAGS,
            ]
            pg19_env = env.copy()
            pg19_env["DOTCACHE_PHASE_TIMING"] = "1"
            report["pg19"].append(_run(pg19_cmd, env=pg19_env))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
