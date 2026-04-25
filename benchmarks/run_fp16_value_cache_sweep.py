"""Sweep bounded FP16 value fallback cache size for the paper Scratch curve.

This runs the PG-19 paired dense/certified harness repeatedly with identical
paper settings while varying ``--fp16-value-cache-blocks``. Use ``full`` for
the legacy full value mirror and integer block counts for bounded scratch.

Example:
    python benchmarks/run_fp16_value_cache_sweep.py --smoke
    python benchmarks/run_fp16_value_cache_sweep.py --context 4096 --num-chunks 5 \
        --sizes 0,16,32,64,128,full
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "benchmarks" / "results" / "fp16_value_cache_sweep"

CERT_ARGS = [
    "--v-tolerance", "0.05",
    "--use-int4-values",
    "--group-size", "16",
    "--tau-cov", "0.995",
    "--k-min", "2",
    "--k-max", "128",
    "--ranking-fallback",
    "--ranking-r", "1",
    "--ranking-fallback-mode", "full",
    "--eps-guard", "0.01",
    "--exploration-rate", "0.02",
    "--rung1-threshold", "0.02",
    "--rung1-multiplier", "2.0",
    "--fp16-key-cache-blocks", "3584",
]


def _parse_sizes(text: str) -> list[int | None]:
    out: list[int | None] = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item in {"full", "none", "mirror"}:
            out.append(None)
        else:
            out.append(int(item))
    if not out:
        raise ValueError("empty --sizes")
    return out


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_parse_error": str(exc)}


def _row(size: int | None, native: dict[str, Any], wall: float, rc: int) -> dict[str, Any]:
    cert = native.get("certified", {}) if isinstance(native, dict) else {}
    telem = cert.get("telemetry", {}) if isinstance(cert, dict) else {}
    dense = native.get("dense", {}) if isinstance(native, dict) else {}
    cache_config = native.get("cache_config", {}) if isinstance(native, dict) else {}
    return {
        "fp16_value_cache_blocks": "full" if size is None else size,
        "exit_code": rc,
        "wall_seconds": wall,
        "dense_ppl": dense.get("perplexity"),
        "certified_ppl": cert.get("perplexity"),
        "ratio": native.get("ratio") if isinstance(native, dict) else None,
        "delta": native.get("delta") if isinstance(native, dict) else None,
        "vram_fp16_value_cache_bytes_max": telem.get("vram_fp16_value_cache_bytes_max"),
        "h2d_value_bytes_total": telem.get("h2d_value_bytes_total"),
        "fp16_value_cache_hit_rate": telem.get("fp16_value_cache_hit_rate"),
        "fp16_value_cache_overflow_steps": telem.get("fp16_value_cache_overflow_steps"),
        "e_val_step_max": telem.get("e_val_step_max"),
        "score_consistency_violation_heads_total": telem.get("score_consistency_violation_heads_total"),
        "cache_config": cache_config,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--num-chunks", type=int, default=5)
    parser.add_argument("--sizes", default="0,16,32,64,128,full")
    parser.add_argument("--smoke", action="store_true",
                        help="Use one PG-19 chunk regardless of --num-chunks.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sizes = _parse_sizes(args.sizes)
    chunks = 1 if args.smoke else int(args.num_chunks)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.output) if args.output else (
        OUT_DIR / f"pg19_{args.context}_ctx_{chunks}_chunks_{stamp}.json"
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for size in sizes:
        label = "full" if size is None else str(size)
        native_json = out_path.with_name(f"{out_path.stem}.{label}.native.json")
        log_path = out_path.with_name(f"{out_path.stem}.{label}.log")
        cmd = [
            sys.executable,
            str(REPO / "benchmarks" / "paper" / "pg19_perplexity.py"),
            "--context", str(args.context),
            "--num-chunks", str(chunks),
            "--telemetry-mode", "summary",
            "--output", str(native_json),
            *CERT_ARGS,
        ]
        cmd += ["--fp16-value-cache-blocks", "full" if size is None else str(size)]

        print(f"[value-cache={label}] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        if args.dry_run:
            continue

        started = dt.datetime.now(dt.timezone.utc).isoformat()
        t0 = time.perf_counter()
        with log_path.open("w") as log_f:
            log_f.write(
                f"cmd: {' '.join(shlex.quote(c) for c in cmd)}\n"
                f"started: {started}\n---\n"
            )
            log_f.flush()
            rc = subprocess.call(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=REPO)
        wall = time.perf_counter() - t0
        native = _load_json(native_json)
        row = _row(size, native, wall, rc)
        rows.append(row)
        out_path.write_text(json.dumps({
            "benchmark": "pg19_fp16_value_cache_sweep",
            "context_length": args.context,
            "num_chunks": chunks,
            "sizes": ["full" if s is None else s for s in sizes],
            "rows": rows,
        }, indent=2))
        print(
            f"[value-cache={label}] exit={rc} wall={wall/60:.1f}m "
            f"cert_ppl={row.get('certified_ppl')} "
            f"scratch_mb={(row.get('vram_fp16_value_cache_bytes_max') or 0) / 1e6:.1f}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
