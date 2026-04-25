"""paper_v1 benchmark sweep orchestrator.

Runs the root-paper experiment matrix on the current branch and writes one
paired Dense-vs-Certified JSON per benchmark/context under
benchmarks/results/paper_v1_20260424/.

The certified side is the paper §7 configuration:
INT8 asymmetric keys, INT4 g=16 values, tau_cov=0.995, K=[2,128],
v_tolerance=0.05, ranking consistency r=1, score-consistency checks enabled,
and a 2% exploration budget.

Usage:
    python benchmarks/run_arxiv_v1_sweep.py --dry-run
    python benchmarks/run_arxiv_v1_sweep.py --smoke --only pg19,4096
    python benchmarks/run_arxiv_v1_sweep.py --from 07 --no-push
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "benchmarks" / "results" / "paper_v1_20260424"
BRANCH = "port-to-paper-20260424"
sys.path.insert(0, str(REPO))

CONTEXTS = [4096, 8192, 16384, 32768]
BENCHES = ["pg19", "niah", "ruler"]

CERT_FLAGS: dict[str, str] = {
    "v_tolerance": "0.05",
    "tau_cov": "0.995",
    "k_min": "2",
    "k_max": "128",
    "ranking_r": "1",
    "eps_guard": "0.01",
    "exploration_rate": "0.02",
    "rung1_threshold": "0.02",
    "rung1_multiplier": "2.0",
    "fp16_value_cache_blocks": "64",
}


def build_cells() -> list[dict[str, Any]]:
    """Return the 12 paired cells used by the paper summary tables."""
    cells: list[dict[str, Any]] = []
    idx = 0
    for ctx in CONTEXTS:
        for bench in BENCHES:
            idx += 1
            cells.append({"idx": f"{idx:02d}", "bench": bench, "ctx": ctx})
    return cells


def _common_cert_args() -> list[str]:
    return [
        "--v-tolerance", CERT_FLAGS["v_tolerance"],
        "--use-int4-values",
        "--group-size", "16",
        "--tau-cov", CERT_FLAGS["tau_cov"],
        "--k-min", CERT_FLAGS["k_min"],
        "--k-max", CERT_FLAGS["k_max"],
        "--ranking-fallback",
        "--ranking-r", CERT_FLAGS["ranking_r"],
        "--ranking-fallback-mode", "full",
        "--score-consistency-check",
        "--eps-guard", CERT_FLAGS["eps_guard"],
        "--exploration-rate", CERT_FLAGS["exploration_rate"],
        "--rung1-threshold", CERT_FLAGS["rung1_threshold"],
        "--rung1-multiplier", CERT_FLAGS["rung1_multiplier"],
        "--fp16-value-cache-blocks", CERT_FLAGS["fp16_value_cache_blocks"],
    ]


def _cli_for_pg19(ctx: int, out_json: Path, smoke: bool) -> list[str]:
    # Paper table: 4K-16K use 5 non-overlapping chunks; 32K uses 20 chunks.
    chunks = 1 if smoke else (20 if ctx == 32768 else 5)
    return [
        sys.executable, str(REPO / "benchmarks" / "paper" / "pg19_perplexity.py"),
        "--context", str(ctx),
        "--num-chunks", str(chunks),
        "--telemetry-mode", "summary",
        "--output", str(out_json),
        *_common_cert_args(),
    ]


def _cli_for_niah(ctx: int, out_json: Path, smoke: bool) -> list[str]:
    # Current NIAH harness has 10 fixed depths. needles=3 => 30 paired trials;
    # needles=10 => the paper's powered 100-paired-trial 8K follow-up.
    needles = 1 if smoke else (10 if ctx == 8192 else 3)
    return [
        sys.executable, str(REPO / "benchmarks" / "paper" / "niah.py"),
        "--contexts", str(ctx),
        "--needles", str(needles),
        "--output", str(out_json),
        *_common_cert_args(),
    ]


def _cli_for_ruler(ctx: int, out_json: Path, smoke: bool) -> list[str]:
    samples = 1 if smoke else 50
    return [
        sys.executable, str(REPO / "benchmarks" / "paper" / "ruler.py"),
        "--contexts", str(ctx),
        "--num-samples", str(samples),
        "--output", str(out_json),
        *_common_cert_args(),
    ]


BENCH_CLIS = {
    "pg19": _cli_for_pg19,
    "niah": _cli_for_niah,
    "ruler": _cli_for_ruler,
}


def _runner_env(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _git(*args: str) -> tuple[int, str]:
    res = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=REPO,
    )
    return res.returncode, (res.stdout + res.stderr)


def git_sha() -> str:
    rc, out = _git("rev-parse", "--short", "HEAD")
    return out.strip() if rc == 0 else "unknown"


def current_branch() -> str:
    rc, out = _git("branch", "--show-current")
    return out.strip() if rc == 0 else "unknown"


def commit_and_push(paths: list[Path], message: str) -> None:
    if current_branch() != BRANCH:
        raise RuntimeError(
            f"Refusing to push benchmark results from branch {current_branch()!r}; "
            f"expected {BRANCH!r}"
        )
    for path in paths:
        rel = path.relative_to(REPO)
        _git("add", str(rel))
    rc, out = _git("commit", "-m", message)
    if rc != 0 and "nothing to commit" in out.lower():
        return
    if rc != 0:
        raise RuntimeError(out)
    rc, out = _git("push", "origin", BRANCH)
    if rc != 0:
        raise RuntimeError(out)


def _hw_tag() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name} sm_{p.major}{p.minor}"
    except Exception:
        pass
    return "unknown"


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_parse_error": str(exc)}


def _quality(bench: str, native: Any) -> dict[str, Any]:
    if not isinstance(native, dict):
        return {"_missing": True}
    if bench == "pg19":
        dense = native.get("dense", {}).get("perplexity")
        cert = native.get("certified", {}).get("perplexity")
        return {
            "metric": "perplexity",
            "dense": dense,
            "certified": cert,
            "delta": (cert - dense) if cert is not None and dense is not None else None,
        }
    if bench == "niah":
        dense = native.get("dense_accuracy")
        cert = native.get("certified_accuracy")
        paired_stats = native.get("paired_stats") or {}
        return {
            "metric": "accuracy",
            "dense": dense,
            "certified": cert,
            "delta": (cert - dense) if cert is not None and dense is not None else None,
            "paired_stats": paired_stats,
            "n": paired_stats.get("n"),
            "delta_pp": paired_stats.get("delta_pp"),
            "bootstrap_ci_pp": (
                [paired_stats.get("bootstrap_ci_pp_lo"), paired_stats.get("bootstrap_ci_pp_hi")]
                if paired_stats.get("bootstrap_ci_pp_lo") is not None
                and paired_stats.get("bootstrap_ci_pp_hi") is not None
                else None
            ),
            "mcnemar_p": paired_stats.get("mcnemar_p"),
            "critical_failures": native.get("critical_failures"),
        }
    if bench == "ruler":
        dense = native.get("overall_dense")
        cert = native.get("overall_cert")
        return {
            "metric": "accuracy",
            "dense": dense,
            "certified": cert,
            "delta": (cert - dense) if cert is not None and dense is not None else None,
            "critical_failures": native.get("critical_failures"),
        }
    return {"_missing_extractor": bench}


def _system(native: Any) -> dict[str, Any]:
    if not isinstance(native, dict):
        return {}
    rfs = native.get("ranking_fallback_summary")
    out: dict[str, Any] = {}
    if rfs:
        out["ranking_disagree_rate_r1"] = rfs.get("disagree_rate_r1")
        out["ranking_disagree_rate_r3"] = rfs.get("disagree_rate_r3")
        out["ranking_fallback_rate"] = rfs.get("fallback_rate")
        out["heads_total"] = rfs.get("heads_total")
    if "certified" in native and isinstance(native["certified"], dict):
        cert = native["certified"]
        out["skip_rate"] = cert.get("skip_rate")
        out["chunks"] = cert.get("num_chunks")
    return out


def run_cell(cell: dict[str, Any], *, smoke: bool, dry_run: bool) -> dict[str, Any]:
    bench = str(cell["bench"])
    ctx = int(cell["ctx"])
    ctx_k = ctx // 1024
    cell_json = OUT_DIR / f"{cell['idx']}_{bench}_{ctx_k}K_paired.json"
    cell_log = OUT_DIR / f"{cell['idx']}_{bench}_{ctx_k}K_paired.log"
    native_json = cell_json.with_suffix(".native.json")
    cli = BENCH_CLIS[bench](ctx, native_json, smoke)
    plan = {
        "idx": cell["idx"],
        "bench": bench,
        "ctx": ctx,
        "cmd": " ".join(shlex.quote(c) for c in cli),
        "out": str(cell_json),
    }
    if dry_run:
        return plan

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.perf_counter()
    print(f"\n[{cell['idx']}] {bench} {ctx_k}K paired — starting")
    print(f"  cmd: {plan['cmd']}")
    print(f"  log: {cell_log}")

    with cell_log.open("w") as log_f:
        log_f.write(f"cmd: {plan['cmd']}\nstarted: {started}\n---\n")
        log_f.flush()
        rc = subprocess.call(cli, stdout=log_f, stderr=subprocess.STDOUT,
                             cwd=REPO, env=_runner_env())

    wall = time.perf_counter() - t0
    ended = dt.datetime.now(dt.timezone.utc).isoformat()
    native = _load_json(native_json)
    wrapped = {
        "benchmark": bench,
        "context_length": ctx,
        "paired_dense_certified": True,
        "smoke": smoke,
        "quality": _quality(bench, native),
        "system": _system(native),
        "native": native,
        "meta": {
            "model": "NousResearch/Meta-Llama-3.1-8B",
            "model_quant": "int8-bitsandbytes",
            "hardware": _hw_tag(),
            "timestamp": ended,
            "started": started,
            "wall_seconds": wall,
            "git_sha": git_sha(),
            "branch": current_branch(),
            "exit_code": rc,
            "paper_config": {
                **CERT_FLAGS,
                "group_size": "16",
                "ranking_fallback_mode": "full",
                "fp16_value_cache_blocks": CERT_FLAGS["fp16_value_cache_blocks"],
            },
        },
    }
    cell_json.write_text(json.dumps(wrapped, indent=2))
    print(f"[{cell['idx']}] {bench} {ctx_k}K paired exit={rc} {wall/60:.1f} min")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Use tiny sample counts for plumbing checks.")
    parser.add_argument("--only", default=None,
                        help="Comma filter, e.g. 'pg19,4096' or '04' or 'ruler'.")
    parser.add_argument("--from", dest="start_from", default=None,
                        help="Resume from cell index, inclusive.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-push", action="store_true",
                        help="Do not commit/push after each completed cell.")
    args = parser.parse_args()

    cells = build_cells()
    if args.only:
        toks = [t.strip().lower() for t in args.only.split(",") if t.strip()]

        def match(c: dict[str, Any]) -> bool:
            bag = {str(c["idx"]).lower(), str(c["bench"]).lower(), str(c["ctx"]).lower()}
            return all(t in bag or any(t in v for v in bag) for t in toks)

        cells = [c for c in cells if match(c)]
    if args.start_from:
        cells = [c for c in cells if int(c["idx"]) >= int(args.start_from)]

    print(f"Plan: {len(cells)} paired cell(s) (smoke={args.smoke})")
    for cell in cells:
        print(f"  {cell['idx']}  {cell['bench']:<5} {cell['ctx']//1024}K")

    if args.dry_run:
        for cell in cells:
            print(run_cell(cell, smoke=args.smoke, dry_run=True)["cmd"])
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from benchmarks._manifest import refresh_cells, write_initial_manifest

    manifest = write_initial_manifest(
        OUT_DIR,
        dotcache_config={
            **CERT_FLAGS,
            "quantization_mode": "asymmetric_int8_keys+int4_g16_values",
            "ranking_fallback_mode": "full",
        },
        repo=REPO,
        notes="paper_v1 paired Dense-vs-Certified sweep from root tex",
    )

    for cell in cells:
        run_cell(cell, smoke=args.smoke, dry_run=False)
        refresh_cells(OUT_DIR)
        if not args.no_push:
            commit_and_push(
                [OUT_DIR],
                f"bench: paper_v1 cell {cell['idx']} {cell['bench']} {cell['ctx']//1024}K",
            )
        else:
            print("[no-push] skipped commit/push for completed cell")

    if not args.no_push:
        commit_and_push([manifest], "bench: refresh paper_v1 manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
