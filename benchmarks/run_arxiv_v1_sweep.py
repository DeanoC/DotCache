"""arXiv v1 Benchmark Sweep orchestrator.

Runs the 18-cell sweep (3 benchmarks × 3 context lengths × 2 configs)
from the paper-alignment spec, writes one JSON per cell under
benchmarks/results/arxiv_v1_20260420/, and commits+pushes after each
completed cell so a pod reset cannot lose more than one cell of work.

Usage:
    python benchmarks/run_arxiv_v1_sweep.py                     # full sweep
    python benchmarks/run_arxiv_v1_sweep.py --smoke             # sanity checks only
    python benchmarks/run_arxiv_v1_sweep.py --only niah,4096,certified
    python benchmarks/run_arxiv_v1_sweep.py --from 07           # resume from cell 07
    python benchmarks/run_arxiv_v1_sweep.py --dry-run           # print the plan
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
OUT_DIR = REPO / "benchmarks" / "results" / "arxiv_v1_20260420"
BRANCH = "feature/interval-ellipsoidal-bounds"

# ---------------------------------------------------------------------------
# Spec knobs (certified side, all features ON per spec).
# ---------------------------------------------------------------------------

CERT_FLAGS: dict[str, str] = {
    "tau_cov": "0.995",
    "k_min": "2",
    "k_max": "128",
    "ranking_r": "1",
    "eps_guard": "0.01",
    "exploration_rate": "0.02",
    "rung1_threshold": "0.02",
    "rung1_multiplier": "2.0",
}

# DEFAULT_V_TOLERANCE in code is 0.5; spec wants 0.05. Env variable is picked up
# by the wrapper so each benchmark run is monkey-patched on startup.
V_TOL_OVERRIDE = "0.05"

# Sample budgets per benchmark. Smoke is intentionally tiny; full matches spec.
SAMPLES = {
    "smoke": {"pg19": {"books": 1}, "niah": {"needles": 3, "trials": 5},
              "ruler": {"num_samples": 3}},
    "full":  {"pg19": {"books": 5}, "niah": {"needles": 3, "trials": 30},
              "ruler": {"num_samples": 50}},
}

CTX_LENGTHS = [4096, 8192, 16384, 32768]
CONFIGS = ["dense", "certified"]
BENCHES = ["pg19", "niah", "ruler"]


# ---------------------------------------------------------------------------
# Cell plan (numbered 01..18)
# ---------------------------------------------------------------------------

def build_cells() -> list[dict]:
    """Return the 18-cell plan in the spec's recommended order
    (shortest contexts first; within a context, all dense then all certified
    per the optimisation note in the spec)."""
    cells: list[dict] = []
    n = 0
    for ctx in CTX_LENGTHS:
        for config in CONFIGS:
            for bench in BENCHES:
                n += 1
                cells.append({
                    "idx": f"{n:02d}",
                    "bench": bench,
                    "ctx": ctx,
                    "config": config,
                })
    return cells


# ---------------------------------------------------------------------------
# Benchmark invocation helpers
# ---------------------------------------------------------------------------

def _runner_env(base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    # Keep the cache + venv paths stable across pod moves.
    env.setdefault("PYTHONUNBUFFERED", "1")
    # NOTE: An earlier version of this orchestrator set DOTCACHE_V_TOL=0.05
    # here in the hope that a PYTHONSTARTUP shim would patch the kernel
    # default — that shim never existed, so every cell silently used
    # v_tolerance=0.5 (FP16 values). The fix is to pass --v-tolerance and
    # --use-int4-values directly on each child CLI; see _cli_for_*.
    return env


def _cli_for_pg19(ctx: int, config: str, out_json: Path, smoke: bool) -> list[str]:
    s = SAMPLES["smoke" if smoke else "full"]["pg19"]
    args = [
        sys.executable, str(REPO / "benchmarks" / "paper" / "pg19_perplexity.py"),
        "--context", str(ctx),
        "--num-chunks", str(s["books"]),
        "--output", str(out_json),
        # Required by every bench (no silent default), even in dense-only:
        "--v-tolerance", V_TOL_OVERRIDE,
    ]
    if config == "dense":
        args.append("--dense-only")
    else:
        args += [
            "--use-int4-values",
            "--group-size", "16",
            "--tau-cov", CERT_FLAGS["tau_cov"],
            "--k-min", CERT_FLAGS["k_min"],
            "--k-max", CERT_FLAGS["k_max"],
            "--ranking-fallback",
            "--ranking-r", CERT_FLAGS["ranking_r"],
            "--score-consistency-check",
            "--eps-guard", CERT_FLAGS["eps_guard"],
            "--exploration-rate", CERT_FLAGS["exploration_rate"],
            "--rung1-threshold", CERT_FLAGS["rung1_threshold"],
            "--rung1-multiplier", CERT_FLAGS["rung1_multiplier"],
        ]
    return args


def _cli_for_niah(ctx: int, config: str, out_json: Path, smoke: bool) -> list[str]:
    s = SAMPLES["smoke" if smoke else "full"]["niah"]
    args = [
        sys.executable, str(REPO / "benchmarks" / "paper" / "niah.py"),
        "--contexts", str(ctx),
        "--needles", str(s["needles"]),
        "--output", str(out_json),
        # Required by every bench (no silent default):
        "--v-tolerance", V_TOL_OVERRIDE,
    ]
    # NIAH always runs dense + certified in one invocation (that's how the
    # harness is built). We'll dispatch differently: for --config dense the
    # orchestrator only records the "dense" half; for certified we get both
    # and compare. Pragmatically: on each cell we run NIAH once, and the
    # orchestrator pulls out the right half for the cell's JSON.
    if config == "certified":
        args += [
            "--use-int4-values",
            "--group-size", "16",
            "--tau-cov", CERT_FLAGS["tau_cov"],
            "--k-min", CERT_FLAGS["k_min"],
            "--k-max", CERT_FLAGS["k_max"],
            "--ranking-fallback",
            "--ranking-r", CERT_FLAGS["ranking_r"],
            "--score-consistency-check",
            "--eps-guard", CERT_FLAGS["eps_guard"],
            "--exploration-rate", CERT_FLAGS["exploration_rate"],
            "--rung1-threshold", CERT_FLAGS["rung1_threshold"],
            "--rung1-multiplier", CERT_FLAGS["rung1_multiplier"],
        ]
    return args


def _cli_for_ruler(ctx: int, config: str, out_json: Path, smoke: bool) -> list[str]:
    s = SAMPLES["smoke" if smoke else "full"]["ruler"]
    args = [
        sys.executable, str(REPO / "benchmarks" / "paper" / "ruler.py"),
        "--contexts", str(ctx),
        "--num-samples", str(s["num_samples"]),
        "--output", str(out_json),
        # Required by every bench (no silent default):
        "--v-tolerance", V_TOL_OVERRIDE,
    ]
    if config == "certified":
        args += [
            "--use-int4-values",
            "--group-size", "16",
            "--tau-cov", CERT_FLAGS["tau_cov"],
            "--k-min", CERT_FLAGS["k_min"],
            "--k-max", CERT_FLAGS["k_max"],
            "--ranking-fallback",
            "--ranking-r", CERT_FLAGS["ranking_r"],
            "--score-consistency-check",
            "--eps-guard", CERT_FLAGS["eps_guard"],
            "--exploration-rate", CERT_FLAGS["exploration_rate"],
            "--rung1-threshold", CERT_FLAGS["rung1_threshold"],
            "--rung1-multiplier", CERT_FLAGS["rung1_multiplier"],
        ]
    return args


BENCH_CLIS = {
    "pg19": _cli_for_pg19,
    "niah": _cli_for_niah,
    "ruler": _cli_for_ruler,
}


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(*args: str) -> tuple[int, str]:
    res = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=REPO,
    )
    return res.returncode, (res.stdout + res.stderr)


def git_sha() -> str:
    rc, out = _git("rev-parse", "--short", "HEAD")
    return out.strip() if rc == 0 else "unknown"


def commit_and_push(path: Path, message: str) -> None:
    rel = path.relative_to(REPO)
    _git("add", str(rel))
    rc, out = _git("commit", "-m", message)
    if rc != 0 and "nothing to commit" in out.lower():
        return
    _git("push", "origin", BRANCH)


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(cell: dict, smoke: bool = False, dry_run: bool = False) -> dict:
    bench, ctx, config = cell["bench"], cell["ctx"], cell["config"]
    ctx_k = ctx // 1024
    cell_json = OUT_DIR / f"{cell['idx']}_{bench}_{ctx_k}K_{config}.json"
    cell_log = OUT_DIR / f"{cell['idx']}_{bench}_{ctx_k}K_{config}.log"

    # Each benchmark writes a "native" JSON plus our wrapper JSON.
    native_json = cell_json.with_suffix(".native.json")

    cli = BENCH_CLIS[bench](ctx, config, native_json, smoke=smoke)

    plan = {
        "idx": cell["idx"],
        "bench": bench,
        "ctx": ctx,
        "config": config,
        "cmd": " ".join(shlex.quote(c) for c in cli),
        "out": str(cell_json),
    }
    if dry_run:
        return plan

    t0 = time.perf_counter()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"\n[{cell['idx']}] {bench} {ctx_k}K {config}  — starting")
    print(f"  cmd: {plan['cmd']}")
    print(f"  log: {cell_log}")

    env = _runner_env()
    with cell_log.open("w") as log_f:
        log_f.write(f"cmd: {plan['cmd']}\nstarted: {started}\n---\n")
        log_f.flush()
        rc = subprocess.call(cli, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    wall = time.perf_counter() - t0
    ended = dt.datetime.now(dt.timezone.utc).isoformat()
    print(f"[{cell['idx']}] {bench} {ctx_k}K {config}  exit={rc}  {wall/60:.1f} min")

    # Assemble the arXiv v1 JSON wrapper around whatever the benchmark emitted.
    native: Any = None
    if native_json.exists():
        try:
            native = json.loads(native_json.read_text())
        except Exception as e:
            native = {"_parse_error": str(e)}

    wrapped = {
        "benchmark": bench,
        "context_length": ctx,
        "config": config,
        "smoke": smoke,
        "quality": _extract_quality(bench, config, native),
        "system": _extract_system(bench, config, native),
        "native": native,
        "meta": {
            # The spec calls for meta-llama/Llama-3.1-8B-Instruct; the
            # benchmark scripts default to NousResearch's non-gated mirror
            # of the base (non-Instruct) weights because HF_TOKEN is unset
            # on this pod. Record what was actually run so the paper can
            # caveat honestly.
            "model": "NousResearch/Meta-Llama-3.1-8B",
            "model_quant": "int8-bitsandbytes",
            "hardware": _hw_tag(),
            "timestamp": ended,
            "started": started,
            "wall_seconds": wall,
            "git_sha": git_sha(),
            "exit_code": rc,
            "params": {
                "v_tol": V_TOL_OVERRIDE,
                **{k: v for k, v in CERT_FLAGS.items()},
            } if config == "certified" else {},
        },
    }
    cell_json.write_text(json.dumps(wrapped, indent=2))

    # Commit + push immediately so a pod reset costs at most this cell.
    commit_and_push(
        OUT_DIR,
        f"arXiv v1 sweep: cell {cell['idx']} {bench} {ctx_k}K {config} (exit={rc})",
    )
    return plan


def _extract_quality(bench: str, config: str, native: Any) -> dict:
    if not isinstance(native, dict):
        return {"_missing": True}
    if bench == "niah":
        # NIAH emits both dense_accuracy and certified_accuracy in the same file.
        if config == "dense":
            return {
                "metric": "accuracy",
                "value": native.get("dense_accuracy"),
                "n_samples": None,
            }
        return {
            "metric": "accuracy",
            "value": native.get("certified_accuracy"),
            "dense_value": native.get("dense_accuracy"),
            "delta": (
                (native.get("certified_accuracy") or 0)
                - (native.get("dense_accuracy") or 0)
            ),
            "critical_failures": native.get("critical_failures"),
        }
    if bench == "pg19":
        # pg19_perplexity.py's JSON has dense {perplexity} and cert {perplexity}
        q: dict[str, Any] = {"metric": "perplexity"}
        if config == "dense":
            q["value"] = native.get("dense", {}).get("perplexity")
        else:
            cert_ppl = native.get("certified", {}).get("perplexity")
            dense_ppl = native.get("dense", {}).get("perplexity")
            q["value"] = cert_ppl
            q["dense_value"] = dense_ppl
            if cert_ppl is not None and dense_ppl is not None:
                q["delta"] = cert_ppl - dense_ppl
        return q
    if bench == "ruler":
        summary = native.get("summary", {})
        overall_d = native.get("overall_dense")
        overall_c = native.get("overall_cert")
        q = {"metric": "accuracy", "summary": summary}
        if config == "dense":
            q["value"] = overall_d
        else:
            q["value"] = overall_c
            q["dense_value"] = overall_d
            if overall_c is not None and overall_d is not None:
                q["delta"] = overall_c - overall_d
            q["critical_failures"] = native.get("critical_failures")
        return q
    return {"_missing_extractor": bench}


def _extract_system(bench: str, config: str, native: Any) -> dict:
    if not isinstance(native, dict) or config == "dense":
        return {}
    out: dict[str, Any] = {}
    rfs = native.get("ranking_fallback_summary") or native.get("cert", {}).get("ranking_fallback_summary")
    if rfs:
        out["ranking_disagree_rate_r1"] = rfs.get("disagree_rate_r1")
        out["ranking_disagree_rate_r3"] = rfs.get("disagree_rate_r3")
        out["ranking_fallback_rate"] = rfs.get("fallback_rate")
        out["heads_total"] = rfs.get("heads_total")
    return out


def _hw_tag() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"{p.name} sm_{p.major}{p.minor}"
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true",
                    help="Run the shorter smoke variant of every cell (5 NIAH trials, 1 PG-19 book, 3 RULER samples)")
    ap.add_argument("--only", default=None,
                    help="Comma-separated cell filter, e.g. 'niah,4096,certified' or idx '04' or 'pg19'.")
    ap.add_argument("--from", dest="start_from", default=None,
                    help="Resume from cell idx (inclusive), e.g. '07'.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan without running anything.")
    ap.add_argument("--no-push", action="store_true",
                    help="Skip git commit+push between cells (for local iteration).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = build_cells()

    if args.only:
        toks = [t.strip().lower() for t in args.only.split(",") if t.strip()]
        def match(c):
            bag = {c["idx"], c["bench"], str(c["ctx"]), c["config"]}
            return all(t in bag or any(t in str(v).lower() for v in bag) for t in toks)
        cells = [c for c in cells if match(c)]

    if args.start_from:
        cells = [c for c in cells if int(c["idx"]) >= int(args.start_from)]

    if args.no_push:
        global commit_and_push
        def commit_and_push(path, message):  # type: ignore
            print(f"[no-push] skipping commit for {message}")

    print(f"Plan: {len(cells)} cell(s) (smoke={args.smoke})")
    for c in cells:
        print(f"  {c['idx']}  {c['bench']:<5} {c['ctx']//1024}K  {c['config']}")

    if args.dry_run:
        return 0

    for cell in cells:
        run_cell(cell, smoke=args.smoke, dry_run=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
