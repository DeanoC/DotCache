"""Shared cache-config provenance helper for paper benches.

Every paper-bench output JSON (pg19, niah, ruler, longbench) embeds the
block returned by ``cache_config_dict()`` so that a downstream auditor can
prove which quantisation config produced the numbers without re-reading
the code at the time of the run. This closes the gap that let the Apr 17–24
results silently use the wrong v_tolerance — see
docs/paper_code_audit_20260424.md for the original bug.

The block is intentionally small and focused on the load-bearing fields
from paper §7. Hardware / git-sha / paper-tex-sha live in the run-level
manifest (benchmarks/_manifest.py), not in every cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from typing import Any


def _git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _quantisation_mode(use_int4_values: bool, group_size: int) -> str:
    keys = "asymmetric_int8_keys"  # set by Step 1; will be the only path
    values = f"int4_g{group_size}_values" if use_int4_values else "fp16_values"
    return f"{keys}+{values}"


def cache_config_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Build the cache-config provenance block from parsed CLI args.

    All paper benches use a common subset of CLI flags
    (--v-tolerance, --use-int4-values, --group-size, plus the §7 knobs);
    this helper consumes that namespace and returns a dict suitable for
    embedding into the cell's output JSON.
    """
    use_int4 = bool(getattr(args, "use_int4_values", False))
    group_size = int(getattr(args, "group_size", 16))
    config = {
        "v_tolerance": float(args.v_tolerance),
        "quantization_mode": _quantisation_mode(use_int4, group_size),
        "asymmetric_keys": True,  # Step 1 makes this the only path
        "use_int4_values": use_int4,
        "group_size": group_size if use_int4 else None,
        "score_consistency_check": bool(getattr(args, "score_consistency_check", True)),
        "tau_cov": float(getattr(args, "tau_cov", 0.0)) or None,
        "k_min": int(getattr(args, "k_min", 2)),
        "k_max": getattr(args, "k_max", None),
        "ranking_fallback": bool(getattr(args, "ranking_fallback", False)),
        "ranking_r": int(getattr(args, "ranking_r", 1)),
        "ranking_fallback_mode": getattr(args, "ranking_fallback_mode", "full"),
        "eps_guard": float(getattr(args, "eps_guard", 0.01)),
        "exploration_rate": float(getattr(args, "exploration_rate", 0.0)),
        "rung1_threshold": float(getattr(args, "rung1_threshold", 0.02)),
        "rung1_multiplier": float(getattr(args, "rung1_multiplier", 2.0)),
        "code_sha": _git_sha(),
    }
    config["dotcache_config_hash"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    return config


def add_paper_cache_args(parser: argparse.ArgumentParser) -> None:
    """Attach the three new paper-port flags to a bench's argparse.

    Used by pg19/niah/ruler/longbench so the flag spelling is identical
    across benches. v_tolerance is REQUIRED — no silent default.
    """
    parser.add_argument(
        "--v-tolerance", type=float, required=True,
        help="INT4-vs-FP16 value-format threshold (paper §7: 0.05). REQUIRED — "
             "no silent default. The kernel raises if this isn't carried through.",
    )
    parser.add_argument(
        "--use-int4-values", action="store_true",
        help="Use INT4 per-group values (paper §3.1/§7). Without this flag, "
             "values stay FP16 (legacy ad-hoc-bench behaviour).",
    )
    parser.add_argument(
        "--group-size", type=int, default=16,
        help="INT4 value group size (paper §7: 16). Ignored unless "
             "--use-int4-values is set.",
    )


def add_paper_section7_args(parser: argparse.ArgumentParser) -> None:
    """Attach the §7 Certified-config knobs. Used by longbench (which was
    missing them); pg19/niah/ruler already declare these directly."""
    parser.add_argument("--tau-cov", type=float, default=0.0,
                        help="Adaptive K* cumulative-mass threshold (paper §7: 0.995)")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=None,
                        help="Adaptive K* upper clamp")
    parser.add_argument("--ranking-fallback", action="store_true")
    parser.add_argument("--ranking-r", type=int, default=1)
    parser.add_argument("--ranking-fallback-mode", default="full",
                        choices=["full", "measure"])
    parser.add_argument("--score-consistency-check", action="store_true")
    parser.add_argument("--eps-guard", type=float, default=0.01)
    parser.add_argument("--exploration-rate", type=float, default=0.0)
    parser.add_argument("--rung1-threshold", type=float, default=0.02)
    parser.add_argument("--rung1-multiplier", type=float, default=2.0)
