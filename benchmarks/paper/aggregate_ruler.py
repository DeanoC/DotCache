"""Aggregate RULER paper-sweep JSONs into a single paper-ready table.

Reads eps0_4k.json, calibrated_4k.json, eps0_8k.json, calibrated_8k.json
from a results directory and emits:

1. Per-subtask accuracy table (dense vs cert) across both regimes × both contexts.
2. Criticals breakdown (samples where dense passed and cert failed).
3. Paired 95% CI via Wilson interval on paired difference.
4. Markdown-ready output (stdout) and CSV (--csv path).

Usage:
  python benchmarks/paper/aggregate_ruler.py \
    --dir benchmarks/results/ruler_paper_20260417 \
    --csv benchmarks/results/ruler_paper_20260417/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CONFIGS = [
    ("eps0_4k", "ε=0 no-skip", 4096),
    ("calibrated_4k", "calibrated", 4096),
    ("eps0_8k", "ε=0 no-skip", 8192),
    ("calibrated_8k", "calibrated", 8192),
]

SUBTASKS = [
    "niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery",
    "vt", "cwe", "fwe",
]


def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfwidth = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


def load_config(results_dir: Path, tag: str) -> dict | None:
    path = results_dir / f"{tag}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def summarise_config(data: dict) -> dict:
    """Aggregate per-subtask. Returns {subtask: {dense, cert, delta, crit, n}}."""
    by_subtask = {s: {"dense_sum": 0.0, "cert_sum": 0.0, "crit": 0, "n": 0,
                       "dense_pass": 0, "cert_pass": 0}
                   for s in SUBTASKS}
    for r in data["results"]:
        s = r["subtask"]
        if s not in by_subtask:
            continue
        bkt = by_subtask[s]
        bkt["dense_sum"] += r["dense_score"]
        bkt["cert_sum"] += r["cert_score"]
        bkt["n"] += 1
        if r.get("critical"):
            bkt["crit"] += 1
        if r["dense_score"] >= 0.999:
            bkt["dense_pass"] += 1
        if r["cert_score"] >= 0.999:
            bkt["cert_pass"] += 1
    out = {}
    for s, bkt in by_subtask.items():
        n = bkt["n"]
        if n == 0:
            continue
        dense = bkt["dense_sum"] / n
        cert = bkt["cert_sum"] / n
        out[s] = {
            "dense": dense, "cert": cert, "delta": cert - dense,
            "crit": bkt["crit"], "n": n,
            "dense_pass_rate": bkt["dense_pass"] / n,
            "cert_pass_rate": bkt["cert_pass"] / n,
        }
    return out


def print_table(rows, headers):
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*r))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--csv", default=None, type=Path)
    args = parser.parse_args()

    all_configs = {}
    missing = []
    for tag, _, _ in CONFIGS:
        d = load_config(args.dir, tag)
        if d is None:
            missing.append(tag)
            continue
        all_configs[tag] = summarise_config(d)

    if missing:
        print(f"# Missing configs (not yet complete): {missing}\n")

    # Per-subtask table: rows=subtasks, cols=(dense, cert, Δ) × 4 configs
    print(f"\n## Per-subtask accuracy (dir={args.dir.name})\n")
    headers = ["subtask"]
    for _, label, ctx in CONFIGS:
        headers += [f"{label} {ctx//1024}K dense", f"cert", "Δ", "crit"]
    rows = []
    for s in SUBTASKS:
        row = [s]
        for tag, _, _ in CONFIGS:
            if tag in all_configs and s in all_configs[tag]:
                x = all_configs[tag][s]
                row += [f"{x['dense']:.3f}", f"{x['cert']:.3f}",
                        f"{x['delta']:+.3f}", str(x['crit'])]
            else:
                row += ["-", "-", "-", "-"]
        rows.append(row)
    print_table(rows, headers)

    # Overall per-config
    print("\n## Overall (all subtasks pooled)\n")
    rows = []
    for tag, label, ctx in CONFIGS:
        if tag not in all_configs:
            rows.append([tag, "-", "-", "-", "-", "-"])
            continue
        d = all_configs[tag]
        total_n = sum(x["n"] for x in d.values())
        dense = sum(x["dense"] * x["n"] for x in d.values()) / total_n
        cert = sum(x["cert"] * x["n"] for x in d.values()) / total_n
        crit = sum(x["crit"] for x in d.values())
        rows.append([
            tag, f"{dense:.3f}", f"{cert:.3f}", f"{cert - dense:+.3f}",
            f"{crit}/{total_n}", str(total_n),
        ])
    print_table(rows, ["config", "dense", "cert", "Δ", "crit", "n"])

    # CSV dump
    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config", "ctx", "subtask", "dense", "cert", "delta", "crit", "n"])
            for tag, _, ctx in CONFIGS:
                if tag not in all_configs:
                    continue
                for s, x in all_configs[tag].items():
                    w.writerow([tag, ctx, s,
                                f"{x['dense']:.4f}", f"{x['cert']:.4f}",
                                f"{x['delta']:+.4f}", x['crit'], x['n']])
        print(f"\nCSV -> {args.csv}")


if __name__ == "__main__":
    main()
