"""Aggregate LongBench paper-sweep JSONs into a single paper-ready table.

Mirrors benchmarks/paper/aggregate_ruler.py for the LongBench sweep layout:

    <dir>/
        eps0_4k.json      calibrated_4k.json
        eps0_8k.json      calibrated_8k.json
        eps0_16k.json     calibrated_16k.json   (optional)

Emits per-subtask dense/cert/Δ table, per-category overall, paired criticals,
and optional CSV.

Usage:
  python benchmarks/paper/aggregate_longbench.py \
    --dir benchmarks/results/longbench_paper_20260417 \
    --csv benchmarks/results/longbench_paper_20260417/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CONFIGS = [
    ("eps0_4k", "ε=0 no-skip", 4096),
    ("calibrated_4k", "calibrated", 4096),
    ("eps0_8k", "ε=0 no-skip", 8192),
    ("calibrated_8k", "calibrated", 8192),
    ("eps0_16k", "ε=0 no-skip", 16384),
    ("calibrated_16k", "calibrated", 16384),
]

SUBTASKS = [
    "narrativeqa", "qasper", "multifieldqa_en",
    "hotpotqa", "2wikimqa",
    "gov_report", "qmsum", "multi_news",
    "trec", "triviaqa", "samsum",
    "lcc", "repobench-p",
]

CATEGORIES = {
    "single_doc_qa": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multi_doc_qa":  ["hotpotqa", "2wikimqa"],
    "summarisation": ["gov_report", "qmsum", "multi_news"],
    "few_shot":      ["trec", "triviaqa", "samsum"],
    "code":          ["lcc", "repobench-p"],
}


def load_config(results_dir: Path, tag: str) -> dict | None:
    path = results_dir / f"{tag}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def summarise_config(data: dict) -> dict:
    by = {s: {"dense_sum": 0.0, "cert_sum": 0.0, "crit": 0, "n": 0}
          for s in SUBTASKS}
    for r in data["results"]:
        s = r["subtask"]
        if s not in by:
            continue
        bkt = by[s]
        bkt["dense_sum"] += r["dense_score"]
        bkt["cert_sum"] += r["cert_score"]
        bkt["n"] += 1
        if r.get("critical"):
            bkt["crit"] += 1
    out = {}
    for s, bkt in by.items():
        n = bkt["n"]
        if n == 0:
            continue
        dense = bkt["dense_sum"] / n
        cert = bkt["cert_sum"] / n
        out[s] = {"dense": dense, "cert": cert, "delta": cert - dense,
                  "crit": bkt["crit"], "n": n}
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
        print(f"# Missing configs (skipped / not yet complete): {missing}\n")

    # Per-subtask table — scores are 0-100 scaled (×100) for readability.
    print(f"\n## Per-subtask scores (×100), dir={args.dir.name}\n")
    present_configs = [(t, l, c) for (t, l, c) in CONFIGS if t in all_configs]
    headers = ["subtask"]
    for _, label, ctx in present_configs:
        headers += [f"{label[:3]}{ctx//1024}K d", "c", "Δ", "crit"]
    rows = []
    for s in SUBTASKS:
        row = [s]
        for tag, _, _ in present_configs:
            if s in all_configs[tag]:
                x = all_configs[tag][s]
                row += [f"{x['dense']*100:5.1f}", f"{x['cert']*100:5.1f}",
                        f"{x['delta']*100:+5.1f}", str(x['crit'])]
            else:
                row += ["-", "-", "-", "-"]
        rows.append(row)
    print_table(rows, headers)

    # Category overall
    print("\n## Category overall (unweighted mean across subtasks in category)\n")
    cat_headers = ["category"]
    for _, label, ctx in present_configs:
        cat_headers += [f"{label[:3]}{ctx//1024}K d", "c", "Δ"]
    cat_rows = []
    for cat, subs in CATEGORIES.items():
        row = [cat]
        for tag, _, _ in present_configs:
            vals = [all_configs[tag][s] for s in subs if s in all_configs[tag]]
            if not vals:
                row += ["-", "-", "-"]
                continue
            dense = sum(v["dense"] for v in vals) / len(vals)
            cert = sum(v["cert"] for v in vals) / len(vals)
            row += [f"{dense*100:5.1f}", f"{cert*100:5.1f}",
                    f"{(cert-dense)*100:+5.1f}"]
        cat_rows.append(row)
    print_table(cat_rows, cat_headers)

    # Overall per-config
    print("\n## Overall (all subtasks pooled, sample-weighted)\n")
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
            tag, f"{dense*100:5.1f}", f"{cert*100:5.1f}",
            f"{(cert-dense)*100:+5.1f}", f"{crit}/{total_n}", str(total_n),
        ])
    print_table(rows, ["config", "dense", "cert", "Δ", "crit", "n"])

    if args.csv is not None:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["config", "ctx", "subtask", "dense", "cert", "delta",
                        "crit", "n"])
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
