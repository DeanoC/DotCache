"""Run-manifest helper for paper_v1 benchmark output directories.

Each paper_v1 results directory carries a `run_manifest.json` at its root,
recording git/code/hardware provenance plus a per-cell sha256 list. This
module is the canonical generator/validator so all bench scripts emit the
same schema. The schema closes the audit's Mismatch 6 ("output JSON has
no cache-config provenance") at the directory level — the per-cell JSON
already carries cache_config from benchmarks/paper/_provenance.py.

Manifest schema (v1):
{
  "manifest_version": 1,
  "git_sha":         "<rev-parse HEAD, with -dirty if uncommitted>",
  "branch":          "<git branch --show-current>",
  "created_utc":     "<ISO 8601>",
  "created_by":      "<env USER or 'unknown'>",
  "paper_tex_sha":   "<sha256 of Certified_Quantised_Attention.tex if present, else null>",
  "hardware":        {"gpu", "cuda", "torch", "triton", "transformers"},
  "dotcache_config": {full §7 knob block — same shape as cache_config_dict()},
  "cells":           [{"name": "<basename>", "sha256": "<hex>", "bytes": <int>}, ...],
}

Usage from a runner script:
    from benchmarks._manifest import write_initial_manifest, refresh_cells
    write_initial_manifest(out_dir, dotcache_config={...})
    # ... run cells ...
    refresh_cells(out_dir)

CLI:
    python -m benchmarks._manifest --validate <dir>
    python -m benchmarks._manifest --refresh <dir>
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

MANIFEST_NAME = "run_manifest.json"
MANIFEST_VERSION = 1


def _git_sha_with_dirty(repo: Path) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_branch(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repo, stderr=subprocess.DEVNULL,
        ).decode().strip() or "detached"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _hardware_info() -> dict[str, str]:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpu": "unknown",
        "cuda": "unknown",
        "torch": "unknown",
        "triton": "unknown",
        "transformers": "unknown",
    }
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda or "unknown"
    except Exception:
        pass
    try:
        import triton
        info["triton"] = triton.__version__
    except Exception:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def _paper_tex_sha(repo: Path) -> str | None:
    """The paper .tex lives outside this branch; return its sha if a copy
    is present, else null. The audit doc is the canonical source.
    """
    candidates = [
        repo / "Certified_Quantised_Attention.tex",
        repo / "docs" / "Certified_Quantised_Attention.tex",
    ]
    for p in candidates:
        if p.exists():
            return _file_sha256(p)
    return None


def _scan_cells(out_dir: Path) -> list[dict[str, Any]]:
    """Per-cell sha256 + size for every JSON file under the directory,
    excluding the manifest itself."""
    cells: list[dict[str, Any]] = []
    for p in sorted(out_dir.rglob("*.json")):
        if p.name == MANIFEST_NAME:
            continue
        try:
            cells.append({
                "name": str(p.relative_to(out_dir)),
                "sha256": _file_sha256(p),
                "bytes": p.stat().st_size,
            })
        except OSError:
            continue
    return cells


def write_initial_manifest(
    out_dir: Path | str,
    *,
    dotcache_config: dict[str, Any] | None = None,
    repo: Path | str | None = None,
    notes: str = "",
) -> Path:
    """Create (or overwrite) run_manifest.json with metadata + empty cells.

    Call this once at the start of a run; refresh_cells() updates the cells
    list as new JSONs are written. dotcache_config is the §7 knob block
    (typically the same dict that benchmarks/paper/_provenance.cache_config_dict
    builds from CLI args), recorded once at the directory level so a
    downstream auditor can prove every cell shares the same config.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(repo) if repo is not None else _find_repo_root(out_dir)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "git_sha": _git_sha_with_dirty(repo),
        "branch": _git_branch(repo),
        "created_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "created_by": os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        "paper_tex_sha": _paper_tex_sha(repo),
        "hardware": _hardware_info(),
        "dotcache_config": dotcache_config or {},
        "notes": notes,
        "cells": _scan_cells(out_dir),
    }
    manifest_path = out_dir / MANIFEST_NAME
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest_path


def refresh_cells(out_dir: Path | str) -> Path:
    """Re-scan the directory's JSON files and rewrite the cells list.

    Other manifest fields (git_sha, hardware, dotcache_config, etc.) are
    preserved from the initial write — only cells are updated.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing — call write_initial_manifest first"
        )
    with manifest_path.open() as f:
        manifest = json.load(f)
    manifest["cells"] = _scan_cells(out_dir)
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest_path


def validate_manifest(out_dir: Path | str) -> dict[str, Any]:
    """Re-compute every cell's sha256 and check it matches the recorded value.

    Returns a report dict with 'ok' bool and a 'mismatches' list. Raises
    if the manifest itself is missing or malformed.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / MANIFEST_NAME
    with manifest_path.open() as f:
        manifest = json.load(f)
    recorded = {c["name"]: c["sha256"] for c in manifest.get("cells", [])}
    actual = {c["name"]: c["sha256"] for c in _scan_cells(out_dir)}
    mismatches: list[dict[str, Any]] = []
    for name, sha in recorded.items():
        if name not in actual:
            mismatches.append({"name": name, "issue": "missing_on_disk"})
        elif actual[name] != sha:
            mismatches.append({
                "name": name, "issue": "sha_mismatch",
                "recorded": sha, "actual": actual[name],
            })
    for name, sha in actual.items():
        if name not in recorded:
            mismatches.append({
                "name": name, "issue": "untracked_on_disk", "actual": sha,
            })
    return {
        "manifest_path": str(manifest_path),
        "ok": len(mismatches) == 0,
        "n_recorded_cells": len(recorded),
        "n_disk_cells": len(actual),
        "mismatches": mismatches,
    }


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="paper_v1 run-manifest tool")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--init", metavar="DIR",
                     help="Create initial manifest in DIR")
    grp.add_argument("--refresh", metavar="DIR",
                     help="Re-scan DIR and update the cells list")
    grp.add_argument("--validate", metavar="DIR",
                     help="Re-hash cells and verify against the manifest")
    ap.add_argument("--config-json", default=None,
                    help="Path to a JSON file with the dotcache_config block "
                         "(used with --init).")
    ap.add_argument("--notes", default="", help="Free-text notes (used with --init)")
    args = ap.parse_args(argv)

    if args.init:
        cfg = {}
        if args.config_json:
            with open(args.config_json) as f:
                cfg = json.load(f)
        path = write_initial_manifest(args.init, dotcache_config=cfg, notes=args.notes)
        print(f"wrote {path}")
        return 0
    if args.refresh:
        path = refresh_cells(args.refresh)
        print(f"refreshed {path}")
        return 0
    if args.validate:
        report = validate_manifest(args.validate)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
