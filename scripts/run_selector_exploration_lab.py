#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotcache.selector_exploration import run_selector_exploration_lab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Apple-first selector exploration lab.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "selector_exploration" / "apple_local_lab.json"),
        help="JSON config describing suite root, strategy list, calibration, and optional serving smoke.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where results and reports should be written.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload = run_selector_exploration_lab(config=config, output_dir=args.output_dir)
    print(payload["json_path"])
    print(payload["markdown_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
