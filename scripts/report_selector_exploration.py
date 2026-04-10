#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotcache.selector_exploration import render_selector_exploration_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render selector exploration lab outputs.")
    parser.add_argument("--input", required=True, help="Path to selector_exploration_results.json.")
    parser.add_argument("--markdown-output", required=True, help="Where to write the markdown report.")
    parser.add_argument("--json-output", default=None, help="Optional normalized JSON copy.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    markdown = render_selector_exploration_markdown(payload)
    Path(args.markdown_output).write_text(markdown + "\n", encoding="utf-8")
    if args.json_output is not None:
        Path(args.json_output).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(args.markdown_output)
    if args.json_output is not None:
        print(args.json_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
