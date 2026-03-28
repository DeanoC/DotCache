from __future__ import annotations

import argparse
import json

from dotcache.kv_quant_registry import get_kv_quant_baseline, list_kv_quant_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit the DotCache KV-quant comparison matrix scaffold.")
    parser.add_argument("--baseline-keys", nargs="*", default=[])
    parser.add_argument("--output-format", choices=["jsonl", "pretty"], default="jsonl")
    return parser.parse_args()


def _selected_rows(keys: list[str]) -> tuple[dict[str, object], ...]:
    if not keys:
        return tuple(row.to_dict() for row in list_kv_quant_baselines())
    return tuple(get_kv_quant_baseline(key).to_dict() for key in keys)


def main() -> None:
    args = parse_args()
    for row in _selected_rows(args.baseline_keys):
        if args.output_format == "pretty":
            print(json.dumps(row, indent=2, sort_keys=True))
        else:
            print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
