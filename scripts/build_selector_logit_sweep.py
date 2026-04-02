#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotcache.selector_baselines import (
    adjust_linear_selector_model_logits,
    load_linear_selector_model,
    save_linear_selector_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build bias-shifted runtime selector artifacts for page-mix sensitivity sweeps.")
    parser.add_argument("--artifact", required=True, help="Base linear_selector_model.json artifact path.")
    parser.add_argument("--output-dir", required=True, help="Directory where adjusted artifacts and manifest should be written.")
    parser.add_argument("--target-candidate", default="M3/affine/4/float16", help="Candidate whose logit bias should be shifted.")
    parser.add_argument("--offsets", type=float, nargs="+", required=True, help="Logit offsets to apply to the target candidate.")
    return parser.parse_args()


def _offset_slug(value: float) -> str:
    formatted = f"{float(value):+.2f}"
    return formatted.replace("+", "p").replace("-", "m").replace(".", "d")


def main() -> int:
    args = parse_args()
    base_model = load_linear_selector_model(args.artifact)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []
    for offset in args.offsets:
        adjusted = adjust_linear_selector_model_logits(
            base_model,
            candidate_logit_offsets={str(args.target_candidate): float(offset)},
        )
        variant_name = f"offset_{_offset_slug(float(offset))}"
        artifact_path = output_dir / f"{variant_name}.json"
        save_linear_selector_model(adjusted, artifact_path)
        manifest_entries.append(
            {
                "variant": variant_name,
                "target_candidate": str(args.target_candidate),
                "logit_offset": float(offset),
                "artifact_path": str(artifact_path),
            }
        )

    manifest = {
        "base_artifact": str(args.artifact),
        "target_candidate": str(args.target_candidate),
        "variants": manifest_entries,
    }
    manifest_path = output_dir / "selector_logit_sweep_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
