from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture and sweep real Qwen3.5 DeltaNet state samples across selected layers.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--prompt-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 12, 22])
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 4, 3])
    parser.add_argument("--renorm-intervals", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--state-kinds", choices=["recurrent", "conv"], nargs="+", default=["recurrent", "conv"])
    parser.add_argument("--output-dir", default="benchmarks/results/statecache_sweeps")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _run_json_lines(cmd: list[str], cwd: Path) -> list[dict[str, object]]:
    completed = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    records: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        records.append(json.loads(stripped))
    return records


def _recommend_mode(summary_records: list[dict[str, object]]) -> dict[str, object]:
    m0_records = [record for record in summary_records if record.get("mode") == "M0"]
    if not m0_records:
        return {
            "recommended_mode": "M3",
            "recommended_bits": None,
            "recommended_renorm_interval": 0,
            "recommendation_reason": "no_m0_records",
        }

    best_by_bits = {
        int(record["bits"]): record
        for record in sorted(m0_records, key=lambda item: (int(item["bits"]), float(item["best_readout_error"])))
    }
    bit8 = best_by_bits.get(8)
    bit4 = best_by_bits.get(4)
    bit3 = best_by_bits.get(3)

    if bit8 is not None and float(bit8["best_readout_error"]) <= 0.15:
        return {
            "recommended_mode": "M0",
            "recommended_bits": 8,
            "recommended_renorm_interval": int(bit8["best_readout_renorm_interval"]),
            "recommendation_reason": "8b_readout_within_safe_band",
        }
    if bit4 is not None and float(bit4["best_readout_error"]) <= 0.3:
        return {
            "recommended_mode": "M0",
            "recommended_bits": 4,
            "recommended_renorm_interval": int(bit4["best_readout_renorm_interval"]),
            "recommendation_reason": "4b_readout_within_experimental_band",
        }
    if bit3 is not None and float(bit3["best_readout_error"]) <= 0.3:
        return {
            "recommended_mode": "M0",
            "recommended_bits": 3,
            "recommended_renorm_interval": int(bit3["best_readout_renorm_interval"]),
            "recommendation_reason": "3b_readout_within_experimental_band",
        }
    if bit8 is not None:
        return {
            "recommended_mode": "M0",
            "recommended_bits": 8,
            "recommended_renorm_interval": int(bit8["best_readout_renorm_interval"]),
            "recommendation_reason": "8b_best_available_m0_fallback",
        }
    best_any = min(m0_records, key=lambda item: float(item["best_readout_error"]))
    return {
        "recommended_mode": "M0",
        "recommended_bits": int(best_any["bits"]),
        "recommended_renorm_interval": int(best_any["best_readout_renorm_interval"]),
        "recommendation_reason": "best_available_m0_fallback",
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    python_bin = repo_root / ".venv" / "bin" / "python"
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_records: list[dict[str, object]] = []
    recommendation_records: list[dict[str, object]] = []
    for state_kind in args.state_kinds:
        for layer_id in args.layers:
            sample_base = output_dir / f"qwen35_layer{layer_id}_{state_kind}.npz"
            inspect_cmd = [
                str(python_bin),
                "benchmarks/bench_qwen35_deltanet_state_inspect.py",
                "--model-id",
                args.model_id,
                "--backend",
                args.backend,
                "--torch-dtype",
                args.torch_dtype,
                "--target-prompt-lengths",
                str(args.prompt_length),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--save-state-sample",
                str(sample_base),
                "--sample-layer-id",
                str(layer_id),
                "--sample-state-kind",
                state_kind,
            ]
            if args.device is not None:
                inspect_cmd.extend(["--device", args.device])
            inspect_records = _run_json_lines(inspect_cmd, repo_root)
            capture_record = next(
                record
                for record in inspect_records
                if record.get("prompt_mode") == "exact_length" and int(record.get("prompt_length", -1)) == args.prompt_length
            )
            captured_sample_path = str(capture_record["captured_state_sample_path"])

            sim_cmd = [
                str(python_bin),
                "benchmarks/bench_state_cache_sim.py",
                "--captured-sample",
                captured_sample_path,
                "--modes",
                "M0",
                "M3",
                "--bits",
                *[str(bit) for bit in args.bits],
                "--renorm-intervals",
                *[str(interval) for interval in args.renorm_intervals],
                "--seed",
                str(args.seed),
            ]
            sim_records = _run_json_lines(sim_cmd, repo_root)
            summary_records = [record for record in sim_records if record.get("benchmark") == "state_cache_sim_summary"]
            for summary in summary_records:
                summary["benchmark"] = "qwen35_statecache_real_sweep"
                summary["model_id"] = args.model_id
                summary["backend"] = args.backend
                summary["device"] = args.device
                summary["torch_dtype"] = args.torch_dtype
                summary["sample_state_kind"] = state_kind
                summary["sample_layer_id"] = layer_id
                summary["sample_prompt_length"] = args.prompt_length
                summary["sample_decode_steps"] = args.max_new_tokens
                aggregate_records.append(summary)
                print(json.dumps(summary, sort_keys=True), flush=True)

            recommendation = {
                "benchmark": "qwen35_statecache_real_sweep_recommendation",
                "model_id": args.model_id,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "sample_state_kind": state_kind,
                "sample_layer_id": layer_id,
                "sample_prompt_length": args.prompt_length,
                "sample_decode_steps": args.max_new_tokens,
            }
            recommendation.update(_recommend_mode(summary_records))
            recommendation_records.append(recommendation)
            print(json.dumps(recommendation, sort_keys=True), flush=True)

    report_path = output_dir / f"qwen35_statecache_prompt{args.prompt_length}_steps{args.max_new_tokens}_summary.json"
    report_payload = {
        "summary_records": aggregate_records,
        "recommendation_records": recommendation_records,
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "benchmark": "qwen35_statecache_real_sweep_report",
                "report_path": str(report_path),
                "record_count": len(aggregate_records),
                "recommendation_count": len(recommendation_records),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
