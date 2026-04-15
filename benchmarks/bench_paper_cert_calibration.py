"""Certificate calibration: measure certified bound values across decode steps.

Collects per-layer, per-step residual certificate data (beta_upper, delta_upper)
to characterise how tight the certified bounds are in practice. Reports
distributions and certified-exit rates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_serving_policy_compare import (
    _DEFAULT_POLICY_PATH,
    _build_prompt_text_inputs,
    _resolve_prompt_records,
)
from benchmarks.bench_qwen35_persistent_real_mixed_probe import (
    real_mixed_probe_dotcache_config,
    real_mixed_probe_serving_config,
)
from dotcache.backends.metal.persistent_types import PersistentServingConfig
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheModelAdapter,
    load_qwen35_text_only_from_pretrained,
    _run_dense_prefill,
    _run_dense_decode_step,
    _clone_qwen35_past_key_values,
)
from dotcache.integrations.llama import _timed_call


_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "benchmarks/manifests/qwen35_stage9_repo_public_validation_broader_20260414.json"
)

_LANES = [
    {"name": "spherical_only", "enable_interval_bound": False, "enable_ellipsoidal_bound": False},
    {"name": "interval", "enable_interval_bound": True, "enable_ellipsoidal_bound": False},
    {"name": "interval_ellip", "enable_interval_bound": True, "enable_ellipsoidal_bound": True},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Certificate calibration benchmark")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", default="torch_cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--lanes", nargs="*", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_lanes = (
        [lane for lane in _LANES if lane["name"] in args.lanes]
        if args.lanes else list(_LANES)
    )
    prompt_records = _resolve_prompt_records(
        manifest_path=args.manifest_path or str(_DEFAULT_MANIFEST),
        prompt_files=[],
        prompt_file_target_length=0,
    )
    print(f"Certificate calibration: {len(prompt_records)} cases, {len(active_lanes)} lanes, {args.decode_steps} steps")

    print(f"Loading model {args.model_id} ...")
    model, tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id, device=args.device, torch_dtype=args.torch_dtype,
    )
    dotcache_config = real_mixed_probe_dotcache_config()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(model, dotcache_config)
    device = next(model.parameters()).device

    all_results: list[dict[str, Any]] = []

    for lane in active_lanes:
        print(f"\n--- Lane: {lane['name']} ---")
        serving_config = real_mixed_probe_serving_config(policy_path=str(_DEFAULT_POLICY_PATH))
        serving_config.enable_interval_bound = bool(lane["enable_interval_bound"])
        serving_config.enable_ellipsoidal_bound = bool(lane["enable_ellipsoidal_bound"])

        for prompt_record in prompt_records:
            case_tag = str(prompt_record["case_tag"])
            prompt_text = Path(str(prompt_record["prompt_file_path"])).read_text(encoding="utf-8")
            input_ids, attention_mask = _build_prompt_text_inputs(
                tokenizer, device=device, prompt_text=prompt_text,
                prompt_length=int(prompt_record.get("prompt_length", 0)),
            )

            # Prefill (dense)
            adapter.set_mode("dense")
            adapter.clear()
            prefill_outputs, _ = _timed_call(
                lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
                device=device,
            )

            # Load into persistent runtime
            pkv_clone = _clone_qwen35_past_key_values(prefill_outputs.past_key_values)
            adapter.persistent_serving_config = serving_config
            adapter.clear()
            adapter.load_attention_subset_persistent_prefill_cache(pkv_clone)
            adapter.set_mode("dotcache_attention_subset_persistent_experimental")
            adapter.configure_persistent_shortlist_policy_context(
                prompt_family=None, prompt_length=int(input_ids.shape[1]),
            )
            runtime_state = adapter.require_persistent_hybrid_runtime_state()

            # Decode steps — collect certificates after each step
            current_input_ids = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1,
            )
            cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=device)

            step_certs: list[dict[str, Any]] = []

            for step_idx in range(args.decode_steps):
                outputs, step_ms = _timed_call(
                    lambda: _run_dense_decode_step(
                        model,
                        decode_input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        past_key_values=runtime_state.model_past_key_values,
                        cache_position=cache_position,
                    ),
                    device=device,
                )
                runtime_state.advance(outputs.past_key_values)

                # Collect per-layer certificate data
                for layer_id in sorted(runtime_state.full_attention.layers.keys()):
                    fa_state = runtime_state.full_attention.layers[layer_id]
                    cert = fa_state.last_residual_certificate
                    first_cert = fa_state.last_first_certified_stop
                    total_blocks = int(len(fa_state.block_token_starts))

                    step_certs.append({
                        "step": step_idx,
                        "layer_id": int(layer_id),
                        "total_blocks": total_blocks,
                        "beta_upper": float(cert["beta_upper"]) if cert else None,
                        "delta_upper": float(cert["delta_upper"]) if cert else None,
                        "processed_blocks": int(cert.get("processed_block_count", 0)) if cert else None,
                        "certified_can_stop": bool(cert.get("certified_can_stop", False)) if cert else False,
                        "mandatory_complete": bool(cert.get("mandatory_complete", True)) if cert else False,
                        "first_cert_stop_blocks": (
                            int(first_cert.get("processed_block_count", 0)) if first_cert else None
                        ),
                    })

                # Advance decode state
                current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                current_attention_mask = torch.cat(
                    [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=device)],
                    dim=1,
                )
                cache_position = cache_position + 1

            # Aggregate
            valid_certs = [c for c in step_certs if c["beta_upper"] is not None]
            beta_values = [c["beta_upper"] for c in valid_certs]
            delta_values = [c["delta_upper"] for c in valid_certs]
            cert_stop_count = sum(1 for c in valid_certs if c["certified_can_stop"])
            first_cert_blocks = [c["first_cert_stop_blocks"] for c in valid_certs if c["first_cert_stop_blocks"] is not None]
            processed_blocks = [c["processed_blocks"] for c in valid_certs if c["processed_blocks"] is not None]
            total_blocks_list = [c["total_blocks"] for c in valid_certs]

            result = {
                "case_tag": case_tag,
                "lane": lane["name"],
                "decode_steps": args.decode_steps,
                "certificate_records": len(valid_certs),
                "cert_stop_fraction": cert_stop_count / max(len(valid_certs), 1),
                "beta_upper_min": float(np.min(beta_values)) if beta_values else None,
                "beta_upper_median": float(np.median(beta_values)) if beta_values else None,
                "beta_upper_p90": float(np.percentile(beta_values, 90)) if beta_values else None,
                "beta_upper_p99": float(np.percentile(beta_values, 99)) if beta_values else None,
                "beta_upper_max": float(np.max(beta_values)) if beta_values else None,
                "delta_upper_min": float(np.min(delta_values)) if delta_values else None,
                "delta_upper_median": float(np.median(delta_values)) if delta_values else None,
                "delta_upper_p90": float(np.percentile(delta_values, 90)) if delta_values else None,
                "delta_upper_p99": float(np.percentile(delta_values, 99)) if delta_values else None,
                "delta_upper_max": float(np.max(delta_values)) if delta_values else None,
                "processed_blocks_median": float(np.median(processed_blocks)) if processed_blocks else None,
                "total_blocks_median": float(np.median(total_blocks_list)) if total_blocks_list else None,
                "first_cert_stop_blocks_median": float(np.median(first_cert_blocks)) if first_cert_blocks else None,
                "first_cert_stop_blocks_max": float(np.max(first_cert_blocks)) if first_cert_blocks else None,
                "per_step_certs": step_certs,
            }
            all_results.append(result)

            summary_str = (
                f"cert_stop={result['cert_stop_fraction']:.1%}, "
                f"beta med={result['beta_upper_median']:.4f} p90={result['beta_upper_p90']:.4f} max={result['beta_upper_max']:.4f}, "
                f"delta med={result['delta_upper_median']:.6f} p90={result['delta_upper_p90']:.6f} max={result['delta_upper_max']:.6f}, "
                f"1st_cert_stop_blocks med={result['first_cert_stop_blocks_median']}"
            ) if result["beta_upper_median"] is not None else "no certificate data"
            print(f"  [{lane['name']}] {case_tag}: {summary_str}")

            # Reset for next case
            adapter.set_mode("dense")
            adapter.clear()

    # Print lane-level summary
    print("\n=== Lane Summaries ===")
    for lane in active_lanes:
        lane_recs = [r for r in all_results if r["lane"] == lane["name"]]
        betas = [r["beta_upper_median"] for r in lane_recs if r["beta_upper_median"] is not None]
        deltas = [r["delta_upper_median"] for r in lane_recs if r["delta_upper_median"] is not None]
        cert_stops = [r["cert_stop_fraction"] for r in lane_recs]
        print(
            f"  {lane['name']}: "
            f"avg_cert_stop={np.mean(cert_stops):.1%}, "
            f"beta_median_across_cases={np.median(betas):.4f}, "
            f"delta_median_across_cases={np.median(deltas):.6f}"
        )

    # Output JSON (without per_step_certs for summary, full in separate key)
    output = {
        "model_id": args.model_id,
        "device": args.device,
        "decode_steps": args.decode_steps,
        "case_count": len(prompt_records),
        "lane_count": len(active_lanes),
        "summaries": [{k: v for k, v in r.items() if k != "per_step_certs"} for r in all_results],
        "per_step_detail": [{
            "case_tag": r["case_tag"],
            "lane": r["lane"],
            "certs": r["per_step_certs"],
        } for r in all_results],
    }

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nJSON -> {args.output_json}")

    if args.output_md:
        lines = ["# Certificate Calibration Results\n"]
        lines.append(f"Model: {args.model_id}, Device: {args.device}, Steps: {args.decode_steps}\n")
        lines.append("| Lane | Case | cert_stop% | beta_med | beta_p90 | beta_max | delta_med | delta_p90 | delta_max | 1st_cert_blks |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in output["summaries"]:
            if r["beta_upper_median"] is not None:
                lines.append(
                    f"| {r['lane']} | {r['case_tag']} "
                    f"| {r['cert_stop_fraction']:.1%} "
                    f"| {r['beta_upper_median']:.4f} | {r['beta_upper_p90']:.4f} | {r['beta_upper_max']:.4f} "
                    f"| {r['delta_upper_median']:.6f} | {r['delta_upper_p90']:.6f} | {r['delta_upper_max']:.6f} "
                    f"| {r['first_cert_stop_blocks_median']} |"
                )
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_md, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"MD   -> {args.output_md}")


if __name__ == "__main__":
    main()
