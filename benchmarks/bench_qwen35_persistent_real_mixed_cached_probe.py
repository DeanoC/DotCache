from __future__ import annotations

import argparse
import copy
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_real_mixed_probe import (
    _DEFAULT_POLICY_PATH,
    _build_prompt_text_inputs,
    _metric_sum,
    _resolve_prompt_records,
    real_mixed_probe_dotcache_config,
    real_mixed_probe_serving_config,
)
from dotcache.backends.metal.persistent_runtime import PersistentHybridRuntimeState
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheModelAdapter,
    Qwen35NativeHybridRuntimeState,
    _build_persistent_prefill_block_metadata,
    _extract_attention_subset_prefill_tensors,
    _install_attention_subset_logical_seq_length,
    _QWEN35_ATTENTION_SUBSET_LAYER_IDS_ATTR,
    _QWEN35_ATTENTION_SUBSET_LOGICAL_SEQ_LENGTH_ATTR,
    _replace_attention_subset_cache_with_placeholders,
    _run_dense_decode_step,
    _run_dense_prefill,
    _timed_call,
    load_qwen35_text_only_from_pretrained,
    transformers_available,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cached-prefill CUDA real-mixed probe.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="torch_cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--weight-quantization", choices=["none", "bnb_8bit"], default="none")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--prompt-files", nargs="*", default=[])
    parser.add_argument("--prompt-file-target-length", type=int, default=0)
    parser.add_argument("--max-k-comp-error-by-layer", default=None)
    parser.add_argument("--detailed-timing", action="store_true")
    parser.add_argument("--warmup-repeats", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--save-bundle-path", default=None)
    parser.add_argument("--load-bundle-path", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def _build_cached_prefill_bundle(
    *,
    model: Any,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    tokenizer: Any,
    prompt_text: str,
    prompt_length: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    input_ids, attention_mask = _build_prompt_text_inputs(
        tokenizer,
        device=device,
        prompt_text=prompt_text,
        prompt_length=int(prompt_length),
    )
    adapter.clear()
    adapter.set_mode("dense")
    adapter.maybe_apply_mps_serving_shortlist_heuristic(prompt_length=int(input_ids.shape[1]))
    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=device,
    )
    source_prefill_partition = adapter.partition_hybrid_state(prefill_outputs.past_key_values)
    attention_layer_ids = source_prefill_partition.token_growing_layer_ids
    extracted = _extract_attention_subset_prefill_tensors(prefill_outputs.past_key_values, attention_layer_ids)
    prefill_block_metadata_by_layer = _build_persistent_prefill_block_metadata(
        prefill_tensors=extracted,
        layer_ids=attention_layer_ids,
        dotcache_config=adapter.dotcache_config,
        block_size=int(adapter.persistent_serving_config.block_size),
        num_attention_heads=int(len(adapter.q_head_to_kv_head)),
    )
    _replace_attention_subset_cache_with_placeholders(prefill_outputs.past_key_values, attention_layer_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prompt_length": int(input_ids.shape[1]),
        "prefill_ms": float(prefill_ms),
        "next_input_ids": prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True),
        "post_handoff_past_key_values": copy.deepcopy(prefill_outputs.past_key_values),
        "prefill_tensors": extracted,
        "prefill_block_metadata_by_layer": prefill_block_metadata_by_layer,
        "linear_layer_ids": list(source_prefill_partition.fixed_resident_layer_ids),
        "serving_shortlist_heuristic_applied": bool(adapter.serving_shortlist_heuristic_applied),
    }


def _serialize_post_handoff_cache(cache: Any) -> dict[str, Any]:
    state = copy.deepcopy(getattr(cache, "__dict__", {}))
    state.pop("get_seq_length", None)
    state.pop("_dotcache_qwen35_attention_subset_original_get_seq_length", None)
    return {
        "cache_module": type(cache).__module__,
        "cache_class": type(cache).__name__,
        "cache_state": state,
        "logical_seq_length": getattr(cache, _QWEN35_ATTENTION_SUBSET_LOGICAL_SEQ_LENGTH_ATTR, None),
        "tracked_layer_ids": list(getattr(cache, _QWEN35_ATTENTION_SUBSET_LAYER_IDS_ATTR, ())),
    }


def _deserialize_post_handoff_cache(payload: dict[str, Any]) -> Any:
    module = importlib.import_module(str(payload["cache_module"]))
    cache_class = getattr(module, str(payload["cache_class"]))
    cache = cache_class.__new__(cache_class)
    cache.__dict__.update(copy.deepcopy(dict(payload["cache_state"])))
    logical_seq_length = payload.get("logical_seq_length")
    tracked_layer_ids = [int(value) for value in payload.get("tracked_layer_ids", [])]
    if logical_seq_length is not None and tracked_layer_ids:
        _install_attention_subset_logical_seq_length(
            cache,
            seq_length=int(logical_seq_length),
            layer_ids=tracked_layer_ids,
        )
    return cache


def _instantiate_cached_runtime(
    *,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    bundle: dict[str, Any],
) -> PersistentHybridRuntimeState:
    past_key_values = _deserialize_post_handoff_cache(bundle["post_handoff_cache_payload"])
    native_state = Qwen35NativeHybridRuntimeState.from_post_handoff_cache(
        past_key_values,
        adapter.model,
    )
    runtime_state = PersistentHybridRuntimeState.from_post_handoff_cache(
        native_state=native_state,
        prefill_tensors=bundle["prefill_tensors"],
        prefill_block_metadata_by_layer=bundle["prefill_block_metadata_by_layer"],
        linear_layer_ids=bundle["linear_layer_ids"],
        q_head_to_kv_head=adapter.q_head_to_kv_head,
        device=adapter.device,
        config=adapter.persistent_serving_config,
        dotcache_config=adapter.dotcache_config,
    )
    adapter.native_hybrid_runtime_state = native_state
    adapter.hybrid_dotcache_runtime_state = None
    adapter.persistent_hybrid_runtime_state = runtime_state
    return runtime_state


def _run_cached_repeat(
    *,
    model: Any,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    tokenizer: Any,
    bundle: dict[str, Any],
    decode_steps: int,
    prompt_family: str,
) -> dict[str, Any]:
    adapter.clear()
    adapter.set_mode("dotcache_attention_subset_persistent_experimental")
    adapter.configure_persistent_shortlist_policy_context(
        prompt_family=prompt_family,
        prompt_length=int(bundle["prompt_length"]),
    )
    runtime_state = _instantiate_cached_runtime(adapter=adapter, bundle=bundle)
    current_input_ids = bundle["next_input_ids"].clone()
    current_attention_mask = torch.cat(
        [
            bundle["attention_mask"],
            torch.ones((1, 1), dtype=bundle["attention_mask"].dtype, device=bundle["attention_mask"].device),
        ],
        dim=1,
    )
    cache_position = torch.tensor([bundle["input_ids"].shape[1]], dtype=torch.long, device=bundle["input_ids"].device)
    generated_ids: list[int] = []
    decode_ms_total = 0.0
    for _ in range(int(decode_steps)):
        generated_ids.append(int(current_input_ids.item()))
        outputs, step_ms = _timed_call(
            lambda: _run_dense_decode_step(
                model,
                decode_input_ids=current_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=runtime_state.model_past_key_values,
                cache_position=cache_position,
            ),
            device=current_input_ids.device,
        )
        decode_ms_total += float(step_ms)
        runtime_state.advance(outputs.past_key_values)
        current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=1,
        )
        cache_position = cache_position + 1
    runtime_summary = {}
    runtime_summary.update(adapter.per_layer_runtime_summary())
    runtime_summary.update(runtime_state.summary())
    return {
        "generated_ids": generated_ids,
        "decode_ms_per_step": float(decode_ms_total / max(int(decode_steps), 1)),
        "direct_m0_query_prep_ms_total": _metric_sum(
            runtime_summary,
            "persistent_full_attention_mixed_execution_direct_m0_query_prep_ms_total_by_layer",
        ),
        "direct_m0_gather_ms_total": _metric_sum(
            runtime_summary,
            "persistent_full_attention_mixed_execution_direct_m0_gather_ms_total_by_layer",
        ),
        "direct_m0_score_ms_total": _metric_sum(
            runtime_summary,
            "persistent_full_attention_mixed_execution_direct_m0_score_ms_total_by_layer",
        ),
        "exact_m3_score_ms_total": _metric_sum(
            runtime_summary,
            "persistent_full_attention_mixed_execution_exact_m3_score_ms_total_by_layer",
        ),
        "aux_exact_m3_score_ms_total": _metric_sum(
            runtime_summary,
            "persistent_full_attention_mixed_execution_aux_exact_m3_score_ms_total_by_layer",
        ),
        "final_mix_ms_total": _metric_sum(
            runtime_summary,
            "persistent_full_attention_mixed_execution_final_mix_ms_total_by_layer",
        ),
        "final_mix_logits_ms_total": _metric_sum(
            runtime_summary,
            "persistent_full_attention_mixed_execution_final_mix_logits_ms_total_by_layer",
        ),
        "final_mix_softmax_ms_total": _metric_sum(
            runtime_summary,
            "persistent_full_attention_mixed_execution_final_mix_softmax_ms_total_by_layer",
        ),
        "final_mix_value_ms_total": _metric_sum(
            runtime_summary,
            "persistent_full_attention_mixed_execution_final_mix_value_ms_total_by_layer",
        ),
    }


def _save_bundle(path: str, bundle: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(bundle)
    serializable.pop("post_handoff_past_key_values", None)
    torch.save(serializable, path)


def _load_bundle(path: str, *, map_location: str | None) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def _average_float(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return float(sum(float(record.get(key, 0.0)) for record in records) / len(records))


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.5 Cached Prefill Real Mixed Probe",
        "",
        f"- device: `{payload['config']['device']}`",
        f"- backend: `{payload['config']['backend']}`",
        f"- decode steps: `{payload['config']['decode_steps']}`",
        f"- warmup repeats: `{payload['config']['warmup_repeats']}`",
        f"- measured repeats: `{payload['config']['repeats']}`",
        "",
    ]
    for record in payload["records"]:
        lines.extend(
            [
                f"## `{record['case_tag']}`",
                "",
                f"- prompt length: `{record['prompt_length']}`",
                f"- one-time prefill ms: `{float(record['prefill_ms']):.4f}`",
                f"- avg decode ms/step: `{float(record['avg_decode_ms_per_step']):.4f}`",
                f"- avg direct-M0 gather ms/repeat: `{float(record['avg_direct_m0_gather_ms_total']):.4f}`",
                f"- avg direct-M0 score ms/repeat: `{float(record['avg_direct_m0_score_ms_total']):.4f}`",
                f"- avg final-mix ms/repeat: `{float(record['avg_final_mix_ms_total']):.4f}`",
                f"- avg final-mix logits ms/repeat: `{float(record['avg_final_mix_logits_ms_total']):.4f}`",
                f"- avg final-mix softmax ms/repeat: `{float(record['avg_final_mix_softmax_ms_total']):.4f}`",
                f"- avg final-mix value ms/repeat: `{float(record['avg_final_mix_value_ms_total']):.4f}`",
                f"- generated ids: `{record['generated_ids']}`",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise RuntimeError("transformers is required for the cached real mixed probe")
    max_k_comp_error_by_layer = (
        {
            int(layer_id): float(value)
            for layer_id, value in json.loads(str(args.max_k_comp_error_by_layer)).items()
        }
        if args.max_k_comp_error_by_layer
        else None
    )
    prompt_records = _resolve_prompt_records(
        manifest_path=str(args.manifest_path) if args.manifest_path else None,
        prompt_files=[str(value) for value in list(args.prompt_files)],
        prompt_file_target_length=int(args.prompt_file_target_length),
    )
    model, tokenizer = load_qwen35_text_only_from_pretrained(
        str(args.model_id),
        device=str(args.device) if args.device else None,
        torch_dtype=str(args.torch_dtype),
        weight_quantization=str(args.weight_quantization),
    )
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=real_mixed_probe_dotcache_config(),
        persistent_serving_config=real_mixed_probe_serving_config(
            policy_path=str(_DEFAULT_POLICY_PATH),
            detailed_timing=bool(args.detailed_timing),
            max_k_comp_error_by_layer=max_k_comp_error_by_layer,
        ),
        backend=str(args.backend),
    )

    records: list[dict[str, Any]] = []
    loaded_bundle = (
        _load_bundle(
            str(args.load_bundle_path),
            map_location=(str(args.device) if args.device else None),
        )
        if args.load_bundle_path
        else None
    )
    if loaded_bundle is not None and len(prompt_records) > 1:
        raise ValueError("--load-bundle-path only supports a single prompt record")
    for prompt_record in prompt_records:
        prompt_path = Path(str(prompt_record["prompt_file_path"]))
        if loaded_bundle is not None:
            bundle = dict(loaded_bundle)
        else:
            prompt_text = prompt_path.read_text(encoding="utf-8")
            bundle = _build_cached_prefill_bundle(
                model=model,
                adapter=adapter,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                prompt_length=int(prompt_record.get("prompt_length", 0)),
            )
            bundle["post_handoff_cache_payload"] = _serialize_post_handoff_cache(bundle["post_handoff_past_key_values"])
            bundle.pop("post_handoff_past_key_values", None)
        bundle["case_tag"] = str(prompt_record["case_tag"])
        bundle["prompt_file_path"] = str(prompt_path)
        if args.save_bundle_path and loaded_bundle is None:
            _save_bundle(str(args.save_bundle_path), bundle)
        for _ in range(max(int(args.warmup_repeats), 0)):
            _run_cached_repeat(
                model=model,
                adapter=adapter,
                tokenizer=tokenizer,
                bundle=bundle,
                decode_steps=int(args.decode_steps),
                prompt_family=str(prompt_record["case_tag"]),
            )
        measured: list[dict[str, Any]] = []
        for _ in range(max(int(args.repeats), 1)):
            measured.append(
                _run_cached_repeat(
                    model=model,
                    adapter=adapter,
                    tokenizer=tokenizer,
                    bundle=bundle,
                    decode_steps=int(args.decode_steps),
                    prompt_family=str(prompt_record["case_tag"]),
                )
            )
        generated_ids = [list(run["generated_ids"]) for run in measured]
        records.append(
            {
                "case_tag": str(prompt_record["case_tag"]),
                "prompt_file_path": str(prompt_path),
                "prompt_length": int(bundle["prompt_length"]),
                "prefill_ms": float(bundle["prefill_ms"]),
                "repeats": int(len(measured)),
                "warmup_repeats": int(max(int(args.warmup_repeats), 0)),
                "generated_ids": generated_ids[0] if generated_ids else [],
                "generated_ids_stable_across_repeats": len({tuple(run) for run in generated_ids}) <= 1,
                "avg_decode_ms_per_step": _average_float(measured, "decode_ms_per_step"),
                "avg_direct_m0_query_prep_ms_total": _average_float(measured, "direct_m0_query_prep_ms_total"),
                "avg_direct_m0_gather_ms_total": _average_float(measured, "direct_m0_gather_ms_total"),
                "avg_direct_m0_score_ms_total": _average_float(measured, "direct_m0_score_ms_total"),
                "avg_exact_m3_score_ms_total": _average_float(measured, "exact_m3_score_ms_total"),
                "avg_aux_exact_m3_score_ms_total": _average_float(measured, "aux_exact_m3_score_ms_total"),
                "avg_final_mix_ms_total": _average_float(measured, "final_mix_ms_total"),
                "avg_final_mix_logits_ms_total": _average_float(measured, "final_mix_logits_ms_total"),
                "avg_final_mix_softmax_ms_total": _average_float(measured, "final_mix_softmax_ms_total"),
                "avg_final_mix_value_ms_total": _average_float(measured, "final_mix_value_ms_total"),
                "measured_runs": measured,
            }
        )

    payload = {
            "config": {
            "model_id": str(args.model_id),
            "device": str(args.device) if args.device else None,
            "backend": str(args.backend),
            "torch_dtype": str(args.torch_dtype),
            "weight_quantization": str(args.weight_quantization),
            "decode_steps": int(args.decode_steps),
            "manifest_path": str(args.manifest_path) if args.manifest_path else None,
            "prompt_file_count": len(prompt_records),
            "warmup_repeats": int(max(int(args.warmup_repeats), 0)),
            "repeats": int(max(int(args.repeats), 1)),
            "save_bundle_path": None if args.save_bundle_path is None else str(args.save_bundle_path),
            "load_bundle_path": None if args.load_bundle_path is None else str(args.load_bundle_path),
            "full_attention_mixed_mode_detailed_timing": bool(args.detailed_timing),
        },
        "records": records,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(_render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
