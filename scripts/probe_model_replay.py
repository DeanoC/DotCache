#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.integrations.llama import (
    _ensure_attention_mask,
    _normalize_input_ids,
    _run_dense_greedy_decode,
    _run_dotcache_decode_inputs,
    resolve_hf_auth_kwargs,
    transformers_available,
)
from dotcache.integrations.qwen2 import Qwen2DotCacheHarness
from dotcache.integrations.llama import LlamaDotCacheHarness
from dotcache.model_kv_cache import ModelPagedKVCache
from dotcache.page_cache import PreparedPageCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe per-step and per-layer replay drift for one DotCache model.")
    parser.add_argument("--family", choices=["llama", "qwen2"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--default-mode-k", choices=["M0", "M1", "M2", "M3", "T3"], default="M0")
    parser.add_argument("--default-mode-v", choices=["M0", "M1", "M3", "T3"], default="M0")
    parser.add_argument("--quant-scheme-k", choices=["affine", "lut", "sketch", "turbo3"], default="affine")
    parser.add_argument("--quant-scheme-v", choices=["affine", "lut", "turbo3"], default="affine")
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    return parser.parse_args()


def _build_exact_length_inputs(harness, *, prompt_unit: str, prompt_length: int) -> tuple[Any, Any]:
    if harness.tokenizer is None:
        raise ValueError("tokenizer is unavailable for exact-length prompt construction")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")

    tokenizer = harness.tokenizer
    unit_ids = tokenizer(prompt_unit, add_special_tokens=False)["input_ids"]
    if not unit_ids:
        raise ValueError("prompt_unit tokenized to an empty sequence")

    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))
    while len(token_ids) < prompt_length:
        token_ids.extend(int(token_id) for token_id in unit_ids)
    token_ids = token_ids[:prompt_length]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=harness.adapter.device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=harness.adapter.device)
    return input_ids, attention_mask


def _build_harness(args: argparse.Namespace):
    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **resolve_hf_auth_kwargs())
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    dotcache_config = DotCacheConfig(
        head_dim=head_dim,
        group_size=args.group_size,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        default_mode_k=args.default_mode_k,
        default_mode_v=args.default_mode_v,
        quant_scheme_k=args.quant_scheme_k,
        quant_scheme_v=args.quant_scheme_v,
        tokens_per_page=args.tokens_per_page,
    )
    harness_cls = LlamaDotCacheHarness if args.family == "llama" else Qwen2DotCacheHarness
    return harness_cls.from_pretrained(
        args.model_id,
        dotcache_config,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )


def _topk_summary(entries: list[dict[str, Any]], key: str, limit: int = 5) -> list[dict[str, Any]]:
    return sorted(entries, key=lambda entry: float(entry[key]), reverse=True)[:limit]


def probe_replay(args: argparse.Namespace) -> dict[str, Any]:
    harness = _build_harness(args)
    model = harness.model
    adapter = harness.adapter
    input_ids, attention_mask = _build_exact_length_inputs(
        harness,
        prompt_unit=args.prompt_unit,
        prompt_length=args.prompt_length,
    )
    input_ids = _normalize_input_ids(input_ids, device=adapter.device)
    attention_mask = _ensure_attention_mask(input_ids, attention_mask, device=adapter.device)

    dense_result = _run_dense_greedy_decode(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.decode_steps + 1,
        capture=True,
    )

    replay_cache = ModelPagedKVCache(
        config=adapter.dotcache_config,
        num_hidden_layers=model.config.num_hidden_layers,
        num_attention_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_key_value_heads,
        backend=adapter.backend,
        cache=PreparedPageCache(),
    )
    for layer_idx, (layer_keys, layer_values) in enumerate(dense_result["prefill_layers"]):
        if torch.is_tensor(layer_keys):
            replay_cache.ingest_prefill_cache_torch(layer_idx, layer_keys, layer_values)
        else:
            replay_cache.ingest_prefill_cache(layer_idx, layer_keys, layer_values)

    entries: list[dict[str, Any]] = []
    step_stats: dict[int, dict[str, Any]] = {}
    layer_stats: dict[int, dict[str, Any]] = defaultdict(lambda: {"max_abs_error": 0.0, "max_rel_error": 0.0, "count": 0})
    replay_context_max_abs = 0.0
    replay_context_max_rel = 0.0

    for step_records in dense_result["capture_records"]:
        for record in step_records:
            replay_cache.append_step(
                record.layer_id,
                record.key_states[:, None, :],
                record.value_states[:, None, :],
                record.token_index,
            )
            replay_context = replay_cache.decode_layer(record.layer_id, record.query_states, adapter.q_head_to_kv_head)
            delta = np.abs(replay_context - record.context_states)
            denom = np.maximum(np.abs(record.context_states), 1e-8)
            max_abs = float(np.max(delta))
            max_rel = float(np.max(delta / denom))
            replay_context_max_abs = max(replay_context_max_abs, max_abs)
            replay_context_max_rel = max(replay_context_max_rel, max_rel)
            entry = {
                "step_index": int(record.step_index),
                "layer_id": int(record.layer_id),
                "token_index": int(record.token_index),
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
            }
            entries.append(entry)
            existing_step = step_stats.get(record.step_index)
            if existing_step is None or max_abs >= float(existing_step["max_abs_error"]):
                step_stats[record.step_index] = dict(entry)
            layer_state = layer_stats[record.layer_id]
            layer_state["max_abs_error"] = max(float(layer_state["max_abs_error"]), max_abs)
            layer_state["max_rel_error"] = max(float(layer_state["max_rel_error"]), max_rel)
            layer_state["count"] = int(layer_state["count"]) + 1

    dotcache_teacher_forced = _run_dotcache_decode_inputs(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefill_layers=dense_result["prefill_layers"],
        decode_inputs=dense_result["decode_inputs"],
    )
    dense_logits = np.stack(dense_result["step_logits"], axis=0) if dense_result["step_logits"] else np.zeros((0, 1))
    dotcache_logits = (
        np.stack(dotcache_teacher_forced["step_logits"], axis=0) if dotcache_teacher_forced["step_logits"] else np.zeros((0, 1))
    )
    if dense_logits.size == 0:
        max_abs_logit_drift = 0.0
        max_rel_logit_drift = 0.0
        token_agreement = 1.0
    else:
        logit_delta = np.abs(dotcache_logits - dense_logits)
        logit_denom = np.maximum(np.abs(dense_logits), 1e-8)
        max_abs_logit_drift = float(np.max(logit_delta))
        max_rel_logit_drift = float(np.max(logit_delta / logit_denom))
        dense_argmax = np.argmax(dense_logits[:, 0, :], axis=-1)
        dotcache_argmax = np.argmax(dotcache_logits[:, 0, :], axis=-1)
        token_agreement = float(np.mean((dense_argmax == dotcache_argmax).astype(np.float32)))

    layer_rows = [
        {
            "layer_id": int(layer_id),
            "max_abs_error": float(stats["max_abs_error"]),
            "max_rel_error": float(stats["max_rel_error"]),
            "count": int(stats["count"]),
        }
        for layer_id, stats in layer_stats.items()
    ]
    layer_rows.sort(key=lambda row: row["layer_id"])
    step_rows = [step_stats[index] for index in sorted(step_stats)]

    return {
        "family": args.family,
        "model_id": args.model_id,
        "backend": args.backend,
        "device": args.device or adapter.device.type,
        "torch_dtype": args.torch_dtype,
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": int(args.decode_steps),
        "default_mode_k": args.default_mode_k,
        "default_mode_v": args.default_mode_v,
        "quant_scheme_k": args.quant_scheme_k,
        "quant_scheme_v": args.quant_scheme_v,
        "tokens_per_page": args.tokens_per_page,
        "replay_context_max_abs_error": replay_context_max_abs,
        "replay_context_max_rel_error": replay_context_max_rel,
        "teacher_forced_logit_max_abs_error": max_abs_logit_drift,
        "teacher_forced_logit_max_rel_error": max_rel_logit_drift,
        "teacher_forced_token_agreement_rate": token_agreement,
        "worst_step_by_abs_error": max(step_rows, key=lambda row: row["max_abs_error"], default=None),
        "worst_layer_by_abs_error": max(layer_rows, key=lambda row: row["max_abs_error"], default=None),
        "top_entries_by_abs_error": _topk_summary(entries, "max_abs_error"),
        "per_step": step_rows,
        "per_layer": layer_rows,
    }


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Replay Probe: {result['model_id']}",
        "",
        f"- Family: `{result['family']}`",
        f"- Backend: `{result['backend']}` on `{result['device']}`",
        f"- Prompt length: `{result['prompt_length']}`",
        f"- Decode steps: `{result['decode_steps']}`",
        f"- Replay max abs error: `{_format_float(result['replay_context_max_abs_error'])}`",
        f"- Replay max rel error: `{_format_float(result['replay_context_max_rel_error'])}`",
        f"- Teacher-forced logit max abs error: `{_format_float(result['teacher_forced_logit_max_abs_error'])}`",
        f"- Teacher-forced token agreement: `{_format_float(result['teacher_forced_token_agreement_rate'])}`",
        "",
        "## Worst Step",
        json.dumps(result["worst_step_by_abs_error"], sort_keys=True),
        "",
        "## Worst Layer",
        json.dumps(result["worst_layer_by_abs_error"], sort_keys=True),
        "",
        "## Top Entries",
    ]
    lines.extend(json.dumps(entry, sort_keys=True) for entry in result["top_entries_by_abs_error"])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("probe_model_replay.py requires the optional transformers dependencies")
    result = probe_replay(args)
    if args.format == "markdown":
        print(render_markdown(result))
        return
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
