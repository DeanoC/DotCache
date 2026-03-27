#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig

from dotcache.attention_runtime import decode_step_with_page_logits
from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.integrations.llama import (
    LlamaDotCacheHarness,
    _ensure_attention_mask,
    _normalize_input_ids,
    _run_dense_greedy_decode,
)
from dotcache.integrations.qwen2 import Qwen2DotCacheHarness
from dotcache.page_cache import PreparedPageCache


VARIANT_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    ("D1", "exact", "exact"),
    ("D2", "M0", "exact"),
    ("D3", "M3", "exact"),
    ("D4", "exact", "M0"),
    ("D5", "M0", "M0"),
    ("D6", "M3", "M0"),
)


@dataclass(slots=True)
class ProbeBundle:
    step_logits: list[np.ndarray]
    prompt_length: int
    q_head_to_kv_head: np.ndarray
    layer_scales: list[float]
    prefill_keys: list[np.ndarray]
    prefill_values: list[np.ndarray]
    step_queries: list[list[np.ndarray]]
    step_keys: list[list[np.ndarray]]
    step_values: list[list[np.ndarray]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe offline attention score fidelity for DotCache K/V modes.")
    parser.add_argument("--family", choices=["llama", "qwen2"], required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--quant-scheme-k", choices=["affine", "lut", "sketch", "turbo3"], default="affine")
    parser.add_argument("--quant-scheme-v", choices=["affine", "lut", "turbo3"], default="affine")
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--layer-rescue-sweep", action="store_true")
    parser.add_argument("--target-layers", type=int, nargs="*", default=None)
    parser.add_argument("--kv-group-rescue-sweep", action="store_true")
    parser.add_argument(
        "--combo-rescue",
        action="append",
        default=[],
        help="Comma-separated rescue spec set like layer:0,layer:27:kv:1. May be repeated; each argument is one combo policy.",
    )
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--include-rows", action="store_true")
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
    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    dotcache_config = DotCacheConfig(
        head_dim=head_dim,
        group_size=args.group_size,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        default_mode_k="M0",
        default_mode_v="M0",
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


def _layer_scaling(layer: Any) -> float:
    attention = layer.self_attn
    if hasattr(attention, "base_attention"):
        attention = attention.base_attention
    return float(attention.scaling)


def _to_numpy_prefill(values: Any) -> np.ndarray:
    if torch.is_tensor(values):
        return values.detach().to(dtype=torch.float32).cpu().numpy()[0]
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 4:
        return array[0]
    return array


def _build_probe_bundle(args: argparse.Namespace) -> tuple[Any, ProbeBundle]:
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

    prefill_keys = [_to_numpy_prefill(layer_keys) for layer_keys, _ in dense_result["prefill_layers"]]
    prefill_values = [_to_numpy_prefill(layer_values) for _, layer_values in dense_result["prefill_layers"]]
    capture_records = dense_result["capture_records"]
    num_layers = model.config.num_hidden_layers
    step_queries = [[] for _ in range(num_layers)]
    step_keys = [[] for _ in range(num_layers)]
    step_values = [[] for _ in range(num_layers)]
    for step_records in capture_records:
        for record in step_records:
            step_queries[record.layer_id].append(np.asarray(record.query_states, dtype=np.float32))
            step_keys[record.layer_id].append(np.asarray(record.key_states, dtype=np.float32))
            step_values[record.layer_id].append(np.asarray(record.value_states, dtype=np.float32))

    layer_scales = [_layer_scaling(layer) for layer in model.model.layers[: model.config.num_hidden_layers]]
    step_logits = [np.asarray(logits, dtype=np.float32) for logits in dense_result["step_logits"]]
    dense_result["prefill_outputs"] = None
    gc.collect()
    if torch.cuda.is_available() and adapter.device.type == "cuda":
        torch.cuda.empty_cache()

    bundle = ProbeBundle(
        step_logits=step_logits,
        prompt_length=int(input_ids.shape[1]),
        q_head_to_kv_head=adapter.q_head_to_kv_head.copy(),
        layer_scales=layer_scales,
        prefill_keys=prefill_keys,
        prefill_values=prefill_values,
        step_queries=step_queries,
        step_keys=step_keys,
        step_values=step_values,
    )
    return harness, bundle


def _rank_correlation(exact: np.ndarray, test: np.ndarray) -> float:
    if exact.size <= 1:
        return 1.0
    exact_rank = np.argsort(np.argsort(exact))
    test_rank = np.argsort(np.argsort(test))
    exact_centered = exact_rank.astype(np.float64) - float(np.mean(exact_rank))
    test_centered = test_rank.astype(np.float64) - float(np.mean(test_rank))
    denom = np.linalg.norm(exact_centered) * np.linalg.norm(test_centered)
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(exact_centered, test_centered) / denom)


def _cosine_similarity(exact: np.ndarray, test: np.ndarray) -> float:
    denom = np.linalg.norm(exact) * np.linalg.norm(test)
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(exact, test) / denom)


def _softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    logits = logits - float(np.max(logits))
    exp = np.exp(logits)
    return (exp / np.maximum(np.sum(exp), 1e-12)).astype(np.float32, copy=False)


def _entropy(weights: np.ndarray) -> float:
    probs = np.clip(np.asarray(weights, dtype=np.float64), 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _kl_divergence(exact: np.ndarray, test: np.ndarray) -> float:
    exact_probs = np.clip(np.asarray(exact, dtype=np.float64), 1e-12, 1.0)
    test_probs = np.clip(np.asarray(test, dtype=np.float64), 1e-12, 1.0)
    return float(np.sum(exact_probs * (np.log(exact_probs) - np.log(test_probs))))


def _topk_overlap(exact: np.ndarray, test: np.ndarray, *, top_k: int) -> float:
    if exact.size == 0:
        return 1.0
    k = min(int(top_k), int(exact.size))
    if k <= 0:
        return 1.0
    exact_indices = set(np.argpartition(exact, -k)[-k:].tolist())
    test_indices = set(np.argpartition(test, -k)[-k:].tolist())
    return float(len(exact_indices & test_indices) / k)


def _top_source_age_bucket(top_index: int, token_count: int) -> str:
    age = int(token_count) - 1 - int(top_index)
    if age <= 127:
        return "recent"
    if age <= 511:
        return "near_mid"
    if age <= 2047:
        return "mid"
    return "far"


def _metric_row(
    *,
    variant_id: str,
    layer_id: int,
    step_index: int,
    q_head_id: int,
    kv_head_id: int,
    token_count: int,
    exact_logits: np.ndarray,
    test_logits: np.ndarray,
    exact_weights: np.ndarray,
    test_weights: np.ndarray,
    exact_output: np.ndarray,
    test_output: np.ndarray,
    top_k: int,
) -> dict[str, Any]:
    delta = test_logits - exact_logits
    output_delta = test_output - exact_output
    top_src_exact = int(np.argmax(exact_logits))
    top_src_test = int(np.argmax(test_logits))
    return {
        "variant_id": variant_id,
        "layer_id": int(layer_id),
        "step_index": int(step_index),
        "q_head_id": int(q_head_id),
        "kv_head_id": int(kv_head_id),
        "token_count": int(token_count),
        "score_max_abs": float(np.max(np.abs(delta))),
        "score_mean_abs": float(np.mean(np.abs(delta))),
        "score_rmse": float(np.sqrt(np.mean(np.square(delta), dtype=np.float64))),
        "score_cosine": _cosine_similarity(exact_logits, test_logits),
        "score_rank_corr": _rank_correlation(exact_logits, test_logits),
        "score_top1_match": bool(top_src_exact == top_src_test),
        "score_topk_overlap": _topk_overlap(exact_logits, test_logits, top_k=top_k),
        "attn_kl": _kl_divergence(exact_weights, test_weights),
        "attn_entropy_exact": _entropy(exact_weights),
        "attn_entropy_test": _entropy(test_weights),
        "attn_entropy_delta": _entropy(test_weights) - _entropy(exact_weights),
        "top_src_exact": top_src_exact,
        "top_src_test": top_src_test,
        "top_src_age_bucket": _top_source_age_bucket(top_src_exact, token_count),
        "output_max_abs": float(np.max(np.abs(output_delta))),
        "output_mean_abs": float(np.mean(np.abs(output_delta))),
        "output_rmse": float(np.sqrt(np.mean(np.square(output_delta), dtype=np.float64))),
    }


def _summary_from_rows(rows: list[dict[str, Any]], *, variant_id: str) -> dict[str, Any]:
    if not rows:
        return {
            "variant_id": variant_id,
            "row_count": 0,
        }
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket_rows[str(row["top_src_age_bucket"])].append(row)

    def _mean(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows], dtype=np.float64))

    def _match_rate(key: str) -> float:
        return float(np.mean([1.0 if bool(row[key]) else 0.0 for row in rows], dtype=np.float64))

    per_layer: dict[int, dict[str, Any]] = {}
    for layer_id in sorted({int(row["layer_id"]) for row in rows}):
        layer_rows = [row for row in rows if int(row["layer_id"]) == layer_id]
        per_layer[layer_id] = {
            "layer_id": layer_id,
            "row_count": len(layer_rows),
            "score_top1_match_rate": float(np.mean([1.0 if row["score_top1_match"] else 0.0 for row in layer_rows])),
            "score_topk_overlap_mean": float(np.mean([float(row["score_topk_overlap"]) for row in layer_rows])),
            "attn_kl_mean": float(np.mean([float(row["attn_kl"]) for row in layer_rows])),
            "score_max_abs_max": float(np.max([float(row["score_max_abs"]) for row in layer_rows])),
            "output_rmse_mean": float(np.mean([float(row["output_rmse"]) for row in layer_rows])),
        }

    top_layers = sorted(
        per_layer.values(),
        key=lambda row: (float(row["attn_kl_mean"]), -float(row["score_top1_match_rate"])),
        reverse=True,
    )[:5]
    by_bucket = {
        bucket: {
            "row_count": len(bucket_entries),
            "score_top1_match_rate": float(np.mean([1.0 if row["score_top1_match"] else 0.0 for row in bucket_entries])),
            "score_topk_overlap_mean": float(np.mean([float(row["score_topk_overlap"]) for row in bucket_entries])),
            "attn_kl_mean": float(np.mean([float(row["attn_kl"]) for row in bucket_entries])),
        }
        for bucket, bucket_entries in sorted(bucket_rows.items())
    }
    worst_rows = sorted(rows, key=lambda row: float(row["attn_kl"]), reverse=True)[:8]
    return {
        "variant_id": variant_id,
        "row_count": len(rows),
        "score_top1_match_rate": _match_rate("score_top1_match"),
        "score_topk_overlap_mean": _mean("score_topk_overlap"),
        "score_cosine_mean": _mean("score_cosine"),
        "score_rank_corr_mean": _mean("score_rank_corr"),
        "attn_kl_mean": _mean("attn_kl"),
        "attn_entropy_delta_mean": _mean("attn_entropy_delta"),
        "score_max_abs_max": float(np.max([float(row["score_max_abs"]) for row in rows])),
        "output_rmse_mean": _mean("output_rmse"),
        "top_layers_by_attn_kl": top_layers,
        "by_top_source_age_bucket": by_bucket,
        "worst_rows_by_attn_kl": worst_rows,
    }


def _compose_rescue_rows(
    *,
    d5_rows: list[dict[str, Any]],
    d6_rows: list[dict[str, Any]],
    selector,
) -> list[dict[str, Any]]:
    d6_lookup = {
        (
            int(row["layer_id"]),
            int(row["step_index"]),
            int(row["q_head_id"]),
            int(row["kv_head_id"]),
        ): row
        for row in d6_rows
    }
    rows: list[dict[str, Any]] = []
    for row in d5_rows:
        key = (
            int(row["layer_id"]),
            int(row["step_index"]),
            int(row["q_head_id"]),
            int(row["kv_head_id"]),
        )
        if selector(row):
            rows.append(d6_lookup[key])
        else:
            rows.append(row)
    return rows


def _parse_combo_rescue_spec(spec: str) -> tuple[str, tuple[int, int | None]]:
    parts = spec.split(":")
    if len(parts) not in {2, 4}:
        raise ValueError(f"invalid combo rescue spec: {spec}")
    if parts[0] != "layer":
        raise ValueError(f"combo rescue must start with layer:<id>: {spec}")
    layer_id = int(parts[1])
    if len(parts) == 2:
        return spec, (layer_id, None)
    if parts[2] != "kv":
        raise ValueError(f"combo rescue KV form must be layer:<id>:kv:<kv_id>: {spec}")
    kv_head_id = int(parts[3])
    return spec, (layer_id, kv_head_id)


def _encode_pages(
    values: np.ndarray,
    *,
    config: DotCacheConfig,
    kind: str,
    mode: str,
    layer_id: int,
    kv_head_id: int,
) -> list[Any]:
    pages: list[Any] = []
    token_count = int(values.shape[0])
    for token_start in range(0, token_count, config.tokens_per_page):
        token_end = min(token_count, token_start + config.tokens_per_page)
        page_values = values[token_start:token_end]
        pages.append(
            encode_page(
                page_values,
                config,
                kind=kind,
                layer_id=layer_id,
                kv_head_id=kv_head_id,
                token_start=token_start,
                mode=mode,
                build_runtime_metadata=False,
            )
        )
    return pages


def _split_page_logits(exact_logits: np.ndarray, page_token_counts: list[int]) -> list[np.ndarray]:
    logits: list[np.ndarray] = []
    offset = 0
    for token_count in page_token_counts:
        logits.append(np.asarray(exact_logits[offset : offset + token_count], dtype=np.float32))
        offset += token_count
    return logits


def _exact_outputs(exact_keys: np.ndarray, exact_values: np.ndarray, scaled_queries: np.ndarray, mapping: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    exact_logits_by_head: list[np.ndarray] = []
    exact_weights_by_head: list[np.ndarray] = []
    exact_outputs_by_head: list[np.ndarray] = []
    for q_head_id, kv_head_id in enumerate(mapping.tolist()):
        query = scaled_queries[q_head_id]
        logits = exact_keys[kv_head_id] @ query
        weights = _softmax(logits)
        output = weights @ exact_values[kv_head_id]
        exact_logits_by_head.append(logits.astype(np.float32, copy=False))
        exact_weights_by_head.append(weights.astype(np.float32, copy=False))
        exact_outputs_by_head.append(output.astype(np.float32, copy=False))
    return exact_logits_by_head, exact_weights_by_head, exact_outputs_by_head


def _collect_variant_rows(
    *,
    variant_id: str,
    key_mode: str,
    value_mode: str,
    config: DotCacheConfig,
    backend: str,
    page_cache: PreparedPageCache,
    bundle: ProbeBundle,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_lists: dict[tuple[str, int, int, int, str], list[Any]] = {}
    num_layers = len(bundle.prefill_keys)
    for layer_id in range(num_layers):
        prefill_keys = bundle.prefill_keys[layer_id]
        prefill_values = bundle.prefill_values[layer_id]
        for step_index, query_states in enumerate(bundle.step_queries[layer_id]):
            step_keys = np.stack(bundle.step_keys[layer_id][: step_index + 1], axis=1).astype(np.float32, copy=False)
            step_values = np.stack(bundle.step_values[layer_id][: step_index + 1], axis=1).astype(np.float32, copy=False)
            exact_keys = np.concatenate([prefill_keys, step_keys], axis=1).astype(np.float32, copy=False)
            exact_values = np.concatenate([prefill_values, step_values], axis=1).astype(np.float32, copy=False)
            scaled_queries = np.asarray(query_states, dtype=np.float32) * np.float32(bundle.layer_scales[layer_id])
            exact_logits_by_head, exact_weights_by_head, exact_outputs_by_head = _exact_outputs(
                exact_keys,
                exact_values,
                scaled_queries,
                bundle.q_head_to_kv_head,
            )
            token_count = int(exact_keys.shape[1])
            for q_head_id, kv_head_id in enumerate(bundle.q_head_to_kv_head.tolist()):
                exact_logits = exact_logits_by_head[q_head_id]
                exact_weights = exact_weights_by_head[q_head_id]
                exact_output = exact_outputs_by_head[q_head_id]
                if key_mode == "exact" and value_mode == "exact":
                    test_logits = exact_logits
                    test_weights = exact_weights
                    test_output = exact_output
                else:
                    page_key = (variant_id, layer_id, step_index, kv_head_id, "K")
                    key_pages = page_lists.get(page_key)
                    if key_pages is None:
                        key_pages = _encode_pages(
                            exact_keys[kv_head_id],
                            config=config,
                            kind="K",
                            mode="M3" if key_mode == "exact" else key_mode,
                            layer_id=layer_id,
                            kv_head_id=kv_head_id,
                        )
                        page_lists[page_key] = key_pages
                    page_value_key = (variant_id, layer_id, step_index, kv_head_id, "V")
                    value_pages = page_lists.get(page_value_key)
                    if value_pages is None:
                        value_pages = _encode_pages(
                            exact_values[kv_head_id],
                            config=config,
                            kind="V",
                            mode="M3" if value_mode == "exact" else value_mode,
                            layer_id=layer_id,
                            kv_head_id=kv_head_id,
                        )
                        page_lists[page_value_key] = value_pages
                    if key_mode == "exact":
                        page_logits = _split_page_logits(exact_logits, [page.header.token_count for page in key_pages])
                        test_logits, test_weights, test_output = decode_step_with_page_logits(
                            scaled_queries[q_head_id],
                            key_pages,
                            value_pages,
                            page_logits=page_logits,
                            backend=backend,
                            cache=page_cache,
                        )
                    elif value_mode == "exact":
                        test_logits, test_weights, _ = decode_step_with_page_logits(
                            scaled_queries[q_head_id],
                            key_pages,
                            value_pages,
                            backend=backend,
                            cache=page_cache,
                        )
                        test_output = test_weights @ exact_values[kv_head_id]
                    else:
                        test_logits, test_weights, test_output = decode_step_with_page_logits(
                            scaled_queries[q_head_id],
                            key_pages,
                            value_pages,
                            backend=backend,
                            cache=page_cache,
                        )
                rows.append(
                    _metric_row(
                        variant_id=variant_id,
                        layer_id=layer_id,
                        step_index=step_index,
                        q_head_id=q_head_id,
                        kv_head_id=kv_head_id,
                        token_count=token_count,
                        exact_logits=np.asarray(exact_logits, dtype=np.float32),
                        test_logits=np.asarray(test_logits, dtype=np.float32),
                        exact_weights=np.asarray(exact_weights, dtype=np.float32),
                        test_weights=np.asarray(test_weights, dtype=np.float32),
                        exact_output=np.asarray(exact_output, dtype=np.float32),
                        test_output=np.asarray(test_output, dtype=np.float32),
                        top_k=top_k,
                    )
                )
    return rows


def probe_attention_score_fidelity(args: argparse.Namespace) -> dict[str, Any]:
    harness, bundle = _build_probe_bundle(args)
    page_cache = PreparedPageCache()
    variant_summaries: list[dict[str, Any]] = []
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    baseline_margins: list[dict[str, Any]] = []
    for step_index, logits in enumerate(bundle.step_logits):
        flat = np.asarray(logits[0], dtype=np.float32)
        top_indices = np.argpartition(flat, -2)[-2:]
        top_sorted = top_indices[np.argsort(flat[top_indices])[::-1]]
        margin = float(flat[top_sorted[0]] - flat[top_sorted[1]]) if top_sorted.size >= 2 else 0.0
        baseline_margins.append(
            {
                "step_index": step_index,
                "top_token_id": int(top_sorted[0]) if top_sorted.size else -1,
                "second_token_id": int(top_sorted[1]) if top_sorted.size >= 2 else -1,
                "top_logit_margin": margin,
            }
        )

    for variant_id, key_mode, value_mode in VARIANT_DEFINITIONS:
        rows = _collect_variant_rows(
            variant_id=variant_id,
            key_mode=key_mode,
            value_mode=value_mode,
            config=harness.adapter.dotcache_config,
            backend=args.backend,
            page_cache=page_cache,
            bundle=bundle,
            top_k=args.top_k,
        )
        rows_by_variant[variant_id] = rows
        variant_summaries.append(_summary_from_rows(rows, variant_id=variant_id))

    layer_rescue_summaries: list[dict[str, Any]] = []
    target_layer_ids = sorted(set(args.target_layers or []))
    combo_rescue_summaries: list[dict[str, Any]] = []
    if args.layer_rescue_sweep:
        d5_rows = rows_by_variant["D5"]
        d6_rows = rows_by_variant["D6"]
        rescue_layer_count = len(bundle.prefill_keys)
        for rescue_layer_id in range(rescue_layer_count):
            rescue_rows = _compose_rescue_rows(
                d5_rows=d5_rows,
                d6_rows=d6_rows,
                selector=lambda row, rescue_layer_id=rescue_layer_id: int(row["layer_id"]) == rescue_layer_id,
            )
            summary = _summary_from_rows(rescue_rows, variant_id=f"rescue_L{rescue_layer_id}")
            summary["rescue_layer_id"] = rescue_layer_id
            layer_rescue_summaries.append(summary)
        layer_rescue_summaries.sort(
            key=lambda row: (
                -float(row["score_top1_match_rate"]),
                -float(row["score_topk_overlap_mean"]),
                float(row["attn_kl_mean"]),
                float(row["output_rmse_mean"]),
            )
        )

    targeted_layer_rescue_summaries: list[dict[str, Any]] = []
    kv_group_rescue_summaries: list[dict[str, Any]] = []
    if target_layer_ids:
        d5_rows = rows_by_variant["D5"]
        d6_rows = rows_by_variant["D6"]
        kv_head_ids = sorted({int(row["kv_head_id"]) for row in d5_rows})
        for target_layer_id in target_layer_ids:
            rescue_rows = _compose_rescue_rows(
                d5_rows=d5_rows,
                d6_rows=d6_rows,
                selector=lambda row, target_layer_id=target_layer_id: int(row["layer_id"]) == target_layer_id,
            )
            summary = _summary_from_rows(rescue_rows, variant_id=f"target_rescue_L{target_layer_id}")
            summary["rescue_layer_id"] = target_layer_id
            targeted_layer_rescue_summaries.append(summary)
            if args.kv_group_rescue_sweep:
                for kv_head_id in kv_head_ids:
                    rescue_rows = _compose_rescue_rows(
                        d5_rows=d5_rows,
                        d6_rows=d6_rows,
                        selector=lambda row, target_layer_id=target_layer_id, kv_head_id=kv_head_id: (
                            int(row["layer_id"]) == target_layer_id and int(row["kv_head_id"]) == kv_head_id
                        ),
                    )
                    summary = _summary_from_rows(
                        rescue_rows,
                        variant_id=f"target_rescue_L{target_layer_id}_KV{kv_head_id}",
                    )
                    summary["rescue_layer_id"] = target_layer_id
                    summary["rescue_kv_head_id"] = kv_head_id
                    kv_group_rescue_summaries.append(summary)
        kv_group_rescue_summaries.sort(
            key=lambda row: (
                int(row["rescue_layer_id"]),
                -float(row["score_top1_match_rate"]),
                -float(row["score_topk_overlap_mean"]),
                float(row["attn_kl_mean"]),
                float(row["output_rmse_mean"]),
            )
        )

    if args.combo_rescue:
        d5_rows = rows_by_variant["D5"]
        d6_rows = rows_by_variant["D6"]
        combo_rules = []
        for combo_spec in args.combo_rescue:
            parts = [part.strip() for part in combo_spec.split(",") if part.strip()]
            parsed_parts = [_parse_combo_rescue_spec(part) for part in parts]
            combo_name = ",".join(name for name, _ in parsed_parts)
            targets = {target for _, target in parsed_parts}
            combo_rules.append((combo_name, targets))
        for combo_name, targets in combo_rules:
            rescue_rows = _compose_rescue_rows(
                d5_rows=d5_rows,
                d6_rows=d6_rows,
                selector=lambda row, targets=targets: (
                    (int(row["layer_id"]), None) in targets
                    or (int(row["layer_id"]), int(row["kv_head_id"])) in targets
                ),
            )
            summary = _summary_from_rows(rescue_rows, variant_id=f"combo_{combo_name}")
            summary["combo_name"] = combo_name
            summary["combo_targets"] = [
                {"layer_id": int(layer_id), "kv_head_id": None if kv_head_id is None else int(kv_head_id)}
                for layer_id, kv_head_id in sorted(targets, key=lambda item: (item[0], -1 if item[1] is None else item[1]))
            ]
            combo_rescue_summaries.append(summary)
        combo_rescue_summaries.sort(
            key=lambda row: (
                -float(row["score_top1_match_rate"]),
                -float(row["score_topk_overlap_mean"]),
                -float(row["score_rank_corr_mean"]),
                float(row["attn_kl_mean"]),
                float(row["output_rmse_mean"]),
            )
        )

    result = {
        "family": args.family,
        "model_id": args.model_id,
        "backend": args.backend,
        "device": args.device or harness.adapter.device.type,
        "prompt_length": bundle.prompt_length,
        "decode_steps": args.decode_steps,
        "top_k": args.top_k,
        "baseline_decode_logit_margins": baseline_margins,
        "variant_summaries": variant_summaries,
    }
    if args.layer_rescue_sweep:
        result["layer_rescue_summaries"] = layer_rescue_summaries
    if targeted_layer_rescue_summaries:
        result["targeted_layer_rescue_summaries"] = targeted_layer_rescue_summaries
    if kv_group_rescue_summaries:
        result["kv_group_rescue_summaries"] = kv_group_rescue_summaries
    if combo_rescue_summaries:
        result["combo_rescue_summaries"] = combo_rescue_summaries
    if args.include_rows:
        result["rows_by_variant"] = rows_by_variant
    return result


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# Attention Score Fidelity: {result['model_id']}",
        "",
        f"- Family: `{result['family']}`",
        f"- Backend: `{result['backend']}` on `{result['device']}`",
        f"- Prompt length: `{result['prompt_length']}`",
        f"- Decode steps: `{result['decode_steps']}`",
        "",
        "## Baseline Margins",
    ]
    lines.extend(
        f"- Step `{entry['step_index']}`: top margin `{entry['top_logit_margin']:.6f}` (`{entry['top_token_id']}` vs `{entry['second_token_id']}`)"
        for entry in result["baseline_decode_logit_margins"]
    )
    lines.extend(["", "## Variant Summary"])
    for summary in result["variant_summaries"]:
        lines.extend(
            [
                f"### {summary['variant_id']}",
                f"- Rows: `{summary['row_count']}`",
                f"- Top-1 score match rate: `{summary['score_top1_match_rate']:.6f}`",
                f"- Top-{result['top_k']} overlap mean: `{summary['score_topk_overlap_mean']:.6f}`",
                f"- Rank correlation mean: `{summary['score_rank_corr_mean']:.6f}`",
                f"- Attention KL mean: `{summary['attn_kl_mean']:.6f}`",
                f"- Score max abs max: `{summary['score_max_abs_max']:.6f}`",
                f"- Output RMSE mean: `{summary['output_rmse_mean']:.6f}`",
                "- Worst layers:",
            ]
        )
        for layer in summary["top_layers_by_attn_kl"]:
            lines.append(
                f"  - Layer `{layer['layer_id']}`: attn_kl_mean `{layer['attn_kl_mean']:.6f}`, top1 `{layer['score_top1_match_rate']:.6f}`, output_rmse `{layer['output_rmse_mean']:.6f}`"
            )
    if result.get("layer_rescue_summaries"):
        lines.extend(["", "## Layer Rescue Sweep"])
        for summary in result["layer_rescue_summaries"][:10]:
            lines.append(
                f"- Layer `{summary['rescue_layer_id']}`: top1 `{summary['score_top1_match_rate']:.6f}`, top{result['top_k']} `{summary['score_topk_overlap_mean']:.6f}`, rank `{summary['score_rank_corr_mean']:.6f}`, KL `{summary['attn_kl_mean']:.6f}`, output_rmse `{summary['output_rmse_mean']:.6f}`"
            )
    if result.get("targeted_layer_rescue_summaries"):
        lines.extend(["", "## Targeted Layers"])
        for summary in result["targeted_layer_rescue_summaries"]:
            lines.append(
                f"- Layer `{summary['rescue_layer_id']}` whole-layer rescue: top1 `{summary['score_top1_match_rate']:.6f}`, top{result['top_k']} `{summary['score_topk_overlap_mean']:.6f}`, rank `{summary['score_rank_corr_mean']:.6f}`, KL `{summary['attn_kl_mean']:.6f}`, output_rmse `{summary['output_rmse_mean']:.6f}`"
            )
    if result.get("kv_group_rescue_summaries"):
        lines.extend(["", "## KV Group Rescue"])
        kv_groups = result["kv_group_rescue_summaries"]
        for layer_id in sorted({int(row["rescue_layer_id"]) for row in kv_groups}):
            for summary in [row for row in kv_groups if int(row["rescue_layer_id"]) == layer_id]:
                lines.append(
                    f"- Layer `{summary['rescue_layer_id']}` KV `{summary['rescue_kv_head_id']}`: top1 `{summary['score_top1_match_rate']:.6f}`, top{result['top_k']} `{summary['score_topk_overlap_mean']:.6f}`, rank `{summary['score_rank_corr_mean']:.6f}`, KL `{summary['attn_kl_mean']:.6f}`, output_rmse `{summary['output_rmse_mean']:.6f}`"
                )
    if result.get("combo_rescue_summaries"):
        lines.extend(["", "## Combo Rescue"])
        for summary in result["combo_rescue_summaries"]:
            lines.append(
                f"- `{summary['combo_name']}`: top1 `{summary['score_top1_match_rate']:.6f}`, top{result['top_k']} `{summary['score_topk_overlap_mean']:.6f}`, rank `{summary['score_rank_corr_mean']:.6f}`, KL `{summary['attn_kl_mean']:.6f}`, output_rmse `{summary['output_rmse_mean']:.6f}`"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = probe_attention_score_fidelity(args)
    if args.format == "markdown":
        print(render_markdown(result))
        return
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
