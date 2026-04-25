"""Dense-vs-certified one-token decode speed comparison.

This is the useful speed comparison for paper planning: both paths run the
same PG-19 chunk, same dense prefix, same warmup token positions, and same
teacher-forced one-token decode window. Setup is reported separately so decode
throughput is not confused with model load, prefill, or cache construction.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pg19_perplexity import load_pg19_chunks
from _provenance import (
    add_paper_cache_args,
    cache_config_dict,
    configure_paper_runtime_defaults,
    resolve_fp16_key_cache_blocks,
    resolve_fp16_value_cache_blocks,
)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _runtime_cache_summary(tiered_caches: dict[int, Any]) -> dict[str, Any]:
    caches = list(tiered_caches.values())
    return {
        "static_resident_key_cache": bool(caches) and all(
            bool(getattr(c, "static_resident_key_cache", False)) for c in caches
        ),
        "static_resident_value_cache": bool(caches) and all(
            bool(getattr(c, "static_resident_value_cache", False)) for c in caches
        ),
        "static_resident_key_prepare_bytes": int(sum(
            int(getattr(c, "static_resident_key_prepare_bytes", 0)) for c in caches
        )),
        "static_resident_value_prepare_bytes": int(sum(
            int(getattr(c, "static_resident_value_prepare_bytes", 0)) for c in caches
        )),
        "vram_fp16_key_cache_bytes": int(sum(
            c.keys_fp16_gpu.nelement() * c.keys_fp16_gpu.element_size()
            for c in caches if getattr(c, "keys_fp16_gpu", None) is not None
        )),
        "vram_fp16_value_cache_bytes": int(sum(
            c.values_fp16_gpu.nelement() * c.values_fp16_gpu.element_size()
            for c in caches if getattr(c, "values_fp16_gpu", None) is not None
        )),
    }


def _ppl(nll: float, tokens: int) -> float:
    return float(math.exp(nll / max(tokens, 1)))


def measure_dense_decode(
    model,
    input_ids: torch.Tensor,
    *,
    prefix_len: int,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    _sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(input_ids=input_ids[:, :prefix_len], use_cache=True)
    _sync()
    prefill_s = time.perf_counter() - t0
    past = outputs.past_key_values
    del outputs

    cache_position = torch.tensor([prefix_len], dtype=torch.long, device=input_ids.device)
    with torch.inference_mode():
        for t in range(warmup_steps):
            token_id = input_ids[:, prefix_len + t:prefix_len + t + 1]
            out = model(
                input_ids=token_id,
                use_cache=True,
                past_key_values=past,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
            past = out.past_key_values
            cache_position.add_(1)
            del out
    _sync()

    nll = torch.zeros((), dtype=torch.float32, device=input_ids.device)
    _sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for t in range(warmup_steps, warmup_steps + measure_steps):
            token_id = input_ids[:, prefix_len + t:prefix_len + t + 1]
            out = model(
                input_ids=token_id,
                use_cache=True,
                past_key_values=past,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
            past = out.past_key_values
            target = input_ids[:, prefix_len + t + 1]
            nll = nll + F.cross_entropy(out.logits[:, -1, :].float(), target, reduction="sum")
            cache_position.add_(1)
            del out
    _sync()
    decode_s = time.perf_counter() - t0
    nll_f = float(nll.item())
    del past
    return {
        "prefill_s": float(prefill_s),
        "decode_s": float(decode_s),
        "decode_steps": int(measure_steps),
        "decode_tok_s": float(measure_steps / max(decode_s, 1e-9)),
        "decode_ms_per_token": float(1000.0 * decode_s / max(measure_steps, 1)),
        "nll": nll_f,
        "perplexity": _ppl(nll_f, measure_steps),
    }


def measure_certified_decode(
    model,
    adapter,
    input_ids: torch.Tensor,
    args: argparse.Namespace,
    *,
    prefix_len: int,
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, Any]:
    from dotcache.integrations.llama import _ensure_certified_imports, CertifiedAttentionState
    from dotcache.kernels.tiered_kv_cache import (
        create_tiered_cache_from_model,
        create_tiered_cache_int4v_from_model,
    )

    _ensure_certified_imports()
    adapter.set_mode("dense")
    _sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        prefix_out = model(input_ids=input_ids[:, :prefix_len], use_cache=True)
    _sync()
    prefill_s = time.perf_counter() - t0
    past_kv = prefix_out.past_key_values
    del prefix_out

    layer_ids = list(range(model.config.num_hidden_layers))
    key_cap = resolve_fp16_key_cache_blocks(
        args.fp16_key_cache_blocks,
        os.environ.get("DOTCACHE_FP16_CACHE_BLOCKS"),
    )
    value_cap = resolve_fp16_value_cache_blocks(
        args.fp16_value_cache_blocks,
        os.environ.get("DOTCACHE_FP16_VALUE_CACHE_BLOCKS"),
    )
    max_new = warmup_steps + measure_steps + 16
    _sync()
    t0 = time.perf_counter()
    if args.use_int4_values:
        tiered_caches = create_tiered_cache_int4v_from_model(
            past_kv,
            layer_ids,
            group_size=args.group_size,
            max_new_tokens=max_new,
            fp16_key_cache_capacity=key_cap,
            fp16_value_cache_capacity=value_cap,
        )
    else:
        tiered_caches = create_tiered_cache_from_model(
            past_kv,
            layer_ids,
            max_new_tokens=max_new,
            fp16_key_cache_capacity=key_cap,
        )
    _sync()
    cache_build_s = time.perf_counter() - t0
    cache_runtime = _runtime_cache_summary(tiered_caches)
    del past_kv
    gc.collect()
    torch.cuda.empty_cache()

    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered_caches,
        collect_stats=False,
        append_kv=True,
        top_k_fp16_keys=args.top_k_fp16,
        v_tolerance=args.v_tolerance,
        tau_cov=(args.tau_cov if args.tau_cov and args.tau_cov > 0 else None),
        k_min=args.k_min,
        k_max=args.k_max,
        ranking_fallback=args.ranking_fallback,
        ranking_r=args.ranking_r,
        ranking_fallback_mode=args.ranking_fallback_mode,
        score_consistency_check=False,
        eps_guard=args.eps_guard,
        exploration_rate=args.exploration_rate,
        rung1_threshold=args.rung1_threshold,
        rung1_multiplier=args.rung1_multiplier,
    )
    if args.phase_profile:
        adapter.certified_state.phase_timings = {}
    adapter.set_mode("certified")
    adapter.reset_runtime_metrics()

    cache_position = torch.tensor([prefix_len], dtype=torch.long, device=input_ids.device)
    with torch.inference_mode():
        for t in range(warmup_steps):
            token_id = input_ids[:, prefix_len + t:prefix_len + t + 1]
            out = model(
                input_ids=token_id,
                use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
            cache_position.add_(1)
            del out
    _sync()
    adapter.reset_runtime_metrics()
    if args.phase_profile:
        adapter.certified_state.phase_timings = {}
    if args.native_profile:
        from dotcache.backends.certified_blackwell import reset_native_profile

        reset_native_profile()

    nll = torch.zeros((), dtype=torch.float32, device=input_ids.device)
    _sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for t in range(warmup_steps, warmup_steps + measure_steps):
            token_id = input_ids[:, prefix_len + t:prefix_len + t + 1]
            out = model(
                input_ids=token_id,
                use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
            target = input_ids[:, prefix_len + t + 1]
            nll = nll + F.cross_entropy(out.logits[:, -1, :].float(), target, reduction="sum")
            cache_position.add_(1)
            del out
    _sync()
    decode_s = time.perf_counter() - t0
    runtime_profile = adapter.runtime_profile_summary(model_forward_ms_total=decode_s * 1000.0)
    phase_timings_us = dict(adapter.certified_state.phase_timings or {}) if args.phase_profile else {}
    phase_timings_ms = {k.removesuffix("_us") + "_ms": float(v) / 1000.0 for k, v in phase_timings_us.items()}
    native_profile = None
    if args.native_profile:
        from dotcache.backends.certified_blackwell import native_profile_summary

        native_profile = native_profile_summary()
    nll_f = float(nll.item())

    adapter.certified_state = None
    adapter.set_mode("dense")
    return {
        "prefill_s": float(prefill_s),
        "cache_build_s": float(cache_build_s),
        "cache_runtime": cache_runtime,
        "decode_s": float(decode_s),
        "decode_steps": int(measure_steps),
        "decode_tok_s": float(measure_steps / max(decode_s, 1e-9)),
        "decode_ms_per_token": float(1000.0 * decode_s / max(measure_steps, 1)),
        "nll": nll_f,
        "perplexity": _ppl(nll_f, measure_steps),
        "runtime_profile": runtime_profile,
        "phase_timings_ms": phase_timings_ms,
        "native_profile": native_profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--chunk-index", type=int, default=0)
    parser.add_argument("--eval-start", type=float, default=0.5)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--measure-steps", type=int, default=128)
    parser.add_argument("--top-k-fp16", type=int, default=4)
    parser.add_argument("--tau-cov", type=float, default=0.995)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=128)
    parser.add_argument("--ranking-fallback", action="store_true")
    parser.add_argument("--ranking-r", type=int, default=1)
    parser.add_argument("--ranking-fallback-mode", default="full", choices=["full", "measure"])
    parser.add_argument("--eps-guard", type=float, default=0.01)
    parser.add_argument("--exploration-rate", type=float, default=0.02)
    parser.add_argument("--rung1-threshold", type=float, default=0.02)
    parser.add_argument("--rung1-multiplier", type=float, default=2.0)
    parser.add_argument("--phase-profile", action="store_true",
                        help="Collect synchronized CUDA phase timings inside certified_attention_layer. Slow; profiling only.")
    parser.add_argument("--native-profile", action="store_true",
                        help="Collect native Blackwell partial-vs-reduce timings. Slow; profiling only.")
    parser.add_argument("--output", default="runs/decode_speed_compare.json")
    add_paper_cache_args(parser)
    args = parser.parse_args()
    configure_paper_runtime_defaults()
    if args.native_profile:
        os.environ["DOTCACHE_NATIVE_PROFILE"] = "1"

    if args.eval_start <= 0.0 or args.eval_start >= 1.0:
        raise SystemExit("--eval-start must be in (0, 1)")
    prefix_len = int(args.context * args.eval_start)
    required = prefix_len + args.warmup_steps + args.measure_steps + 1
    if required > args.context:
        raise SystemExit(
            f"context too short for prefix+warmup+measure+target: need {required}, have {args.context}"
        )

    token = os.environ.get("HF_TOKEN") or None
    warnings.filterwarnings(
        "ignore",
        message=r"MatMul8bitLt: inputs will be cast from .* during quantization",
        category=UserWarning,
    )
    print(f"Loading {args.model} (INT8)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
        dtype=torch.float16,
        token=token,
    )
    model.eval()

    from dotcache.config import DotCacheConfig
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    adapter = LlamaDotCacheModelAdapter(model, DotCacheConfig(head_dim=head_dim))

    chunks, book_indices = load_pg19_chunks(
        tokenizer,
        args.context,
        args.chunk_index + 1,
    )
    input_ids = chunks[args.chunk_index].unsqueeze(0).to("cuda")
    print(
        f"Compare context={args.context} prefix={prefix_len} "
        f"warmup={args.warmup_steps} measured={args.measure_steps} "
        f"backend={os.environ.get('DOTCACHE_CERTIFIED_BACKEND')}",
        flush=True,
    )

    dense = measure_dense_decode(
        model,
        input_ids,
        prefix_len=prefix_len,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )
    gc.collect()
    torch.cuda.empty_cache()
    certified = measure_certified_decode(
        model,
        adapter,
        input_ids,
        args,
        prefix_len=prefix_len,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )

    output = {
        "benchmark": "decode_speed_compare",
        "model": args.model,
        "context_length": args.context,
        "chunk_index": args.chunk_index,
        "book_idx": book_indices[args.chunk_index] if book_indices else None,
        "prefix_len": prefix_len,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "measured_token_start": prefix_len + args.warmup_steps,
        "measured_token_end_exclusive": prefix_len + args.warmup_steps + args.measure_steps,
        "cache_config": cache_config_dict(args),
        "dense": dense,
        "certified": certified,
        "certified_vs_dense_decode_speed": {
            "tok_s_ratio": float(certified["decode_tok_s"] / max(dense["decode_tok_s"], 1e-9)),
            "slowdown": float(dense["decode_tok_s"] / max(certified["decode_tok_s"], 1e-9)),
            "dense_tok_s": dense["decode_tok_s"],
            "certified_tok_s": certified["decode_tok_s"],
        },
        "quality_window": {
            "dense_ppl": dense["perplexity"],
            "certified_ppl": certified["perplexity"],
            "ppl_ratio": float(certified["perplexity"] / max(dense["perplexity"], 1e-9)),
            "delta_ppl": float(certified["perplexity"] - dense["perplexity"]),
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "dense_tok_s": dense["decode_tok_s"],
        "certified_tok_s": certified["decode_tok_s"],
        "certified_vs_dense_ratio": output["certified_vs_dense_decode_speed"]["tok_s_ratio"],
        "slowdown": output["certified_vs_dense_decode_speed"]["slowdown"],
        "dense_ppl": dense["perplexity"],
        "certified_ppl": certified["perplexity"],
        "ppl_ratio": output["quality_window"]["ppl_ratio"],
        "phase_timings_ms": certified.get("phase_timings_ms"),
        "native_profile": certified.get("native_profile"),
        "json": str(out_path),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
