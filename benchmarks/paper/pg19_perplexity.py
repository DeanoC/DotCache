"""PG-19 perplexity benchmark: dense vs certified attention.

Measures teacher-forced perplexity on PG-19 test books at various
context lengths. For each document chunk:
  1. Dense: standard HF forward pass → per-token NLL
  2. Certified: dense prefill → tiered KV cache → certified decode
     (teacher-forced, one token at a time) → per-token NLL

Reports: perplexity (dense & certified), ratio, skip rate.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_pg19_chunks(tokenizer, context_length: int, num_chunks: int,
                     stride: int = None) -> list[torch.Tensor]:
    """Load PG-19 test set and chunk into fixed-length token sequences.

    Uses strided windowing: each chunk starts `stride` tokens after the
    previous one. Default stride = context_length (no overlap).
    """
    from datasets import load_dataset

    if stride is None:
        stride = context_length

    chunks = []
    ds = load_dataset("emozilla/pg19", split="test", streaming=True)

    for book in ds:
        text = book["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Slide window across the book
        for start in range(0, len(tokens) - context_length, stride):
            chunk = tokens[start : start + context_length]
            chunks.append(torch.tensor(chunk, dtype=torch.long))
            if len(chunks) >= num_chunks:
                return chunks

    print(f"Warning: only found {len(chunks)} chunks (requested {num_chunks})")
    return chunks


def compute_dense_perplexity(model, chunks: list[torch.Tensor],
                             device: str = "cuda") -> dict:
    """Compute perplexity using standard dense forward pass."""
    total_nll = 0.0
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        input_ids = chunk.unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids)
        # Compute loss in chunks to avoid materialising full FP32 logit tensor
        # (8K × 128K × 4 bytes = 4 GB — too large when sharing GPU)
        logits = outputs.logits[:, :-1, :]  # keep native dtype
        targets = input_ids[:, 1:]
        chunk_nll = 0.0
        chunk_size = 512  # process 512 positions at a time
        for start in range(0, logits.shape[1], chunk_size):
            end = min(start + chunk_size, logits.shape[1])
            chunk_logits = logits[:, start:end, :].float()
            chunk_targets = targets[:, start:end]
            nll = F.cross_entropy(chunk_logits.reshape(-1, chunk_logits.size(-1)),
                                  chunk_targets.reshape(-1), reduction="sum")
            chunk_nll += nll.item()
            del chunk_logits, nll
        total_nll += chunk_nll
        total_tokens += targets.numel()

        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            ppl_so_far = math.exp(total_nll / total_tokens)
            print(f"  Dense [{i+1}/{len(chunks)}]: ppl={ppl_so_far:.2f}")

        del outputs, logits
        gc.collect()
        torch.cuda.empty_cache()

    ppl = math.exp(total_nll / total_tokens)
    return {
        "perplexity": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
    }


def compute_certified_perplexity(
    model, adapter, chunks: list[torch.Tensor],
    calibrated_profile=None,
    eval_start_frac: float = 0.5,
    epsilon_override: float = None,
    top_k_override: int = None,
    concentration_threshold: float = 0.0,
    device: str = "cuda",
    # Paper-alignment features (T4/T7/Rung1/T9/T10).
    tau_cov: float | None = None,
    k_min: int = 2,
    k_max: int | None = None,
    ranking_fallback: bool = False,
    ranking_r: int = 1,
    ranking_fallback_mode: str = "full",
    score_consistency_check: bool = False,
    eps_guard: float = 0.01,
    exploration_rate: float = 0.0,
    rung1_threshold: float = 0.02,
    rung1_multiplier: float = 2.0,
    telemetry_collector=None,
) -> dict:
    """Compute perplexity using certified attention decode.

    For each chunk:
      1. Dense prefill on the first `prefix_len` tokens
      2. Build tiered KV cache from dense KV pairs
      3. Teacher-forced certified decode for remaining tokens
      4. Compute NLL from certified logits

    Args:
        eval_start_frac: fraction of context to use as dense prefix.
            Tokens after this point are evaluated with certified attention.
    """
    from dotcache.integrations.llama import _ensure_certified_imports, CertifiedAttentionState
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model

    total_nll = 0.0
    total_tokens = 0
    total_skipped = 0
    total_blocks = 0
    total_steps = 0

    for i, chunk in enumerate(chunks):
        seq_len = chunk.shape[0]
        prefix_len = int(seq_len * eval_start_frac)
        eval_len = seq_len - prefix_len

        input_ids = chunk.unsqueeze(0).to(device)

        # Phase 1: Dense prefill for prefix
        adapter.set_mode("dense")
        with torch.inference_mode():
            prefix_out = model(input_ids=input_ids[:, :prefix_len], use_cache=True)
        past_kv = prefix_out.past_key_values

        # Compute prefix NLL in chunks (same technique as dense path)
        prefix_logits = prefix_out.logits[:, :-1, :]
        prefix_targets = input_ids[:, 1:prefix_len]
        prefix_nll = 0.0
        pchunk = 512
        for pstart in range(0, prefix_logits.shape[1], pchunk):
            pend = min(pstart + pchunk, prefix_logits.shape[1])
            pl = prefix_logits[:, pstart:pend, :].float()
            pt = prefix_targets[:, pstart:pend]
            prefix_nll += F.cross_entropy(
                pl.reshape(-1, pl.size(-1)), pt.reshape(-1), reduction="sum"
            ).item()
            del pl
        del prefix_out, prefix_logits

        # Phase 2: Build tiered cache with enough room for eval tokens
        _ensure_certified_imports()
        layer_ids = list(range(model.config.num_hidden_layers))
        _cap = int(os.environ.get("DOTCACHE_FP16_CACHE_BLOCKS", "0")) or None
        tiered_caches = create_tiered_cache_from_model(
            past_kv, layer_ids, max_new_tokens=eval_len + 16,
            fp16_key_cache_capacity=_cap,
        )
        del past_kv
        gc.collect()
        torch.cuda.empty_cache()

        # Get layer epsilons from calibrated profile or override
        if epsilon_override is not None:
            layer_epsilons = {}
            default_eps = epsilon_override
        elif calibrated_profile is not None:
            layer_epsilons = calibrated_profile.get_layer_epsilons_min(prefix_len)
            default_eps = 1e-4
        else:
            layer_epsilons = {}
            default_eps = 1e-4

        top_k = top_k_override if top_k_override is not None else 4

        adapter.certified_state = CertifiedAttentionState(
            tiered_caches=tiered_caches,
            layer_epsilons=layer_epsilons,
            default_epsilon=default_eps,
            collect_stats=True,
            append_kv=True,
            top_k_fp16_keys=top_k,
            concentration_threshold=concentration_threshold,
            tau_cov=tau_cov,
            k_min=k_min,
            k_max=k_max,
            ranking_fallback=ranking_fallback,
            ranking_r=ranking_r,
            ranking_fallback_mode=ranking_fallback_mode,
            score_consistency_check=score_consistency_check,
            eps_guard=eps_guard,
            exploration_rate=exploration_rate,
            rung1_threshold=rung1_threshold,
            rung1_multiplier=rung1_multiplier,
        )
        adapter.set_mode("certified")

        # Phase 3: Teacher-forced certified decode
        cache_position = torch.tensor([prefix_len], dtype=torch.long, device=device)
        suffix_nll = 0.0
        chunk_skipped = 0
        chunk_blocks = 0

        for t in range(eval_len - 1):
            token_id = input_ids[:, prefix_len + t]
            with torch.inference_mode():
                out = model(
                    input_ids=token_id.unsqueeze(0),
                    use_cache=False,
                    cache_position=cache_position,
                    position_ids=cache_position.unsqueeze(0),
                )
            if telemetry_collector is not None:
                telemetry_collector.record_step()
            # Loss: predict token at prefix_len + t + 1
            logits = out.logits[:, -1, :].float()
            target = input_ids[:, prefix_len + t + 1]
            nll = F.cross_entropy(logits, target, reduction="sum")
            suffix_nll += nll.item()
            cache_position = cache_position + 1

            # Drain per-step skip stats (aggregated across layers)
            step = adapter.certified_state.aggregate_step_stats()
            chunk_blocks += step["total_blocks"]
            chunk_skipped += step["skipped_blocks"]
            adapter.certified_state.clear_step_stats()
            total_steps += 1

        adapter.certified_state = None
        adapter.set_mode("dense")

        # Total NLL = prefix (dense) + suffix (certified)
        chunk_nll = prefix_nll + suffix_nll
        chunk_tokens = (prefix_len - 1) + (eval_len - 1)
        total_nll += chunk_nll
        total_tokens += chunk_tokens

        total_skipped += chunk_skipped
        total_blocks += chunk_blocks

        chunk_ppl = math.exp(chunk_nll / chunk_tokens)
        suffix_ppl = math.exp(suffix_nll / max(eval_len - 1, 1))
        chunk_skip_rate = chunk_skipped / chunk_blocks if chunk_blocks else 0.0
        if (i + 1) % 5 == 0 or i == len(chunks) - 1:
            ppl_so_far = math.exp(total_nll / total_tokens)
            overall_skip = total_skipped / total_blocks if total_blocks else 0.0
            print(f"  Certified [{i+1}/{len(chunks)}]: chunk_ppl={chunk_ppl:.2f}, "
                  f"suffix_ppl={suffix_ppl:.2f}, running_ppl={ppl_so_far:.2f}, "
                  f"skip={chunk_skip_rate:.3f} (overall {overall_skip:.3f})")

        del tiered_caches
        gc.collect()
        torch.cuda.empty_cache()

    ppl = math.exp(total_nll / total_tokens)
    skip_rate = total_skipped / total_blocks if total_blocks else 0.0
    return {
        "perplexity": ppl,
        "total_nll": total_nll,
        "total_tokens": total_tokens,
        "skip_rate": skip_rate,
        "skipped_blocks": total_skipped,
        "total_blocks": total_blocks,
        "decode_steps": total_steps,
    }


def main():
    import argparse
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter
    from dotcache.config import DotCacheConfig
    from dotcache.calibration.calibrated_profile import CalibratedProfile

    parser = argparse.ArgumentParser(description="PG-19 perplexity: dense vs certified")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    parser.add_argument("--context", type=int, default=8192,
                        help="Context length per chunk")
    parser.add_argument("--num-chunks", type=int, default=20,
                        help="Number of document chunks to evaluate")
    parser.add_argument("--eval-start", type=float, default=0.5,
                        help="Fraction of context for dense prefix (rest is certified)")
    parser.add_argument("--profile", default=None,
                        help="Path to calibrated profile .npz")
    parser.add_argument("--output", default="benchmarks/results/pg19_perplexity.json")
    parser.add_argument("--dense-only", action="store_true",
                        help="Only run dense baseline (skip certified)")
    parser.add_argument("--epsilon-override", type=float, default=None,
                        help="Override all layer epsilons (e.g. 0.0 for no-skip diagnostic)")
    parser.add_argument("--top-k-override", type=int, default=None,
                        help="Override top_k_fp16_keys (default: 4)")
    parser.add_argument("--concentration-threshold", type=float, default=0.0,
                        help="If max block mass fraction < this, disable skip for that head (0=off, 0.02=2%%)")
    # Paper-alignment flags (T4/T7/Rung1/T9/T10).
    parser.add_argument("--tau-cov", type=float, default=0.0,
                        help="Adaptive K* cumulative-mass threshold (0 or omitted = disabled; paper default 0.995)")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=None,
                        help="Adaptive K* upper clamp (None = no cap)")
    parser.add_argument("--ranking-fallback", action="store_true",
                        help="Enable Rung-3 ranking-consistency fallback")
    parser.add_argument("--ranking-r", type=int, default=1)
    parser.add_argument("--ranking-fallback-mode", default="full", choices=["full", "measure"])
    parser.add_argument("--score-consistency-check", action="store_true")
    parser.add_argument("--eps-guard", type=float, default=0.01)
    parser.add_argument("--exploration-rate", type=float, default=0.0)
    parser.add_argument("--rung1-threshold", type=float, default=0.02)
    parser.add_argument("--rung1-multiplier", type=float, default=2.0)
    parser.add_argument("--pagein-telemetry", action="store_true",
                        help="Collect per-step page-in / rung / VRAM-cache telemetry (Test 3)")
    parser.add_argument("--telemetry-output", default=None,
                        help="Path to write per-step telemetry JSON (default: <output>.pagein.json)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading {args.model} (INT8)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant_config, device_map="auto", token=token,
    )
    model.eval()
    print(f"Model: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    config = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, config)

    profile = None
    if args.profile:
        profile = CalibratedProfile.load(args.profile)
        print(f"Loaded profile: {profile.summary()[:200]}")

    # Load PG-19 chunks
    print(f"\nLoading PG-19 test set: {args.num_chunks} chunks × {args.context} tokens...")
    t0 = time.perf_counter()
    chunks = load_pg19_chunks(tokenizer, args.context, args.num_chunks)
    print(f"Loaded {len(chunks)} chunks in {time.perf_counter()-t0:.1f}s")

    # Dense perplexity
    print(f"\n{'='*50}")
    print("Dense perplexity")
    print(f"{'='*50}")
    t0 = time.perf_counter()
    dense_result = compute_dense_perplexity(model, chunks)
    t_dense = time.perf_counter() - t0
    print(f"Dense: ppl={dense_result['perplexity']:.2f} "
          f"({dense_result['total_tokens']} tokens, {t_dense:.1f}s)")

    # Certified perplexity
    cert_result = None
    telemetry_collector = None
    if args.pagein_telemetry and not args.dense_only:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        from _pagein_telemetry import PageinTelemetry
        telemetry_collector = PageinTelemetry(adapter, enabled=True)
        telemetry_collector.start()
    if not args.dense_only:
        print(f"\n{'='*50}")
        print("Certified perplexity")
        print(f"{'='*50}")
        t0 = time.perf_counter()
        cert_result = compute_certified_perplexity(
            model, adapter, chunks,
            calibrated_profile=profile,
            eval_start_frac=args.eval_start,
            epsilon_override=args.epsilon_override,
            top_k_override=args.top_k_override,
            concentration_threshold=args.concentration_threshold,
            tau_cov=(args.tau_cov if args.tau_cov and args.tau_cov > 0 else None),
            k_min=args.k_min,
            k_max=args.k_max,
            ranking_fallback=args.ranking_fallback,
            ranking_r=args.ranking_r,
            ranking_fallback_mode=args.ranking_fallback_mode,
            score_consistency_check=args.score_consistency_check,
            eps_guard=args.eps_guard,
            exploration_rate=args.exploration_rate,
            rung1_threshold=args.rung1_threshold,
            rung1_multiplier=args.rung1_multiplier,
            telemetry_collector=telemetry_collector,
        )
        t_cert = time.perf_counter() - t0
        print(f"Certified: ppl={cert_result['perplexity']:.2f} "
              f"({cert_result['total_tokens']} tokens, {t_cert:.1f}s)")
        if telemetry_collector is not None:
            telemetry_collector.finish()
            tele_path = args.telemetry_output or (str(args.output).replace(".json", ".pagein.json"))
            telemetry_collector.write_json(tele_path)
            s = telemetry_collector.summary()
            print(f"Page-in telemetry: n_steps={s.get('n_steps',0)} "
                  f"h2d_mean={s.get('h2d_total_bytes_mean',0)/1024:.1f} KB/step, "
                  f"pct_zero_pagein={s.get('pct_steps_zero_pagein',0):.1%}, "
                  f"rung1_rate={s.get('rung1_rate',0):.2%}, rung2_rate={s.get('rung2_rate',0):.2%}, "
                  f"rung3_rate={s.get('rung3_rate',0):.2%}, rung4_rate={s.get('rung4_rate',0):.2%}")

    # Summary
    print(f"\n{'='*50}")
    print("Summary")
    print(f"{'='*50}")
    print(f"Context: {args.context} tokens, {len(chunks)} chunks")
    print(f"Dense perplexity:     {dense_result['perplexity']:.4f}")
    if cert_result:
        ratio = cert_result['perplexity'] / dense_result['perplexity']
        delta = cert_result['perplexity'] - dense_result['perplexity']
        print(f"Certified perplexity: {cert_result['perplexity']:.4f}")
        print(f"Ratio (cert/dense):   {ratio:.6f}")
        print(f"Delta:                {delta:+.4f}")
        print(f"Skip rate:            {cert_result.get('skip_rate', 0.0):.4f} "
              f"({cert_result.get('skipped_blocks', 0)}/{cert_result.get('total_blocks', 0)} blocks)")
        print(f"Concentration thr:    {args.concentration_threshold}")

    # Save results
    output = {
        "benchmark": "pg19_perplexity",
        "model": args.model,
        "context_length": args.context,
        "num_chunks": len(chunks),
        "eval_start_frac": args.eval_start,
        "concentration_threshold": args.concentration_threshold,
        "default_epsilon_override": args.epsilon_override,
        "top_k_override": args.top_k_override,
        "dense": dense_result,
        "certified": cert_result,
    }
    if cert_result:
        output["ratio"] = cert_result['perplexity'] / dense_result['perplexity']
        output["delta"] = cert_result['perplexity'] - dense_result['perplexity']

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
