"""Per-token FP16 cache trace — drives the decode-mode transition figure.

Runs an 8K pg19 prefill + 256-token argmax decode at a fixed cache capacity,
snapshotting cumulative cache counters across all layer caches between each
decode step. Emits per-token arrays of:

  per_token_hits       — hits this decode step (summed across 32 layers)
  per_token_misses     — misses this decode step
  per_token_h2d_bytes  — bytes H2D'd this decode step
  per_token_hit_rate   — hits / (hits + misses) this step

The purpose is to characterise where argmax-generated decode transitions
from concentrated (early tokens anchored to the pg19 prefix) to scattered
(later tokens drifted out-of-distribution). The resulting curve is the
figure referenced in §9.8.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--decode-tokens", type=int, default=256)
    ap.add_argument("--cache-blocks", type=int, default=64,
                    help="Cache capacity to measure at (choose low enough that hits/misses are both visible).")
    ap.add_argument("--prompt-source", choices=["pg19", "filler"], default="pg19")
    ap.add_argument("--teacher-forced", action="store_true",
                    help="Feed ground-truth pg19 tokens instead of argmax (requires --prompt-source pg19).")
    ap.add_argument("--output", default="benchmarks/results/perf_tests_20260422/per_token_trace_pg19_cap64.json")
    args = ap.parse_args()

    # DOTCACHE_V_TOL was attempted as a runtime override but the env var was
    # never read by any kernel — see docs/paper_code_audit_20260424.md.
    # The CertifiedAttentionState construction below hardcodes 0.5.
    os.environ["DOTCACHE_FP16_CACHE_BLOCKS"] = str(args.cache_blocks)

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import (
        LlamaDotCacheModelAdapter, CertifiedAttentionState, _ensure_certified_imports,
    )
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model
    from dotcache.config import DotCacheConfig

    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading {args.model} (INT8)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    quant = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto", token=token,
    )
    model.eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    adapter = LlamaDotCacheModelAdapter(model, DotCacheConfig(head_dim=head_dim))
    device = next(model.parameters()).device

    # Prefill
    ref_tokens = None
    if args.prompt_source == "pg19":
        from datasets import load_dataset
        ds = load_dataset("emozilla/pg19", split="test", streaming=True)
        text = None
        need = args.context_length + (args.decode_tokens + 4 if args.teacher_forced else 0)
        for book in ds:
            tokens = tokenizer.encode(book["text"], add_special_tokens=False)
            if len(tokens) >= need:
                text = tokens[:args.context_length]
                if args.teacher_forced:
                    ref_tokens = torch.tensor(
                        tokens[args.context_length:args.context_length + args.decode_tokens + 4],
                        dtype=torch.long, device=device,
                    )
                break
        if text is None:
            raise RuntimeError("no suitable pg19 book")
        ids = {"input_ids": torch.tensor(text, dtype=torch.long, device=device).unsqueeze(0)}
    else:
        FILLER = ("The history of mathematics spans thousands of years and encompasses many "
                  "different cultures and civilizations. ")
        prompt = FILLER * (args.context_length // 20)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=args.context_length).to(device)
    seq_len = ids["input_ids"].shape[1]
    print(f"Prefill seq_len = {seq_len}  ({args.prompt_source}, cap={args.cache_blocks})")

    # Dense prefill → tiered cache
    adapter.set_mode("dense")
    with torch.inference_mode():
        out = model(**ids, use_cache=True)
    past_kv = out.past_key_values
    first_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del out

    _ensure_certified_imports()
    layer_ids = list(range(model.config.num_hidden_layers))
    tiered = create_tiered_cache_from_model(
        past_kv, layer_ids, fp16_key_cache_capacity=args.cache_blocks,
    )
    del past_kv
    gc.collect()
    torch.cuda.empty_cache()

    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered, layer_epsilons={},
        v_tolerance=0.5,
        collect_stats=False, append_kv=True, top_k_fp16_keys=4,
        tau_cov=0.995, k_min=2, k_max=128,
        ranking_fallback=True, ranking_r=1,
        score_consistency_check=True, eps_guard=0.01,
        exploration_rate=0.02,
        rung1_threshold=0.02, rung1_multiplier=2.0,
    )
    adapter.set_mode("certified")

    def snapshot_cache() -> tuple[int, int, int]:
        """Sum cumulative cache counters across all 32 layer caches."""
        h = m = b = 0
        for c in tiered.values():
            h += int(c._fp16_key_cache_hits)
            m += int(c._fp16_key_cache_misses)
            b += int(c._fp16_key_cache_h2d_bytes)
        return h, m, b

    # Decode with per-step cache snapshots
    cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
    current_input = first_token
    gen_ids = []
    per_token_hits: list[int] = []
    per_token_misses: list[int] = []
    per_token_h2d_bytes: list[int] = []
    per_token_tok_ms: list[float] = []

    prev_h, prev_m, prev_b = snapshot_cache()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    print("Tracing decode…")
    for t in range(args.decode_tokens):
        start_evt.record()
        with torch.inference_mode():
            o = model(
                input_ids=current_input, use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
        end_evt.record()
        torch.cuda.synchronize()
        per_token_tok_ms.append(start_evt.elapsed_time(end_evt))

        h, m, b = snapshot_cache()
        per_token_hits.append(h - prev_h)
        per_token_misses.append(m - prev_m)
        per_token_h2d_bytes.append(b - prev_b)
        prev_h, prev_m, prev_b = h, m, b

        if ref_tokens is not None and t < ref_tokens.shape[0]:
            tid = ref_tokens[t].view(1)
        else:
            tid = o.logits[:, -1, :].argmax(dim=-1)
        gen_ids.append(int(tid.item()))
        current_input = tid.view(1, 1)
        cache_position = cache_position + 1

    per_token_hit_rate = [
        (h / (h + m)) if (h + m) > 0 else 0.0
        for h, m in zip(per_token_hits, per_token_misses)
    ]
    payload = {
        "model": args.model,
        "hardware": torch.cuda.get_device_name(0),
        "context_length": args.context_length,
        "decode_tokens": args.decode_tokens,
        "cache_blocks": args.cache_blocks,
        "prompt_source": args.prompt_source,
        "seq_len": seq_len,
        "generated_text_preview": tokenizer.decode(gen_ids[:64], skip_special_tokens=True),
        "per_token_hits": per_token_hits,
        "per_token_misses": per_token_misses,
        "per_token_h2d_bytes": per_token_h2d_bytes,
        "per_token_hit_rate": per_token_hit_rate,
        "per_token_tok_ms": per_token_tok_ms,
    }
    p = Path(args.output); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {p}")

    # Print a coarse ASCII summary.
    print("\n=== Per-token hit rate (every 16 steps) ===")
    for i in range(0, args.decode_tokens, 16):
        hr = per_token_hit_rate[i]
        mb = per_token_h2d_bytes[i] / (1024 ** 2)
        ms = per_token_tok_ms[i]
        bar = "█" * int(hr * 40)
        print(f"  step {i:>3d}: hit={hr*100:5.1f}%  h2d={mb:6.1f}MB  {ms:6.1f}ms  {bar}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
