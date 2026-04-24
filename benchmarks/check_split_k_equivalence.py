"""Compare next-token output between original hybrid and split-K kernels.

Runs a short certified decode with each kernel (toggled via env) and checks
generated tokens match. The kernels differ in summation order (not
algorithm), so logits should match to within FP32 noise; argmax should be
identical for almost all steps.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys

import torch


def run_decode(ctx_len: int, steps: int, fast: bool):
    os.environ["DOTCACHE_FAST_ATTEND"] = "1" if fast else "0"

    # Force a fresh import of certified_attention so the env var is
    # re-read inside the call site (it's read per-step so reload not needed,
    # but we invalidate just to be defensive).
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from dotcache.integrations.llama import (
        LlamaDotCacheModelAdapter, CertifiedAttentionState, _ensure_certified_imports,
    )
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model
    from dotcache.config import DotCacheConfig

    tok_env = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B", token=tok_env)
    quant = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Meta-Llama-3.1-8B", quantization_config=quant, device_map="auto", token=tok_env,
    )
    model.eval()
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    cfg = DotCacheConfig(head_dim=head_dim)
    adapter = LlamaDotCacheModelAdapter(model, cfg)
    device = next(model.parameters()).device

    FILLER = "The history of mathematics spans thousands of years. "
    question = "\nContinue:"
    ft = len(tokenizer.encode(FILLER, add_special_tokens=False))
    qt = len(tokenizer.encode(question, add_special_tokens=False))
    avail = ctx_len - qt - 50
    nb = max(avail // ft, 2)
    prompt = FILLER * nb + question

    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to(device)
    adapter.set_mode("dense")
    with torch.inference_mode():
        out = model(**ids, use_cache=True)
    past_kv = out.past_key_values
    first_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del out

    _ensure_certified_imports()
    layer_ids = list(range(model.config.num_hidden_layers))
    tiered = create_tiered_cache_from_model(past_kv, layer_ids, fp16_key_cache_capacity=None)
    del past_kv
    gc.collect(); torch.cuda.empty_cache()

    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered,
        layer_epsilons={},
        v_tolerance=0.5,
        collect_stats=False,
        append_kv=True,
        top_k_fp16_keys=4,
        tau_cov=0.995, k_min=2, k_max=128,
        ranking_fallback=True, ranking_r=1,
        score_consistency_check=True, eps_guard=0.01,
        exploration_rate=0.02,
        rung1_threshold=0.02, rung1_multiplier=2.0,
    )
    adapter.set_mode("certified")

    seq_len = ids["input_ids"].shape[1]
    cache_position = torch.tensor([seq_len], dtype=torch.long, device=device)
    current_input = first_tok
    toks = [int(first_tok.item())]
    for _ in range(steps):
        with torch.inference_mode():
            out = model(
                input_ids=current_input, use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
        tid = out.logits[:, -1, :].argmax(dim=-1)
        toks.append(int(tid.item()))
        current_input = tid.view(1, 1)
        cache_position = cache_position + 1

    del model, adapter
    gc.collect(); torch.cuda.empty_cache()
    return toks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context-length", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()

    print(f"ctx={args.context_length} steps={args.steps}")
    print("Running original hybrid kernel (DOTCACHE_FAST_ATTEND=0)...")
    toks_old = run_decode(args.context_length, args.steps, fast=False)
    print(f"  tokens: {toks_old}")

    print("Running split-K kernel (DOTCACHE_FAST_ATTEND=1)...")
    toks_new = run_decode(args.context_length, args.steps, fast=True)
    print(f"  tokens: {toks_new}")

    matches = sum(1 for a, b in zip(toks_old, toks_new) if a == b)
    print(f"\nMatch: {matches}/{len(toks_old)} tokens identical")
    if toks_old != toks_new:
        first_div = next(i for i, (a, b) in enumerate(zip(toks_old, toks_new)) if a != b)
        print(f"  First divergence at step {first_div}: old={toks_old[first_div]} new={toks_new[first_div]}")
    return 0 if toks_old == toks_new else 1


if __name__ == "__main__":
    sys.exit(main())
