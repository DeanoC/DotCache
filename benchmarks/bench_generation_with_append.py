"""Generation with KV append: certified decode that grows context.

Validates that certified attention with KV append produces coherent text
and matches dense generation token-for-token.
"""
from __future__ import annotations

import os
import sys
import gc
import time
import json
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.build_heterogeneous_context import build_context

LAYER_EPS = {}
for _lid in range(6):
    LAYER_EPS[_lid] = 1e-3
for _lid in range(6, 29):
    LAYER_EPS[_lid] = 1e-4
for _lid in range(29, 32):
    LAYER_EPS[_lid] = 1e-5


def run_generation_comparison(model, adapter, tokenizer, ctx_len, gen_steps=32):
    """Compare dense vs certified-with-append generation."""
    from dotcache.integrations.llama import _ensure_certified_imports, CertifiedAttentionState
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model

    prompt = build_context(ctx_len, question_idx=0)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=ctx_len).to("cuda")
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Prompt: {seq_len} tokens")

    # === Dense generation (ground truth) ===
    adapter.set_mode("dense")
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    past_kv_dense = outputs.past_key_values
    current_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    dense_tokens = []

    for _ in range(gen_steps):
        with torch.inference_mode():
            out = model(input_ids=current_token, past_key_values=past_kv_dense, use_cache=True)
        tid = out.logits[:, -1, :].argmax(dim=-1).item()
        dense_tokens.append(tid)
        past_kv_dense = out.past_key_values
        current_token = torch.tensor([[tid]], dtype=torch.long, device="cuda")

    del past_kv_dense
    gc.collect()
    torch.cuda.empty_cache()

    # === Certified generation with KV append ===
    adapter.set_mode("dense")
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Build tiered caches
    _ensure_certified_imports()
    layer_ids = list(range(model.config.num_hidden_layers))
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model
    tiered_caches = create_tiered_cache_from_model(past_kv, layer_ids)
    del outputs, past_kv
    gc.collect()
    torch.cuda.empty_cache()

    adapter.certified_state = CertifiedAttentionState(
        tiered_caches=tiered_caches,
        layer_epsilons=LAYER_EPS,
        default_epsilon=1e-4,
        v_tolerance=0.5,
        collect_stats=True,
        append_kv=True,  # Enable KV append
    )
    adapter.set_mode("certified")

    cache_position = torch.tensor([seq_len], dtype=torch.long, device="cuda")
    current_token = first_token
    cert_tokens = []
    step_stats = []

    for step in range(gen_steps):
        adapter.certified_state.clear_step_stats()
        with torch.inference_mode():
            out = model(
                input_ids=current_token,
                use_cache=False,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
            )
        tid = out.logits[:, -1, :].argmax(dim=-1).item()
        cert_tokens.append(tid)
        agg = adapter.certified_state.aggregate_step_stats()
        step_stats.append(agg)
        current_token = torch.tensor([[tid]], dtype=torch.long, device="cuda")
        cache_position = cache_position + 1

    # Compare
    token_match = sum(1 for a, b in zip(dense_tokens, cert_tokens) if a == b)
    skip_rates = [s.get("skip_rate", 0) for s in step_stats]

    return {
        "seq_len": seq_len,
        "gen_steps": gen_steps,
        "token_match": f"{token_match}/{gen_steps}",
        "token_match_rate": token_match / gen_steps,
        "skip_rate_mean": float(np.mean(skip_rates)),
        "dense_text": tokenizer.decode(dense_tokens, skip_special_tokens=True)[:120],
        "cert_text": tokenizer.decode(cert_tokens, skip_special_tokens=True)[:120],
        "first_diverge": next(
            (i for i, (a, b) in enumerate(zip(dense_tokens, cert_tokens)) if a != b),
            gen_steps,
        ),
    }


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from dotcache.integrations.llama import LlamaDotCacheModelAdapter
    from dotcache.config import DotCacheConfig

    model_id = "NousResearch/Meta-Llama-3.1-8B"
    token = os.environ.get("HF_TOKEN") or None

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=token,
    ).to("cuda")
    model.eval()

    config = DotCacheConfig(head_dim=model.config.hidden_size // model.config.num_attention_heads)
    adapter = LlamaDotCacheModelAdapter(model, config)

    results = []

    for ctx_len, label in [(4096, "4K"), (8192, "8K")]:
        print(f"\n{'='*60}")
        print(f"  {label} context, generation with KV append")
        print(f"{'='*60}")

        r = run_generation_comparison(model, adapter, tokenizer, ctx_len, gen_steps=32)
        r["context"] = label
        results.append(r)

        print(f"  Token match: {r['token_match']}")
        print(f"  First diverge: step {r['first_diverge']}")
        print(f"  Skip rate: {r['skip_rate_mean']:.1%}")
        print(f"  Dense: {r['dense_text'][:80]}")
        print(f"  Cert:  {r['cert_text'][:80]}")

        adapter.clear()
        adapter.set_mode("dense")
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"GENERATION WITH KV APPEND")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['context']}: match={r['token_match']}, diverge@{r['first_diverge']}, skip={r['skip_rate_mean']:.1%}")

    out_path = Path("benchmarks/results/generation_with_append.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON -> {out_path}")


if __name__ == "__main__":
    main()
