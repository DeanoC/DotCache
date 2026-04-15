"""Phase 1: Measure execution floor per (context_bucket, layer, head).

The execution floor is the cosine similarity when skip is disabled (ε=0)
but the full compressed execution path is active. Any error below the
floor is from execution (INT8 keys, top-K FP16 fallback), not skipping.
"""
from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def measure_execution_floor(
    model: Any,
    tokenizer: Any,
    prompts: list[tuple[str, int]],  # (text, target_ctx_len)
    ctx_buckets: list[int],
    device: str = "cuda",
) -> np.ndarray:
    """Measure per-layer, per-head execution floor with no skipping.

    For each prompt, runs dense FP16 (oracle) and certified at ε=0
    (compressed execution only, no skip). Compares per-head outputs.

    Returns:
        execution_floor: [num_ctx_buckets, num_layers, num_heads] float32
        Each entry is the worst (minimum) cosine across all prompts.
    """
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model
    from dotcache.kernels.certified_attention import certified_attention_layer

    config = model.config
    num_layers = config.num_hidden_layers
    num_q = config.num_attention_heads
    num_kv = config.num_key_value_heads
    head_dim = config.hidden_size // num_q
    gqa = num_q // num_kv
    q_scale = 1.0 / (head_dim ** 0.5)
    num_buckets = len(ctx_buckets)

    floor = np.ones((num_buckets, num_layers, num_q), dtype=np.float32)

    for prompt_idx, (prompt_text, target_ctx) in enumerate(prompts):
        bucket_idx = _find_bucket(target_ctx, ctx_buckets)
        print(f"  Floor prompt {prompt_idx+1}/{len(prompts)}: "
              f"{target_ctx//1024}K (bucket {bucket_idx})")

        # Prefill + one decode step for hidden states
        inputs = tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=target_ctx,
        ).to(device)
        seq_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model(**inputs, use_cache=True)
        past_kv = outputs.past_key_values
        first_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        with torch.inference_mode():
            dec = model(
                input_ids=first_token, past_key_values=past_kv,
                use_cache=True, output_hidden_states=True,
            )
        hidden_cpu = [h.detach().cpu() for h in dec.hidden_states]
        del dec

        # Build tiered caches
        caches = create_tiered_cache_from_model(past_kv, list(range(num_layers)))
        del past_kv, outputs
        gc.collect()
        torch.cuda.empty_cache()

        # Per-layer measurement
        for lid in range(num_layers):
            cache = caches[lid]
            attn_wrapper = model.model.layers[lid].self_attn
            attn = attn_wrapper.base_attention if hasattr(attn_wrapper, 'base_attention') else attn_wrapper

            with torch.inference_mode():
                h = hidden_cpu[lid].to(device)
                q_all = attn.q_proj(h).view(num_q, head_dim).to(torch.float32)
                del h

            # Oracle: FP16 per-KV-head
            out_oracle = torch.empty(num_q, head_dim, dtype=torch.float32, device=device)
            for kv in range(num_kv):
                kf = cache.keys_fp16_cpu[kv].to(dtype=torch.float32, device=device)
                vf = cache.values_fp16[kv].to(torch.float32)
                for qh in range(kv * gqa, (kv + 1) * gqa):
                    s = (kf @ q_all[qh]) * q_scale
                    w = torch.softmax(s, dim=0)
                    out_oracle[qh] = w @ vf
                del kf, vf

            # Certified at ε=0 (no skip, but INT8 keys + top-K FP16 fallback)
            out_cert, _ = certified_attention_layer(
                cache, q_all, gqa, q_scale,
                block_epsilon=0.0,  # disable all skipping
                collect_stats=False,
            )

            # Per-head cosine
            cos = F.cosine_similarity(out_oracle, out_cert, dim=1)  # [num_q]
            for qh in range(num_q):
                floor[bucket_idx, lid, qh] = min(
                    floor[bucket_idx, lid, qh],
                    cos[qh].item(),
                )

            del out_oracle, out_cert
            gc.collect()
            torch.cuda.empty_cache()

        del caches, hidden_cpu
        gc.collect()
        torch.cuda.empty_cache()

    return floor


def diagnose_floor_failures(
    floor: np.ndarray,
    ctx_buckets: list[int],
    target_cos: float = 0.999,
) -> list[dict]:
    """Report layers/heads where execution floor < target."""
    failures = []
    num_buckets, num_layers, num_heads = floor.shape
    for bi in range(num_buckets):
        for lid in range(num_layers):
            for hid in range(num_heads):
                if floor[bi, lid, hid] < target_cos:
                    failures.append({
                        "ctx_bucket": ctx_buckets[bi],
                        "layer": lid,
                        "head": hid,
                        "floor_cos": float(floor[bi, lid, hid]),
                        "gap": float(target_cos - floor[bi, lid, hid]),
                    })
    return failures


def _find_bucket(ctx_len: int, ctx_buckets: list[int]) -> int:
    for i, b in enumerate(ctx_buckets):
        if ctx_len <= b:
            return i
    return len(ctx_buckets) - 1
