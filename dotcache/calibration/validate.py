"""Phase 3: End-to-end validation on held-out prompts."""
from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dotcache.calibration.calibrated_profile import CalibratedProfile


def validate_profile(
    model: Any,
    tokenizer: Any,
    profile: CalibratedProfile,
    validation_prompts: list[tuple[str, int]],
    device: str = "cuda",
) -> dict:
    """End-to-end validation on held-out prompts.

    Returns dict with per-bucket worst cosine, per-layer stats, and pass/fail.
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

    # Track worst per (bucket, layer, head)
    num_buckets = len(profile.ctx_buckets)
    worst_cos = np.ones((num_buckets, num_layers, num_q), dtype=np.float32)
    skip_counts = np.zeros((num_buckets, num_layers), dtype=np.int64)
    block_counts = np.zeros((num_buckets, num_layers), dtype=np.int64)

    for pidx, (ptxt, pctx) in enumerate(validation_prompts):
        bucket_idx = profile._find_bucket(pctx)
        print(f"  Validate prompt {pidx+1}/{len(validation_prompts)}: "
              f"{pctx//1024}K (bucket {bucket_idx})")

        inputs = tokenizer(
            ptxt, return_tensors="pt", truncation=True, max_length=pctx,
        ).to(device)

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

        caches = create_tiered_cache_from_model(past_kv, list(range(num_layers)))
        del past_kv, outputs
        gc.collect()
        torch.cuda.empty_cache()

        # Get per-layer epsilons from profile
        layer_eps = profile.get_layer_epsilons_min(pctx)

        for lid in range(num_layers):
            cache = caches[lid]
            eps = layer_eps[lid]
            attn_wrapper = model.model.layers[lid].self_attn
            attn = attn_wrapper.base_attention if hasattr(attn_wrapper, 'base_attention') else attn_wrapper

            with torch.inference_mode():
                h = hidden_cpu[lid].to(device)
                q_all = attn.q_proj(h).view(num_q, head_dim).to(torch.float32)
                del h

            # Oracle
            out_oracle = torch.empty(num_q, head_dim, dtype=torch.float32, device=device)
            for kv in range(num_kv):
                kf = cache.keys_fp16_cpu[kv].to(dtype=torch.float32, device=device)
                vf = cache.values_fp16[kv].to(torch.float32)
                for qh in range(kv * gqa, (kv + 1) * gqa):
                    s = (kf @ q_all[qh]) * q_scale
                    w = torch.softmax(s, dim=0)
                    out_oracle[qh] = w @ vf
                del kf, vf

            # Certified
            out_cert, stats = certified_attention_layer(
                cache, q_all, gqa, q_scale,
                block_epsilon=eps, collect_stats=True,
            )

            cos = F.cosine_similarity(out_oracle, out_cert, dim=1)
            for qh in range(num_q):
                worst_cos[bucket_idx, lid, qh] = min(
                    worst_cos[bucket_idx, lid, qh], cos[qh].item(),
                )

            if stats:
                skip_counts[bucket_idx, lid] += stats.get("skipped_blocks", 0)
                block_counts[bucket_idx, lid] += stats.get("total_blocks", 0)

            del out_oracle, out_cert
            gc.collect()
            torch.cuda.empty_cache()

        del caches, hidden_cpu
        gc.collect()
        torch.cuda.empty_cache()

    # Aggregate results
    violations = []
    for bi in range(num_buckets):
        for lid in range(num_layers):
            for qh in range(num_q):
                if worst_cos[bi, lid, qh] < profile.target_cos:
                    violations.append({
                        "bucket": profile.ctx_buckets[bi],
                        "layer": lid,
                        "head": qh,
                        "cos": float(worst_cos[bi, lid, qh]),
                    })

    per_bucket = {}
    for bi, ctx in enumerate(profile.ctx_buckets):
        bucket_cos = worst_cos[bi]
        total_blocks = block_counts[bi].sum()
        total_skipped = skip_counts[bi].sum()
        per_bucket[ctx] = {
            "cos_min": float(bucket_cos.min()),
            "cos_mean": float(bucket_cos.mean()),
            "cos_p5": float(np.percentile(bucket_cos, 5)),
            "skip_rate": float(total_skipped / max(total_blocks, 1)),
        }

    passed = len(violations) == 0
    margin = float(worst_cos.min() - (1.0 - (1.0 - profile.target_cos)))

    return {
        "passed": passed,
        "violations": violations,
        "num_violations": len(violations),
        "global_cos_min": float(worst_cos.min()),
        "margin": margin,
        "per_bucket": per_bucket,
    }
