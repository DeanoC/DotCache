"""Phase 2: Sequential layer-by-layer epsilon calibration.

For each layer (in order), sweep epsilon while all preceding layers run
with their locked epsilons. Uses batched-heads optimization: evaluate all
heads at the same epsilon simultaneously, then check each head's cosine.
"""
from __future__ import annotations

import gc
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


# Binary search over log-space epsilon values
EPSILON_LO = 1e-8
EPSILON_HI = 1e-1
BINARY_SEARCH_STEPS = 10


def sequential_epsilon_calibration(
    model: Any,
    tokenizer: Any,
    prompts: list[tuple[str, int]],  # (text, target_ctx_len)
    ctx_buckets: list[int],
    execution_floor: np.ndarray,  # [buckets, layers, heads]
    target_cos: float = 0.999,
    search_steps: int = BINARY_SEARCH_STEPS,
    device: str = "cuda",
) -> np.ndarray:
    """Sequential layer-by-layer epsilon calibration with batched heads.

    For layer L:
      - Layers 0..L-1 run with locked (calibrated) epsilons
      - Layer L is swept (all heads at same candidate ε)
      - Layers L+1..N-1 run at ε=0

    Returns:
        epsilon: [num_ctx_buckets, num_layers, num_heads] float32
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

    epsilon = np.zeros((num_buckets, num_layers, num_q), dtype=np.float32)

    for bucket_idx, ctx_target in enumerate(ctx_buckets):
        print(f"\n{'='*50}")
        print(f"  Calibrating {ctx_target//1024}K context")
        print(f"{'='*50}")

        # Collect prompts for this bucket
        bucket_prompts = [
            (t, c) for t, c in prompts
            if _find_bucket(c, ctx_buckets) == bucket_idx
        ]
        if not bucket_prompts:
            # Copy from nearest smaller bucket
            if bucket_idx > 0:
                epsilon[bucket_idx] = epsilon[bucket_idx - 1]
                print(f"  No prompts, copying from {ctx_buckets[bucket_idx-1]//1024}K")
            continue

        # Prepare prompts lazily (one at a time) to manage VRAM at 32K+
        # Store only lightweight data (hidden states CPU, oracle outputs CPU)
        # Rebuild tiered caches on demand per-layer
        prompt_data = []
        for pidx, (ptxt, pctx) in enumerate(bucket_prompts):
            print(f"    Preparing prompt {pidx+1}/{len(bucket_prompts)}")
            data = _prepare_prompt(model, tokenizer, ptxt, pctx, num_layers, device)
            prompt_data.append(data)
            gc.collect()
            torch.cuda.empty_cache()

        # Sequential layer calibration
        for lid in range(num_layers):
            layer_floor = execution_floor[bucket_idx, lid, :]  # [num_heads]

            # Determine per-head skip budget
            has_budget = layer_floor >= target_cos
            no_budget_heads = (~torch.tensor(has_budget)).sum().item()
            if no_budget_heads > 0:
                pass  # Some heads have no budget — they'll get ε=0

            # Binary search: find largest ε where all-heads cos ≥ target
            # (batched: same ε for all heads, check individually)
            lo = math.log10(EPSILON_LO)
            hi = math.log10(EPSILON_HI)
            best_per_head = np.zeros(num_q, dtype=np.float32)

            for step in range(search_steps):
                mid = (lo + hi) / 2
                eps_candidate = 10 ** mid

                # Evaluate this ε on all prompts, all heads
                worst_cos_per_head = np.ones(num_q, dtype=np.float32)

                for pdata in prompt_data:
                    cos_heads = _evaluate_layer_epsilon(
                        model, pdata, lid, eps_candidate,
                        epsilon[bucket_idx],
                        num_q, num_kv, head_dim, gqa, q_scale, device,
                    )
                    for qh in range(num_q):
                        worst_cos_per_head[qh] = min(
                            worst_cos_per_head[qh], cos_heads[qh],
                        )
                    gc.collect()
                    torch.cuda.empty_cache()

                # Check: how many heads pass at this ε?
                passes = worst_cos_per_head >= target_cos
                all_budgeted_pass = np.all(passes | ~has_budget)

                if all_budgeted_pass:
                    # All heads with budget pass — try larger ε
                    for qh in range(num_q):
                        if has_budget[qh] and passes[qh]:
                            best_per_head[qh] = eps_candidate
                    lo = mid
                else:
                    hi = mid

            # For heads without budget, ε stays 0
            for qh in range(num_q):
                if not has_budget[qh]:
                    best_per_head[qh] = 0.0

            epsilon[bucket_idx, lid, :] = best_per_head

            # Report
            active = (best_per_head > 0).sum()
            if active > 0:
                mean_eps = best_per_head[best_per_head > 0].mean()
                print(f"  L{lid:2d}: {active}/{num_q} heads active, "
                      f"ε=[{best_per_head[best_per_head>0].min():.1e}, "
                      f"{best_per_head.max():.1e}]")
            else:
                print(f"  L{lid:2d}: 0/{num_q} heads active (no budget)")

        # Cleanup prompt data
        del prompt_data
        gc.collect()
        torch.cuda.empty_cache()

    return epsilon


def _prepare_prompt(model, tokenizer, prompt_text, ctx_len, num_layers, device):
    """Prefill, get hidden states, build caches, compute oracle outputs."""
    from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model

    config = model.config
    num_q = config.num_attention_heads
    num_kv = config.num_key_value_heads
    head_dim = config.hidden_size // num_q
    gqa = num_q // num_kv
    q_scale = 1.0 / (head_dim ** 0.5)

    inputs = tokenizer(
        prompt_text, return_tensors="pt",
        truncation=True, max_length=ctx_len,
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
    del dec, outputs

    # Build tiered caches, compute oracle outputs, then free caches
    # This keeps VRAM usage low at long contexts
    caches = create_tiered_cache_from_model(past_kv, list(range(num_layers)))
    del past_kv
    gc.collect()
    torch.cuda.empty_cache()

    oracle_outputs = {}
    # Store per-layer cache data on CPU for reconstruction
    cache_cpu_data = {}
    for lid in range(num_layers):
        cache = caches[lid]
        attn_wrapper = model.model.layers[lid].self_attn
        attn = attn_wrapper.base_attention if hasattr(attn_wrapper, 'base_attention') else attn_wrapper
        with torch.inference_mode():
            h = hidden_cpu[lid].to(device)
            q_all = attn.q_proj(h).view(num_q, head_dim).to(torch.float32)
            del h

        out_oracle = torch.empty(num_q, head_dim, dtype=torch.float32, device=device)
        for kv in range(num_kv):
            kf = cache.keys_fp16_cpu[kv].to(dtype=torch.float32, device=device)
            vf = cache.values_fp16[kv].to(torch.float32)
            for qh in range(kv * gqa, (kv + 1) * gqa):
                s = (kf @ q_all[qh]) * q_scale
                w = torch.softmax(s, dim=0)
                out_oracle[qh] = w @ vf
            del kf, vf

        oracle_outputs[lid] = out_oracle.cpu()
        del out_oracle

        # Save cache tensors to CPU for later reconstruction
        cache_cpu_data[lid] = {
            "keys_int8": cache.keys_int8.cpu(),
            "keys_scale": cache.keys_scale.cpu(),
            "correction": cache.correction.cpu(),
            "values_fp16": cache.values_fp16.cpu(),
            "keys_fp16_cpu": cache.keys_fp16_cpu,
            "kv_heads": cache.kv_heads,
            "num_tokens": cache.num_tokens,
            "head_dim": cache.head_dim,
            "d_v": cache.d_v,
            "block_size": cache.block_size,
            "num_blocks": cache.num_blocks,
        }

    del caches
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "hidden_cpu": hidden_cpu,
        "cache_cpu_data": cache_cpu_data,
        "oracle_outputs": oracle_outputs,
    }


def _evaluate_layer_epsilon(
    model, pdata, layer_id, epsilon_candidate,
    locked_epsilon_array,
    num_q, num_kv, head_dim, gqa, q_scale, device,
) -> np.ndarray:
    """Evaluate one epsilon candidate on one prompt for one layer.

    Reconstructs the tiered cache on GPU from CPU data, runs certified
    attention, then frees. This keeps VRAM usage bounded.

    Returns: [num_heads] cosine values.
    """
    from dotcache.kernels.certified_attention import certified_attention_layer
    from dotcache.kernels.tiered_kv_cache import TieredKeyCacheLayer

    # Reconstruct cache on GPU for this layer
    cd = pdata["cache_cpu_data"][layer_id]
    cache = TieredKeyCacheLayer(
        keys_int8=cd["keys_int8"].to(device),
        keys_scale=cd["keys_scale"].to(device),
        correction=cd["correction"].to(device),
        values_fp16=cd["values_fp16"].to(device),
        keys_fp16_cpu=cd["keys_fp16_cpu"],
        kv_heads=cd["kv_heads"],
        num_tokens=cd["num_tokens"],
        head_dim=cd["head_dim"],
        d_v=cd["d_v"],
        block_size=cd["block_size"],
        num_blocks=cd["num_blocks"],
    )

    attn_wrapper = model.model.layers[layer_id].self_attn
    attn = attn_wrapper.base_attention if hasattr(attn_wrapper, 'base_attention') else attn_wrapper
    with torch.inference_mode():
        h = pdata["hidden_cpu"][layer_id].to(device)
        q_all = attn.q_proj(h).view(num_q, head_dim).to(torch.float32)
        del h

    out_cert, _ = certified_attention_layer(
        cache, q_all, gqa, q_scale,
        block_epsilon=epsilon_candidate,
        collect_stats=False,
    )

    oracle = pdata["oracle_outputs"][layer_id].to(device)
    cos = F.cosine_similarity(oracle, out_cert, dim=1)

    result = cos.cpu().numpy()
    del out_cert, oracle, cache
    return result


def _find_bucket(ctx_len: int, ctx_buckets: list[int]) -> int:
    for i, b in enumerate(ctx_buckets):
        if ctx_len <= b:
            return i
    return len(ctx_buckets) - 1
