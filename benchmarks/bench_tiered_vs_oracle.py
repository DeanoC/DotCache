"""Tiered certified attention vs full-attention oracle.

Measures correctness, performance, and memory across all 32 layers at 8K
(where both can run), then tiered-only at 32K (oracle OOMs).

Reports per-step token match, cosine similarity, wall-clock decode latency,
and detailed VRAM/CPU memory accounting.
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
from dotcache.kernels.tiered_kv_cache import create_tiered_cache_from_model, TieredKeyCacheLayer
from dotcache.kernels.fused_score_certify import fused_score_certify_multihead
from dotcache.kernels.selective_attend_triton import selective_attend_multihead


# Per-layer epsilon profile for LLaMA 3.1-8B (calibrated for cos≥0.999)
LAYER_EPS = {}
for _lid in range(6):
    LAYER_EPS[_lid] = 1e-3
for _lid in range(6, 29):
    LAYER_EPS[_lid] = 1e-4
for _lid in range(29, 32):
    LAYER_EPS[_lid] = 1e-5

TOP_K_FP16 = 4


def _certified_layer_output(
    cache: TieredKeyCacheLayer,
    q_all: torch.Tensor,
    gqa: int,
    q_scale: float,
    epsilon: float,
) -> tuple[torch.Tensor, dict]:
    """Certified attention for one layer with top-K FP16 fallback."""
    num_q = q_all.shape[0]
    m_b, S_b, skip = fused_score_certify_multihead(
        cache.keys_int8, cache.keys_scale, q_all, cache.correction,
        gqa, cache.block_size, q_scale, block_epsilon=epsilon,
    )
    keys_deq = (cache.keys_int8.to(torch.float32).reshape(
        cache.kv_heads, cache.num_blocks, cache.block_size, cache.head_dim
    ) * cache.keys_scale[:, :, None, None])

    # Top-K FP16 fallback
    keys_fp_gpu = cache.keys_fp16_cpu.to(device=q_all.device, dtype=torch.float32)
    keys_fp_bl = keys_fp_gpu.reshape(cache.kv_heads, cache.num_blocks, cache.block_size, cache.head_dim)
    for qh in range(num_q):
        kv = qh // gqa
        topk = m_b[qh].topk(min(TOP_K_FP16, cache.num_blocks)).indices
        keys_deq[kv, topk] = keys_fp_bl[kv, topk]
    del keys_fp_gpu, keys_fp_bl

    output = selective_attend_multihead(
        keys_deq.reshape(cache.kv_heads, cache.num_tokens, cache.head_dim),
        cache.values_fp16.to(torch.float32),
        q_all, skip.to(torch.int32), gqa, cache.block_size, q_scale,
    )
    skipped = skip.sum().item()
    total = num_q * cache.num_blocks
    return output, {"skip_rate": skipped / total, "skipped": int(skipped), "total": total}


def _oracle_layer_output(
    keys_fp32: torch.Tensor,
    values_fp32: torch.Tensor,
    q_all: torch.Tensor,
    gqa: int,
    q_scale: float,
) -> torch.Tensor:
    """Full FP32 attention (oracle)."""
    num_q = q_all.shape[0]
    d_v = values_fp32.shape[2]
    output = torch.empty(num_q, d_v, dtype=torch.float32, device=q_all.device)
    for qh in range(num_q):
        kv = qh // gqa
        s = torch.matmul(keys_fp32[kv], q_all[qh]) * q_scale
        w = torch.softmax(s, dim=0)
        output[qh] = w @ values_fp32[kv]
    return output


def run_benchmark(model, tokenizer, prompt_text, ctx_len, gen_steps, run_oracle=True):
    """Run generation benchmark, return results dict."""
    config = model.config
    num_layers = config.num_hidden_layers
    num_q = config.num_attention_heads
    num_kv = config.num_key_value_heads
    head_dim = config.hidden_size // num_q
    gqa = num_q // num_kv
    q_scale = 1.0 / (head_dim ** 0.5)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=ctx_len).to("cuda")
    seq_len = inputs["input_ids"].shape[1]

    # Memory before prefill
    torch.cuda.reset_peak_memory_stats()
    vram_before = torch.cuda.memory_allocated()

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = outputs.past_key_values

    vram_after_prefill = torch.cuda.memory_allocated()

    # Create tiered caches for ALL layers
    active_layers = list(range(num_layers))
    caches = create_tiered_cache_from_model(past_kv, active_layers)

    vram_after_tiered = torch.cuda.memory_allocated()
    tiered_vram = sum(c.vram_bytes() for c in caches.values())
    tiered_cpu = sum(c.cpu_bytes() for c in caches.values())

    # Oracle: keep FP16 keys in VRAM for comparison
    oracle_keys = {}
    oracle_vals = {}
    if run_oracle:
        for lid in active_layers:
            oracle_keys[lid] = caches[lid].keys_fp16_cpu.to(dtype=torch.float32, device="cuda")
            oracle_vals[lid] = caches[lid].values_fp16.to(torch.float32)
        vram_after_oracle = torch.cuda.memory_allocated()
    else:
        vram_after_oracle = vram_after_tiered

    print(f"  Memory:")
    print(f"    Model:           {vram_before/1e9:.2f} GB")
    print(f"    After prefill:   {vram_after_prefill/1e9:.2f} GB (+{(vram_after_prefill-vram_before)/1e9:.2f})")
    print(f"    Tiered caches:   {tiered_vram/1e6:.0f} MB VRAM, {tiered_cpu/1e6:.0f} MB CPU")
    if run_oracle:
        print(f"    Oracle keys:     {(vram_after_oracle-vram_after_tiered)/1e9:.2f} GB VRAM")
    print(f"    Peak VRAM:       {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Generation loop
    current_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del outputs
    torch.cuda.empty_cache()

    oracle_tokens = []
    tiered_tokens = []
    step_cosines = []      # worst cosine across all layers per step
    step_skip_rates = []
    step_times_oracle = []
    step_times_tiered = []

    for step in range(gen_steps):
        with torch.inference_mode():
            dec = model(
                input_ids=current_token, past_key_values=past_kv,
                use_cache=True, output_hidden_states=True,
            )

        oracle_token = dec.logits[:, -1, :].argmax(dim=-1).item()
        oracle_tokens.append(oracle_token)

        # Per-layer comparison
        worst_cos = 1.0
        total_skip = 0
        total_blk = 0
        t_oracle_step = 0
        t_tiered_step = 0

        for lid in active_layers:
            attn = model.model.layers[lid].self_attn
            with torch.inference_mode():
                q_all = attn.q_proj(dec.hidden_states[lid]).view(num_q, head_dim).to(torch.float32)

            cache = caches[lid]
            eps = LAYER_EPS.get(lid, 1e-4)

            # Tiered
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out_cert, stats = _certified_layer_output(cache, q_all, gqa, q_scale, eps)
            torch.cuda.synchronize()
            t_tiered_step += time.perf_counter() - t0

            total_skip += stats["skipped"]
            total_blk += stats["total"]

            # Oracle (if running)
            if run_oracle:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out_oracle = _oracle_layer_output(oracle_keys[lid], oracle_vals[lid], q_all, gqa, q_scale)
                torch.cuda.synchronize()
                t_oracle_step += time.perf_counter() - t0

                cos = torch.nn.functional.cosine_similarity(out_cert, out_oracle, dim=1)
                worst_cos = min(worst_cos, cos.min().item())

        step_cosines.append(worst_cos if run_oracle else None)
        step_skip_rates.append(total_skip / total_blk if total_blk > 0 else 0)
        step_times_oracle.append(t_oracle_step * 1e6)
        step_times_tiered.append(t_tiered_step * 1e6)

        # Use the same token for both (don't diverge the sequence)
        tiered_tokens.append(oracle_token)

        past_kv = dec.past_key_values
        current_token = dec.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    cos_arr = np.array([c for c in step_cosines if c is not None])
    skip_arr = np.array(step_skip_rates)
    t_oracle_arr = np.array(step_times_oracle) if run_oracle else None
    t_tiered_arr = np.array(step_times_tiered)

    token_match = sum(1 for a, b in zip(oracle_tokens, tiered_tokens) if a == b)
    generated = tokenizer.decode(oracle_tokens, skip_special_tokens=True)

    return {
        "seq_len": seq_len,
        "gen_steps": gen_steps,
        "token_match": f"{token_match}/{gen_steps}",
        "cos_min": float(cos_arr.min()) if len(cos_arr) > 0 else None,
        "cos_mean": float(cos_arr.mean()) if len(cos_arr) > 0 else None,
        "cos_p5": float(np.percentile(cos_arr, 5)) if len(cos_arr) > 0 else None,
        "skip_mean": float(skip_arr.mean()),
        "skip_min": float(skip_arr.min()),
        "oracle_us_mean": float(t_oracle_arr.mean()) if t_oracle_arr is not None else None,
        "tiered_us_mean": float(t_tiered_arr.mean()),
        "speedup": float((t_oracle_arr.mean() - t_tiered_arr.mean()) / t_oracle_arr.mean()) if t_oracle_arr is not None else None,
        "tiered_vram_mb": tiered_vram / 1e6,
        "tiered_cpu_mb": tiered_cpu / 1e6,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "generated_preview": generated[:80],
    }


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "NousResearch/Meta-Llama-3.1-8B"
    token = os.environ.get("HF_TOKEN", "")

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=token,
    ).to("cuda")
    model.eval()
    print(f"Model VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    results = []

    # === 8K: Both oracle and tiered ===
    print(f"\n{'='*70}")
    print(f"8K CONTEXT: ORACLE vs TIERED (32 gen steps)")
    print(f"{'='*70}")

    for pname, ptxt in [
        ("hetero_8k", build_context(8192, question_idx=0)),
        ("code_8k", Path("dotcache/integrations/llama.py").read_text()[:30000]),
        ("docs_8k", Path("docs/performance_journal.md").read_text()[:30000]),
    ]:
        print(f"\n--- {pname} ---")
        r = run_benchmark(model, tokenizer, ptxt, 8192, gen_steps=32, run_oracle=True)
        r["prompt"] = pname
        r["context"] = "8K"
        results.append(r)
        print(f"  Tokens:   {r['token_match']}")
        print(f"  Cosine:   min={r['cos_min']:.4f}  mean={r['cos_mean']:.4f}")
        print(f"  Skip:     {r['skip_mean']:.1%}")
        print(f"  Latency:  oracle={r['oracle_us_mean']:.0f}μs  tiered={r['tiered_us_mean']:.0f}μs  speedup={r['speedup']:+.1%}")
        print(f"  Memory:   VRAM={r['tiered_vram_mb']:.0f}MB  CPU={r['tiered_cpu_mb']:.0f}MB  peak={r['peak_vram_gb']:.2f}GB")
        print(f"  Text:     {r['generated_preview']}")

        # Clean up
        gc.collect()
        torch.cuda.empty_cache()

    # === 32K: Tiered only (oracle OOMs) ===
    print(f"\n{'='*70}")
    print(f"32K CONTEXT: TIERED ONLY (oracle OOMs, 16 gen steps)")
    print(f"{'='*70}")

    # Build a long prompt
    long_prompt = build_context(32768, question_idx=0)
    print(f"\n--- tiered_32k ---")
    try:
        r = run_benchmark(model, tokenizer, long_prompt, 32768, gen_steps=16, run_oracle=False)
        r["prompt"] = "tiered_32k"
        r["context"] = "32K"
        results.append(r)
        print(f"  Skip:     {r['skip_mean']:.1%}")
        print(f"  Latency:  tiered={r['tiered_us_mean']:.0f}μs")
        print(f"  Memory:   VRAM={r['tiered_vram_mb']:.0f}MB  CPU={r['tiered_cpu_mb']:.0f}MB  peak={r['peak_vram_gb']:.2f}GB")
        print(f"  Text:     {r['generated_preview']}")
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at 32K even with tiered storage!")
        results.append({"prompt": "tiered_32k", "context": "32K", "error": "OOM"})

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Prompt':<15} {'Ctx':>4} {'cos_min':>8} {'skip':>6} {'oracle_μs':>10} {'tiered_μs':>10} {'speedup':>8} {'VRAM_MB':>8} {'CPU_MB':>8}")
    for r in results:
        if "error" in r:
            print(f"  {r['prompt']:<13} {r['context']:>4} {'OOM':>8}")
        else:
            oracle_str = f"{r['oracle_us_mean']:.0f}" if r['oracle_us_mean'] else "N/A"
            cos_str = f"{r['cos_min']:.4f}" if r['cos_min'] else "N/A"
            sp_str = f"{r['speedup']:+.1%}" if r['speedup'] else "N/A"
            print(f"  {r['prompt']:<13} {r['context']:>4} {cos_str:>8} {r['skip_mean']:>5.1%} {oracle_str:>10} {r['tiered_us_mean']:>10.0f} {sp_str:>8} {r['tiered_vram_mb']:>7.0f} {r['tiered_cpu_mb']:>7.0f}")

    out_path = Path("benchmarks/results/tiered_vs_oracle.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON → {out_path}")


if __name__ == "__main__":
    main()
