"""Bound win rates on LLaMA 3.1-8B (head_dim=128) — vectorized.

Computes spherical, interval, and ellipsoidal bounds per block per q_head
using batched GPU operations. Reports which bound wins and the gap at d=128.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _compute_bounds_vectorized(
    keys_by_block: torch.Tensor,  # [num_blocks, block_size, head_dim]
    q_vec: torch.Tensor,          # [head_dim]
    q_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute all three bounds for all blocks at once.

    Returns (actual_max, U_sph, U_int, U_ellip) each [num_blocks].
    """
    num_blocks, bs, hd = keys_by_block.shape
    q = q_vec.unsqueeze(0)  # [1, head_dim]
    q_norm = q_vec.norm().item()

    # Actual max(q·k) per block
    logits = torch.matmul(keys_by_block, q_vec) * q_scale  # [num_blocks, block_size]
    actual_max = logits.max(dim=1).values  # [num_blocks]

    # Centroids
    centers = keys_by_block.mean(dim=1)  # [num_blocks, head_dim]
    center_sim = (centers @ q_vec * q_scale)  # [num_blocks]

    # Spherical: q·μ + ‖q‖·r
    deviations = keys_by_block - centers.unsqueeze(1)  # [num_blocks, bs, hd]
    radii = deviations.norm(dim=-1).max(dim=1).values  # [num_blocks]
    U_sph = center_sim + q_norm * radii * abs(q_scale)

    # Interval: Σ_d max(q_d * k_max_d, q_d * k_min_d)
    k_min = keys_by_block.min(dim=1).values  # [num_blocks, head_dim]
    k_max = keys_by_block.max(dim=1).values  # [num_blocks, head_dim]
    q_scaled = q_vec * q_scale  # [head_dim]
    U_int = torch.where(
        q_scaled.unsqueeze(0) >= 0,
        q_scaled.unsqueeze(0) * k_max,
        q_scaled.unsqueeze(0) * k_min,
    ).sum(dim=-1)  # [num_blocks]

    # Ellipsoidal: q·μ + |q·pc1|·r_along + ‖q_perp‖·r_perp
    # Batched SVD on centered keys
    # For blocks with bs > 1, compute PC1 via batched SVD
    U_ellip = U_sph.clone()  # fallback
    if bs > 1:
        # Batched SVD: [num_blocks, bs, hd]
        try:
            U_svd, S_svd, Vh_svd = torch.linalg.svd(deviations, full_matrices=False)
            pc1 = Vh_svd[:, 0, :]  # [num_blocks, head_dim]

            # Project onto PC1
            proj_along = torch.bmm(
                deviations, pc1.unsqueeze(-1)
            ).squeeze(-1)  # [num_blocks, bs]
            r_along = proj_along.abs().max(dim=1).values  # [num_blocks]

            # Perpendicular residual
            perp = deviations - proj_along.unsqueeze(-1) * pc1.unsqueeze(1)  # [num_blocks, bs, hd]
            r_perp = perp.norm(dim=-1).max(dim=1).values  # [num_blocks]

            # Query projections
            q_dot_pc1 = (pc1 @ q_vec)  # [num_blocks]
            q_perp_sq = q_norm**2 - q_dot_pc1**2
            q_perp_norm = torch.clamp(q_perp_sq, min=0.0).sqrt()

            U_ellip = center_sim + (q_dot_pc1.abs() * r_along + q_perp_norm * r_perp) * abs(q_scale)
        except Exception:
            pass  # fallback to spherical

    return actual_max, U_sph, U_int, U_ellip


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_id = "NousResearch/Meta-Llama-3.1-8B"
    token = os.environ.get("HF_TOKEN", "")
    block_size = 16

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, token=token,
    ).to("cuda")
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_q_heads
    gqa_group = num_q_heads // num_kv_heads

    prompt = Path("docs/performance_journal.md").read_text()[:40000]
    filler = "The quick brown fox jumps over the lazy dog. " * 3000
    prompt = filler + "\n\n" + prompt

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    seq_len = inputs["input_ids"].shape[1]
    num_blocks = seq_len // block_size  # drop partial last block for vectorization
    print(f"Tokens: {seq_len}, Blocks: {num_blocks}, head_dim: {head_dim}")

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = outputs.past_key_values

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.inference_mode():
        decode_out = model(
            input_ids=next_token, past_key_values=past_kv,
            use_cache=True, output_hidden_states=True,
        )
    print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    q_scale = 1.0 / (head_dim ** 0.5)

    total_sph = 0
    total_int = 0
    total_ellip = 0
    total_evals = 0
    per_layer_wins = {}
    all_gaps_sph = []
    all_gaps_int = []
    all_gaps_ellip = []

    for layer_idx in range(num_layers):
        if hasattr(past_kv, 'layers'):
            key_cache = past_kv.layers[layer_idx].keys[0]  # [kv_heads, seq_len, head_dim]
        else:
            key_cache = past_kv[layer_idx][0][0]

        if hasattr(model, 'model'):
            attn = model.model.layers[layer_idx].self_attn
        else:
            attn = model.layers[layer_idx].self_attn

        hidden = decode_out.hidden_states[layer_idx]
        with torch.inference_mode():
            q_proj = attn.q_proj(hidden)
            q_all = q_proj.view(num_q_heads, head_dim).to(torch.float32)

        layer_sph = 0
        layer_int = 0
        layer_ellip = 0
        layer_n = 0

        for kv_head_idx in range(num_kv_heads):
            keys = key_cache[kv_head_idx, :num_blocks * block_size, :].to(dtype=torch.float32)
            # Reshape into blocks: [num_blocks, block_size, head_dim]
            keys_blocked = keys.reshape(num_blocks, block_size, head_dim)

            # Process all q_heads that map to this kv_head
            for q_head_idx in range(kv_head_idx * gqa_group, (kv_head_idx + 1) * gqa_group):
                q_vec = q_all[q_head_idx]

                actual_max, U_sph, U_int, U_ellip = _compute_bounds_vectorized(
                    keys_blocked, q_vec, q_scale
                )

                # Winner per block
                stacked = torch.stack([U_sph, U_int, U_ellip], dim=0)  # [3, num_blocks]
                winners = stacked.argmin(dim=0)  # [num_blocks] — 0=sph, 1=int, 2=ellip

                sph_wins = (winners == 0).sum().item()
                int_wins = (winners == 1).sum().item()
                ellip_wins = (winners == 2).sum().item()

                layer_sph += sph_wins
                layer_int += int_wins
                layer_ellip += ellip_wins
                layer_n += num_blocks

                # Gaps
                all_gaps_sph.extend((U_sph - actual_max).cpu().tolist())
                all_gaps_int.extend((U_int - actual_max).cpu().tolist())
                all_gaps_ellip.extend((U_ellip - actual_max).cpu().tolist())

        total_sph += layer_sph
        total_int += layer_int
        total_ellip += layer_ellip
        total_evals += layer_n
        per_layer_wins[layer_idx] = {
            "sph": layer_sph, "int": layer_int, "ellip": layer_ellip, "n": layer_n,
        }

        if layer_idx % 8 == 0 or layer_idx >= 28:
            n = layer_n
            print(f"  layer {layer_idx:>2}: sph={layer_sph/n:.1%}  int={layer_int/n:.1%}  ellip={layer_ellip/n:.1%}  ({n:,} evals)")

    # Results
    total = total_sph + total_int + total_ellip
    print(f"\n{'='*70}")
    print(f"BOUND WIN RATES: LLaMA 3.1-8B @ {seq_len} tokens (d={head_dim})")
    print(f"{'='*70}")
    print(f"Total evaluations: {total_evals:,}")
    print(f"\n  Spherical:    {total_sph:>8,} ({total_sph/total:.1%})")
    print(f"  Interval:     {total_int:>8,} ({total_int/total:.1%})")
    print(f"  Ellipsoidal:  {total_ellip:>8,} ({total_ellip/total:.1%})")

    gs = np.array(all_gaps_sph)
    gi = np.array(all_gaps_int)
    ge = np.array(all_gaps_ellip)
    best_gap = np.minimum(np.minimum(gs, gi), ge)

    print(f"\nBound gaps (U - actual_max):")
    print(f"  Spherical:   median={np.median(gs):.2f}  p90={np.percentile(gs, 90):.2f}")
    print(f"  Interval:    median={np.median(gi):.2f}  p90={np.percentile(gi, 90):.2f}")
    print(f"  Ellipsoidal: median={np.median(ge):.2f}  p90={np.percentile(ge, 90):.2f}")
    print(f"  Best of 3:   median={np.median(best_gap):.2f}  p90={np.percentile(best_gap, 90):.2f}")

    print(f"\nComparison:")
    print(f"  Qwen3.5 (d=256): sph=40%  int=21%  ellip=39%  best_gap≈9.0")
    print(f"  LLaMA   (d={head_dim}):  sph={total_sph/total:.0%}  int={total_int/total:.0%}  ellip={total_ellip/total:.0%}  best_gap≈{np.median(best_gap):.1f}")


if __name__ == "__main__":
    main()
