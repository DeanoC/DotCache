"""Benchmark: cuBLAS matmul + Triton block-reduce certification pipeline.

Uses cuBLAS for the optimal K@q matmul, then a lightweight Triton kernel
for the per-block max/exp/sum reduction and certification. This avoids
competing with cuBLAS on the matmul while eliminating PyTorch per-op
overhead on the certification arithmetic.
"""
from __future__ import annotations

import os
import sys
import time
import torch
import triton
import triton.language as tl
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Triton kernel: block-level reduce + certification
# ---------------------------------------------------------------------------
@triton.jit
def _block_reduce_kernel(
    Scores_ptr,     # [N] float32 — flat scores from cuBLAS matmul
    M_b_ptr,        # [num_blocks] float32 output
    S_b_ptr,        # [num_blocks] float32 output
    block_size: tl.constexpr,
    num_blocks: tl.constexpr,
    BLOCKS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    for local_b in range(BLOCKS_PER_PROG):
        block_id = pid * BLOCKS_PER_PROG + local_b
        still_valid = block_id < num_blocks
        if still_valid:
            base = block_id * block_size
            offsets = base + tl.arange(0, block_size)
            scores = tl.load(Scores_ptr + offsets)
            m_b = tl.max(scores)
            s_b = tl.sum(tl.exp(scores - m_b))
            tl.store(M_b_ptr + block_id, m_b)
            tl.store(S_b_ptr + block_id, s_b)


@triton.jit
def _certify_kernel(
    M_b_ptr, S_b_ptr, Corr_ptr, Skip_ptr,
    num_blocks: tl.constexpr,
    block_epsilon: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Global max
    m_global = tl.full((), float("-inf"), dtype=tl.float32)
    for start in range(0, num_blocks, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < num_blocks
        m_global = tl.maximum(m_global, tl.max(tl.load(M_b_ptr + offs, mask=mask, other=float("-inf"))))

    # Total mass + skip mask
    total = tl.full((), 0.0, dtype=tl.float32)
    for start in range(0, num_blocks, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < num_blocks
        s = tl.load(S_b_ptr + offs, mask=mask, other=0.0)
        m = tl.load(M_b_ptr + offs, mask=mask, other=float("-inf"))
        c = tl.load(Corr_ptr + offs, mask=mask, other=1.0)
        total += tl.sum(tl.where(mask, s * c * tl.exp(m - m_global), 0.0))

    for start in range(0, num_blocks, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < num_blocks
        s = tl.load(S_b_ptr + offs, mask=mask, other=0.0)
        m = tl.load(M_b_ptr + offs, mask=mask, other=float("-inf"))
        c = tl.load(Corr_ptr + offs, mask=mask, other=1.0)
        res = s * c * tl.exp(m - m_global)
        skip = (res / total) < block_epsilon
        tl.store(Skip_ptr + offs, tl.where(mask, skip.to(tl.int32), 0), mask=mask)


def hybrid_phase1(scores_flat, correction, num_blocks, block_size=16, block_epsilon=0.001):
    """cuBLAS scores → Triton block reduce → Triton certify → skip mask."""
    device = scores_flat.device
    m_b = torch.empty(num_blocks, dtype=torch.float32, device=device)
    S_b = torch.empty(num_blocks, dtype=torch.float32, device=device)
    skip_i32 = torch.empty(num_blocks, dtype=torch.int32, device=device)

    BLOCKS_PER_PROG = max(1, num_blocks // 32)
    n_progs = (num_blocks + BLOCKS_PER_PROG - 1) // BLOCKS_PER_PROG
    _block_reduce_kernel[(n_progs,)](
        scores_flat, m_b, S_b,
        block_size=block_size,
        num_blocks=num_blocks,
        BLOCKS_PER_PROG=BLOCKS_PER_PROG,
    )

    BLOCK_N = min(triton.next_power_of_2(num_blocks), 1024)
    _certify_kernel[(1,)](
        m_b, S_b, correction, skip_i32,
        num_blocks=num_blocks,
        block_epsilon=block_epsilon,
        BLOCK_N=BLOCK_N,
    )
    return m_b, S_b, skip_i32.bool()


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from benchmarks.build_heterogeneous_context import build_context

    token = os.environ.get("HF_TOKEN", "")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B", token=token)
    model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Meta-Llama-3.1-8B", torch_dtype=torch.float16, token=token,
    ).to("cuda")
    model.eval()

    head_dim = 128; block_size = 16
    q_scale_val = 1.0 / (head_dim ** 0.5)

    prompt = build_context(8192, question_idx=0)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    seq_len = inputs["input_ids"].shape[1]
    num_blocks = seq_len // block_size
    N = num_blocks * block_size

    with torch.inference_mode():
        outputs = model(**inputs, use_cache=True, output_hidden_states=True)
    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    with torch.inference_mode():
        decode_out = model(
            input_ids=next_token, past_key_values=past_kv,
            use_cache=True, output_hidden_states=True,
        )

    kc31 = past_kv.layers[31].keys[0]
    vc31 = past_kv.layers[31].values[0]
    attn31 = model.model.layers[31].self_attn
    with torch.inference_mode():
        q_all = attn31.q_proj(decode_out.hidden_states[31]).view(32, head_dim).to(torch.float32)

    kv_idx = 14 // 4
    q_vec = q_all[14]
    keys_fp = kc31[kv_idx, :N, :].to(torch.float32)
    vals_fp = vc31[kv_idx, :N, :].to(torch.float32)
    keys_bl = keys_fp.reshape(num_blocks, block_size, head_dim)

    # INT8 quantise
    k_max = keys_bl.abs().amax(dim=(1, 2)).clamp(min=1e-8)
    k_scale = k_max / 127.0
    keys_i8 = (keys_bl / k_scale.unsqueeze(1).unsqueeze(2)).round().clamp(-127, 127).to(torch.int8)
    keys_deq_flat = (keys_i8.to(torch.float32) * k_scale.unsqueeze(1).unsqueeze(2)).reshape(N, head_dim)

    # Correction
    scores_fp = torch.matmul(keys_bl, q_vec) * q_scale_val
    scores_i8 = torch.matmul(keys_deq_flat.reshape(num_blocks, block_size, head_dim), q_vec) * q_scale_val
    delta = (scores_i8 - scores_fp).abs().max(dim=1).values
    correction = torch.exp(3 * delta)

    # Warmup
    s_test = torch.matmul(keys_deq_flat, q_vec) * q_scale_val
    for _ in range(50):
        hybrid_phase1(s_test, correction, num_blocks)
    torch.cuda.synchronize()

    # Validate
    m_b, S_b, skip = hybrid_phase1(s_test, correction, num_blocks)
    m_ref = s_test.reshape(num_blocks, block_size).max(dim=1).values
    S_ref = torch.exp(s_test.reshape(num_blocks, block_size) - m_ref.unsqueeze(1)).sum(dim=1)
    print(f"Correct: m_b err={( m_b - m_ref).abs().max():.6f}  S_b err={(S_b - S_ref).abs().max():.6f}  skip={skip.sum()}/{num_blocks}")

    iters = 5000

    # 1. Full fp16 attention
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        s = torch.matmul(keys_fp, q_vec) * q_scale_val
        w = torch.softmax(s, dim=0)
        o = w @ vals_fp
    torch.cuda.synchronize()
    t_full = (time.perf_counter() - t0) / iters * 1e6

    # 2. cuBLAS matmul (K_int8_deq @ q)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        s = torch.matmul(keys_deq_flat, q_vec) * q_scale_val
    torch.cuda.synchronize()
    t_matmul = (time.perf_counter() - t0) / iters * 1e6

    # 3. Triton reduce+certify only
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        m_b, S_b, skip = hybrid_phase1(s_test, correction, num_blocks)
    torch.cuda.synchronize()
    t_reduce = (time.perf_counter() - t0) / iters * 1e6

    # 4. Full Phase 1 (cuBLAS + Triton)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        s = torch.matmul(keys_deq_flat, q_vec) * q_scale_val
        m_b, S_b, skip = hybrid_phase1(s, correction, num_blocks)
    torch.cuda.synchronize()
    t_p1 = (time.perf_counter() - t0) / iters * 1e6

    # 5. Phase 2 (selective attend)
    attend_idx = (~skip).nonzero(as_tuple=True)[0]
    ak = keys_fp.reshape(num_blocks, block_size, head_dim)[attend_idx].reshape(-1, head_dim)
    av = vals_fp.reshape(num_blocks, block_size, -1)[attend_idx].reshape(-1, vals_fp.shape[-1])
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        s2 = torch.matmul(ak, q_vec) * q_scale_val
        w2 = torch.softmax(s2, dim=0)
        o2 = w2 @ av
    torch.cuda.synchronize()
    t_p2 = (time.perf_counter() - t0) / iters * 1e6

    # 6. End-to-end
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        s = torch.matmul(keys_deq_flat, q_vec) * q_scale_val
        m_b, S_b, skip = hybrid_phase1(s, correction, num_blocks)
        idx = (~skip).nonzero(as_tuple=True)[0]
        akk = keys_fp.reshape(num_blocks, block_size, head_dim)[idx].reshape(-1, head_dim)
        avv = vals_fp.reshape(num_blocks, block_size, -1)[idx].reshape(-1, vals_fp.shape[-1])
        s2 = torch.matmul(akk, q_vec) * q_scale_val
        w2 = torch.softmax(s2, dim=0)
        o2 = w2 @ avv
    torch.cuda.synchronize()
    t_e2e = (time.perf_counter() - t0) / iters * 1e6

    n_attend = int((~skip).sum().item())
    print(f"\n=== cuBLAS + Triton Pipeline ({iters} iters, μs) ===")
    print(f"  Full fp16 attention:       {t_full:>7.1f} μs (baseline)")
    print(f"  cuBLAS K_i8@q matmul:      {t_matmul:>7.1f} μs ({t_matmul/t_full:.0%})")
    print(f"  Triton reduce+certify:     {t_reduce:>7.1f} μs ({t_reduce/t_full:.0%})")
    print(f"  Phase 1 (matmul+reduce):   {t_p1:>7.1f} μs ({t_p1/t_full:.0%})")
    print(f"  Phase 2 ({n_attend} blocks):        {t_p2:>7.1f} μs ({t_p2/t_full:.0%})")
    print(f"  End-to-end (P1+P2):        {t_e2e:>7.1f} μs")
    print(f"  vs baseline:               {(t_full-t_e2e)/t_full:+.1%}")
    print(f"  Skip: {skip.sum()}/{num_blocks} ({skip.sum()/num_blocks:.1%})")


if __name__ == "__main__":
    main()
