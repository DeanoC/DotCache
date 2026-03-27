from __future__ import annotations

import argparse
import json
import math
import time

import torch
from transformers import AutoConfig

from dotcache.backends.torch_mps import _mix_m0_contribution_torch, _mix_m0_contribution_two_group64_torch, _score_m0_logits_flat_torch, _score_m0_logits_two_group64_torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic microbenchmark for the torch M0 decode math path.")
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--bench-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _synchronize(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _bench(device: str, fn, *, warmup_iters: int, bench_iters: int) -> tuple[float, object]:
    for _ in range(max(warmup_iters, 0)):
        fn()
    _synchronize(device)
    start = time.perf_counter()
    result = None
    for _ in range(max(bench_iters, 1)):
        result = fn()
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms / max(bench_iters, 1), result


def _score_m0_logits_einsum_grouped(codes, queries, scales, bias, query_group_sums):
    logits = torch.einsum("bptg,bqg->bqpt", codes, queries).reshape(codes.shape[0], queries.shape[1], -1)
    return logits * scales.reshape(codes.shape[0], 1, -1) + query_group_sums.reshape(codes.shape[0], -1, 1) * bias.reshape(
        codes.shape[0], 1, -1
    )


def _mix_m0_contribution_einsum_grouped(weights, codes, scales, bias):
    weights_flat = weights.reshape(codes.shape[0], weights.shape[1], -1)
    weighted_scales = weights_flat * scales.reshape(codes.shape[0], 1, -1)
    contribution = torch.einsum("bqt,btg->bqg", weighted_scales, codes.reshape(codes.shape[0], -1, codes.shape[-1]))
    bias_term = (weights_flat * bias.reshape(codes.shape[0], 1, -1)).sum(dim=-1, keepdim=True)
    return contribution + bias_term


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")

    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_id)
    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = int(config.num_key_value_heads)
    head_dim = hidden_size // num_attention_heads
    query_count = num_attention_heads // num_key_value_heads
    num_groups = head_dim // int(args.group_size)
    page_count = int(math.ceil(args.prompt_length / args.tokens_per_page))
    token_count = int(args.tokens_per_page)
    device = args.device

    if head_dim % int(args.group_size) != 0:
        raise SystemExit("head_dim must be divisible by group_size")

    codes = torch.randint(
        0,
        16,
        (num_key_value_heads, page_count, token_count, args.group_size),
        dtype=torch.int32,
        device=device,
    ).to(dtype=torch.float32)
    scales = torch.rand((num_key_value_heads, page_count, token_count), dtype=torch.float32, device=device)
    bias = torch.randn((num_key_value_heads, page_count, token_count), dtype=torch.float32, device=device)
    queries = torch.randn((num_key_value_heads, query_count, args.group_size), dtype=torch.float32, device=device)
    query_group_sums = queries.sum(dim=-1)
    def score_flat():
        return _score_m0_logits_flat_torch(codes, queries, scales, bias, query_group_sums)

    def score_einsum():
        return _score_m0_logits_einsum_grouped(codes, queries, scales, bias, query_group_sums)

    def score_fused():
        if fused_scaled_codes is None or fused_queries is None or fused_bias_groups is None:
            raise RuntimeError("fused path is unavailable for this shape")
        return _score_m0_logits_two_group64_torch(fused_scaled_codes, fused_queries, fused_bias_groups, query_group_sums)

    score_flat_ms, score_logits = _bench(
        device,
        score_flat,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    score_einsum_ms, score_logits_einsum = _bench(
        device,
        score_einsum,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    score_max_abs_error = float((score_logits - score_logits_einsum).abs().max().item())

    weights = torch.softmax(score_logits, dim=-1).reshape(num_key_value_heads, query_count, page_count, token_count)

    def mix_flat():
        return _mix_m0_contribution_torch(weights, codes, scales, bias)

    def mix_einsum():
        return _mix_m0_contribution_einsum_grouped(weights, codes, scales, bias)

    def mix_fused():
        if fused_scaled_codes is None or fused_bias_groups is None:
            raise RuntimeError("fused path is unavailable for this shape")
        return _mix_m0_contribution_two_group64_torch(weights, fused_scaled_codes, fused_bias_groups)

    mix_flat_ms, mix_output = _bench(
        device,
        mix_flat,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    mix_einsum_ms, mix_output_einsum = _bench(
        device,
        mix_einsum,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    mix_max_abs_error = float((mix_output - mix_output_einsum).abs().max().item())

    combined_flat_ms, _ = _bench(
        device,
        lambda: _mix_m0_contribution_torch(
            torch.softmax(_score_m0_logits_flat_torch(codes, queries, scales, bias, query_group_sums), dim=-1).reshape(
                num_key_value_heads, query_count, page_count, token_count
            ),
            codes,
            scales,
            bias,
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    combined_einsum_ms, _ = _bench(
        device,
        lambda: _mix_m0_contribution_einsum_grouped(
            torch.softmax(_score_m0_logits_einsum_grouped(codes, queries, scales, bias, query_group_sums), dim=-1).reshape(
                num_key_value_heads, query_count, page_count, token_count
            ),
            codes,
            scales,
            bias,
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )

    two_group_score_flat_ms = None
    two_group_score_fused_ms = None
    two_group_score_fused_speedup = None
    two_group_score_max_abs_error = None
    two_group_mix_flat_ms = None
    two_group_mix_fused_ms = None
    two_group_mix_fused_speedup = None
    two_group_mix_max_abs_error = None
    two_group_combined_flat_ms = None
    two_group_combined_fused_ms = None
    two_group_combined_fused_speedup = None
    if num_groups == 2 and head_dim == 64 and args.group_size == 32:
        pair_codes = torch.randint(
            0,
            16,
            (num_key_value_heads, page_count, token_count, 2, args.group_size),
            dtype=torch.int32,
            device=device,
        ).to(dtype=torch.float32)
        pair_scales = torch.rand((num_key_value_heads, page_count, token_count, 2), dtype=torch.float32, device=device)
        pair_bias = torch.randn((num_key_value_heads, page_count, token_count, 2), dtype=torch.float32, device=device)
        pair_queries = torch.randn((num_key_value_heads, query_count, 2, args.group_size), dtype=torch.float32, device=device)
        pair_query_sums = pair_queries.sum(dim=-1)
        pair_fused_scaled_codes = torch.cat(
            [
                pair_codes[:, :, :, 0, :] * pair_scales[:, :, :, 0, None],
                pair_codes[:, :, :, 1, :] * pair_scales[:, :, :, 1, None],
            ],
            dim=-1,
        ).contiguous()
        pair_fused_queries = pair_queries.reshape(num_key_value_heads, query_count, head_dim).contiguous()
        pair_bias_groups = (pair_bias[:, :, :, 0], pair_bias[:, :, :, 1])

        def two_group_score_flat():
            return (
                _score_m0_logits_flat_torch(
                    pair_codes[:, :, :, 0, :],
                    pair_queries[:, :, 0, :],
                    pair_scales[:, :, :, 0],
                    pair_bias[:, :, :, 0],
                    pair_query_sums[:, :, 0],
                )
                + _score_m0_logits_flat_torch(
                    pair_codes[:, :, :, 1, :],
                    pair_queries[:, :, 1, :],
                    pair_scales[:, :, :, 1],
                    pair_bias[:, :, :, 1],
                    pair_query_sums[:, :, 1],
                )
            )

        def two_group_score_fused():
            return _score_m0_logits_two_group64_torch(
                pair_fused_scaled_codes,
                pair_fused_queries,
                pair_bias_groups,
                pair_query_sums,
            )

        two_group_score_flat_ms, two_group_score_logits = _bench(
            device,
            two_group_score_flat,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
        two_group_score_fused_ms, two_group_score_logits_fused = _bench(
            device,
            two_group_score_fused,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
        two_group_score_fused_speedup = two_group_score_flat_ms / max(two_group_score_fused_ms, 1e-9)
        two_group_score_max_abs_error = float((two_group_score_logits - two_group_score_logits_fused).abs().max().item())

        two_group_weights = torch.softmax(two_group_score_logits, dim=-1).reshape(num_key_value_heads, query_count, page_count, token_count)

        def two_group_mix_flat():
            return torch.cat(
                [
                    _mix_m0_contribution_torch(
                        two_group_weights,
                        pair_codes[:, :, :, 0, :],
                        pair_scales[:, :, :, 0],
                        pair_bias[:, :, :, 0],
                    ),
                    _mix_m0_contribution_torch(
                        two_group_weights,
                        pair_codes[:, :, :, 1, :],
                        pair_scales[:, :, :, 1],
                        pair_bias[:, :, :, 1],
                    ),
                ],
                dim=-1,
            )

        def two_group_mix_fused():
            return _mix_m0_contribution_two_group64_torch(
                two_group_weights,
                pair_fused_scaled_codes,
                pair_bias_groups,
            )

        two_group_mix_flat_ms, two_group_mix_output = _bench(
            device,
            two_group_mix_flat,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
        two_group_mix_fused_ms, two_group_mix_output_fused = _bench(
            device,
            two_group_mix_fused,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
        two_group_mix_fused_speedup = two_group_mix_flat_ms / max(two_group_mix_fused_ms, 1e-9)
        two_group_mix_max_abs_error = float((two_group_mix_output - two_group_mix_output_fused).abs().max().item())

        def two_group_combined_flat():
            weights = torch.softmax(two_group_score_flat(), dim=-1).reshape(num_key_value_heads, query_count, page_count, token_count)
            return torch.cat(
                [
                    _mix_m0_contribution_torch(
                        weights,
                        pair_codes[:, :, :, 0, :],
                        pair_scales[:, :, :, 0],
                        pair_bias[:, :, :, 0],
                    ),
                    _mix_m0_contribution_torch(
                        weights,
                        pair_codes[:, :, :, 1, :],
                        pair_scales[:, :, :, 1],
                        pair_bias[:, :, :, 1],
                    ),
                ],
                dim=-1,
            )

        def two_group_combined_fused():
            weights = torch.softmax(two_group_score_fused(), dim=-1).reshape(num_key_value_heads, query_count, page_count, token_count)
            return _mix_m0_contribution_two_group64_torch(
                weights,
                pair_fused_scaled_codes,
                pair_bias_groups,
            )

        two_group_combined_flat_ms, _ = _bench(
            device,
            two_group_combined_flat,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
        two_group_combined_fused_ms, _ = _bench(
            device,
            two_group_combined_fused,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
        two_group_combined_fused_speedup = two_group_combined_flat_ms / max(two_group_combined_fused_ms, 1e-9)

    print(
        json.dumps(
            {
                "benchmark": "torch_decode_micro",
                "model_id": args.model_id,
                "device": args.device,
                "prompt_length": args.prompt_length,
                "tokens_per_page": args.tokens_per_page,
                "group_count": num_key_value_heads,
                "query_count": query_count,
                "page_count": page_count,
                "token_count": token_count,
                "head_dim": head_dim,
                "group_size": args.group_size,
                "num_groups": num_groups,
                "warmup_iters": args.warmup_iters,
                "bench_iters": args.bench_iters,
                "score_flat_ms": score_flat_ms,
                "score_einsum_ms": score_einsum_ms,
                "score_speedup_vs_einsum": score_einsum_ms / max(score_flat_ms, 1e-9),
                "score_max_abs_error": score_max_abs_error,
                "mix_flat_ms": mix_flat_ms,
                "mix_einsum_ms": mix_einsum_ms,
                "mix_speedup_vs_einsum": mix_einsum_ms / max(mix_flat_ms, 1e-9),
                "mix_max_abs_error": mix_max_abs_error,
                "combined_flat_ms": combined_flat_ms,
                "combined_einsum_ms": combined_einsum_ms,
                "combined_speedup_vs_einsum": combined_einsum_ms / max(combined_flat_ms, 1e-9),
                "two_group_score_flat_ms": two_group_score_flat_ms,
                "two_group_score_fused_ms": two_group_score_fused_ms,
                "two_group_score_fused_speedup_vs_flat": two_group_score_fused_speedup,
                "two_group_score_max_abs_error": two_group_score_max_abs_error,
                "two_group_mix_flat_ms": two_group_mix_flat_ms,
                "two_group_mix_fused_ms": two_group_mix_fused_ms,
                "two_group_mix_fused_speedup_vs_flat": two_group_mix_fused_speedup,
                "two_group_mix_max_abs_error": two_group_mix_max_abs_error,
                "two_group_combined_flat_ms": two_group_combined_flat_ms,
                "two_group_combined_fused_ms": two_group_combined_fused_ms,
                "two_group_combined_fused_speedup_vs_flat": two_group_combined_fused_speedup,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
