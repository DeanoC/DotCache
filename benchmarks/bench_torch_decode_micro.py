from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig

from dotcache.backends.torch_mps import (
    _mix_m0_contribution_fused_torch,
    _mix_m0_contribution_two_group64_torch,
    _score_exact_logits_flat_torch,
    _score_exact_logits_paged_torch,
    _score_exact_logits_transposed_torch,
    _score_m0_logits_fused_torch,
    _score_m0_logits_fused_transposed_torch,
    _score_m0_logits_two_group64_torch,
)
from dotcache.integrations.llama import resolve_hf_auth_kwargs
from dotcache.modes.m0_affine import dequantize_groups, quantize_tensor


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Microbenchmark the Torch M0 decode kernels against the current mixed-execution reconstruct path."
    )
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=_default_device())
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--quant-scheme-k", choices=["affine", "symmetric"], default="affine")
    parser.add_argument("--quant-scheme-v", choices=["affine", "symmetric"], default="affine")
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--query-count", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--runtime-shaped-profile",
        choices=[
            "none",
            "qwen35_bias_broad",
            "qwen35_hand_broad",
            "qwen35_bias_large",
            "qwen35_hand_large",
        ],
        default="none",
    )
    parser.add_argument("--runtime-m0-pages", type=int, default=0)
    parser.add_argument("--runtime-m3-pages", type=int, default=0)
    parser.add_argument("--direct-m0-crossover-sweep-pages", default="")
    parser.add_argument("--output-format", choices=["pretty", "json"], default="pretty")
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


def _resolve_shape(args: argparse.Namespace) -> tuple[int, int, int]:
    if args.head_dim is not None and args.num_key_value_heads is not None and args.query_count is not None:
        return int(args.head_dim), int(args.num_key_value_heads), int(args.query_count)
    config = AutoConfig.from_pretrained(args.model_id, **resolve_hf_auth_kwargs())
    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    num_key_value_heads = int(getattr(config, "num_key_value_heads", num_attention_heads))
    head_dim = int(args.head_dim) if args.head_dim is not None else hidden_size // num_attention_heads
    kv_head_count = int(args.num_key_value_heads) if args.num_key_value_heads is not None else num_key_value_heads
    query_count = int(args.query_count) if args.query_count is not None else num_attention_heads // num_key_value_heads
    return head_dim, kv_head_count, query_count

def _mix_exact_torch(weights: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    batch_size, _page_count, _token_count, head_dim = map(int, values.shape)
    value_flat = values.reshape(batch_size, -1, head_dim)
    return torch.bmm(weights.reshape(batch_size, int(weights.shape[1]), -1), value_flat).to(torch.float32)


def _quantize_grouped_blocks_to_numpy(
    *,
    values: torch.Tensor,
    group_size: int,
    bits: int,
    scheme: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_cpu = values.detach().cpu()
    batch_size, page_count, _token_count, _head_dim = map(int, values.shape)
    codes_rows: list[list[np.ndarray]] = []
    scales_rows: list[list[np.ndarray]] = []
    bias_rows: list[list[np.ndarray]] = []
    for batch_index in range(batch_size):
        batch_codes: list[np.ndarray] = []
        batch_scales: list[np.ndarray] = []
        batch_bias: list[np.ndarray] = []
        for page_index in range(page_count):
            page_values = np.asarray(values_cpu[batch_index, page_index].numpy(), dtype=np.float32)
            codes, scales, bias, _padded_head_dim = quantize_tensor(
                page_values,
                group_size=group_size,
                bits=bits,
                scheme=scheme,
            )
            batch_codes.append(np.asarray(codes, dtype=np.float32))
            batch_scales.append(np.asarray(scales, dtype=np.float32))
            batch_bias.append(np.asarray(bias, dtype=np.float32))
        codes_rows.append(batch_codes)
        scales_rows.append(batch_scales)
        bias_rows.append(batch_bias)
    return (
        np.asarray(codes_rows, dtype=np.float32),
        np.asarray(scales_rows, dtype=np.float32),
        np.asarray(bias_rows, dtype=np.float32),
    )


def _prepare_fused_m0_inputs(
    *,
    codes_np: np.ndarray,
    scales_np: np.ndarray,
    bias_np: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    codes = torch.as_tensor(codes_np, dtype=torch.float32, device=device)
    scales = torch.as_tensor(scales_np, dtype=torch.float32, device=device)
    bias = torch.as_tensor(bias_np, dtype=torch.float32, device=device)
    num_groups = int(codes.shape[-2])
    fused_scaled_codes = torch.cat(
        [codes[..., group_index, :] * scales[..., group_index, None] for group_index in range(num_groups)],
        dim=-1,
    ).contiguous()
    bias_groups = tuple(bias[..., group_index].contiguous() for group_index in range(num_groups))
    return fused_scaled_codes, bias_groups


def _cached_dequantize_blocks_to_device(
    *,
    codes_np: np.ndarray,
    scales_np: np.ndarray,
    bias_np: np.ndarray,
    bits: int,
    scheme: str,
    head_dim: int,
    device: str,
) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    batch_size, page_count = map(int, codes_np.shape[:2])
    for batch_index in range(batch_size):
        page_blocks: list[torch.Tensor] = []
        for page_index in range(page_count):
            reconstructed = dequantize_groups(
                codes_np[batch_index, page_index],
                scales=scales_np[batch_index, page_index],
                bias=bias_np[batch_index, page_index],
                bits=bits,
                scheme=scheme,
            ).reshape(int(codes_np.shape[2]), -1)[:, :head_dim]
            page_blocks.append(torch.as_tensor(reconstructed, dtype=torch.float32, device=device))
        blocks.append(torch.stack(page_blocks, dim=0))
    return torch.stack(blocks, dim=0)


def _blockwise_quantize_only(
    *,
    values: torch.Tensor,
    group_size: int,
    bits: int,
    scheme: str,
) -> None:
    batch_size, page_count = map(int, values.shape[:2])
    for batch_index in range(batch_size):
        for page_index in range(page_count):
            page_values = np.asarray(values[batch_index, page_index].detach().cpu().numpy(), dtype=np.float32)
            quantize_tensor(
                page_values,
                group_size=group_size,
                bits=bits,
                scheme=scheme,
            )


def _blockwise_quantize_dequantize_to_device(
    *,
    values: torch.Tensor,
    group_size: int,
    bits: int,
    scheme: str,
    head_dim: int,
    device: str,
) -> torch.Tensor:
    blocks: list[torch.Tensor] = []
    batch_size, page_count = map(int, values.shape[:2])
    for batch_index in range(batch_size):
        page_blocks: list[torch.Tensor] = []
        for page_index in range(page_count):
            page_values = np.asarray(values[batch_index, page_index].detach().cpu().numpy(), dtype=np.float32)
            codes, scales, bias, _padded_head_dim = quantize_tensor(
                page_values,
                group_size=group_size,
                bits=bits,
                scheme=scheme,
            )
            reconstructed = dequantize_groups(
                codes,
                scales=scales,
                bias=bias,
                bits=bits,
                scheme=scheme,
            ).reshape(int(values.shape[2]), -1)[:, :head_dim]
            page_blocks.append(torch.as_tensor(reconstructed, dtype=torch.float32, device=device))
        blocks.append(torch.stack(page_blocks, dim=0))
    return torch.stack(blocks, dim=0)


def _direct_score(
    *,
    fused_scaled_codes: torch.Tensor,
    queries: torch.Tensor,
    bias_groups: tuple[torch.Tensor, ...],
    query_group_sums: torch.Tensor,
    group_size: int,
) -> tuple[str, torch.Tensor]:
    head_dim = int(fused_scaled_codes.shape[-1])
    num_groups = head_dim // int(group_size)
    if num_groups == 2 and head_dim == 64 and int(group_size) == 32:
        return (
            "fused_two_group64",
            _score_m0_logits_two_group64_torch(
                fused_scaled_codes,
                queries,
                bias_groups,
                query_group_sums,
            ),
        )
    return (
        "fused_generic",
        _score_m0_logits_fused_torch(
            fused_scaled_codes,
            queries,
            bias_groups,
            query_group_sums,
        ),
    )


def _direct_mix(
    *,
    weights: torch.Tensor,
    fused_scaled_codes: torch.Tensor,
    bias_groups: tuple[torch.Tensor, ...],
    group_size: int,
    variant: str,
) -> torch.Tensor:
    if variant == "fused_two_group64":
        return _mix_m0_contribution_two_group64_torch(weights, fused_scaled_codes, bias_groups)
    return _mix_m0_contribution_fused_torch(
        weights,
        fused_scaled_codes,
        bias_groups,
        group_size=group_size,
    )


def _max_abs_error(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return float((lhs - rhs).abs().max().item())


def _runtime_shaped_profile_pages(profile: str) -> tuple[int, int]:
    resolved = str(profile or "none").strip().lower()
    if resolved == "qwen35_bias_broad":
        return 16, 16
    if resolved == "qwen35_hand_broad":
        return 64, 64
    if resolved == "qwen35_bias_large":
        return 24, 24
    if resolved == "qwen35_hand_large":
        return 96, 96
    return 0, 0


def _mix_exact_flat_torch(weights: torch.Tensor, values_flat: torch.Tensor) -> torch.Tensor:
    return torch.bmm(weights.to(dtype=torch.float32), values_flat.to(dtype=torch.float32)).to(torch.float32)


def _runtime_score_breakdown_result(
    *,
    device: str,
    head_dim: int,
    num_key_value_heads: int,
    query_count: int,
    tokens_per_page: int,
    group_size: int,
    bits_k: int,
    quant_scheme_k: str,
    warmup_iters: int,
    bench_iters: int,
    m0_pages: int,
    m3_pages: int,
) -> dict[str, Any] | None:
    if int(m0_pages) <= 0 and int(m3_pages) <= 0:
        return None
    total_pages = max(int(m0_pages), 0) + max(int(m3_pages), 0)
    if total_pages <= 0:
        return None

    queries = torch.randn((num_key_value_heads, query_count, head_dim), dtype=torch.float32, device=device)
    value_pages = torch.randn(
        (num_key_value_heads, total_pages, int(tokens_per_page), head_dim),
        dtype=torch.float32,
        device=device,
    )
    value_flat = value_pages.reshape(num_key_value_heads, total_pages * int(tokens_per_page), head_dim)

    m0_logits_reference = None
    direct_variant = "none"
    direct_score_ms = 0.0
    direct_score_logits = None
    direct_score_transposed_ms = 0.0
    direct_score_transposed_logits = None
    m0_page_tensor = None
    if int(m0_pages) > 0:
        m0_key_pages = torch.randn(
            (num_key_value_heads, int(m0_pages), int(tokens_per_page), head_dim),
            dtype=torch.float32,
            device=device,
        )
        m0_logits_reference = _score_exact_logits_paged_torch(m0_key_pages, queries)
        key_codes_np, key_scales_np, key_bias_np = _quantize_grouped_blocks_to_numpy(
            values=m0_key_pages,
            group_size=int(group_size),
            bits=int(bits_k),
            scheme=str(quant_scheme_k),
        )
        key_fused_scaled_codes, key_bias_groups = _prepare_fused_m0_inputs(
            codes_np=key_codes_np,
            scales_np=key_scales_np,
            bias_np=key_bias_np,
            device=device,
        )
        key_fused_scaled_codes_transposed = (
            key_fused_scaled_codes.reshape(num_key_value_heads, -1, head_dim).transpose(1, 2).contiguous()
        )
        queries_grouped = queries.reshape(num_key_value_heads, query_count, head_dim // int(group_size), int(group_size))
        query_group_sums = queries_grouped.sum(dim=-1)
        direct_variant, _ = _direct_score(
            fused_scaled_codes=key_fused_scaled_codes,
            queries=queries,
            bias_groups=key_bias_groups,
            query_group_sums=query_group_sums,
            group_size=int(group_size),
        )
        direct_score_ms, direct_score_logits = _bench(
            device,
            lambda: _direct_score(
                fused_scaled_codes=key_fused_scaled_codes,
                queries=queries,
                bias_groups=key_bias_groups,
                query_group_sums=query_group_sums,
                group_size=int(group_size),
            )[1],
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )
        direct_score_transposed_ms, direct_score_transposed_logits = _bench(
            device,
            lambda: _score_m0_logits_fused_transposed_torch(
                key_fused_scaled_codes_transposed,
                queries,
                key_bias_groups,
                query_group_sums,
            ),
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )
        m0_page_tensor = m0_key_pages

    exact_m3_score_ms = 0.0
    exact_m3_logits = None
    exact_m3_flat_score_ms = 0.0
    exact_m3_transposed_score_ms = 0.0
    m3_page_tensor = None
    if int(m3_pages) > 0:
        m3_key_pages = torch.randn(
            (num_key_value_heads, int(m3_pages), int(tokens_per_page), head_dim),
            dtype=torch.float32,
            device=device,
        )
        m3_key_flat = m3_key_pages.reshape(num_key_value_heads, int(m3_pages) * int(tokens_per_page), head_dim).contiguous()
        m3_key_transposed = m3_key_flat.transpose(1, 2).contiguous()
        exact_m3_score_ms, exact_m3_logits = _bench(
            device,
            lambda: _score_exact_logits_paged_torch(m3_key_pages, queries),
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )
        exact_m3_flat_score_ms, exact_m3_flat_logits = _bench(
            device,
            lambda: _score_exact_logits_flat_torch(m3_key_flat, queries),
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )
        exact_m3_transposed_score_ms, exact_m3_transposed_logits = _bench(
            device,
            lambda: _score_exact_logits_transposed_torch(m3_key_transposed, queries),
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
        )
        if exact_m3_logits is not None:
            assert exact_m3_flat_logits is not None
            assert exact_m3_transposed_logits is not None
        m3_page_tensor = m3_key_pages
    else:
        exact_m3_flat_logits = None
        exact_m3_transposed_logits = None

    def _combined_logits_from_parts() -> torch.Tensor:
        parts = []
        if direct_score_logits is not None:
            parts.append(direct_score_logits)
        if exact_m3_logits is not None:
            parts.append(exact_m3_logits)
        return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    combined_logits = _combined_logits_from_parts()
    total_selected_tokens = int(combined_logits.shape[-1])
    mix_weights = torch.softmax(combined_logits, dim=-1)
    final_mix_ms, final_mix_output = _bench(
        device,
        lambda: _mix_exact_flat_torch(mix_weights, value_flat),
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
    )

    def _combined_runtime_path() -> torch.Tensor:
        parts = []
        if int(m0_pages) > 0:
            assert m0_page_tensor is not None
            key_codes_np, key_scales_np, key_bias_np = _quantize_grouped_blocks_to_numpy(
                values=m0_page_tensor,
                group_size=int(group_size),
                bits=int(bits_k),
                scheme=str(quant_scheme_k),
            )
            key_fused_scaled_codes, key_bias_groups = _prepare_fused_m0_inputs(
                codes_np=key_codes_np,
                scales_np=key_scales_np,
                bias_np=key_bias_np,
                device=device,
            )
            queries_grouped = queries.reshape(
                num_key_value_heads,
                query_count,
                head_dim // int(group_size),
                int(group_size),
            )
            query_group_sums = queries_grouped.sum(dim=-1)
            parts.append(
                _direct_score(
                    fused_scaled_codes=key_fused_scaled_codes,
                    queries=queries,
                    bias_groups=key_bias_groups,
                    query_group_sums=query_group_sums,
                    group_size=int(group_size),
                )[1]
            )
        if int(m3_pages) > 0:
            assert m3_page_tensor is not None
            parts.append(_score_exact_logits_paged_torch(m3_page_tensor, queries))
        logits = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        weights = torch.softmax(logits, dim=-1)
        return _mix_exact_flat_torch(weights, value_flat)

    combined_ms, combined_output = _bench(
        device,
        _combined_runtime_path,
        warmup_iters=warmup_iters,
        bench_iters=bench_iters,
    )

    dense_reference_keys = []
    if m0_page_tensor is not None:
        dense_reference_keys.append(m0_page_tensor)
    if m3_page_tensor is not None:
        dense_reference_keys.append(m3_page_tensor)
    dense_reference_keys_tensor = (
        torch.cat(dense_reference_keys, dim=1) if len(dense_reference_keys) > 1 else dense_reference_keys[0]
    )
    dense_reference_logits = _score_exact_logits_paged_torch(dense_reference_keys_tensor, queries)
    dense_reference_weights = torch.softmax(dense_reference_logits, dim=-1)
    dense_reference_output = _mix_exact_flat_torch(dense_reference_weights, value_flat)

    result = {
        "enabled": True,
        "m0_pages": int(m0_pages),
        "m3_pages": int(m3_pages),
        "m0_selected_tokens": int(max(int(m0_pages), 0) * int(tokens_per_page)),
        "m3_selected_tokens": int(max(int(m3_pages), 0) * int(tokens_per_page)),
        "total_selected_tokens": int(total_selected_tokens),
        "direct_m0_variant": str(direct_variant),
        "direct_m0_score_ms": float(direct_score_ms),
        "direct_m0_transposed_score_ms": float(direct_score_transposed_ms),
        "exact_m3_score_ms": float(exact_m3_score_ms),
        "exact_m3_flat_score_ms": float(exact_m3_flat_score_ms),
        "exact_m3_transposed_score_ms": float(exact_m3_transposed_score_ms),
        "final_mix_ms": float(final_mix_ms),
        "combined_ms": float(combined_ms),
        "direct_vs_dense_score_max_abs_error": (
            0.0
            if direct_score_logits is None or m0_logits_reference is None
            else _max_abs_error(direct_score_logits, m0_logits_reference)
        ),
        "direct_m0_transposed_vs_dense_score_max_abs_error": (
            0.0
            if direct_score_transposed_logits is None or m0_logits_reference is None
            else _max_abs_error(direct_score_transposed_logits, m0_logits_reference)
        ),
        "exact_m3_flat_vs_page_score_max_abs_error": (
            0.0
            if exact_m3_logits is None or exact_m3_flat_logits is None
            else _max_abs_error(exact_m3_flat_logits, exact_m3_logits)
        ),
        "exact_m3_transposed_vs_page_score_max_abs_error": (
            0.0
            if exact_m3_logits is None or exact_m3_transposed_logits is None
            else _max_abs_error(exact_m3_transposed_logits, exact_m3_logits)
        ),
        "combined_vs_dense_mix_max_abs_error": _max_abs_error(combined_output, dense_reference_output),
    }
    return result


def _parse_positive_int_csv(value: str) -> list[int]:
    parts = [part.strip() for part in str(value or "").split(",")]
    resolved: list[int] = []
    for part in parts:
        if not part:
            continue
        count = int(part)
        if count <= 0:
            continue
        resolved.append(count)
    return resolved


def _direct_m0_crossover_sweep_result(
    *,
    device: str,
    head_dim: int,
    num_key_value_heads: int,
    query_count: int,
    tokens_per_page: int,
    group_size: int,
    bits_k: int,
    quant_scheme_k: str,
    warmup_iters: int,
    bench_iters: int,
    page_counts: list[int],
) -> dict[str, Any] | None:
    if not page_counts:
        return None
    rows: list[dict[str, Any]] = []
    for page_count in page_counts:
        breakdown = _runtime_score_breakdown_result(
            device=device,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
            query_count=query_count,
            tokens_per_page=tokens_per_page,
            group_size=group_size,
            bits_k=bits_k,
            quant_scheme_k=quant_scheme_k,
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            m0_pages=int(page_count),
            m3_pages=0,
        )
        assert breakdown is not None
        flat_ms = float(breakdown["direct_m0_score_ms"])
        transposed_ms = float(breakdown["direct_m0_transposed_score_ms"])
        winner = "flat" if flat_ms <= transposed_ms else "transposed"
        rows.append(
            {
                "m0_pages": int(page_count),
                "selected_tokens": int(page_count) * int(tokens_per_page),
                "flat_ms": flat_ms,
                "transposed_ms": transposed_ms,
                "winner": winner,
                "speedup": (
                    transposed_ms / max(flat_ms, 1e-9)
                    if winner == "flat"
                    else flat_ms / max(transposed_ms, 1e-9)
                ),
                "flat_vs_dense_score_max_abs_error": float(breakdown["direct_vs_dense_score_max_abs_error"]),
                "transposed_vs_dense_score_max_abs_error": float(
                    breakdown["direct_m0_transposed_vs_dense_score_max_abs_error"]
                ),
            }
        )
    first_transposed = next((row for row in rows if row["winner"] == "transposed"), None)
    consistent_transposed = None
    for row_index, row in enumerate(rows):
        if row["winner"] != "transposed":
            continue
        if all(next_row["winner"] == "transposed" for next_row in rows[row_index:]):
            consistent_transposed = row
            break
    return {
        "enabled": True,
        "page_counts": [int(row["m0_pages"]) for row in rows],
        "rows": rows,
        "first_transposed_winner_m0_pages": (
            None if first_transposed is None else int(first_transposed["m0_pages"])
        ),
        "first_transposed_winner_selected_tokens": (
            None if first_transposed is None else int(first_transposed["selected_tokens"])
        ),
        "first_consistent_transposed_winner_m0_pages": (
            None if consistent_transposed is None else int(consistent_transposed["m0_pages"])
        ),
        "first_consistent_transposed_winner_selected_tokens": (
            None if consistent_transposed is None else int(consistent_transposed["selected_tokens"])
        ),
        "recommended_custom_kernel_layout": (
            "transposed" if consistent_transposed is not None else "flat"
        ),
        "recommended_custom_kernel_min_selected_tokens": (
            None if consistent_transposed is None else int(consistent_transposed["selected_tokens"])
        ),
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable")

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    head_dim, num_key_value_heads, query_count = _resolve_shape(args)
    if head_dim % int(args.group_size) != 0:
        raise SystemExit("head_dim must be divisible by group_size")

    device = args.device
    page_count = int(math.ceil(args.prompt_length / args.tokens_per_page))
    token_count = int(args.tokens_per_page)
    num_groups = head_dim // int(args.group_size)
    runtime_profile = str(args.runtime_shaped_profile or "none").strip().lower()
    profile_m0_pages, profile_m3_pages = _runtime_shaped_profile_pages(runtime_profile)
    runtime_m0_pages = int(args.runtime_m0_pages) if int(args.runtime_m0_pages) > 0 else int(profile_m0_pages)
    runtime_m3_pages = int(args.runtime_m3_pages) if int(args.runtime_m3_pages) > 0 else int(profile_m3_pages)

    keys = torch.randn((num_key_value_heads, page_count, token_count, head_dim), dtype=torch.float32, device=device)
    values = torch.randn((num_key_value_heads, page_count, token_count, head_dim), dtype=torch.float32, device=device)
    queries = torch.randn((num_key_value_heads, query_count, head_dim), dtype=torch.float32, device=device)
    queries_grouped = queries.reshape(num_key_value_heads, query_count, num_groups, int(args.group_size))
    query_group_sums = queries_grouped.sum(dim=-1)

    key_codes_np, key_scales_np, key_bias_np = _quantize_grouped_blocks_to_numpy(
        values=keys,
        group_size=int(args.group_size),
        bits=int(args.bits_k),
        scheme=str(args.quant_scheme_k),
    )
    value_codes_np, value_scales_np, value_bias_np = _quantize_grouped_blocks_to_numpy(
        values=values,
        group_size=int(args.group_size),
        bits=int(args.bits_v),
        scheme=str(args.quant_scheme_v),
    )
    key_fused_scaled_codes, key_bias_groups = _prepare_fused_m0_inputs(
        codes_np=key_codes_np,
        scales_np=key_scales_np,
        bias_np=key_bias_np,
        device=device,
    )
    value_fused_scaled_codes, value_bias_groups = _prepare_fused_m0_inputs(
        codes_np=value_codes_np,
        scales_np=value_scales_np,
        bias_np=value_bias_np,
        device=device,
    )

    def dense_exact_score():
        return _score_exact_logits_paged_torch(keys, queries)

    def dense_exact_combined():
        logits = _score_exact_logits_paged_torch(keys, queries)
        weights = torch.softmax(logits, dim=-1).reshape(num_key_value_heads, query_count, page_count, token_count)
        return _mix_exact_torch(weights, values)

    dense_exact_score_ms, dense_logits = _bench(
        device,
        dense_exact_score,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    dense_exact_mix_weights = torch.softmax(dense_logits, dim=-1).reshape(num_key_value_heads, query_count, page_count, token_count)
    dense_exact_mix_ms, dense_mix_output = _bench(
        device,
        lambda: _mix_exact_torch(dense_exact_mix_weights, values),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    dense_exact_combined_ms, dense_combined_output = _bench(
        device,
        dense_exact_combined,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )

    cached_dequantize_keys_ms, cached_keys = _bench(
        device,
        lambda: _cached_dequantize_blocks_to_device(
            codes_np=key_codes_np,
            scales_np=key_scales_np,
            bias_np=key_bias_np,
            bits=int(args.bits_k),
            scheme=str(args.quant_scheme_k),
            head_dim=head_dim,
            device=device,
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    cached_dequantize_values_ms, cached_values = _bench(
        device,
        lambda: _cached_dequantize_blocks_to_device(
            codes_np=value_codes_np,
            scales_np=value_scales_np,
            bias_np=value_bias_np,
            bits=int(args.bits_v),
            scheme=str(args.quant_scheme_v),
            head_dim=head_dim,
            device=device,
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    cached_reconstruct_combined_ms, cached_combined_output = _bench(
        device,
            lambda: _mix_exact_torch(
            torch.softmax(
                _score_exact_logits_paged_torch(
                    _cached_dequantize_blocks_to_device(
                        codes_np=key_codes_np,
                        scales_np=key_scales_np,
                        bias_np=key_bias_np,
                        bits=int(args.bits_k),
                        scheme=str(args.quant_scheme_k),
                        head_dim=head_dim,
                        device=device,
                    ),
                    queries,
                ),
                dim=-1,
            ).reshape(num_key_value_heads, query_count, page_count, token_count),
            _cached_dequantize_blocks_to_device(
                codes_np=value_codes_np,
                scales_np=value_scales_np,
                bias_np=value_bias_np,
                bits=int(args.bits_v),
                scheme=str(args.quant_scheme_v),
                head_dim=head_dim,
                device=device,
            ),
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )

    blockwise_quantize_only_keys_ms, _ = _bench(
        device,
        lambda: _blockwise_quantize_only(
            values=keys,
            group_size=int(args.group_size),
            bits=int(args.bits_k),
            scheme=str(args.quant_scheme_k),
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    blockwise_quantize_only_values_ms, _ = _bench(
        device,
        lambda: _blockwise_quantize_only(
            values=values,
            group_size=int(args.group_size),
            bits=int(args.bits_v),
            scheme=str(args.quant_scheme_v),
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    blockwise_qdq_keys_ms, blockwise_keys = _bench(
        device,
        lambda: _blockwise_quantize_dequantize_to_device(
            values=keys,
            group_size=int(args.group_size),
            bits=int(args.bits_k),
            scheme=str(args.quant_scheme_k),
            head_dim=head_dim,
            device=device,
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    blockwise_qdq_values_ms, blockwise_values = _bench(
        device,
        lambda: _blockwise_quantize_dequantize_to_device(
            values=values,
            group_size=int(args.group_size),
            bits=int(args.bits_v),
            scheme=str(args.quant_scheme_v),
            head_dim=head_dim,
            device=device,
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    blockwise_qdq_combined_ms, blockwise_combined_output = _bench(
        device,
            lambda: _mix_exact_torch(
            torch.softmax(
                _score_exact_logits_paged_torch(
                    _blockwise_quantize_dequantize_to_device(
                        values=keys,
                        group_size=int(args.group_size),
                        bits=int(args.bits_k),
                        scheme=str(args.quant_scheme_k),
                        head_dim=head_dim,
                        device=device,
                    ),
                    queries,
                ),
                dim=-1,
            ).reshape(num_key_value_heads, query_count, page_count, token_count),
            _blockwise_quantize_dequantize_to_device(
                values=values,
                group_size=int(args.group_size),
                bits=int(args.bits_v),
                scheme=str(args.quant_scheme_v),
                head_dim=head_dim,
                device=device,
            ),
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )

    direct_variant, direct_score_logits = _direct_score(
        fused_scaled_codes=key_fused_scaled_codes,
        queries=queries,
        bias_groups=key_bias_groups,
        query_group_sums=query_group_sums,
        group_size=int(args.group_size),
    )
    direct_score_ms, direct_score_logits = _bench(
        device,
        lambda: _direct_score(
            fused_scaled_codes=key_fused_scaled_codes,
            queries=queries,
            bias_groups=key_bias_groups,
            query_group_sums=query_group_sums,
            group_size=int(args.group_size),
        )[1],
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    direct_weights = torch.softmax(direct_score_logits, dim=-1).reshape(num_key_value_heads, query_count, page_count, token_count)
    direct_mix_ms, direct_mix_output = _bench(
        device,
        lambda: _direct_mix(
            weights=direct_weights,
            fused_scaled_codes=value_fused_scaled_codes,
            bias_groups=value_bias_groups,
            group_size=int(args.group_size),
            variant=direct_variant,
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    direct_combined_ms, direct_combined_output = _bench(
        device,
        lambda: _direct_mix(
            weights=torch.softmax(
                _direct_score(
                    fused_scaled_codes=key_fused_scaled_codes,
                    queries=queries,
                    bias_groups=key_bias_groups,
                    query_group_sums=query_group_sums,
                    group_size=int(args.group_size),
                )[1],
                dim=-1,
            ).reshape(num_key_value_heads, query_count, page_count, token_count),
            fused_scaled_codes=value_fused_scaled_codes,
            bias_groups=value_bias_groups,
            group_size=int(args.group_size),
            variant=direct_variant,
        ),
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )

    cached_score_logits = _score_exact_logits_paged_torch(cached_keys, queries)
    cached_weights = torch.softmax(cached_score_logits, dim=-1).reshape(num_key_value_heads, query_count, page_count, token_count)
    cached_mix_output = _mix_exact_torch(cached_weights, cached_values)

    blockwise_score_logits = _score_exact_logits_paged_torch(blockwise_keys, queries)
    blockwise_weights = torch.softmax(blockwise_score_logits, dim=-1).reshape(num_key_value_heads, query_count, page_count, token_count)
    blockwise_mix_output = _mix_exact_torch(blockwise_weights, blockwise_values)

    result: dict[str, Any] = {
        "benchmark": "torch_decode_micro",
        "mode": "m0_execution",
        "model_id": args.model_id,
        "device": args.device,
        "prompt_length": args.prompt_length,
        "tokens_per_page": args.tokens_per_page,
        "page_count": page_count,
        "selected_token_count": page_count * token_count,
        "num_key_value_heads": num_key_value_heads,
        "query_count": query_count,
        "head_dim": head_dim,
        "group_size": args.group_size,
        "num_groups": num_groups,
        "bits_k": args.bits_k,
        "bits_v": args.bits_v,
        "quant_scheme_k": args.quant_scheme_k,
        "quant_scheme_v": args.quant_scheme_v,
        "warmup_iters": args.warmup_iters,
        "bench_iters": args.bench_iters,
        "runtime_shaped_profile": runtime_profile,
        "direct_m0_variant": direct_variant,
        "dense_exact_score_ms": dense_exact_score_ms,
        "dense_exact_mix_ms": dense_exact_mix_ms,
        "dense_exact_combined_ms": dense_exact_combined_ms,
        "cached_dequantize_keys_ms": cached_dequantize_keys_ms,
        "cached_dequantize_values_ms": cached_dequantize_values_ms,
        "cached_dequantize_kv_ms": cached_dequantize_keys_ms + cached_dequantize_values_ms,
        "cached_reconstruct_exact_combined_ms": cached_reconstruct_combined_ms,
        "blockwise_quantize_only_keys_ms": blockwise_quantize_only_keys_ms,
        "blockwise_quantize_only_values_ms": blockwise_quantize_only_values_ms,
        "blockwise_quantize_only_kv_ms": blockwise_quantize_only_keys_ms + blockwise_quantize_only_values_ms,
        "blockwise_qdq_keys_ms": blockwise_qdq_keys_ms,
        "blockwise_qdq_values_ms": blockwise_qdq_values_ms,
        "blockwise_qdq_kv_ms": blockwise_qdq_keys_ms + blockwise_qdq_values_ms,
        "blockwise_qdq_exact_combined_ms": blockwise_qdq_combined_ms,
        "direct_m0_score_ms": direct_score_ms,
        "direct_m0_mix_ms": direct_mix_ms,
        "direct_m0_combined_ms": direct_combined_ms,
        "direct_m0_speedup_vs_blockwise_qdq_exact_combined": blockwise_qdq_combined_ms / max(direct_combined_ms, 1e-9),
        "direct_m0_speedup_vs_cached_reconstruct_exact_combined": cached_reconstruct_combined_ms / max(direct_combined_ms, 1e-9),
        "direct_m0_speedup_vs_dense_exact_combined": dense_exact_combined_ms / max(direct_combined_ms, 1e-9),
        "direct_vs_cached_score_max_abs_error": _max_abs_error(direct_score_logits, cached_score_logits),
        "direct_vs_cached_mix_max_abs_error": _max_abs_error(direct_mix_output, cached_mix_output),
        "cached_vs_blockwise_mix_max_abs_error": _max_abs_error(cached_mix_output, blockwise_mix_output),
        "direct_vs_dense_mix_max_abs_error": _max_abs_error(direct_combined_output, dense_combined_output),
        "cached_vs_dense_mix_max_abs_error": _max_abs_error(cached_combined_output, dense_combined_output),
        "blockwise_vs_dense_mix_max_abs_error": _max_abs_error(blockwise_combined_output, dense_combined_output),
    }
    runtime_score_breakdown = _runtime_score_breakdown_result(
        device=device,
        head_dim=head_dim,
        num_key_value_heads=num_key_value_heads,
        query_count=query_count,
        tokens_per_page=token_count,
        group_size=int(args.group_size),
        bits_k=int(args.bits_k),
        quant_scheme_k=str(args.quant_scheme_k),
        warmup_iters=int(args.warmup_iters),
        bench_iters=int(args.bench_iters),
        m0_pages=runtime_m0_pages,
        m3_pages=runtime_m3_pages,
    )
    if runtime_score_breakdown is not None:
        result["runtime_score_breakdown"] = runtime_score_breakdown
    crossover_sweep = _direct_m0_crossover_sweep_result(
        device=device,
        head_dim=head_dim,
        num_key_value_heads=num_key_value_heads,
        query_count=query_count,
        tokens_per_page=token_count,
        group_size=int(args.group_size),
        bits_k=int(args.bits_k),
        quant_scheme_k=str(args.quant_scheme_k),
        warmup_iters=int(args.warmup_iters),
        bench_iters=int(args.bench_iters),
        page_counts=_parse_positive_int_csv(args.direct_m0_crossover_sweep_pages),
    )
    if crossover_sweep is not None:
        result["direct_m0_crossover_sweep"] = crossover_sweep

    if args.output_format == "json":
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
