from __future__ import annotations

import argparse
import json

from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.integrations.llama import LlamaDotCacheHarness, resolve_hf_auth_kwargs, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile multiple M0 K/V bit-width combinations against one loaded Llama-family model."
    )
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v-options", type=int, nargs="+", default=[4, 3])
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--target-prompt-lengths", type=int, nargs="+", default=[577])
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    return parser.parse_args()


def _build_config(*, head_dim: int, group_size: int, bits_k: int, bits_v: int, tokens_per_page: int) -> DotCacheConfig:
    return DotCacheConfig(
        head_dim=head_dim,
        group_size=group_size,
        bits_k=bits_k,
        bits_v=bits_v,
        default_mode_k="M0",
        default_mode_v="M0",
        quant_scheme_k="affine",
        quant_scheme_v="affine",
        tokens_per_page=tokens_per_page,
    )


def _build_exact_length_inputs(harness: LlamaDotCacheHarness, *, prompt_unit: str, prompt_length: int):
    if harness.tokenizer is None:
        raise ValueError("tokenizer is unavailable for exact-length prompt construction")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    tokenizer = harness.tokenizer
    unit_ids = tokenizer(prompt_unit, add_special_tokens=False)["input_ids"]
    if not unit_ids:
        raise ValueError("prompt_unit tokenized to an empty sequence")

    token_ids: list[int] = []
    if tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))
    while len(token_ids) < prompt_length:
        token_ids.extend(int(token_id) for token_id in unit_ids)
    token_ids = token_ids[:prompt_length]

    import torch

    device = harness.adapter.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_llama_mixed_bits_profile.py requires the optional transformers dependencies")

    model_config = AutoConfig.from_pretrained(args.model_id, **resolve_hf_auth_kwargs())
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    initial_config = _build_config(
        head_dim=head_dim,
        group_size=args.group_size,
        bits_k=args.bits_k,
        bits_v=args.bits_v_options[0],
        tokens_per_page=args.tokens_per_page,
    )
    harness = LlamaDotCacheHarness.from_pretrained(
        args.model_id,
        initial_config,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    for bits_v in args.bits_v_options:
        harness.adapter.reconfigure(
            _build_config(
                head_dim=head_dim,
                group_size=args.group_size,
                bits_k=args.bits_k,
                bits_v=bits_v,
                tokens_per_page=args.tokens_per_page,
            ),
            backend=args.backend,
        )
        for prompt_length in args.target_prompt_lengths:
            input_ids, attention_mask = _build_exact_length_inputs(
                harness,
                prompt_unit=args.prompt_unit,
                prompt_length=prompt_length,
            )
            record = harness.generate_greedy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                profile=False,
            )
            record.update(
                {
                    "benchmark": "llama_mixed_bits_profile",
                    "mode_profile": "m0_mixed_bits",
                    "model_id": args.model_id,
                    "backend": args.backend,
                    "device": args.device,
                    "torch_dtype": args.torch_dtype,
                    "tokens_per_page": args.tokens_per_page,
                    "requested_prompt_length": prompt_length,
                    "prompt_mode": "exact_length",
                    "prompt_unit": args.prompt_unit,
                    "bits_k": args.bits_k,
                    "bits_v": bits_v,
                }
            )
            print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
