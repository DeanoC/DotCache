from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.integrations.llama import LlamaDotCacheHarness, _prefill_prompt, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect chosen page modes after one dense prefill on a Llama-family model.")
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--key-policy-tier", choices=["exact", "strict", "balanced", "aggressive"], default="exact")
    parser.add_argument("--value-policy-tier", choices=["exact", "strict", "balanced", "aggressive"], default="exact")
    return parser.parse_args()


def _build_exact_length_inputs(
    harness: LlamaDotCacheHarness,
    *,
    prompt_unit: str,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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

    device = harness.adapter.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("inspect_policy_prefill.py requires the optional transformers dependencies")

    model_config = AutoConfig.from_pretrained(args.model_id)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    dotcache_config = DotCacheConfig(
        head_dim=head_dim,
        group_size=args.group_size,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        default_mode_k="M0",
        default_mode_v="M0",
        key_policy_tier=args.key_policy_tier,
        value_policy_tier=args.value_policy_tier,
        tokens_per_page=args.tokens_per_page,
    )
    harness = LlamaDotCacheHarness.from_pretrained(
        args.model_id,
        dotcache_config,
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )
    input_ids, attention_mask = _build_exact_length_inputs(
        harness,
        prompt_unit=args.prompt_unit,
        prompt_length=args.prompt_length,
    )

    prefill_outputs, _, _ = _prefill_prompt(harness.model, harness.adapter, input_ids, attention_mask)
    harness.adapter.load_prefill_cache(prefill_outputs.past_key_values)
    summary = harness.adapter.model_kv_cache.page_mode_summary()
    result = {
        "benchmark": "inspect_policy_prefill",
        "model_id": args.model_id,
        "backend": args.backend,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "prompt_length": args.prompt_length,
        "tokens_per_page": args.tokens_per_page,
        "key_policy_tier": args.key_policy_tier,
        "value_policy_tier": args.value_policy_tier,
        **summary,
    }
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
