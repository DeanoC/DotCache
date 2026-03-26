from __future__ import annotations

import argparse
import json

from dotcache.config import DotCacheConfig
from dotcache.integrations.llama import (
    LlamaDotCacheHarness,
    LlamaDotCacheModelAdapter,
    run_llama_generation_harness,
    transformers_available,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Phase 5 Llama DotCache integration path.")
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--prompt", default="Write one short sentence about cache locality.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--backend", choices=["cpu_ref", "torch_mps", "torch_cuda", "auto"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--random-tiny", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def _build_random_harness(args: argparse.Namespace):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config)
    model.to(args.device)
    model.eval()
    dotcache_config = DotCacheConfig(
        head_dim=32,
        group_size=args.group_size,
        bits_k=args.bits_k,
        bits_v=args.bits_v,
        tokens_per_page=args.tokens_per_page,
    )
    adapter = LlamaDotCacheModelAdapter(model, dotcache_config, backend=args.backend)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long, device=args.device)
    return model, adapter, input_ids


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_llama_decode.py requires the optional transformers dependencies")

    if args.random_tiny:
        model, adapter, input_ids = _build_random_harness(args)
        record = run_llama_generation_harness(
            model,
            adapter,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            profile=args.profile,
        )
        record.update(
            {
                "backend": args.backend,
                "benchmark": "llama_decode",
                "device": args.device,
                "model_id": "tiny-random-llama",
                "tokens_per_page": args.tokens_per_page,
            }
        )
    else:
        import torch
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(args.model_id)
        head_dim = model_config.hidden_size // model_config.num_attention_heads
        dotcache_config = DotCacheConfig(
            head_dim=head_dim,
            group_size=args.group_size,
            bits_k=args.bits_k,
            bits_v=args.bits_v,
            tokens_per_page=args.tokens_per_page,
        )
        harness = LlamaDotCacheHarness.from_pretrained(
            args.model_id,
            dotcache_config,
            backend=args.backend,
            device=args.device,
            torch_dtype=args.torch_dtype,
        )
        record = harness.generate_greedy(prompt=args.prompt, max_new_tokens=args.max_new_tokens, profile=args.profile)
        record.update(
            {
                "backend": args.backend,
                "benchmark": "llama_decode",
                "device": args.device,
                "model_id": args.model_id,
                "tokens_per_page": args.tokens_per_page,
                "prompt": args.prompt,
                "torch_dtype": args.torch_dtype,
            }
        )
        del torch

    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
