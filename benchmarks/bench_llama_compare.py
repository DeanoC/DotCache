from __future__ import annotations

import argparse
import json

from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.integrations.llama import LlamaDotCacheHarness, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dense KV vs DotCache on one loaded Llama-family model.")
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", choices=["torch_mps", "cpu_ref", "auto"], default="torch_mps")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--bits-k", type=int, default=4)
    parser.add_argument("--bits-v", type=int, default=4)
    parser.add_argument("--tokens-per-page", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--repeat-counts", type=int, nargs="+", default=[1, 32, 64])
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_llama_compare.py requires the optional transformers dependencies")

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

    for repeat_count in args.repeat_counts:
        prompt = " ".join([args.prompt_unit] * repeat_count)
        input_ids, attention_mask = harness.tokenize_prompt(prompt)
        record = harness.generate_greedy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
        )
        record.update(
            {
                "benchmark": "llama_compare",
                "model_id": args.model_id,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "tokens_per_page": args.tokens_per_page,
                "repeat_count": repeat_count,
                "prompt_unit": args.prompt_unit,
            }
        )
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
