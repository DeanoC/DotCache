from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig

from dotcache.integrations.llama import resolve_hf_auth_kwargs
from dotcache.integrations.qwen35 import Qwen35TextHarness, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense-only text-generation smoke and timing harness for Qwen3.5 hybrid models.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="auto")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--repeat-counts", type=int, nargs="*", default=[1, 32, 64])
    parser.add_argument("--target-prompt-lengths", type=int, nargs="+", default=[])
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    return parser.parse_args()


def _build_exact_length_inputs(
    harness: Qwen35TextHarness,
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


def _run_case(
    harness: Qwen35TextHarness,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    base_record: dict[str, object],
    continue_on_error: bool,
) -> None:
    try:
        record = harness.generate_greedy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:  # pragma: no cover - benchmark failure path
        if not continue_on_error:
            raise
        error_record = dict(base_record)
        error_record.update(
            {
                "benchmark": "qwen35_text",
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "prompt_length": int(input_ids.shape[1]),
            }
        )
        print(json.dumps(error_record, sort_keys=True), flush=True)
        return

    record.update(base_record)
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_text.py requires the optional transformers dependencies")

    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **resolve_hf_auth_kwargs())
    text_config = getattr(model_config, "text_config", model_config)
    max_position_embeddings = int(getattr(text_config, "max_position_embeddings", 0) or 0)

    harness = Qwen35TextHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    for repeat_count in args.repeat_counts:
        prompt = " ".join([args.prompt_unit] * repeat_count)
        input_ids, attention_mask = harness.tokenize_prompt(prompt)
        _run_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            base_record={
                "benchmark": "qwen35_text",
                "model_id": args.model_id,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "prompt_mode": "repeat_count",
                "repeat_count": repeat_count,
                "prompt_unit": args.prompt_unit,
                "model_max_position_embeddings": max_position_embeddings,
                "text_only": True,
                "dotcache_ready": False,
                "hybrid_family": "qwen3_5",
            },
            continue_on_error=args.continue_on_error,
        )

    for prompt_length in sorted(set(length for length in args.target_prompt_lengths if length > 0)):
        input_ids, attention_mask = _build_exact_length_inputs(
            harness,
            prompt_unit=args.prompt_unit,
            prompt_length=prompt_length,
        )
        _run_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            base_record={
                "benchmark": "qwen35_text",
                "model_id": args.model_id,
                "backend": args.backend,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "prompt_mode": "exact_length",
                "prompt_length": prompt_length,
                "prompt_unit": args.prompt_unit,
                "model_max_position_embeddings": max_position_embeddings,
                "text_only": True,
                "dotcache_ready": False,
                "hybrid_family": "qwen3_5",
            },
            continue_on_error=args.continue_on_error,
        )


if __name__ == "__main__":
    main()
