#!/usr/bin/env python3

import argparse
import json
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-ids", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_ids = [int(part) for part in args.prompt_ids.split(",") if part]
    if not prompt_ids:
        raise SystemExit("prompt ids must not be empty")

    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    load_elapsed_ms = (time.perf_counter() - load_started) * 1000.0

    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    prefill_started = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    prefill_elapsed_ms = (time.perf_counter() - prefill_started) * 1000.0

    embedding_output = None
    first_layer_output = None

    def embed_hook(_module, _inputs, output):
        nonlocal embedding_output
        embedding_output = output.detach().to(dtype=torch.float32).cpu()

    def layer_hook(_module, _inputs, output):
        nonlocal first_layer_output
        layer_output = output[0] if isinstance(output, tuple) else output
        first_layer_output = layer_output.detach().to(dtype=torch.float32).cpu()

    embed_handle = model.model.embed_tokens.register_forward_hook(embed_hook)
    layer_handle = model.model.layers[0].register_forward_hook(layer_hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=True)
    finally:
        embed_handle.remove()
        layer_handle.remove()

    if embedding_output is None or first_layer_output is None:
        raise RuntimeError("failed to capture embedding or first layer output from PyTorch model")

    prefill_last_token_logits = (
        outputs.logits[0, -1, :].to(dtype=torch.float32).cpu().tolist()
    )

    decode_started = time.perf_counter()
    decode_logits: list[list[float]] = []
    generated_token_ids: list[int] = []
    past_key_values = outputs.past_key_values
    next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
    for _ in range(args.max_new_tokens):
        generated_token_ids.append(next_token)
        decode_input_ids = torch.tensor([[next_token]], dtype=torch.long)
        with torch.no_grad():
            outputs = model(
                input_ids=decode_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
        step_logits = outputs.logits[0, -1, :].to(dtype=torch.float32).cpu().tolist()
        decode_logits.append(step_logits)
        next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        past_key_values = outputs.past_key_values
    decode_elapsed_ms = (time.perf_counter() - decode_started) * 1000.0

    payload = {
        "load_ms": load_elapsed_ms,
        "prefill_ms": prefill_elapsed_ms,
        "decode_ms": decode_elapsed_ms,
        "embedding_output": embedding_output.tolist(),
        "first_layer_output": first_layer_output.tolist(),
        "prefill_last_token_logits": prefill_last_token_logits,
        "decode_last_token_logits": decode_logits,
        "generated_token_ids": generated_token_ids,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
