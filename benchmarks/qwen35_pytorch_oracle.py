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
    first_layer_input_layernorm_output = None
    first_layer_linear_qkv_output = None
    first_layer_linear_z_output = None
    first_layer_linear_b_output = None
    first_layer_linear_a_output = None
    first_layer_linear_post_conv_output = None
    first_layer_linear_pre_norm_output = None
    first_layer_linear_norm_gate_input = None
    first_layer_linear_norm_output = None
    first_layer_token_mixer_output = None
    first_layer_post_attention_layernorm_output = None
    first_layer_mlp_output = None

    def embed_hook(_module, _inputs, output):
        nonlocal embedding_output
        embedding_output = output.detach().to(dtype=torch.float32).cpu()

    def layer_hook(_module, _inputs, output):
        nonlocal first_layer_output
        layer_output = output[0] if isinstance(output, tuple) else output
        first_layer_output = layer_output.detach().to(dtype=torch.float32).cpu()

    def capture_tensor(output):
        tensor = output[0] if isinstance(output, tuple) else output
        return tensor.detach().to(dtype=torch.float32).cpu()

    def input_layernorm_hook(_module, _inputs, output):
        nonlocal first_layer_input_layernorm_output
        first_layer_input_layernorm_output = capture_tensor(output)

    def token_mixer_hook(_module, _inputs, output):
        nonlocal first_layer_token_mixer_output
        first_layer_token_mixer_output = capture_tensor(output)

    def linear_qkv_hook(_module, _inputs, output):
        nonlocal first_layer_linear_qkv_output
        first_layer_linear_qkv_output = capture_tensor(output)

    def linear_z_hook(_module, _inputs, output):
        nonlocal first_layer_linear_z_output
        first_layer_linear_z_output = capture_tensor(output)

    def linear_b_hook(_module, _inputs, output):
        nonlocal first_layer_linear_b_output
        first_layer_linear_b_output = capture_tensor(output)

    def linear_a_hook(_module, _inputs, output):
        nonlocal first_layer_linear_a_output
        first_layer_linear_a_output = capture_tensor(output)

    def linear_conv_hook(_module, _inputs, output):
        nonlocal first_layer_linear_post_conv_output
        tensor = capture_tensor(output)
        seq_len = input_ids.shape[1]
        first_layer_linear_post_conv_output = (
            tensor.transpose(1, 2)[:, -seq_len:, :].contiguous()
        )

    def linear_norm_hook(_module, _inputs, output):
        nonlocal first_layer_linear_norm_output
        tensor = capture_tensor(output)
        first_layer_linear_norm_output = tensor.reshape(input_ids.shape[0], input_ids.shape[1], -1)

    def linear_norm_pre_hook(_module, inputs):
        nonlocal first_layer_linear_pre_norm_output
        tensor = capture_tensor(inputs[0])
        first_layer_linear_pre_norm_output = tensor.reshape(input_ids.shape[0], input_ids.shape[1], -1)
        nonlocal first_layer_linear_norm_gate_input
        gate_tensor = capture_tensor(inputs[1])
        first_layer_linear_norm_gate_input = gate_tensor.reshape(
            input_ids.shape[0], input_ids.shape[1], -1
        )

    def post_attention_layernorm_hook(_module, _inputs, output):
        nonlocal first_layer_post_attention_layernorm_output
        first_layer_post_attention_layernorm_output = capture_tensor(output)

    def mlp_hook(_module, _inputs, output):
        nonlocal first_layer_mlp_output
        first_layer_mlp_output = capture_tensor(output)

    embed_handle = model.model.embed_tokens.register_forward_hook(embed_hook)
    layer_handle = model.model.layers[0].register_forward_hook(layer_hook)
    input_layernorm_handle = model.model.layers[0].input_layernorm.register_forward_hook(
        input_layernorm_hook
    )
    token_mixer_handle = model.model.layers[0].linear_attn.register_forward_hook(
        token_mixer_hook
    )
    linear_qkv_handle = model.model.layers[0].linear_attn.in_proj_qkv.register_forward_hook(
        linear_qkv_hook
    )
    linear_z_handle = model.model.layers[0].linear_attn.in_proj_z.register_forward_hook(
        linear_z_hook
    )
    linear_b_handle = model.model.layers[0].linear_attn.in_proj_b.register_forward_hook(
        linear_b_hook
    )
    linear_a_handle = model.model.layers[0].linear_attn.in_proj_a.register_forward_hook(
        linear_a_hook
    )
    linear_conv_handle = model.model.layers[0].linear_attn.conv1d.register_forward_hook(
        linear_conv_hook
    )
    linear_norm_handle = model.model.layers[0].linear_attn.norm.register_forward_hook(
        linear_norm_hook
    )
    linear_norm_pre_handle = model.model.layers[0].linear_attn.norm.register_forward_pre_hook(
        linear_norm_pre_hook
    )
    post_attention_layernorm_handle = (
        model.model.layers[0]
        .post_attention_layernorm.register_forward_hook(post_attention_layernorm_hook)
    )
    mlp_handle = model.model.layers[0].mlp.register_forward_hook(mlp_hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=True)
    finally:
        embed_handle.remove()
        layer_handle.remove()
        input_layernorm_handle.remove()
        token_mixer_handle.remove()
        linear_qkv_handle.remove()
        linear_z_handle.remove()
        linear_b_handle.remove()
        linear_a_handle.remove()
        linear_conv_handle.remove()
        linear_norm_pre_handle.remove()
        linear_norm_handle.remove()
        post_attention_layernorm_handle.remove()
        mlp_handle.remove()

    if (
        embedding_output is None
        or first_layer_output is None
        or first_layer_input_layernorm_output is None
        or first_layer_linear_qkv_output is None
        or first_layer_linear_z_output is None
        or first_layer_linear_b_output is None
        or first_layer_linear_a_output is None
        or first_layer_linear_post_conv_output is None
        or first_layer_linear_pre_norm_output is None
        or first_layer_linear_norm_gate_input is None
        or first_layer_linear_norm_output is None
        or first_layer_token_mixer_output is None
        or first_layer_post_attention_layernorm_output is None
        or first_layer_mlp_output is None
    ):
        raise RuntimeError("failed to capture staged first-layer outputs from PyTorch model")

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
        "first_layer_input_layernorm_output": first_layer_input_layernorm_output.tolist(),
        "first_layer_linear_qkv_output": first_layer_linear_qkv_output.tolist(),
        "first_layer_linear_z_output": first_layer_linear_z_output.tolist(),
        "first_layer_linear_b_output": first_layer_linear_b_output.tolist(),
        "first_layer_linear_a_output": first_layer_linear_a_output.tolist(),
        "first_layer_linear_post_conv_output": first_layer_linear_post_conv_output.tolist(),
        "first_layer_linear_pre_norm_output": first_layer_linear_pre_norm_output.tolist(),
        "first_layer_linear_norm_gate_input": first_layer_linear_norm_gate_input.tolist(),
        "first_layer_linear_norm_output": first_layer_linear_norm_output.tolist(),
        "first_layer_token_mixer_output": first_layer_token_mixer_output.tolist(),
        "first_layer_post_attention_layernorm_output": first_layer_post_attention_layernorm_output.tolist(),
        "first_layer_mlp_output": first_layer_mlp_output.tolist(),
        "prefill_last_token_logits": prefill_last_token_logits,
        "decode_last_token_logits": decode_logits,
        "generated_token_ids": generated_token_ids,
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
