from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the actual Luce Qwen3.5 megakernel as an external control lane."
    )
    parser.add_argument("model_id")
    parser.add_argument("prompt_text")
    parser.add_argument("out_prefix")
    parser.add_argument("--luce-repo", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt-token-target", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def ensure_cuda_device(device: str) -> int:
    normalized = device.strip().lower()
    if not normalized.startswith("cuda"):
        raise SystemExit(f"luce external lane requires cuda device, got `{device}`")
    if ":" not in normalized:
        return 0
    return int(normalized.split(":", 1)[1])


def ensure_extension(repo: Path) -> None:
    if any(repo.glob("qwen35_megakernel_bf16_C*.so")):
        return
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=repo,
        check=True,
    )


def build_prompt_ids(tokenizer: Any, prompt_text: str, target: int | None) -> list[int]:
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not token_ids:
        raise SystemExit("prompt encoding produced no tokens")
    if target is None:
        return token_ids
    if len(token_ids) > target:
        return token_ids[:target]
    if len(token_ids) == target:
        return token_ids
    filler_ids = tokenizer.encode(f" {prompt_text}", add_special_tokens=False)
    if not filler_ids:
        raise SystemExit("prompt filler encoding produced no tokens")
    while len(token_ids) < target:
        token_ids.extend(filler_ids)
    return token_ids[:target]


def valid_token_id(tokenizer: Any, token_id: int) -> bool:
    return 0 <= int(token_id) < len(tokenizer)


def main() -> None:
    args = parse_args()
    repo = Path(args.luce_repo).resolve()
    if not repo.exists():
        raise SystemExit(f"luce repo not found: {repo}")
    device_ordinal = ensure_cuda_device(args.device)

    ensure_extension(repo)
    sys.path.insert(0, str(repo))

    import torch
    from transformers import AutoTokenizer

    model = importlib.import_module("model")
    qwen_ext = importlib.import_module("qwen35_megakernel_bf16_C")
    _ = qwen_ext
    torch.cuda.set_device(device_ordinal)

    if args.model_id != "Qwen/Qwen3.5-0.8B":
        raise SystemExit(
            f"luce external lane currently supports only Qwen/Qwen3.5-0.8B, got {args.model_id}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    prompt_ids = build_prompt_ids(tokenizer, args.prompt_text, args.prompt_token_target)
    if len(prompt_ids) > model.MAX_SEQ_LEN:
        raise SystemExit(
            f"luce external lane max context is {model.MAX_SEQ_LEN}, got {len(prompt_ids)}"
        )

    decoder = model.Decoder(model_name=args.model_id, verbose=False)
    prefill_op = torch.ops.qwen35_megakernel_bf16_C.prefill_bf16

    S = len(prompt_ids)
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    f32 = dict(dtype=torch.float32, device="cuda")
    i32 = dict(dtype=torch.int32, device="cuda")
    mx = max(model.DN_CONV_CHANNELS, model.FA_QPROJ_SIZE, model.INTERMEDIATE_SIZE)
    bufs = {
        "hidden": torch.empty(S * model.HIDDEN_SIZE, **bf16),
        "residual": torch.empty(S * model.HIDDEN_SIZE, **bf16),
        "normalized": torch.empty(S * model.HIDDEN_SIZE, **bf16),
        "proj_buf": torch.empty(S * mx, **bf16),
        "proj_buf2": torch.empty(S * mx, **bf16),
        "attn_buf": torch.empty(S * max(model.FA_Q_SIZE, model.FA_KV_SIZE), **bf16),
        "mlp_buf": torch.empty(S * model.INTERMEDIATE_SIZE, **bf16),
        "dn_out_buf": torch.empty(S * model.DN_V_SIZE, **bf16),
        "beta_buf": torch.empty(S * model.DN_NUM_HEADS, **f32),
        "alpha_buf": torch.empty(S * model.DN_NUM_HEADS, **f32),
        "final_normed": torch.empty(model.HIDDEN_SIZE, **bf16),
        "hidden_bf16_out": torch.empty(model.HIDDEN_SIZE, **bf16),
        "lm_bmv": torch.empty(1024, **f32),
        "lm_bmi": torch.empty(1024, **i32),
    }
    ids_t = torch.tensor(prompt_ids, dtype=torch.int32, device="cuda")

    def prefill_once() -> int:
        decoder.reset()
        prefill_op(
            decoder._out_token,
            ids_t,
            decoder._embed_weight,
            decoder._layer_weights_packed,
            decoder._final_norm_weight,
            decoder._lm_head_weight,
            decoder._fa_k_cache,
            decoder._fa_v_cache,
            decoder._dn_states,
            decoder._conv_bufs,
            bufs["hidden"],
            bufs["residual"],
            bufs["normalized"],
            bufs["proj_buf"],
            bufs["proj_buf2"],
            bufs["attn_buf"],
            bufs["mlp_buf"],
            bufs["dn_out_buf"],
            bufs["beta_buf"],
            bufs["alpha_buf"],
            bufs["final_normed"],
            bufs["hidden_bf16_out"],
            bufs["lm_bmv"],
            bufs["lm_bmi"],
        )
        decoder._hidden.copy_(bufs["hidden_bf16_out"])
        decoder._position = len(prompt_ids)
        return int(decoder._out_token.item())

    warmup_millis = 0.0
    for _ in range(args.warmup_runs):
        torch.cuda.synchronize()
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        _ = prefill_once()
        ended.record()
        torch.cuda.synchronize()
        warmup_millis += started.elapsed_time(ended)

    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    first_token = prefill_once()
    ended.record()
    torch.cuda.synchronize()
    prefill_millis = started.elapsed_time(ended)

    out_ids: list[int] = []
    next_id = first_token
    invalid_token_id: int | None = None
    invalid_token_step: int | None = None
    if args.max_new_tokens > 0:
        if valid_token_id(tokenizer, first_token):
            out_ids.append(first_token)
        else:
            invalid_token_id = int(first_token)
            invalid_token_step = 0
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    started.record()
    for step in range(max(args.max_new_tokens - 1, 0)):
        if invalid_token_id is not None:
            break
        candidate_id = decoder.step(next_id)
        if not valid_token_id(tokenizer, candidate_id):
            invalid_token_id = int(candidate_id)
            invalid_token_step = step + 1
            break
        next_id = int(candidate_id)
        out_ids.append(next_id)
    ended.record()
    torch.cuda.synchronize()
    decode_millis = started.elapsed_time(ended)

    total_millis = prefill_millis + decode_millis
    generated_text = tokenizer.decode(out_ids, skip_special_tokens=True) if out_ids else ""
    generated_token_count = len(out_ids)

    summary = {
        "model_id": args.model_id,
        "device": args.device,
        "lane": "luce_external_megakernel",
        "luce_repo": str(repo),
        "prompt": args.prompt_text,
        "prompt_token_count": len(prompt_ids),
        "prompt_token_target": args.prompt_token_target,
        "generated_token_count": generated_token_count,
        "warmup_runs": args.warmup_runs,
        "warmup_millis": warmup_millis,
        "max_new_tokens": args.max_new_tokens,
        "prefill_millis": float(prefill_millis),
        "decode_millis": float(decode_millis),
        "total_millis": float(total_millis),
        "prefill_tokens_per_second": len(prompt_ids) / (prefill_millis / 1000.0),
        "decode_tokens_per_second": (
            generated_token_count / (decode_millis / 1000.0) if decode_millis > 0 else 0.0
        ),
        "total_tokens_per_second": (
            (len(prompt_ids) + generated_token_count) / (total_millis / 1000.0)
            if total_millis > 0
            else 0.0
        ),
        "generated_text": generated_text,
        "invalid_token_id": invalid_token_id,
        "invalid_token_step": invalid_token_step,
        "terminated_due_to_invalid_token": invalid_token_id is not None,
    }

    summary_path = Path(f"{args.out_prefix}.summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(generated_text)


if __name__ == "__main__":
    main()
