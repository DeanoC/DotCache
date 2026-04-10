from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import torch

from dotcache.integrations.qwen35 import (
    Qwen35TextHarness,
    _run_dense_decode_step,
    _run_dense_prefill,
    transformers_available,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense-only mixed workload harness for Qwen3.5 text models."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", choices=["torch_mps", "torch_cuda", "cpu_ref", "auto"], default="torch_mps")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--shared-prompt-length", type=int, default=None)
    parser.add_argument("--shared-prompt-text", default=None)
    parser.add_argument("--shared-prompt-token-target", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument("--total-sessions", type=int, default=4)
    parser.add_argument("--wave-size", type=int, default=2)
    parser.add_argument("--decode-rounds-per-wave", type=int, default=1)
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--suffix-prefix", default="session")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--stress-suffix-repeats", type=int, default=6)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.shared_prompt_text is None and args.shared_prompt_length is None:
        parser.error("either --shared-prompt-length or --shared-prompt-text is required")
    return args


def _build_exact_length_input_ids(
    harness: Qwen35TextHarness,
    *,
    prompt_unit: str,
    prompt_length: int,
    add_bos: bool,
) -> torch.Tensor:
    if harness.tokenizer is None:
        raise ValueError("tokenizer is unavailable for exact-length prompt construction")
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    tokenizer = harness.tokenizer
    unit_ids = tokenizer(prompt_unit, add_special_tokens=False)["input_ids"]
    if not unit_ids:
        raise ValueError("prompt_unit tokenized to an empty sequence")

    token_ids: list[int] = []
    if add_bos and tokenizer.bos_token_id is not None:
        token_ids.append(int(tokenizer.bos_token_id))
    while len(token_ids) < prompt_length:
        token_ids.extend(int(token_id) for token_id in unit_ids)
    token_ids = token_ids[:prompt_length]
    return torch.tensor([token_ids], dtype=torch.long, device=harness.adapter.device)


def _build_prompt_text_input_ids(
    harness: Qwen35TextHarness,
    *,
    prompt_text: str,
    prompt_token_target: int | None,
) -> torch.Tensor:
    if harness.tokenizer is None:
        raise ValueError("tokenizer is unavailable for prompt-text construction")
    tokenizer = harness.tokenizer
    token_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    if not token_ids:
        raise ValueError("shared_prompt_text tokenized to an empty sequence")
    if prompt_token_target is not None:
        if prompt_token_target <= 0:
            raise ValueError("shared_prompt_token_target must be positive")
        if len(token_ids) > prompt_token_target:
            token_ids = token_ids[:prompt_token_target]
        elif len(token_ids) < prompt_token_target:
            filler_ids = tokenizer(f" {prompt_text}", add_special_tokens=False)["input_ids"]
            if not filler_ids:
                raise ValueError("shared_prompt_text filler tokenized to an empty sequence")
            while len(token_ids) < prompt_token_target:
                token_ids.extend(int(token_id) for token_id in filler_ids)
            token_ids = token_ids[:prompt_token_target]
    return torch.tensor([token_ids], dtype=torch.long, device=harness.adapter.device)


def _suffix_text(prefix: str, logical_index: int, stress_mode: bool, stress_suffix_repeats: int) -> str:
    if not stress_mode:
        return f" {prefix}-{logical_index}"
    parts: list[str] = []
    for repeat in range(stress_suffix_repeats):
        parts.append(
            f" {prefix}-{logical_index}-segment-{repeat} detail-{logical_index} load-{repeat}"
        )
    return "".join(parts)


def _target_decode_tokens(logical_index: int, max_new_tokens: int) -> int:
    if max_new_tokens <= 1:
        return 1
    spread = min(max_new_tokens - 1, 2)
    return max_new_tokens - (logical_index % (spread + 1))


def _timed_call(fn):
    start = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - start) * 1000.0


@dataclass
class DenseSession:
    logical_index: int
    arrival_wave: int
    suffix_text: str
    suffix_token_count: int
    target_decode_tokens: int
    past_key_values: object
    attention_mask: torch.Tensor
    cache_position: torch.Tensor
    logits: torch.Tensor
    generated_token_ids: list[int]
    completed_by_eos: bool = False


def _prefill_session(
    harness: Qwen35TextHarness,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor]:
    harness.adapter.begin_dense_stage_phase("prefill")
    try:
        outputs = _run_dense_prefill(
            harness.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    finally:
        harness.adapter.end_dense_stage_phase()
    next_attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            ),
        ],
        dim=1,
    )
    cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    logits = outputs.logits[:, -1, :]
    return outputs.past_key_values, next_attention_mask, cache_position, logits


def _decode_one_step(harness: Qwen35TextHarness, session: DenseSession, next_token: int):
    decode_input_ids = torch.tensor([[next_token]], dtype=torch.long, device=session.attention_mask.device)
    harness.adapter.begin_dense_stage_phase("decode")
    try:
        outputs = _run_dense_decode_step(
            harness.model,
            decode_input_ids=decode_input_ids,
            attention_mask=session.attention_mask,
            past_key_values=session.past_key_values,
            cache_position=session.cache_position,
        )
    finally:
        harness.adapter.end_dense_stage_phase()
    session.past_key_values = outputs.past_key_values
    session.attention_mask = torch.cat(
        [
            session.attention_mask,
            torch.ones((1, 1), dtype=session.attention_mask.dtype, device=session.attention_mask.device),
        ],
        dim=1,
    )
    session.cache_position = session.cache_position + 1
    session.logits = outputs.logits[:, -1, :]


def _run_workload_pass(harness: Qwen35TextHarness, args: argparse.Namespace) -> dict[str, object]:
    harness.adapter.reset_dense_stage_profile()
    if args.shared_prompt_text is not None:
        shared_prompt_ids = _build_prompt_text_input_ids(
            harness,
            prompt_text=args.shared_prompt_text,
            prompt_token_target=args.shared_prompt_token_target,
        )
    else:
        shared_prompt_ids = _build_exact_length_input_ids(
            harness,
            prompt_unit=args.prompt_unit,
            prompt_length=args.shared_prompt_length,
            add_bos=True,
        )
    shared_prompt_token_count = int(shared_prompt_ids.shape[1])
    eos_token_ids = {
        int(token_id)
        for token_id in ([harness.tokenizer.eos_token_id] if harness.tokenizer and harness.tokenizer.eos_token_id is not None else [])
    }

    total_start = time.perf_counter()
    cold_prefix_prefill_ms = 0.0
    seed_suffix_prefill_ms = 0.0
    attached_suffix_prefill_ms = 0.0
    decode_ms = 0.0

    sessions: list[DenseSession] = []
    active: list[DenseSession] = []
    peak_active_sessions = 0

    def _cold_prefill():
        harness.adapter.begin_dense_stage_phase("prefill")
        try:
            return _run_dense_prefill(
                harness.model,
                input_ids=shared_prompt_ids,
                attention_mask=torch.ones_like(shared_prompt_ids, dtype=torch.long),
            )
        finally:
            harness.adapter.end_dense_stage_phase()

    _, cold_prefix_prefill_ms = _timed_call(_cold_prefill)

    def make_session(logical_index: int, arrival_wave: int) -> DenseSession:
        suffix_text = _suffix_text(
            args.suffix_prefix,
            logical_index,
            args.stress,
            args.stress_suffix_repeats,
        )
        suffix_ids, suffix_mask = harness.tokenize_prompt(suffix_text)
        suffix_ids = suffix_ids[:, 1:] if suffix_ids.shape[1] > 0 else suffix_ids
        suffix_mask = suffix_mask[:, 1:] if suffix_mask.shape[1] > 0 else suffix_mask
        input_ids = torch.cat([shared_prompt_ids, suffix_ids], dim=1)
        attention_mask = torch.cat(
            [torch.ones_like(shared_prompt_ids, dtype=torch.long), suffix_mask], dim=1
        )
        (past_key_values, next_attention_mask, cache_position, logits), elapsed_ms = _timed_call(
            lambda: _prefill_session(
                harness,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )
        nonlocal seed_suffix_prefill_ms, attached_suffix_prefill_ms
        if logical_index == 0:
            seed_suffix_prefill_ms += elapsed_ms
        else:
            attached_suffix_prefill_ms += elapsed_ms
        return DenseSession(
            logical_index=logical_index,
            arrival_wave=arrival_wave,
            suffix_text=suffix_text,
            suffix_token_count=int(suffix_ids.shape[1]),
            target_decode_tokens=_target_decode_tokens(logical_index, args.max_new_tokens),
            past_key_values=past_key_values,
            attention_mask=next_attention_mask,
            cache_position=cache_position,
            logits=logits,
            generated_token_ids=[],
        )

    seed = make_session(0, 0)
    sessions.append(seed)
    active.append(seed)
    peak_active_sessions = 1

    def run_decode_round() -> None:
        nonlocal decode_ms
        next_active: list[DenseSession] = []
        for session in active:
            next_token = int(torch.argmax(session.logits, dim=-1).item())
            session.generated_token_ids.append(next_token)
            hit_eos = next_token in eos_token_ids
            hit_limit = len(session.generated_token_ids) >= session.target_decode_tokens
            if hit_eos or hit_limit:
                session.completed_by_eos = hit_eos
                continue
            _, step_ms = _timed_call(lambda: _decode_one_step(harness, session, next_token))
            decode_ms += step_ms
            next_active.append(session)
        active[:] = next_active

    if args.total_sessions > 1:
        run_decode_round()

    if args.wave_size <= 0:
        raise ValueError("--wave-size must be positive")

    next_logical = 1
    wave_index = 1
    while next_logical < args.total_sessions:
        arrivals = min(args.wave_size, args.total_sessions - next_logical)
        for _ in range(arrivals):
            session = make_session(next_logical, wave_index)
            sessions.append(session)
            active.append(session)
            next_logical += 1
        peak_active_sessions = max(peak_active_sessions, len(active))
        for _ in range(args.decode_rounds_per_wave):
            if not active:
                break
            run_decode_round()
        wave_index += 1

    while active:
        run_decode_round()

    total_ms = (time.perf_counter() - total_start) * 1000.0
    total_generated = sum(len(session.generated_token_ids) for session in sessions)
    total_input_token_count = shared_prompt_token_count * len(sessions) + sum(
        session.suffix_token_count for session in sessions
    ) + total_generated
    result = {
        "benchmark": "qwen35_text_workload",
        "model_id": args.model_id,
        "backend": args.backend,
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "prompt_mode": "prompt_text" if args.shared_prompt_text is not None else "exact_length",
        "shared_prompt_token_count": shared_prompt_token_count,
        "shared_prompt_text": args.shared_prompt_text,
        "shared_prompt_token_target": args.shared_prompt_token_target,
        "total_sessions": args.total_sessions,
        "wave_size": args.wave_size,
        "decode_rounds_per_wave": args.decode_rounds_per_wave,
        "max_new_tokens": args.max_new_tokens,
        "stress_mode": bool(args.stress),
        "stress_suffix_repeats": int(args.stress_suffix_repeats),
        "cold_prefix_prefill_ms": float(cold_prefix_prefill_ms),
        "seed_suffix_prefill_ms": float(seed_suffix_prefill_ms),
        "attached_suffix_prefill_ms": float(attached_suffix_prefill_ms),
        "decode_ms": float(decode_ms),
        "total_ms": float(total_ms),
        "peak_active_sessions": int(peak_active_sessions),
        "total_generated_token_count": int(total_generated),
        "total_input_token_count": int(total_input_token_count),
        "total_tokens_per_second": float(total_input_token_count / max(total_ms / 1000.0, 1e-8)),
        "sessions": [
            {
                "logical_index": session.logical_index,
                "arrival_wave": session.arrival_wave,
                "suffix_token_count": session.suffix_token_count,
                "target_decode_tokens": session.target_decode_tokens,
                "generated_token_count": len(session.generated_token_ids),
                "completed_by_eos": session.completed_by_eos,
            }
            for session in sessions
        ],
        "status": "ok",
    }
    result.update(harness.adapter.dense_stage_summary())
    return result


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit(
            "bench_qwen35_text_workload.py requires the optional transformers dependencies"
        )

    harness = Qwen35TextHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
        profile_dense_stages=args.profile_stages,
    )

    warmup_ms = 0.0
    for _ in range(max(args.warmup_runs, 0)):
        _, elapsed_ms = _timed_call(lambda: _run_workload_pass(harness, args))
        warmup_ms += elapsed_ms

    try:
        record = _run_workload_pass(harness, args)
    except Exception as exc:  # pragma: no cover - benchmark failure path
        if not args.continue_on_error:
            raise
        record = {
            "benchmark": "qwen35_text_workload",
            "model_id": args.model_id,
            "backend": args.backend,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "prompt_mode": "prompt_text" if args.shared_prompt_text is not None else "exact_length",
            "shared_prompt_token_count": args.shared_prompt_token_target or args.shared_prompt_length,
            "shared_prompt_text": args.shared_prompt_text,
            "shared_prompt_token_target": args.shared_prompt_token_target,
            "total_sessions": args.total_sessions,
            "wave_size": args.wave_size,
            "decode_rounds_per_wave": args.decode_rounds_per_wave,
            "max_new_tokens": args.max_new_tokens,
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    record["warmup_runs"] = int(args.warmup_runs)
    record["warmup_ms"] = float(warmup_ms)
    print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
