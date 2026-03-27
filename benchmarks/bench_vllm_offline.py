from __future__ import annotations

import argparse
import json
import time
from typing import Any

from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.integrations.vllm_adapter import (
    install_dotcache_on_vllm_runtime,
    require_supported_vllm_version,
    vllm_available,
)
from dotcache.tracing import ExecutionTrace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline vLLM dense vs DotCache benchmark for the Phase 6 CUDA adapter.")
    parser.add_argument("--model-id", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--mode", choices=["dense", "dotcache_shadow", "dotcache_active", "all"], default="all")
    parser.add_argument("--backend", choices=["torch_cuda"], default="torch_cuda")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", default="half")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default="Write one short sentence about cache locality.")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def _require_vllm_runtime() -> None:
    if not vllm_available():
        raise SystemExit("bench_vllm_offline.py requires the optional vllm extra on the CUDA machine")
    require_supported_vllm_version()


def _sampling_params(max_new_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(temperature=0.0, max_tokens=max_new_tokens)


def _build_llm(args: argparse.Namespace):
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model_id,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "block_size": args.block_size,
        "enforce_eager": True,
        "trust_remote_code": bool(args.trust_remote_code),
    }
    if args.max_model_len is not None:
        kwargs["max_model_len"] = int(args.max_model_len)
    return LLM(**kwargs)


def _extract_generated_token_ids(output: Any) -> list[int]:
    outputs = getattr(output, "outputs", None)
    if outputs:
        first = outputs[0]
        token_ids = getattr(first, "token_ids", None)
        if token_ids is not None:
            return [int(token_id) for token_id in token_ids]
    token_ids = getattr(output, "token_ids", None)
    if token_ids is not None:
        return [int(token_id) for token_id in token_ids]
    raise RuntimeError("could not extract generated token ids from the vLLM output object")


def _run_mode(args: argparse.Namespace, *, mode: str) -> dict[str, Any]:
    llm = _build_llm(args)
    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=bool(args.trust_remote_code))
    head_dim = int(model_config.hidden_size // model_config.num_attention_heads)
    dotcache_config = DotCacheConfig(
        head_dim=head_dim,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=args.block_size,
    )
    adapter = None
    if mode != "dense":
        adapter = install_dotcache_on_vllm_runtime(
            llm,
            dotcache_config,
            block_size=args.block_size,
            backend=args.backend,
            mode=mode,
        )
        adapter.clear()
        adapter.reset_runtime_metrics()
    sampling_params = _sampling_params(args.max_new_tokens)
    start = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling_params=sampling_params)
    wall_ms = (time.perf_counter() - start) * 1000.0
    if not outputs:
        raise RuntimeError("vLLM returned no outputs")
    token_ids = _extract_generated_token_ids(outputs[0])
    record: dict[str, Any] = {
        "benchmark": "vllm_offline",
        "model_id": args.model_id,
        "mode": mode,
        "backend": args.backend,
        "block_size": args.block_size,
        "prompt": args.prompt,
        "prompt_length": None,
        "decode_steps": len(token_ids),
        "generated_token_ids": token_ids,
        "wall_ms_total": wall_ms,
        "decode_ms_per_step": wall_ms / max(len(token_ids), 1),
    }
    if adapter is None:
        return record
    trace = adapter.runtime_trace if adapter is not None else ExecutionTrace()
    record.update(
        {
            "prefill_block_encode_ms": adapter.prefill_block_encode_ms_total,
            "append_runtime_ms_total": adapter.append_runtime_ms_total,
            "decode_runtime_ms_total": adapter.decode_runtime_ms_total,
            "dotcache_resident_bytes": adapter.resident_bytes,
            "dotcache_host_to_device_bytes": trace.host_to_device_bytes,
            "shadow_output_max_abs_error": adapter.shadow_output_max_abs_error if mode == "dotcache_shadow" else None,
            "shadow_output_max_rel_error": adapter.shadow_output_max_rel_error if mode == "dotcache_shadow" else None,
            "teacher_forced_logit_max_abs_error": None,
        }
    )
    return record


def main() -> None:
    args = parse_args()
    _require_vllm_runtime()
    modes = ["dense", "dotcache_shadow", "dotcache_active"] if args.mode == "all" else [args.mode]
    records = [_run_mode(args, mode=mode) for mode in modes]
    dense_record = next((record for record in records if record["mode"] == "dense"), None)
    if dense_record is not None:
        dense_ids = dense_record["generated_token_ids"]
        for record in records:
            if record["mode"] == "dense":
                continue
            compared = min(len(dense_ids), len(record["generated_token_ids"]))
            agreement = 1.0
            if compared > 0:
                agreement = sum(
                    int(dense_ids[index] == record["generated_token_ids"][index]) for index in range(compared)
                ) / compared
            record["greedy_agreement_vs_dense"] = agreement
            record["dense_decode_ms_per_step"] = dense_record["decode_ms_per_step"]
            record["dotcache_vs_dense_decode_speedup"] = (
                dense_record["decode_ms_per_step"] / record["decode_ms_per_step"]
                if record["decode_ms_per_step"] > 0
                else None
            )
    print(json.dumps(records if len(records) > 1 else records[0], sort_keys=True))


if __name__ == "__main__":
    main()
