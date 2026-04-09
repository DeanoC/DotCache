from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotcache.integrations.llama import resolve_hf_auth_kwargs
from dotcache.integrations.qwen35 import Qwen35AttentionSubsetHarness, transformers_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a broader real Qwen paged-attention replay corpus.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--repeat-counts", type=int, nargs="*", default=[])
    parser.add_argument("--target-prompt-lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--prompt-files", nargs="*", default=[])
    parser.add_argument("--prompt-file-target-length", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[3, 7, 11, 15, 19, 23])
    parser.add_argument("--kv-head-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--step-indices", type=int, nargs="+", default=[0, -1])
    parser.add_argument("--tokens-per-page", type=int, default=64)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def _build_exact_length_inputs(
    harness: Qwen35AttentionSubsetHarness,
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


def _build_prompt_text_inputs(
    harness: Qwen35AttentionSubsetHarness,
    *,
    prompt_text: str,
    prompt_length: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if harness.tokenizer is None:
        raise ValueError("tokenizer is unavailable for prompt-text construction")
    tokenizer = harness.tokenizer
    token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError("prompt text tokenized to an empty sequence")
    if prompt_length > 0:
        token_ids = token_ids[:prompt_length]
        if not token_ids:
            raise ValueError("prompt text does not contain enough tokens for the requested prompt_length")
    if tokenizer.bos_token_id is not None:
        token_ids = [int(tokenizer.bos_token_id)] + [int(token_id) for token_id in token_ids]
    else:
        token_ids = [int(token_id) for token_id in token_ids]
    device = harness.adapter.device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    return input_ids, attention_mask


def _case_tag(base_record: dict[str, object]) -> str:
    if "repeat_count" in base_record:
        return f"repeat{int(base_record['repeat_count']):04d}"
    if "prompt_file_label" in base_record:
        return str(base_record["prompt_file_label"])
    prompt_length = int(base_record["prompt_length"])
    return f"prompt{prompt_length:05d}"


def _build_summary(records: list[dict[str, object]]) -> dict[str, object]:
    success_records = [record for record in records if record.get("status", "ok") == "ok"]
    error_records = [record for record in records if record.get("status") == "error"]
    counts_by_prompt_mode: dict[str, int] = {}
    unique_layers: set[int] = set()
    unique_kv_heads: set[int] = set()
    unique_steps: set[int] = set()
    total_snapshots = 0
    for record in success_records:
        prompt_mode = str(record.get("prompt_mode", "unknown"))
        counts_by_prompt_mode[prompt_mode] = counts_by_prompt_mode.get(prompt_mode, 0) + 1
        total_snapshots += int(record.get("paged_attention_snapshot_corpus_count", 0))
        for layer_id in record.get("paged_attention_snapshot_corpus_layer_ids", []):
            unique_layers.add(int(layer_id))
        for kv_head_id in record.get("paged_attention_snapshot_corpus_kv_head_ids", []):
            unique_kv_heads.add(int(kv_head_id))
        for step_index in record.get("paged_attention_snapshot_corpus_resolved_step_indices", []):
            unique_steps.add(int(step_index))
    return {
        "case_count": len(records),
        "success_case_count": len(success_records),
        "error_case_count": len(error_records),
        "snapshot_count": int(total_snapshots),
        "counts_by_prompt_mode": dict(sorted(counts_by_prompt_mode.items())),
        "layer_ids": sorted(unique_layers),
        "kv_head_ids": sorted(unique_kv_heads),
        "resolved_step_indices": sorted(unique_steps),
    }


def _run_case(
    harness: Qwen35AttentionSubsetHarness,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    base_record: dict[str, object],
    output_root: Path,
    layer_ids: list[int],
    kv_head_ids: list[int],
    step_indices: list[int],
    tokens_per_page: int,
    continue_on_error: bool,
) -> dict[str, object]:
    case_dir = output_root / _case_tag(base_record)
    try:
        record = harness.capture_attention_subset_paged_attention_snapshot_corpus(
            output_dir=case_dir,
            layer_ids=layer_ids,
            kv_head_ids=kv_head_ids,
            step_indices=step_indices,
            tokens_per_page=tokens_per_page,
            input_ids=input_ids,
            attention_mask=attention_mask,
            decode_steps=max_new_tokens,
        )
    except Exception as exc:  # pragma: no cover - benchmark failure path
        if not continue_on_error:
            raise
        record = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "prompt_length": int(input_ids.shape[1]),
        }
    else:
        record["status"] = "ok"
    record.update(base_record)
    record["case_tag"] = _case_tag(base_record)
    record["output_dir"] = str(case_dir)
    return record


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("bench_qwen35_paged_attention_replay_corpus.py requires the optional transformers dependencies")

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else output_root / "manifest.json"
    summary_path = Path(args.summary_path).resolve() if args.summary_path else output_root / "summary.json"

    model_config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=False, **resolve_hf_auth_kwargs())
    text_config = getattr(model_config, "text_config", model_config)
    max_position_embeddings = int(getattr(text_config, "max_position_embeddings", 0) or 0)

    harness = Qwen35AttentionSubsetHarness.from_pretrained(
        args.model_id,
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    records: list[dict[str, object]] = []
    for repeat_count in args.repeat_counts:
        prompt = " ".join([args.prompt_unit] * repeat_count)
        input_ids, attention_mask = harness.tokenize_prompt(prompt)
        record = _run_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            base_record={
                "benchmark": "qwen35_paged_attention_replay_corpus",
                "model_id": args.model_id,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "prompt_mode": "repeat_count",
                "repeat_count": int(repeat_count),
                "prompt_unit": args.prompt_unit,
                "model_max_position_embeddings": max_position_embeddings,
            },
            output_root=output_root,
            layer_ids=[int(layer_id) for layer_id in args.layer_ids],
            kv_head_ids=[int(kv_head_id) for kv_head_id in args.kv_head_ids],
            step_indices=[int(step_index) for step_index in args.step_indices],
            tokens_per_page=int(args.tokens_per_page),
            continue_on_error=bool(args.continue_on_error),
        )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    for prompt_length in sorted(set(length for length in args.target_prompt_lengths if length > 0)):
        input_ids, attention_mask = _build_exact_length_inputs(
            harness,
            prompt_unit=args.prompt_unit,
            prompt_length=int(prompt_length),
        )
        record = _run_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            base_record={
                "benchmark": "qwen35_paged_attention_replay_corpus",
                "model_id": args.model_id,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "prompt_mode": "exact_length",
                "prompt_length": int(prompt_length),
                "prompt_unit": args.prompt_unit,
                "model_max_position_embeddings": max_position_embeddings,
            },
            output_root=output_root,
            layer_ids=[int(layer_id) for layer_id in args.layer_ids],
            kv_head_ids=[int(kv_head_id) for kv_head_id in args.kv_head_ids],
            step_indices=[int(step_index) for step_index in args.step_indices],
            tokens_per_page=int(args.tokens_per_page),
            continue_on_error=bool(args.continue_on_error),
        )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    for prompt_file in [Path(path).resolve() for path in args.prompt_files]:
        prompt_text = prompt_file.read_text(encoding="utf-8", errors="ignore")
        input_ids, attention_mask = _build_prompt_text_inputs(
            harness,
            prompt_text=prompt_text,
            prompt_length=max(int(args.prompt_file_target_length), 0),
        )
        record = _run_case(
            harness,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            base_record={
                "benchmark": "qwen35_paged_attention_replay_corpus",
                "model_id": args.model_id,
                "device": args.device,
                "torch_dtype": args.torch_dtype,
                "prompt_mode": "prompt_file",
                "prompt_file_path": str(prompt_file),
                "prompt_file_label": prompt_file.stem.replace(" ", "_").lower(),
                "prompt_length": int(input_ids.shape[1]),
                "prompt_unit": None,
                "model_max_position_embeddings": max_position_embeddings,
            },
            output_root=output_root,
            layer_ids=[int(layer_id) for layer_id in args.layer_ids],
            kv_head_ids=[int(kv_head_id) for kv_head_id in args.kv_head_ids],
            step_indices=[int(step_index) for step_index in args.step_indices],
            tokens_per_page=int(args.tokens_per_page),
            continue_on_error=bool(args.continue_on_error),
        )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    summary = _build_summary(records)
    manifest = {
        "output_dir": str(output_root),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "records": records,
        "summary": summary,
        "layer_ids": [int(layer_id) for layer_id in args.layer_ids],
        "kv_head_ids": [int(kv_head_id) for kv_head_id in args.kv_head_ids],
        "requested_step_indices": [int(step_index) for step_index in args.step_indices],
        "tokens_per_page": int(args.tokens_per_page),
        "max_new_tokens": int(args.max_new_tokens),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
