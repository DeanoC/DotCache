#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig

from dotcache.config import DotCacheConfig
from dotcache.integrations.llama import LlamaDotCacheHarness, resolve_hf_auth_kwargs, transformers_available
from dotcache.page_oracle import merge_page_trace_manifests, save_page_trace_manifest


DEFAULT_PROMPT_FAMILIES = {
    "cache": {
        "locality": "Cache locality matters for fast decoding and stable long-context attention.",
        "bandwidth": "Paged serving wins when memory movement is predictable, compact, and aligned with the active working set.",
    },
    "reasoning": {
        "arithmetic": "A careful mathematician writes down each intermediate step, checks the arithmetic twice, and only then states the final answer.",
        "logic": "A patient logician separates premises from conclusions, tests counterexamples, and revises the argument before committing.",
    },
    "instruction": {
        "constraints": "You are a precise assistant. Follow the numbered constraints, keep the answer grounded, and end with a concise response.",
        "formatting": "Return the answer in the requested format, keep the scope narrow, and do not add extra assumptions beyond the prompt.",
    },
    "retrieval": {
        "memo": "The archive memo mentions river permits, bridge repairs, zoning appeals, and school budgets across several quarters of updates.",
        "transcript": "Meeting notes reference subsidy waivers, inspection backlogs, permit renewals, and maintenance schedules across multiple departments.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Llama-family page-trace capture sweep and merge the resulting manifests.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--prompt-unit", default="Cache locality matters for fast decoding.")
    parser.add_argument("--prompt-family", action="append", choices=sorted(DEFAULT_PROMPT_FAMILIES.keys()), default=[])
    parser.add_argument("--prompt-length", action="append", type=int, default=[])
    parser.add_argument("--decode-steps", action="append", type=int, default=[])
    parser.add_argument("--tokens-per-page", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind", action="append", choices=["K", "V"], default=[])
    return parser.parse_args()


def _build_dotcache_config(*, model_id: str, tokens_per_page: int, group_size: int) -> DotCacheConfig:
    auth_kwargs = resolve_hf_auth_kwargs()
    model_config = AutoConfig.from_pretrained(model_id, **auth_kwargs)
    head_dim = int(model_config.hidden_size) // int(model_config.num_attention_heads)
    return DotCacheConfig(
        head_dim=head_dim,
        group_size=int(group_size),
        bits_k=4,
        bits_v=4,
        tokens_per_page=int(tokens_per_page),
    )


def _build_exact_length_inputs(harness: LlamaDotCacheHarness, *, prompt_unit: str, prompt_length: int):
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


def main() -> int:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("run_llama_page_trace_capture_sweep.py requires the optional transformers dependencies")

    prompt_families = args.prompt_family or ["cache", "reasoning", "instruction", "retrieval"]
    prompt_lengths = args.prompt_length or [128, 256, 512, 1024]
    decode_steps = args.decode_steps or [4, 8]
    kinds = tuple(args.kind) if args.kind else ("K", "V")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    harness = LlamaDotCacheHarness.from_pretrained(
        args.model_id,
        _build_dotcache_config(
            model_id=args.model_id,
            tokens_per_page=args.tokens_per_page,
            group_size=args.group_size,
        ),
        backend="cpu_ref",
        device=args.device,
        torch_dtype=args.torch_dtype,
    )

    manifests: list[dict] = []
    runs: list[dict] = []
    for prompt_family in prompt_families:
        family_variants = DEFAULT_PROMPT_FAMILIES.get(prompt_family, {"default": args.prompt_unit})
        for prompt_variant, prompt_unit in family_variants.items():
            for prompt_length in prompt_lengths:
                for decode_step_count in decode_steps:
                    run_name = f"family-{prompt_family}_variant-{prompt_variant}_prompt{prompt_length:03d}_decode{decode_step_count:02d}"
                    run_dir = output_dir / run_name
                    input_ids, attention_mask = _build_exact_length_inputs(
                        harness,
                        prompt_unit=prompt_unit,
                        prompt_length=prompt_length,
                    )
                    result = harness.capture_page_traces(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decode_steps=decode_step_count,
                        output_dir=run_dir,
                        tokens_per_page=args.tokens_per_page,
                        kinds=kinds,
                    )
                    manifests.append(
                        {
                            "output_dir": str(run_dir),
                            "page_trace_count": int(result["page_trace_count"]),
                            "page_trace_paths": list(result["page_trace_paths"]),
                            "page_trace_counts_by_kind": dict(result["page_trace_counts_by_kind"]),
                            "page_trace_counts_by_stage": dict(result["page_trace_counts_by_stage"]),
                            "page_trace_counts_by_layer": dict(result["page_trace_counts_by_layer"]),
                            "tokens_per_page": int(result["tokens_per_page"]),
                            "kinds": list(result["kinds"]),
                            "source": str(result["source"]),
                        }
                    )
                    runs.append(
                        {
                            "run_name": run_name,
                            "prompt_family": prompt_family,
                            "prompt_variant": prompt_variant,
                            "prompt_length": int(prompt_length),
                            "decode_steps": int(decode_step_count),
                            "output_dir": str(run_dir),
                            "page_trace_count": int(result["page_trace_count"]),
                        }
                    )

    merged_manifest = merge_page_trace_manifests(
        manifests,
        output_dir=output_dir,
        source="llama_page_trace_capture_sweep",
    )
    merged_manifest["runs"] = runs
    merged_manifest["prompt_families"] = list(prompt_families)
    merged_manifest["prompt_variants_by_family"] = {
        family: sorted(DEFAULT_PROMPT_FAMILIES.get(family, {"default": args.prompt_unit}).keys())
        for family in prompt_families
    }
    merged_manifest["prompt_lengths"] = [int(value) for value in prompt_lengths]
    merged_manifest["decode_steps_swept"] = [int(value) for value in decode_steps]
    merged_manifest["tokens_per_page"] = int(args.tokens_per_page)
    merged_manifest["kinds"] = list(kinds)
    save_page_trace_manifest(merged_manifest, output_dir / "manifest.json")
    (output_dir / "runs.json").write_text(json.dumps(runs, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(merged_manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
