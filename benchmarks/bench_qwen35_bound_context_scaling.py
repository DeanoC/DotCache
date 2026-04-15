"""Context-length scaling benchmark for certified bound modes.

Tests how interval-bound win rates, certified-exit efficiency, and ms/step scale
with context length. Designed to run on Mac Mini MPS as a pre-CUDA predictor.

Hypothesis
----------
Longer contexts → more block diversity → interval and ellipsoidal bounds win a
larger fraction of evaluations → certified exit fires earlier → bigger speedup
relative to spherical. If confirmed here, the CUDA gain from interval alone
should exceed the 5.7% measured on the short 1.5K-token benchmark.

Design
------
- Primary file: performance_journal.md (130K tokens) sliced at each target length
  from offset 0. Same text type at every length isolates the length effect.
- Control files: benchmark_report.md and qwen35_stage9_thesis_status sliced at
  their natural lengths (up to 8K), to catch any file-specific variance.
- Lengths: 2K, 4K, 8K tokens (16K excluded on MPS — prefill OOMs; defer to CUDA box).
- Lanes: spherical_only and interval (ellipsoidal excluded — too slow on MPS
  for a 4-length sweep; its win rate is already characterised at 1.5K).
- Decode steps: 16 per case (more stable timing than 8).

Expected runtime on Mac Mini MPS: ~25–35 minutes.

Key outputs
-----------
- Per-length summary: ms/step, cert_stop_block_rate (cert_stop / total_blocks),
  interval_win_frac, exact_match_vs_dense
- Scaling table: how each metric changes as context doubles
- Raw JSON + MD artefacts for the CUDA run notes
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_serving_policy_compare import (
    _DEFAULT_POLICY_PATH,
    _build_prompt_text_inputs,
    _resolve_prompt_records,
)
from benchmarks.bench_qwen35_persistent_real_mixed_probe import (
    real_mixed_probe_dotcache_config,
    real_mixed_probe_serving_config,
)
from dotcache.backends.metal.persistent_types import PersistentServingConfig
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheModelAdapter,
    Qwen35TextModelAdapter,
    load_qwen35_text_only_from_pretrained,
    run_qwen35_attention_subset_persistent_serving_harness,
    run_qwen35_text_generation_harness,
    transformers_available,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Primary file: 130K tokens — sliced at each target length from the start
_PRIMARY_FILE = _REPO_ROOT / "docs" / "performance_journal.md"

# Control files: natural-length diversity; capped to avoid exceeding their size
_CONTROL_FILES = [
    (_REPO_ROOT / "docs" / "benchmark_report.md",              8192),   # 8.6K tok
    (_REPO_ROOT / "docs" / "qwen35_stage9_thesis_status_20260412.md", 6144),  # 6.6K tok
]

# Target lengths to sweep.
# 16K is excluded from the MPS default: the persistent prefill at 16K uses full SDPA
# (block-selective attention only applies during decode), which OOMs on Mac Mini MPS
# (~20 GB unified memory limit). 16K+ should be run on the CUDA box.
_DEFAULT_LENGTHS = [2048, 4096, 8192]

# Lanes
_LANES = [
    {
        "name": "spherical_only",
        "label": "Spherical only (baseline)",
        "enable_interval_bound": False,
        "enable_ellipsoidal_bound": False,
    },
    {
        "name": "interval",
        "label": "Interval bound",
        "enable_interval_bound": True,
        "enable_ellipsoidal_bound": False,
    },
    {
        "name": "interval_ellip",
        "label": "Interval + Ellipsoidal (full paper)",
        "enable_interval_bound": True,
        "enable_ellipsoidal_bound": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Context-length scaling benchmark for certified bound modes."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--backend", default="torch_mps")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=_DEFAULT_LENGTHS,
        help="Context lengths to sweep (in tokens).",
    )
    parser.add_argument(
        "--lanes",
        nargs="*",
        default=None,
        help="Subset of lane names (default: spherical_only interval).",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument(
        "--max-dense-length",
        type=int,
        default=8192,
        help=(
            "Skip the dense reference pass for cases with target_length > this value. "
            "Full-attention dense prefill at 16K+ tokens OOMs on Mac Mini MPS (unified "
            "memory limit ~20 GB). The persistent path is fine since it never materialises "
            "the full attention matrix. Default: 8192."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Decode steps to run as a throwaway warmup before the first timed case "
            "in each lane.  Primes Triton JIT compilation and CUDA kernel caches so "
            "the timed run is not polluted by cold-start overhead.  "
            "Default: 1.  Set to 0 to skip (e.g. on MPS where JIT is not an issue)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel worker processes.  Each worker loads the model "
            "independently and processes a subset of cases.  Improves GPU utilisation "
            "by overlapping Python overhead across workers.  Requires ~1.6 GB VRAM "
            "per worker (Qwen3.5-0.8B float16).  Default: 1."
        ),
    )
    return parser.parse_args()


def _build_serving_config(lane: dict[str, Any]) -> PersistentServingConfig:
    config = real_mixed_probe_serving_config(policy_path=str(_DEFAULT_POLICY_PATH))
    config.enable_interval_bound = bool(lane["enable_interval_bound"])
    config.enable_ellipsoidal_bound = bool(lane["enable_ellipsoidal_bound"])
    return config


def _build_cases(lengths: list[int]) -> list[dict[str, Any]]:
    """Build (case_tag, prompt_file, target_length) list for the sweep.

    Each target length gets:
      - A slice of performance_journal.md (primary; always available)
      - Slices of the control files up to their natural cap
    """
    cases: list[dict[str, Any]] = []
    for length in sorted(set(lengths)):
        # Primary slice
        cases.append({
            "case_tag": f"perf_journal_{length // 1024}k",
            "prompt_file": str(_PRIMARY_FILE),
            "target_length": length,
        })
        # Controls (only when the file is long enough for this target)
        for ctrl_file, max_len in _CONTROL_FILES:
            if length <= max_len and ctrl_file.exists():
                short_name = ctrl_file.stem[:20]
                cases.append({
                    "case_tag": f"{short_name}_{length // 1024}k",
                    "prompt_file": str(ctrl_file),
                    "target_length": length,
                })
    return cases


def _sum_by_layer(result: dict[str, Any], key: str) -> float:
    return float(sum(float(v) for v in result.get(key, {}).values()))


def _sum_by_layer_int(result: dict[str, Any], key: str) -> int:
    return int(sum(int(v) for v in result.get(key, {}).values() if v is not None))


def _run_one_case(
    *,
    case: dict[str, Any],
    lane: dict[str, Any],
    model: Any,
    tokenizer: Any,
    adapter: Qwen35AttentionSubsetDotCacheModelAdapter,
    dense_ids: list[int],
    decode_steps: int,
) -> dict[str, Any]:
    serving_config = _build_serving_config(lane)
    device = next(model.parameters()).device
    input_ids, attention_mask = _build_prompt_text_inputs(
        tokenizer,
        device=device,
        prompt_text=Path(case["prompt_file"]).read_text(encoding="utf-8"),
        prompt_length=int(case["target_length"]),
    )
    result = run_qwen35_attention_subset_persistent_serving_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=decode_steps,
        persistent_serving_config=serving_config,
    )
    decode_ms = float(result.get("persistent_decode_ms_per_step", 0.0))
    generated_ids = [int(t) for t in result.get("persistent_generated_ids", [])]
    if not dense_ids:
        exact_match = None  # dense reference skipped (context too long for MPS dense prefill)
    else:
        exact_match = bool(generated_ids == dense_ids[:decode_steps] and len(generated_ids) >= decode_steps)

    # Block counts: executed and cert-stop
    executed_m0 = _sum_by_layer_int(result, "persistent_full_attention_executed_m0_block_count_total_by_layer")
    executed_m3 = _sum_by_layer_int(result, "persistent_full_attention_executed_m3_block_count_total_by_layer")
    cert_stop   = _sum_by_layer_int(result, "persistent_full_attention_last_first_certified_stop_block_count_by_layer")
    checkpoints = _sum_by_layer_int(result, "persistent_full_attention_last_checkpoint_count_by_layer")

    # Bound winner counts
    sph_count  = _sum_by_layer_int(result, "persistent_full_attention_bound_spherical_active_count_by_layer")
    int_count  = _sum_by_layer_int(result, "persistent_full_attention_bound_interval_active_count_by_layer")
    ellip_count = _sum_by_layer_int(result, "persistent_full_attention_bound_ellipsoidal_active_count_by_layer")
    total_count = sph_count + int_count + ellip_count

    # cert_stop_block_rate: blocks processed at first certified stop / total blocks executed
    # (lower = earlier exit = more efficient)
    total_executed = executed_m0 + executed_m3
    cert_stop_rate = float(cert_stop) / float(total_executed) if total_executed > 0 else 1.0

    # Actual token count in this case (input_ids includes BOS)
    actual_tokens = int(input_ids.shape[-1])
    # Blocks per layer = ceil(actual_tokens / block_size) with block_size=16
    blocks_per_layer = (actual_tokens + 15) // 16

    record = {
        "case_tag": str(case["case_tag"]),
        "target_length": int(case["target_length"]),
        "actual_tokens": actual_tokens,
        "blocks_per_layer": blocks_per_layer,
        "lane": str(lane["name"]),
        "decode_ms_per_step": float(decode_ms),
        "exact_match_vs_dense": exact_match,  # None when dense reference was skipped
        "executed_m0_blocks": int(executed_m0),
        "executed_m3_blocks": int(executed_m3),
        "cert_stop_blocks": int(cert_stop),
        "cert_stop_rate": float(cert_stop_rate),
        "checkpoints": int(checkpoints),
        "bound_spherical_count": int(sph_count),
        "bound_interval_count": int(int_count),
        "bound_ellipsoidal_count": int(ellip_count),
        "bound_total_count": int(total_count),
        "bound_spherical_frac": float(sph_count) / float(total_count) if total_count > 0 else 0.0,
        "bound_interval_frac": float(int_count) / float(total_count) if total_count > 0 else 0.0,
        "bound_ellipsoidal_frac": float(ellip_count) / float(total_count) if total_count > 0 else 0.0,
        # Certificate bound values (per-layer aggregates)
        "beta_upper_by_layer": result.get("persistent_full_attention_last_beta_upper_by_layer", {}),
        "delta_upper_by_layer": result.get("persistent_full_attention_last_delta_upper_by_layer", {}),
        "first_cert_stop_blocks_by_layer": result.get(
            "persistent_full_attention_last_first_certified_stop_block_count_by_layer", {}
        ),
        "certified_can_stop_by_layer": result.get(
            "persistent_full_attention_last_certified_can_stop_by_layer", {}
        ),
        # Fallback counts
        "fallback_process_more_count": _sum_by_layer_int(
            result, "persistent_full_attention_fallback_process_more_count_by_layer"
        ),
        "dense_fallback_count": _sum_by_layer_int(
            result, "persistent_full_attention_dense_fallback_count_by_layer"
        ),
    }

    bound_str = ""
    if total_count > 0:
        bound_str = (
            f"  bound: sph={record['bound_spherical_frac']:.0%}"
            f" int={record['bound_interval_frac']:.0%}"
        )
    print(
        f"  [{lane['name']}] {case['case_tag']}"
        f" ({actual_tokens} tok, {blocks_per_layer} blk/layer):"
        f" {decode_ms:.1f} ms/step, exact={exact_match if exact_match is not None else 'skipped'},"
        f" cert_stop_rate={cert_stop_rate:.1%}{bound_str}"
    )
    return record


def _summarise_by_length(records: list[dict[str, Any]], lane_name: str) -> dict[int, dict[str, Any]]:
    """Group records by target_length for a given lane, compute mean metrics."""
    from collections import defaultdict
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["lane"] == lane_name:
            groups[r["target_length"]].append(r)
    out: dict[int, dict[str, Any]] = {}
    for length, recs in sorted(groups.items()):
        n = len(recs)
        total_bound = sum(r["bound_total_count"] for r in recs)
        out[length] = {
            "case_count": n,
            "avg_ms_per_step": sum(r["decode_ms_per_step"] for r in recs) / n,
            "exact_match": (
            sum(int(r["exact_match_vs_dense"]) for r in recs if r["exact_match_vs_dense"] is not None)
            / max(sum(1 for r in recs if r["exact_match_vs_dense"] is not None), 1)
        ),
            "avg_cert_stop_rate": sum(r["cert_stop_rate"] for r in recs) / n,
            "avg_blocks_per_layer": sum(r["blocks_per_layer"] for r in recs) / n,
            "bound_spherical_frac": sum(r["bound_spherical_count"] for r in recs) / total_bound if total_bound > 0 else 0.0,
            "bound_interval_frac": sum(r["bound_interval_count"] for r in recs) / total_bound if total_bound > 0 else 0.0,
            "bound_ellipsoidal_frac": sum(r["bound_ellipsoidal_count"] for r in recs) / total_bound if total_bound > 0 else 0.0,
        }
    return out


def _render_markdown(
    *,
    payload: dict[str, Any],
    active_lanes: list[dict[str, Any]],
    lengths: list[int],
) -> str:
    records = payload.get("records", [])
    lines = [
        "# Qwen3.5 Bound Mode — Context-Length Scaling",
        "",
        "Measures how interval-bound win rate, certified-exit efficiency, and ms/step",
        "scale with context length on Mac Mini MPS.",
        "",
        "## Lane definitions",
        "",
        "| Lane | enable_interval_bound | enable_ellipsoidal_bound |",
        "|---|---|---|",
    ]
    for lane in active_lanes:
        lines.append(
            f"| `{lane['name']}` | {lane['enable_interval_bound']} | {lane['enable_ellipsoidal_bound']} |"
        )

    # Per-length scaling table for each lane
    for lane in active_lanes:
        by_length = _summarise_by_length(records, lane["name"])
        if not by_length:
            continue
        lines += [
            "",
            f"## `{lane['name']}` — scaling table",
            "",
            "| tokens | blocks/layer | ms/step | exact | cert_stop_rate"
            + (" | sph_win | int_win |" if lane["enable_interval_bound"] else " |"),
            "|---|---|---|---|---" + ("|----|------|" if lane["enable_interval_bound"] else "|"),
        ]
        for length in sorted(by_length):
            s = by_length[length]
            row = (
                f"| {length:,} "
                f"| {s['avg_blocks_per_layer']:.0f} "
                f"| {s['avg_ms_per_step']:.1f} "
                f"| {s['exact_match']:.3f} "
                f"| {s['avg_cert_stop_rate']:.1%} "
            )
            if lane["enable_interval_bound"]:
                row += f"| {s['bound_spherical_frac']:.1%} | {s['bound_interval_frac']:.1%} |"
            else:
                row += "|"
            lines.append(row)

    # Speedup table: interval vs spherical at each length
    sph_by_len  = _summarise_by_length(records, "spherical_only")
    int_by_len  = _summarise_by_length(records, "interval")
    if sph_by_len and int_by_len:
        lines += [
            "",
            "## Interval speedup vs spherical by context length",
            "",
            "| tokens | spherical ms/step | interval ms/step | speedup | int_win_frac |",
            "|---|---|---|---|---|",
        ]
        for length in sorted(sph_by_len):
            if length not in int_by_len:
                continue
            s_ms = sph_by_len[length]["avg_ms_per_step"]
            i_ms = int_by_len[length]["avg_ms_per_step"]
            speedup = (s_ms - i_ms) / s_ms * 100.0
            int_frac = int_by_len[length]["bound_interval_frac"]
            lines.append(
                f"| {length:,} "
                f"| {s_ms:.1f} "
                f"| {i_ms:.1f} "
                f"| {speedup:+.1f}% "
                f"| {int_frac:.1%} |"
            )

    # Per-case detail
    lines += ["", "## Per-case results", ""]
    case_tags = sorted({r["case_tag"] for r in records})
    for case_tag in case_tags:
        case_recs = [r for r in records if r["case_tag"] == case_tag]
        if not case_recs:
            continue
        lines.append(f"### {case_tag}")
        lines.append("")
        for r in sorted(case_recs, key=lambda x: x["lane"]):
            bound_str = ""
            if r["bound_total_count"] > 0:
                bound_str = (
                    f", bound: sph={r['bound_spherical_frac']:.0%}"
                    f"/int={r['bound_interval_frac']:.0%}"
                )
            lines.append(
                f"- `{r['lane']}` ({r['actual_tokens']} tok, {r['blocks_per_layer']} blk/layer):"
                f" {r['decode_ms_per_step']:.1f} ms/step,"
                f" exact={r['exact_match_vs_dense'] if r['exact_match_vs_dense'] is not None else 'skipped'},"
                f" cert_stop_rate={r['cert_stop_rate']:.1%}"
                f"{bound_str}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Multi-process worker infrastructure
# ---------------------------------------------------------------------------
_CS_WORKER_STATE: dict[str, Any] = {}


def _cs_worker_init(
    model_id: str,
    device: str,
    backend: str,
    torch_dtype: str,
    first_lane: dict[str, Any],
) -> None:
    """Initialiser: load model once per worker process."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _repo = str(Path(__file__).resolve().parents[1])
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    from dotcache.integrations.qwen35 import (
        Qwen35AttentionSubsetDotCacheModelAdapter,
        load_qwen35_text_only_from_pretrained,
    )
    from benchmarks.bench_qwen35_persistent_real_mixed_probe import (
        real_mixed_probe_dotcache_config,
    )

    model, tokenizer = load_qwen35_text_only_from_pretrained(
        model_id, device=device, torch_dtype=torch_dtype
    )
    dotcache_config = real_mixed_probe_dotcache_config()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=dotcache_config,
        persistent_serving_config=_build_serving_config(first_lane),
        backend=backend,
    )
    _CS_WORKER_STATE["model"] = model
    _CS_WORKER_STATE["tokenizer"] = tokenizer
    _CS_WORKER_STATE["adapter"] = adapter
    print(f"[cs-worker pid={os.getpid()}] model loaded on {device}", flush=True)


def _cs_worker_run_case(
    args_tuple: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Run all lanes for a single case record.  Returns a list of result dicts."""
    (case, dense_ids, decode_steps, lane_dicts) = args_tuple

    model = _CS_WORKER_STATE["model"]
    tokenizer = _CS_WORKER_STATE["tokenizer"]
    adapter = _CS_WORKER_STATE["adapter"]

    records: list[dict[str, Any]] = []
    for lane in lane_dicts:
        rec = _run_one_case(
            case=case,
            lane=lane,
            model=model,
            tokenizer=tokenizer,
            adapter=adapter,
            dense_ids=dense_ids,
            decode_steps=decode_steps,
        )
        records.append(rec)
    return records


def main() -> None:
    args = parse_args()
    if not transformers_available():
        raise SystemExit("requires optional transformers dependencies")

    active_lanes = (
        [l for l in _LANES if l["name"] in args.lanes]
        if args.lanes
        else list(_LANES)
    )
    if not active_lanes:
        raise SystemExit(f"no valid lanes; choices: {[l['name'] for l in _LANES]}")

    lengths = sorted(set(args.lengths))
    cases = _build_cases(lengths)
    if not cases:
        raise SystemExit("no cases built — check that primary/control files exist")

    print(f"Context-length scaling benchmark")
    print(f"  Lengths : {lengths}")
    print(f"  Cases   : {len(cases)} total ({[c['case_tag'] for c in cases]})")
    print(f"  Lanes   : {[l['name'] for l in active_lanes]}")
    print(f"  Decode  : {args.decode_steps} steps/case")
    print()

    max_dense_length = int(args.max_dense_length)
    print(f"Loading model {args.model_id} ...")
    dense_model, dense_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id, device=args.device, torch_dtype=args.torch_dtype,
    )
    persistent_model, persistent_tokenizer = load_qwen35_text_only_from_pretrained(
        args.model_id, device=args.device, torch_dtype=args.torch_dtype,
    )
    dotcache_config = real_mixed_probe_dotcache_config()
    dense_adapter = Qwen35TextModelAdapter(model=dense_model)
    persistent_adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=persistent_model,
        dotcache_config=dotcache_config,
        persistent_serving_config=_build_serving_config(active_lanes[0]),
        backend=str(args.backend),
    )

    # Dense reference pass — skipped for cases above max_dense_length.
    # Full-attention dense prefill at 16K+ tokens OOMs on MPS (O(n²) attention matrix
    # exhausts unified memory when two models are loaded simultaneously).
    # The persistent path is fine: it uses block-selective attention and never
    # materialises the full attention matrix.
    dense_cases = [c for c in cases if int(c["target_length"]) <= max_dense_length]
    skipped_dense = [c["case_tag"] for c in cases if int(c["target_length"]) > max_dense_length]
    if skipped_dense:
        print(
            f"NOTE: skipping dense reference for {len(skipped_dense)} case(s) "
            f"with target_length > {max_dense_length}: {skipped_dense}"
        )

    print("--- Dense reference pass ---")
    dense_ids_by_case: dict[str, list[int]] = {}
    for case in dense_cases:
        if case["case_tag"] in dense_ids_by_case:
            continue
        device = next(dense_model.parameters()).device
        input_ids, attention_mask = _build_prompt_text_inputs(
            dense_tokenizer,
            device=device,
            prompt_text=Path(case["prompt_file"]).read_text(encoding="utf-8"),
            prompt_length=int(case["target_length"]),
        )
        dense_result = run_qwen35_text_generation_harness(
            dense_model,
            dense_adapter,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(args.decode_steps) + 1,
            tokenizer=dense_tokenizer,
        )
        all_ids = [int(t) for t in dense_result.get("dense_generated_ids", [])]
        dense_ids_by_case[case["case_tag"]] = all_ids[:int(args.decode_steps)]
        print(f"  [dense] {case['case_tag']} ({case['target_length']} tok): {len(all_ids)} tokens generated")

    # Free dense model memory before persistent runs at long contexts.
    del dense_model, dense_adapter
    if args.device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # Per-lane/case serving runs
    warmup_steps = int(args.warmup_steps)
    n_workers = max(1, int(args.workers))
    all_records: list[dict[str, Any]] = []

    if n_workers > 1:
        # Multi-process path: dispatch (case, lane) pairs across workers.
        # Each worker loads the model once, then processes its task list.
        # Tasks are ordered lane-inner so results come back case-ordered per lane.
        worker_tasks = [
            (case, dense_ids_by_case.get(case["case_tag"], []), int(args.decode_steps), active_lanes)
            for case in cases
        ]
        print(
            f"\n--- Parallel serving runs ({len(worker_tasks)} case×all-lanes tasks, "
            f"{n_workers} workers) ---"
        )
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_cs_worker_init,
            initargs=(
                str(args.model_id),
                str(args.device),
                str(args.backend),
                str(args.torch_dtype),
                active_lanes[0],
            ),
        ) as pool:
            for case_lane_recs in pool.map(_cs_worker_run_case, worker_tasks):
                all_records.extend(case_lane_recs)
    else:
        # Single-process path (default).
        for lane in active_lanes:
            print(f"\n--- Lane: {lane['label']} ---")
            # Warmup: throwaway run before timing to prime Triton JIT / CUDA kernel caches.
            if warmup_steps > 0 and cases:
                first_case = cases[0]
                _wu_device = next(persistent_model.parameters()).device
                _wu_ids, _wu_mask = _build_prompt_text_inputs(
                    persistent_tokenizer,
                    device=_wu_device,
                    prompt_text=Path(first_case["prompt_file"]).read_text(encoding="utf-8"),
                    prompt_length=int(first_case["target_length"]),
                )
                run_qwen35_attention_subset_persistent_serving_harness(
                    persistent_model,
                    persistent_adapter,
                    input_ids=_wu_ids,
                    attention_mask=_wu_mask,
                    decode_steps=warmup_steps,
                    persistent_serving_config=_build_serving_config(lane),
                )
                del _wu_ids, _wu_mask
            for case in cases:
                rec = _run_one_case(
                    case=case,
                    lane=lane,
                    model=persistent_model,
                    tokenizer=persistent_tokenizer,
                    adapter=persistent_adapter,
                    dense_ids=dense_ids_by_case.get(case["case_tag"], []),
                    decode_steps=int(args.decode_steps),
                )
                all_records.append(rec)

    # Summary tables
    print("\n=== Interval win rate and speedup by context length ===")
    sph_by_len = _summarise_by_length(all_records, "spherical_only")
    int_by_len = _summarise_by_length(all_records, "interval")
    for length in lengths:
        sph = sph_by_len.get(length, {})
        intv = int_by_len.get(length, {})
        if not sph or not intv:
            continue
        speedup = (sph["avg_ms_per_step"] - intv["avg_ms_per_step"]) / sph["avg_ms_per_step"] * 100.0
        print(
            f"  {length:>6} tok: "
            f"sph={sph['avg_ms_per_step']:.1f}ms  "
            f"int={intv['avg_ms_per_step']:.1f}ms  "
            f"speedup={speedup:+.1f}%  "
            f"int_win={intv['bound_interval_frac']:.1%}  "
            f"cert_stop_rate={intv['avg_cert_stop_rate']:.1%}"
        )

    payload: dict[str, Any] = {
        "records": all_records,
        "active_lanes": [l["name"] for l in active_lanes],
        "lengths": lengths,
        "decode_steps": int(args.decode_steps),
        "warmup_steps": warmup_steps,
        "model_id": str(args.model_id),
        "device": str(args.device),
        "backend": str(args.backend),
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"\nJSON -> {out}")

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            _render_markdown(payload=payload, active_lanes=active_lanes, lengths=lengths),
            encoding="utf-8",
        )
        print(f"MD   -> {out}")


if __name__ == "__main__":
    main()
