"""Calibration and validation prompt builders.

Must cover ALL attention patterns the model will encounter in evaluation:
  1. Dense factual (PG-19 style — uniform attention, language modelling)
  2. Code (LongBench code tasks — structured, function-level attention)
  3. Needle retrieval (NIAH — extreme sparsity, single fact in haystack)
  4. Multi-fact retrieval (RULER multi-key — multiple needles)
  5. QA over document (LongBench QA — focused attention on evidence spans)
  6. Summarisation (LongBench — broad attention with recency bias)
  7. Conversational (multi-turn — recent-token bias + periodic lookback)
  8. Adversarial quantisation (high-entropy, outlier distributions)

The calibration set must be HARDER than the evaluation set. If calibration
is too easy, the epsilons will be too loose for harder eval prompts.
"""
from __future__ import annotations

import random
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# Filler text for NIAH-style haystack (innocuous, no overlap with needles)
_FILLER = (
    "The history of mathematics spans thousands of years and encompasses many "
    "different cultures and civilizations. From the earliest counting systems "
    "developed by ancient peoples to the sophisticated abstract algebras of the "
    "modern era, mathematical knowledge has grown through a process of discovery, "
    "invention, and refinement. The Babylonians developed a base-60 number system "
    "that still influences how we measure time and angles today. The ancient Greeks "
    "made foundational contributions to geometry, number theory, and logic. During "
    "the Islamic Golden Age, scholars preserved and extended Greek mathematics while "
    "making original advances in algebra and trigonometry. The Renaissance saw a "
    "flowering of mathematical activity in Europe, leading eventually to the "
    "development of calculus by Newton and Leibniz in the seventeenth century. "
    "In the nineteenth and twentieth centuries, mathematics became increasingly "
    "abstract and specialized, with new fields emerging at a rapid pace. Today, "
    "mathematics is essential to science, engineering, economics, and many other "
    "domains of human activity.\n\n"
)

# Different needles for calibration (NOT the same as NIAH benchmark needles)
_CAL_NEEDLES = [
    ("The classified satellite frequency is 47.3 GHz.", "47.3 GHz"),
    ("The backup database password is xK9#mP2v.", "xK9#mP2v"),
    ("The emergency meeting point is the north parking garage, level B2.", "north parking garage"),
    ("The quarterly revenue target was revised to $12.8 million.", "12.8 million"),
    ("The prototype's maximum operating temperature is 847 degrees Celsius.", "847 degrees"),
    ("The research grant number is NSF-2024-07831.", "NSF-2024-07831"),
]


def _read_chunk(path: Path, max_chars: int = 8000, offset: int = 0) -> str:
    if not path.exists():
        return f"[File not found: {path}]\n" * (max_chars // 30)
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[offset : offset + max_chars]


def _pad_source(src: str, target_chars: int) -> str:
    """Pad source to at least target_chars by repeating."""
    while len(src) < target_chars:
        src = src + "\n" + src
    return src


def _build_niah_prompt(needle_text: str, filler: str, ctx_chars: int, depth: float) -> str:
    """Build NIAH-style prompt with needle at specified depth."""
    filler_padded = _pad_source(filler, ctx_chars)
    insert_pos = int(len(filler_padded[:ctx_chars]) * depth)
    before = filler_padded[:insert_pos]
    after = filler_padded[insert_pos : ctx_chars]
    question = (
        "\n\nBased on the above text, what is the specific value, code, location, "
        "temperature, number, or password mentioned in the special statement?\nAnswer:"
    )
    return before + f"\n[IMPORTANT] {needle_text}\n" + after + question


def _build_multi_needle_prompt(needles: list[tuple[str, str]], filler: str, ctx_chars: int) -> str:
    """Build prompt with multiple needles at different depths."""
    filler_padded = _pad_source(filler, ctx_chars)
    parts = []
    chunk_size = ctx_chars // (len(needles) + 1)
    for i, (needle_text, _) in enumerate(needles):
        start = i * chunk_size
        end = start + chunk_size
        parts.append(filler_padded[start:end])
        parts.append(f"\n[FACT {i+1}] {needle_text}\n")
    parts.append(filler_padded[len(needles) * chunk_size : ctx_chars])
    question = "\n\nList all the specific facts, values, and numbers mentioned in the [FACT] statements above.\nAnswer:"
    return "".join(parts) + question


def _build_qa_prompt(document: str, ctx_chars: int) -> str:
    """Build QA-style prompt: long document + question."""
    doc = _pad_source(document, ctx_chars - 200)[:ctx_chars - 200]
    return (
        "Read the following document carefully:\n\n"
        + doc
        + "\n\nBased on the document above, what are the main technical concepts discussed? "
        "Provide a detailed summary.\nAnswer:"
    )


def _build_summarisation_prompt(document: str, ctx_chars: int) -> str:
    """Build summarisation prompt: long document + summary request."""
    doc = _pad_source(document, ctx_chars - 100)[:ctx_chars - 100]
    return doc + "\n\nTL;DR:"


def build_calibration_prompts(
    ctx_buckets: list[int],
    prompts_per_bucket: int = 8,
) -> list[tuple[str, int]]:
    """Build calibration prompt set covering all attention patterns.

    Returns list of (prompt_text, target_ctx_len) pairs.
    8 prompts per bucket by default, covering 8 pattern categories.
    """
    prompts = []

    # Source material (diverse)
    sources = {
        "code_llama": _read_chunk(_REPO / "dotcache" / "integrations" / "llama.py", 80000),
        "code_kernels": _read_chunk(_REPO / "dotcache" / "kernels" / "selective_attend_triton.py", 40000),
        "docs_perf": _read_chunk(_REPO / "docs" / "performance_journal.md", 80000),
        "docs_readme": _read_chunk(_REPO / "README.md", 80000),
        "paper": _read_chunk(_REPO / "DotCacheArXiv.tex", 80000),
        "config": _read_chunk(_REPO / "dotcache" / "config.py", 30000),
        "backend_mps": _read_chunk(_REPO / "dotcache" / "backends" / "torch_mps.py", 30000),
        "model_kv": _read_chunk(_REPO / "dotcache" / "model_kv_cache.py", 80000),
    }
    source_list = list(sources.values())

    for bucket_idx, ctx_len in enumerate(ctx_buckets):
        ctx_chars = ctx_len * 4  # rough chars-per-token estimate
        bucket_prompts = []

        # 1. Dense factual (PG-19 style) — just text, no question
        src = _pad_source(source_list[bucket_idx % len(source_list)], ctx_chars)
        bucket_prompts.append(src[:ctx_chars])

        # 2. Code — pure code, structured attention
        code_src = _pad_source(sources["code_llama"] + sources["code_kernels"], ctx_chars)
        bucket_prompts.append(code_src[:ctx_chars])

        # 3. NIAH — single needle at depth 50%
        needle = _CAL_NEEDLES[bucket_idx % len(_CAL_NEEDLES)]
        bucket_prompts.append(_build_niah_prompt(needle[0], _FILLER, ctx_chars, 0.5))

        # 4. NIAH — single needle at depth 10% (near start)
        needle2 = _CAL_NEEDLES[(bucket_idx + 1) % len(_CAL_NEEDLES)]
        bucket_prompts.append(_build_niah_prompt(needle2[0], _FILLER, ctx_chars, 0.1))

        # 5. NIAH — single needle at depth 90% (near end)
        needle3 = _CAL_NEEDLES[(bucket_idx + 2) % len(_CAL_NEEDLES)]
        bucket_prompts.append(_build_niah_prompt(needle3[0], _FILLER, ctx_chars, 0.9))

        # 6. Multi-needle retrieval (RULER-style)
        multi_needles = _CAL_NEEDLES[:3]
        bucket_prompts.append(_build_multi_needle_prompt(multi_needles, _FILLER, ctx_chars))

        # 7. QA over document (LongBench-style)
        qa_src = _pad_source(sources["paper"] + sources["docs_readme"], ctx_chars)
        bucket_prompts.append(_build_qa_prompt(qa_src, ctx_chars))

        # 8. Summarisation (LongBench-style)
        sum_src = _pad_source(sources["docs_perf"] + sources["model_kv"], ctx_chars)
        bucket_prompts.append(_build_summarisation_prompt(sum_src, ctx_chars))

        # Take up to prompts_per_bucket
        for pi in range(min(prompts_per_bucket, len(bucket_prompts))):
            prompts.append((bucket_prompts[pi], ctx_len))

    return prompts


def build_validation_prompts(
    ctx_buckets: list[int],
    prompts_per_bucket: int = 4,
) -> list[tuple[str, int]]:
    """Build held-out validation prompt set (different source material + needles).

    Covers the same pattern categories as calibration but with different content.
    """
    prompts = []

    sources = {
        "selector": _read_chunk(_REPO / "dotcache" / "selector_baselines.py", 40000),
        "tex_full": _read_chunk(_REPO / "dotcache_full.tex", 80000),
        "bench_32k": _read_chunk(_REPO / "benchmarks" / "bench_certified_32k.py", 30000),
        "calibrate": _read_chunk(_REPO / "dotcache" / "calibration" / "calibrate.py", 30000),
    }
    source_list = list(sources.values())

    # Validation needles (different from calibration needles)
    val_needles = [
        ("The decryption key for the archive is ZETA-4491-OMEGA.", "ZETA-4491-OMEGA"),
        ("The server migration is scheduled for March 15, 2027.", "March 15, 2027"),
        ("The maximum batch size for training is 2048 samples.", "2048 samples"),
    ]

    for bucket_idx, ctx_len in enumerate(ctx_buckets):
        ctx_chars = ctx_len * 4
        bucket_prompts = []

        # Dense text
        src = _pad_source(source_list[bucket_idx % len(source_list)], ctx_chars)
        bucket_prompts.append(src[:ctx_chars] + "\n\nSummary:\n")

        # NIAH retrieval
        needle = val_needles[bucket_idx % len(val_needles)]
        bucket_prompts.append(_build_niah_prompt(needle[0], _FILLER, ctx_chars, 0.5))

        # NIAH retrieval (different depth)
        needle2 = val_needles[(bucket_idx + 1) % len(val_needles)]
        bucket_prompts.append(_build_niah_prompt(needle2[0], _FILLER, ctx_chars, 0.2))

        # QA
        qa_src = _pad_source(source_list[(bucket_idx + 1) % len(source_list)], ctx_chars)
        bucket_prompts.append(_build_qa_prompt(qa_src, ctx_chars))

        for pi in range(min(prompts_per_bucket, len(bucket_prompts))):
            prompts.append((bucket_prompts[pi], ctx_len))

    return prompts
