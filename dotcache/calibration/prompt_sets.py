"""Calibration and validation prompt builders.

Builds diverse prompt sets covering different attention patterns:
  - Dense factual (Wikipedia-style, uniform attention)
  - Code (structured, block-level attention)
  - Retrieval/QA (needle in haystack, very sparse)
  - Conversational (recent-token bias + lookback)
"""
from __future__ import annotations

from pathlib import Path
import sys

# Use the existing build_context for heterogeneous prompts
_REPO = Path(__file__).resolve().parents[2]


def _read_chunk(path: Path, max_chars: int = 8000, offset: int = 0) -> str:
    if not path.exists():
        return f"[File not found: {path}]\n" * (max_chars // 30)
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[offset : offset + max_chars]


def build_calibration_prompts(
    ctx_buckets: list[int],
    prompts_per_bucket: int = 4,
) -> list[tuple[str, int]]:
    """Build calibration prompt set.

    Returns list of (prompt_text, target_ctx_len) pairs.
    At least `prompts_per_bucket` prompts per context bucket.
    """
    prompts = []

    # Source material
    sources = {
        "code_llama": _read_chunk(_REPO / "dotcache" / "integrations" / "llama.py", 60000),
        "code_kernels": _read_chunk(_REPO / "dotcache" / "kernels" / "selective_attend_triton.py", 20000),
        "code_cache": _read_chunk(_REPO / "dotcache" / "kernels" / "tiered_kv_cache.py", 20000),
        "docs_perf": _read_chunk(_REPO / "docs" / "performance_journal.md", 60000),
        "docs_readme": _read_chunk(_REPO / "README.md", 60000),
        "paper": _read_chunk(_REPO / "DotCacheArXiv.tex", 60000),
        "config": _read_chunk(_REPO / "dotcache" / "config.py", 20000),
        "backend": _read_chunk(_REPO / "dotcache" / "backends" / "torch_mps.py", 20000),
    }

    # Prompt templates for different attention patterns
    templates = [
        # Dense: concatenated source, question at end
        lambda ctx, src: src[:ctx * 3] + "\n\nQuestion: What is the main function in this code?\nAnswer:",
        # Code-heavy: pure code
        lambda ctx, src: src[:ctx * 4],
        # Retrieval: needle buried in padding
        lambda ctx, src: (
            "CONTEXT START\n" + src[:ctx * 2]
            + "\nIMPORTANT FACT: The secret key is 42.\n"
            + src[ctx * 2 : ctx * 3]
            + "\nCONTEXT END\nQuestion: What is the secret key?\nAnswer:"
        ),
        # Mixed: alternating segments
        lambda ctx, src: "\n---\n".join(
            [src[i * 2000 : (i + 1) * 2000] for i in range(ctx * 4 // 2000 + 1)]
        ),
    ]

    source_keys = list(sources.keys())

    for bucket_idx, ctx_len in enumerate(ctx_buckets):
        for pi in range(prompts_per_bucket):
            template = templates[pi % len(templates)]
            src_key = source_keys[(bucket_idx * prompts_per_bucket + pi) % len(source_keys)]
            src = sources[src_key]

            # Pad source if too short
            while len(src) < ctx_len * 5:
                src = src + "\n" + src

            prompt = template(ctx_len, src)
            prompts.append((prompt, ctx_len))

    return prompts


def build_validation_prompts(
    ctx_buckets: list[int],
    prompts_per_bucket: int = 2,
) -> list[tuple[str, int]]:
    """Build held-out validation prompt set (different sources)."""
    prompts = []

    # Different source material than calibration
    sources = {
        "model_kv": _read_chunk(_REPO / "dotcache" / "model_kv_cache.py", 60000),
        "benchmark": _read_chunk(_REPO / "benchmarks" / "bench_certified_32k.py", 20000),
        "tex_full": _read_chunk(_REPO / "dotcache_full.tex", 60000),
        "selector": _read_chunk(_REPO / "dotcache" / "selector_baselines.py", 30000),
    }

    source_keys = list(sources.keys())

    for bucket_idx, ctx_len in enumerate(ctx_buckets):
        for pi in range(prompts_per_bucket):
            src_key = source_keys[(bucket_idx + pi) % len(source_keys)]
            src = sources[src_key]
            while len(src) < ctx_len * 5:
                src = src + "\n" + src
            prompt = src[: ctx_len * 4] + "\n\nSummary:\n"
            prompts.append((prompt, ctx_len))

    return prompts
