"""Build heterogeneous long contexts from repo content for sparsity testing.

Creates prompts by stitching together semantically distinct segments:
  - Python source code (model code, backend kernels, benchmark scripts)
  - Technical documentation (architecture docs, benchmark reports)
  - LaTeX paper content
  - Configuration and registry data
  - Performance analysis and results

Each segment is 1-4K tokens of distinct content. The final segment is
a question about one specific segment, forcing the model to attend
selectively to the relevant portion.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _read_chunk(path: Path, max_chars: int = 8000, offset: int = 0) -> str:
    """Read a chunk of a file, starting at offset."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[offset:offset + max_chars]


# Segments: (label, file_path, offset, max_chars)
_SEGMENTS = [
    ("CUDA_KERNEL_CODE", "dotcache/backends/torch_mps.py", 0, 6000),
    ("MODEL_REGISTRY", "dotcache/model_kv_cache.py", 5000, 6000),
    ("BENCHMARK_REPORT", "docs/benchmark_report.md", 0, 6000),
    ("PAPER_SECTION", "DotCacheArXiv.tex", 3000, 6000),
    ("PERFORMANCE_JOURNAL_A", "docs/performance_journal.md", 0, 6000),
    ("BACKEND_RUNTIME", "dotcache/backends/metal/persistent_runtime.py", 10000, 6000),
    ("SELECTOR_CODE", "dotcache/selector_baselines.py", 0, 6000),
    ("ARCHITECTURE_DOC", "docs/model_roadmap.md", 0, 6000),
    ("STAGE9_THESIS", "docs/qwen35_stage9_thesis_status_20260412.md", 0, 6000),
    ("LLAMA_INTEGRATION", "dotcache/integrations/llama.py", 5000, 6000),
    ("PERFORMANCE_JOURNAL_B", "docs/performance_journal.md", 50000, 6000),
    ("PAGE_SELECTION_DOC", "docs/dotcache_page_selection_standardized_evaluation.md", 0, 6000),
    ("GEMMA_INTEGRATION", "dotcache/integrations/gemma4.py", 0, 6000),
    ("LAYER_PROFILES", "docs/local_layer_profiles.md", 0, 6000),
    ("BENCHMARK_SCRIPT", "benchmarks/bench_qwen35_persistent_real_mixed_probe.py", 0, 6000),
    ("CUDA_RESULTS", "docs/qwen35_bound_cuda_results_20260414.md", 0, 6000),
    ("PERFORMANCE_JOURNAL_C", "docs/performance_journal.md", 100000, 6000),
    ("TURBOQUANT_SWEEP", "docs/qwen35_turboquant_full_sweep_20260329.md", 0, 6000),
    ("QWEN_INTEGRATION", "dotcache/integrations/qwen35.py", 20000, 6000),
    ("FULL_PAPER", "dotcache_full.tex", 5000, 6000),
    ("PERFORMANCE_JOURNAL_D", "docs/performance_journal.md", 200000, 6000),
    ("COMPRESSED_RFC", "docs/dotcache_compressed_page_test_readiness_rfc.md", 0, 6000),
]

# Questions targeting specific segments
_QUESTIONS = [
    ("CUDA_RESULTS", "Based on the CUDA benchmark results above, what was the speedup of the interval bound compared to the spherical baseline?"),
    ("PAPER_SECTION", "According to the paper section above, what is the main contribution of the DotCache system?"),
    ("BENCHMARK_REPORT", "What were the key findings in the benchmark report section?"),
    ("ARCHITECTURE_DOC", "Based on the model roadmap, what models are planned for future support?"),
    ("LLAMA_INTEGRATION", "Looking at the LLaMA integration code, how does the adapter handle KV cache compression?"),
]


def build_context(target_tokens: int, question_idx: int = 0) -> str:
    """Build a heterogeneous context of approximately target_tokens tokens.

    Rough estimate: 1 token ≈ 4 characters.
    """
    target_chars = target_tokens * 4
    segments_needed = max(1, target_chars // 6000)

    parts = []
    chars_used = 0

    for i, (label, rel_path, offset, max_chars) in enumerate(_SEGMENTS):
        if chars_used >= target_chars:
            break
        path = _REPO / rel_path
        if not path.exists():
            continue
        chunk = _read_chunk(path, max_chars=max_chars, offset=offset)
        if not chunk.strip():
            continue

        parts.append(f"\n\n--- BEGIN SEGMENT: {label} ---\n\n{chunk}\n\n--- END SEGMENT: {label} ---\n")
        chars_used += len(parts[-1])

    # Add question at the end
    q_label, q_text = _QUESTIONS[question_idx % len(_QUESTIONS)]
    parts.append(f"\n\nQUESTION: {q_text}\n\nANSWER:")

    return "".join(parts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--question", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    text = build_context(args.target_tokens, args.question)
    actual_chars = len(text)
    est_tokens = actual_chars // 4

    if args.output:
        Path(args.output).write_text(text)
        print(f"Written {actual_chars} chars (~{est_tokens} tokens) to {args.output}")
    else:
        print(f"Context: {actual_chars} chars (~{est_tokens} tokens)")
        print(f"Segments: {text.count('BEGIN SEGMENT')}")
        print(f"First 200 chars: {text[:200]}...")
        print(f"Last 200 chars: ...{text[-200:]}")
