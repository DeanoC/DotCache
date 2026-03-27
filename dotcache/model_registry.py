from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

ModelFamily = Literal["llama", "qwen2", "qwen3_5_hybrid"]
SourceFormat = Literal["hf", "gguf"]
RuntimeName = Literal["transformers", "dotcache_hf", "vllm", "llama_cpp"]
LocalTier = Literal["works_here", "stretch_here", "reference_only"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    display_name: str
    model_id: str
    tokenizer_model_id: str | None
    family: ModelFamily
    source_format: SourceFormat
    runtime: RuntimeName
    context_window: int
    local_tier: LocalTier
    dotcache_ready: bool
    benchmark_harness: str | None
    prompt_lengths: tuple[int, ...]
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_MODEL_REGISTRY: dict[str, ModelSpec] = {
    "tinyllama_hf": ModelSpec(
        key="tinyllama_hf",
        display_name="TinyLlama 1.1B Chat",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        family="llama",
        source_format="hf",
        runtime="dotcache_hf",
        context_window=2048,
        local_tier="works_here",
        dotcache_ready=True,
        benchmark_harness="llama_compare",
        prompt_lengths=(289, 577, 1536),
        notes="Current smallest real-model regression lane for exact HF DotCache on this Mac.",
    ),
    "smollm2_360m_hf": ModelSpec(
        key="smollm2_360m_hf",
        display_name="SmolLM2 360M Instruct",
        model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        tokenizer_model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
        family="llama",
        source_format="hf",
        runtime="dotcache_hf",
        context_window=8192,
        local_tier="works_here",
        dotcache_ready=True,
        benchmark_harness="llama_compare",
        prompt_lengths=(1024, 2048),
        notes="Best higher-context real-model lane currently working on this Mac.",
    ),
    "llama32_3b_hf": ModelSpec(
        key="llama32_3b_hf",
        display_name="Llama 3.2 3B Instruct",
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        tokenizer_model_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        source_format="hf",
        runtime="dotcache_hf",
        context_window=131072,
        local_tier="stretch_here",
        dotcache_ready=True,
        benchmark_harness="llama_compare",
        prompt_lengths=(1024, 2048, 4096),
        notes="Best next proper-model HF target because it matches the current Llama-family integration path.",
    ),
    "qwen25_3b_hf": ModelSpec(
        key="qwen25_3b_hf",
        display_name="Qwen2.5 3B Instruct",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        tokenizer_model_id="Qwen/Qwen2.5-3B-Instruct",
        family="qwen2",
        source_format="hf",
        runtime="dotcache_hf",
        context_window=32768,
        local_tier="stretch_here",
        dotcache_ready=True,
        benchmark_harness="qwen2_compare",
        prompt_lengths=(1024, 2048, 4096),
        notes="First non-Llama native-weight DotCache target on the HF path, using a Qwen2-specific attention adapter.",
    ),
    "qwen35_4b_hf": ModelSpec(
        key="qwen35_4b_hf",
        display_name="Qwen3.5 4B",
        model_id="Qwen/Qwen3.5-4B",
        tokenizer_model_id="Qwen/Qwen3.5-4B",
        family="qwen3_5_hybrid",
        source_format="hf",
        runtime="transformers",
        context_window=262144,
        local_tier="reference_only",
        dotcache_ready=False,
        benchmark_harness=None,
        prompt_lengths=(2048, 4096),
        notes="Reference-only for now; hybrid attention/delta architecture is not a next-step DotCache target.",
    ),
    "llama32_3b_gguf": ModelSpec(
        key="llama32_3b_gguf",
        display_name="Llama 3.2 3B Instruct GGUF",
        model_id="ggml-org/Llama-3.2-3B-Instruct-GGUF",
        tokenizer_model_id="meta-llama/Llama-3.2-3B-Instruct",
        family="llama",
        source_format="gguf",
        runtime="llama_cpp",
        context_window=131072,
        local_tier="reference_only",
        dotcache_ready=False,
        benchmark_harness="gguf_external",
        prompt_lengths=(1024, 2048, 4096),
        notes="External reference baseline for llama.cpp / GGUF comparisons.",
    ),
    "qwen25_3b_gguf": ModelSpec(
        key="qwen25_3b_gguf",
        display_name="Qwen2.5 3B Instruct GGUF",
        model_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
        tokenizer_model_id="Qwen/Qwen2.5-3B-Instruct",
        family="qwen2",
        source_format="gguf",
        runtime="llama_cpp",
        context_window=32768,
        local_tier="reference_only",
        dotcache_ready=False,
        benchmark_harness="gguf_external",
        prompt_lengths=(1024, 2048, 4096),
        notes="External GGUF reference lane for future TurboQuant / llama.cpp comparisons.",
    ),
}


def list_model_specs() -> tuple[ModelSpec, ...]:
    return tuple(_MODEL_REGISTRY.values())


def get_model_spec(key: str) -> ModelSpec:
    try:
        return _MODEL_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"unknown model registry key: {key}") from exc
