from .attention_reference import (
    explicit_dequantized_attention,
    mix_page_ref,
    run_attention_reference,
    score_page_ref,
)
from .attention_runtime import decode_step, mix_page, prepare_page, prepare_pages, score_page
from .config import DotCacheConfig
from .page_cache import PreparedPageCache
from .encode import encode_page
from .planner import choose_mode
from .session_runtime import PagedDecodeSession
from .tracing import ExecutionTrace
from .types import EncodedPage, PageHeader
from .model_kv_cache import ModelPagedKVCache
from .model_registry import ModelSpec, get_model_spec, list_model_specs

__all__ = [
    "DotCacheConfig",
    "EncodedPage",
    "ExecutionTrace",
    "ModelSpec",
    "ModelPagedKVCache",
    "PageHeader",
    "PagedDecodeSession",
    "PreparedPageCache",
    "choose_mode",
    "decode_step",
    "encode_page",
    "explicit_dequantized_attention",
    "get_model_spec",
    "list_model_specs",
    "mix_page",
    "mix_page_ref",
    "prepare_page",
    "prepare_pages",
    "run_attention_reference",
    "score_page",
    "score_page_ref",
]

try:  # pragma: no cover - optional HF path
    from .integrations import (
        LlamaDotCacheHarness,
        LlamaDotCacheModelAdapter,
        LlamaReplayRecord,
        Qwen2DotCacheHarness,
        Qwen2DotCacheModelAdapter,
        VllmAdapterConfig,
        VllmDotCacheModelAdapter,
        VllmPagedKVCache,
        VLLM_V1_MULTIPROCESSING_ENV,
        configure_vllm_inprocess_runtime,
        get_vllm_version,
        install_dotcache_on_vllm_model,
        install_dotcache_on_vllm_runtime,
        require_supported_vllm_version,
        run_llama_generation_harness,
        run_llama_replay_harness,
        run_qwen2_generation_harness,
        run_qwen2_loss_harness,
        run_qwen2_replay_harness,
        transformers_available,
        vllm_available,
    )
except ImportError:  # pragma: no cover - exercised when optional deps are absent
    pass
else:
    __all__.extend(
        [
            "LlamaDotCacheHarness",
            "LlamaDotCacheModelAdapter",
            "LlamaReplayRecord",
            "Qwen2DotCacheHarness",
            "Qwen2DotCacheModelAdapter",
            "VllmAdapterConfig",
            "VllmDotCacheModelAdapter",
            "VllmPagedKVCache",
            "VLLM_V1_MULTIPROCESSING_ENV",
            "configure_vllm_inprocess_runtime",
            "get_vllm_version",
            "install_dotcache_on_vllm_model",
            "install_dotcache_on_vllm_runtime",
            "require_supported_vllm_version",
            "run_llama_generation_harness",
            "run_llama_replay_harness",
            "run_qwen2_generation_harness",
            "run_qwen2_loss_harness",
            "run_qwen2_replay_harness",
            "transformers_available",
            "vllm_available",
        ]
    )
