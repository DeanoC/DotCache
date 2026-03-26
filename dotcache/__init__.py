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

__all__ = [
    "DotCacheConfig",
    "EncodedPage",
    "ExecutionTrace",
    "ModelPagedKVCache",
    "PageHeader",
    "PagedDecodeSession",
    "PreparedPageCache",
    "choose_mode",
    "decode_step",
    "encode_page",
    "explicit_dequantized_attention",
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
        run_llama_generation_harness,
        run_llama_replay_harness,
        transformers_available,
    )
except ImportError:  # pragma: no cover - exercised when optional deps are absent
    pass
else:
    __all__.extend(
        [
            "LlamaDotCacheHarness",
            "LlamaDotCacheModelAdapter",
            "LlamaReplayRecord",
            "run_llama_generation_harness",
            "run_llama_replay_harness",
            "transformers_available",
        ]
    )
