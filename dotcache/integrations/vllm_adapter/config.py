from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ...config import DotCacheConfig

VllmAdapterMode = Literal["dense", "dotcache_shadow", "dotcache_active"]
VllmModelFamily = Literal["llama"]


@dataclass(frozen=True, slots=True)
class VllmAdapterConfig:
    dotcache_config: DotCacheConfig
    block_size: int
    mode: VllmAdapterMode = "dense"
    model_family: VllmModelFamily = "llama"
    supported_vllm_minor: str = "0.18"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.dotcache_config.tokens_per_page != self.block_size:
            raise ValueError("DotCache tokens_per_page must equal the vLLM block_size for this phase")
        if self.model_family != "llama":
            raise ValueError("Phase 6 supports only the Llama-family vLLM path")
