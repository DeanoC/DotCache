from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True, slots=True)
class DotCacheConfig:
    head_dim: int
    group_size: int = 32
    bits_k: int = 4
    bits_v: int = 4
    tokens_per_page: int = 64
    recent_window: int = 128
    sink_window: int = 0
    store_scales_dtype: str = "float16"
    store_bias_dtype: str = "float16"
    payload_layout_k: str = "group_major"
    payload_layout_v: str = "group_major"
    default_mode_k: str = "M0"
    default_mode_v: str = "M0"
    quant_scheme_k: str = "affine"
    quant_scheme_v: str = "affine"
    escape_dtype: str = "float16"
    m2_sketch_dim_k: int = 8
    m2_center_k: bool = False
    m2_segment_count_k: int = 1
    m2_adaptive_segments_k: bool = False
    m2_adaptive_min_improvement_k: float = 0.1
    m2_prefilter_top_k: int = 0
    m2_prefilter_min_pages: int = 8
    lut_refine_steps: int = 6
    preconditioner: str = "none"
    precondition_strength: float = 2.0
    m1_segment_count_k: int = 1
    m1_segment_count_v: int = 1
    m1_fallback_to_m0: bool = True
    m1_error_threshold: float = 0.35
    m1_token_p95_error_threshold: float = 1000000.0

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.bits_k not in (2, 4):
            raise ValueError("bits_k must be 2 or 4 for the MVP")
        if self.bits_v not in (2, 4):
            raise ValueError("bits_v must be 2 or 4 for the MVP")
        if self.tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if self.payload_layout_k not in ("group_major", "token_major"):
            raise ValueError("payload_layout_k must be group_major or token_major")
        if self.payload_layout_v not in ("group_major", "token_major"):
            raise ValueError("payload_layout_v must be group_major or token_major")
        if self.default_mode_k not in ("M0", "M1", "M2", "M3", "T3"):
            raise ValueError("default_mode_k must be M0, M1, M2, M3, or T3")
        if self.default_mode_v not in ("M0", "M1", "M3", "T3"):
            raise ValueError("default_mode_v must be M0, M1, M3, or T3")
        if self.quant_scheme_k not in ("affine", "symmetric", "lut", "sketch", "turbo3"):
            raise ValueError("quant_scheme_k must be affine, symmetric, lut, sketch, or turbo3")
        if self.quant_scheme_v not in ("affine", "symmetric", "lut", "turbo3"):
            raise ValueError("quant_scheme_v must be affine, symmetric, lut, or turbo3")
        if self.m2_sketch_dim_k <= 0:
            raise ValueError("m2_sketch_dim_k must be positive")
        if not isinstance(self.m2_center_k, bool):
            raise ValueError("m2_center_k must be a bool")
        if self.m2_segment_count_k <= 0:
            raise ValueError("m2_segment_count_k must be positive")
        if not isinstance(self.m2_adaptive_segments_k, bool):
            raise ValueError("m2_adaptive_segments_k must be a bool")
        if self.m2_adaptive_min_improvement_k < 0:
            raise ValueError("m2_adaptive_min_improvement_k must be non-negative")
        if self.m2_prefilter_top_k < 0:
            raise ValueError("m2_prefilter_top_k must be non-negative")
        if self.m2_prefilter_min_pages < 0:
            raise ValueError("m2_prefilter_min_pages must be non-negative")
        if self.lut_refine_steps < 0:
            raise ValueError("lut_refine_steps must be non-negative")
        if self.preconditioner not in ("none", "tanh"):
            raise ValueError("preconditioner must be none or tanh")
        if self.precondition_strength <= 0:
            raise ValueError("precondition_strength must be positive")
        if self.m1_segment_count_k <= 0:
            raise ValueError("m1_segment_count_k must be positive")
        if self.m1_segment_count_v <= 0:
            raise ValueError("m1_segment_count_v must be positive")
        if self.m1_error_threshold <= 0:
            raise ValueError("m1_error_threshold must be positive")
        if self.m1_token_p95_error_threshold <= 0:
            raise ValueError("m1_token_p95_error_threshold must be positive")

    @property
    def num_groups(self) -> int:
        return ceil(self.head_dim / self.group_size)

    @property
    def padded_head_dim(self) -> int:
        return self.num_groups * self.group_size
