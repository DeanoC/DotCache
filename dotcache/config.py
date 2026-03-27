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
        if self.default_mode_k not in ("M0", "M1", "M3"):
            raise ValueError("default_mode_k must be M0, M1, or M3")
        if self.default_mode_v not in ("M0", "M1", "M3"):
            raise ValueError("default_mode_v must be M0, M1, or M3")
        if self.quant_scheme_k not in ("affine", "symmetric", "lut"):
            raise ValueError("quant_scheme_k must be affine, symmetric, or lut")
        if self.quant_scheme_v not in ("affine", "symmetric", "lut"):
            raise ValueError("quant_scheme_v must be affine, symmetric, or lut")

    @property
    def num_groups(self) -> int:
        return ceil(self.head_dim / self.group_size)

    @property
    def padded_head_dim(self) -> int:
        return self.num_groups * self.group_size
