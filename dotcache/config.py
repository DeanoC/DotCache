from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .modes.m4_key_project import valid_m4_basis_families
from .planner import LayerPolicy, PageModeSpec, make_explicit_policy, make_tier_candidates, parse_page_mode_token


_VALID_KEY_MODES = ("M0", "M1", "M2", "M3", "M4", "T3")
_VALID_VALUE_MODES = ("M0", "M1", "M3", "T3")
_VALID_M4_BASIS_FAMILIES = valid_m4_basis_families()


def _parse_mode_override_spec(spec: str, *, allowed_modes: tuple[str, ...], field_name: str) -> tuple[int, int | None, str]:
    if "=" not in spec:
        raise ValueError(f"{field_name} entries must use layer:<id>=<mode> or layer:<id>:kv:<id>=<mode>")
    target, mode = spec.split("=", 1)
    mode = mode.strip()
    if mode not in allowed_modes:
        allowed = ", ".join(allowed_modes)
        raise ValueError(f"{field_name} mode must be one of {allowed}")
    parts = target.strip().split(":")
    if len(parts) == 2 and parts[0] == "layer":
        return int(parts[1]), None, mode
    if len(parts) == 4 and parts[0] == "layer" and parts[2] == "kv":
        return int(parts[1]), int(parts[3]), mode
    raise ValueError(f"{field_name} entries must use layer:<id>=<mode> or layer:<id>:kv:<id>=<mode>")


def _parse_layer_value_spec(
    spec: str,
    *,
    field_name: str,
    allowed_values: tuple[str, ...],
) -> tuple[int, str]:
    if "=" not in spec:
        raise ValueError(f"{field_name} entries must use layer:<id>=<value>")
    target, value = spec.split("=", 1)
    parts = target.strip().split(":")
    if len(parts) != 2 or parts[0] != "layer":
        raise ValueError(f"{field_name} entries must use layer:<id>=<value>")
    value = value.strip()
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise ValueError(f"{field_name} values must be one of {allowed}")
    return int(parts[1]), value


def _parse_layer_candidate_spec(spec: str, *, field_name: str) -> tuple[int, tuple[PageModeSpec, ...]]:
    if "=" not in spec:
        raise ValueError(f"{field_name} entries must use layer:<id>=MODE/SCHEME/BITS[,MODE/SCHEME/BITS...]")
    target, raw_candidates = spec.split("=", 1)
    parts = target.strip().split(":")
    if len(parts) != 2 or parts[0] != "layer":
        raise ValueError(f"{field_name} entries must use layer:<id>=MODE/SCHEME/BITS[,MODE/SCHEME/BITS...]")
    candidates = tuple(
        parse_page_mode_token(token.strip())
        for token in raw_candidates.split(",")
        if token.strip()
    )
    if not candidates:
        raise ValueError(f"{field_name} entries must include at least one candidate")
    return int(parts[1]), candidates


def _parse_layer_positive_int_spec(spec: str, *, field_name: str) -> tuple[int, int]:
    if "=" not in spec:
        raise ValueError(f"{field_name} entries must use layer:<id>=<positive_int>")
    target, value = spec.split("=", 1)
    parts = target.strip().split(":")
    if len(parts) != 2 or parts[0] != "layer":
        raise ValueError(f"{field_name} entries must use layer:<id>=<positive_int>")
    parsed_value = int(value.strip())
    if parsed_value <= 0:
        raise ValueError(f"{field_name} values must be positive integers")
    return int(parts[1]), parsed_value


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
    recent_page_escape_dtype: str = "float16"
    m2_sketch_dim_k: int = 8
    m4_project_basis_k: str = "hadamard"
    m4_project_basis_k_overrides: tuple[str, ...] = ()
    m4_project_dim_k_overrides: tuple[str, ...] = ()
    m2_center_k: bool = False
    m2_segment_count_k: int = 1
    m2_adaptive_segments_k: bool = False
    m2_adaptive_min_improvement_k: float = 0.1
    m2_prefilter_top_k: int = 0
    m2_prefilter_min_pages: int = 8
    prefer_m4_project_k: bool = False
    lut_refine_steps: int = 6
    preconditioner: str = "none"
    precondition_strength: float = 2.0
    m1_segment_count_k: int = 1
    m1_segment_count_v: int = 1
    m1_fallback_to_m0: bool = True
    m1_error_threshold: float = 0.35
    m1_token_p95_error_threshold: float = 1000000.0
    prepared_chunk_cache_budget_ratio: float = 0.5
    prepared_chunk_cache_min_bytes: int = 1 * 1024 * 1024
    prepared_chunk_cache_max_bytes: int = 64 * 1024 * 1024
    key_mode_overrides: tuple[str, ...] = ()
    value_mode_overrides: tuple[str, ...] = ()
    key_policy_tier: str = "exact"
    value_policy_tier: str = "exact"
    key_layer_sensitivity: tuple[str, ...] = ()
    value_layer_sensitivity: tuple[str, ...] = ()
    key_policy_overrides: tuple[str, ...] = ()
    value_policy_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.bits_k not in (2, 3, 4):
            raise ValueError("bits_k must be 2, 3, or 4 for the current runtime")
        if self.bits_v not in (2, 3, 4):
            raise ValueError("bits_v must be 2, 3, or 4 for the current runtime")
        if self.tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be positive")
        if self.payload_layout_k not in ("group_major", "token_major"):
            raise ValueError("payload_layout_k must be group_major or token_major")
        if self.payload_layout_v not in ("group_major", "token_major"):
            raise ValueError("payload_layout_v must be group_major or token_major")
        if self.default_mode_k not in _VALID_KEY_MODES:
            raise ValueError("default_mode_k must be M0, M1, M2, M3, M4, or T3")
        if self.default_mode_v not in _VALID_VALUE_MODES:
            raise ValueError("default_mode_v must be M0, M1, M3, or T3")
        if self.quant_scheme_k not in ("affine", "symmetric", "lut", "sketch", "project", "turbo3"):
            raise ValueError("quant_scheme_k must be affine, symmetric, lut, sketch, project, or turbo3")
        if self.quant_scheme_v not in ("affine", "symmetric", "lut", "turbo3"):
            raise ValueError("quant_scheme_v must be affine, symmetric, lut, or turbo3")
        if self.escape_dtype not in ("float16", "float32", "int8"):
            raise ValueError("escape_dtype must be float16, float32, or int8")
        if self.recent_page_escape_dtype not in ("float16", "float32", "int8"):
            raise ValueError("recent_page_escape_dtype must be float16, float32, or int8")
        if self.m2_sketch_dim_k <= 0:
            raise ValueError("m2_sketch_dim_k must be positive")
        if self.m4_project_basis_k not in _VALID_M4_BASIS_FAMILIES:
            allowed = ", ".join(_VALID_M4_BASIS_FAMILIES)
            raise ValueError(f"m4_project_basis_k must be one of {allowed}")
        for spec in self.m4_project_basis_k_overrides:
            _parse_layer_value_spec(
                spec,
                field_name="m4_project_basis_k_overrides",
                allowed_values=_VALID_M4_BASIS_FAMILIES,
            )
        for spec in self.m4_project_dim_k_overrides:
            _parse_layer_positive_int_spec(spec, field_name="m4_project_dim_k_overrides")
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
        if self.prepared_chunk_cache_budget_ratio < 0:
            raise ValueError("prepared_chunk_cache_budget_ratio must be non-negative")
        if self.prepared_chunk_cache_min_bytes < 0:
            raise ValueError("prepared_chunk_cache_min_bytes must be non-negative")
        if self.prepared_chunk_cache_max_bytes < 0:
            raise ValueError("prepared_chunk_cache_max_bytes must be non-negative")
        if (
            self.prepared_chunk_cache_max_bytes > 0
            and self.prepared_chunk_cache_min_bytes > self.prepared_chunk_cache_max_bytes
        ):
            raise ValueError("prepared_chunk_cache_min_bytes must not exceed prepared_chunk_cache_max_bytes")
        for spec in self.key_mode_overrides:
            _parse_mode_override_spec(spec, allowed_modes=_VALID_KEY_MODES, field_name="key_mode_overrides")
        for spec in self.value_mode_overrides:
            _parse_mode_override_spec(spec, allowed_modes=_VALID_VALUE_MODES, field_name="value_mode_overrides")
        for field_name, tier in (("key_policy_tier", self.key_policy_tier), ("value_policy_tier", self.value_policy_tier)):
            if tier not in ("exact", "strict", "balanced", "aggressive"):
                raise ValueError(f"{field_name} must be exact, strict, balanced, or aggressive")
        for spec in self.key_layer_sensitivity:
            _parse_layer_value_spec(
                spec,
                field_name="key_layer_sensitivity",
                allowed_values=("strict", "balanced", "aggressive"),
            )
        for spec in self.value_layer_sensitivity:
            _parse_layer_value_spec(
                spec,
                field_name="value_layer_sensitivity",
                allowed_values=("strict", "balanced", "aggressive"),
            )
        for spec in self.key_policy_overrides:
            _parse_layer_candidate_spec(spec, field_name="key_policy_overrides")
        for spec in self.value_policy_overrides:
            _parse_layer_candidate_spec(spec, field_name="value_policy_overrides")

    @property
    def num_groups(self) -> int:
        return ceil(self.head_dim / self.group_size)

    @property
    def padded_head_dim(self) -> int:
        return self.num_groups * self.group_size

    def has_mode_overrides(self, *, kind: str | None = None) -> bool:
        if kind == "K":
            return bool(self.key_mode_overrides)
        if kind == "V":
            return bool(self.value_mode_overrides)
        return bool(self.key_mode_overrides or self.value_mode_overrides)

    def has_policy_overrides(self, *, kind: str | None = None) -> bool:
        if kind == "K":
            return bool(self.key_layer_sensitivity or self.key_policy_overrides or self.key_policy_tier != "exact")
        if kind == "V":
            return bool(self.value_layer_sensitivity or self.value_policy_overrides or self.value_policy_tier != "exact")
        return bool(
            self.key_layer_sensitivity
            or self.value_layer_sensitivity
            or self.key_policy_overrides
            or self.value_policy_overrides
            or self.key_policy_tier != "exact"
            or self.value_policy_tier != "exact"
        )

    def resolve_page_mode(self, *, kind: str, layer_id: int, kv_head_id: int) -> str:
        if kind == "K":
            resolved = self.default_mode_k
            specs = self.key_mode_overrides
            allowed_modes = _VALID_KEY_MODES
            field_name = "key_mode_overrides"
        elif kind == "V":
            resolved = self.default_mode_v
            specs = self.value_mode_overrides
            allowed_modes = _VALID_VALUE_MODES
            field_name = "value_mode_overrides"
        else:
            raise ValueError("kind must be K or V")
        for spec in specs:
            override_layer_id, override_kv_head_id, override_mode = _parse_mode_override_spec(
                spec,
                allowed_modes=allowed_modes,
                field_name=field_name,
            )
            if override_layer_id != int(layer_id):
                continue
            if override_kv_head_id is not None and override_kv_head_id != int(kv_head_id):
                continue
            resolved = override_mode
        return resolved

    def resolve_m4_project_dim_k(self, *, layer_id: int) -> int:
        resolved = int(self.m2_sketch_dim_k)
        for spec in self.m4_project_dim_k_overrides:
            override_layer_id, override_dim = _parse_layer_positive_int_spec(
                spec,
                field_name="m4_project_dim_k_overrides",
            )
            if override_layer_id == int(layer_id):
                resolved = int(override_dim)
        return resolved

    def resolve_m4_project_basis_k(self, *, layer_id: int) -> str:
        resolved = self.m4_project_basis_k
        for spec in self.m4_project_basis_k_overrides:
            override_layer_id, override_basis = _parse_layer_value_spec(
                spec,
                field_name="m4_project_basis_k_overrides",
                allowed_values=_VALID_M4_BASIS_FAMILIES,
            )
            if override_layer_id == int(layer_id):
                resolved = override_basis
        return resolved

    def resolve_layer_policy(self, *, kind: str, layer_id: int, kv_head_id: int) -> LayerPolicy:
        if kind == "K":
            default_mode = self.default_mode_k
            default_bits = self.bits_k
            default_quant_scheme = self.quant_scheme_k
            default_tier = self.key_policy_tier
            sensitivity_specs = self.key_layer_sensitivity
            explicit_specs = self.key_policy_overrides
            mode_overrides = self.key_mode_overrides
        elif kind == "V":
            default_mode = self.default_mode_v
            default_bits = self.bits_v
            default_quant_scheme = self.quant_scheme_v
            default_tier = self.value_policy_tier
            sensitivity_specs = self.value_layer_sensitivity
            explicit_specs = self.value_policy_overrides
            mode_overrides = self.value_mode_overrides
        else:
            raise ValueError("kind must be K or V")

        resolved_mode = self.resolve_page_mode(kind=kind, layer_id=layer_id, kv_head_id=kv_head_id)
        if resolved_mode != default_mode:
            override_scheme = (
                "lut" if resolved_mode == "M1"
                else "sketch" if resolved_mode == "M2"
                else "project" if resolved_mode == "M4"
                else "turbo3" if resolved_mode == "T3"
                else default_quant_scheme
            )
            return make_explicit_policy(
                kind=kind,
                policy_id=f"{kind.lower()}_mode_override_layer_{int(layer_id)}",
                sensitivity_tier="exact",
                candidates=(PageModeSpec(mode=resolved_mode, bits=default_bits, quant_scheme=override_scheme),),
                recent_escape_dtype=self.recent_page_escape_dtype,
                recent_window=0,
            )

        for spec in explicit_specs:
            override_layer_id, candidates = _parse_layer_candidate_spec(spec, field_name="key_policy_overrides" if kind == "K" else "value_policy_overrides")
            if override_layer_id == int(layer_id):
                return make_explicit_policy(
                    kind=kind,
                    policy_id=f"{kind.lower()}_policy_override_layer_{int(layer_id)}",
                    sensitivity_tier="balanced",
                    candidates=candidates,
                    recent_escape_dtype=self.recent_page_escape_dtype,
                    recent_window=self.recent_window,
                )

        tier = default_tier
        for spec in sensitivity_specs:
            override_layer_id, override_tier = _parse_layer_value_spec(
                spec,
                field_name="key_layer_sensitivity" if kind == "K" else "value_layer_sensitivity",
                allowed_values=("strict", "balanced", "aggressive"),
            )
            if override_layer_id == int(layer_id):
                tier = override_tier
        return make_tier_candidates(
            kind=kind,
            sensitivity_tier=tier,
            default_bits=default_bits,
            default_quant_scheme=default_quant_scheme,
            default_mode=default_mode,
            recent_escape_dtype=self.recent_page_escape_dtype,
            recent_window=self.recent_window,
            prefer_project_key_mode=self.prefer_m4_project_k if kind == "K" else False,
        )
