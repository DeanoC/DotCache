from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

BaselineKind = Literal["local_mechanism", "external_runner", "paper_reference_only"]


@dataclass(frozen=True, slots=True)
class KvQuantBaselineSpec:
    key: str
    display_name: str
    kind: BaselineKind
    family: str
    asymmetry_support: str
    local_status: str
    benchmark_harness: str | None
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_KV_QUANT_REGISTRY: dict[str, KvQuantBaselineSpec] = {
    "dense_kv": KvQuantBaselineSpec(
        key="dense_kv",
        display_name="Dense FP16/BF16 KV",
        kind="local_mechanism",
        family="dense",
        asymmetry_support="none",
        local_status="implemented",
        benchmark_harness="llama_compare",
        notes="Baseline dense KV path for model-level comparisons.",
    ),
    "dequant_then_attend": KvQuantBaselineSpec(
        key="dequant_then_attend",
        display_name="Explicit dequantize-then-attend",
        kind="local_mechanism",
        family="dequantized",
        asymmetry_support="mode-specific",
        local_status="implemented",
        benchmark_harness="explicit_reference",
        notes="Reference path used to compare compressed-domain execution with widening-first execution.",
    ),
    "dotcache_m0": KvQuantBaselineSpec(
        key="dotcache_m0",
        display_name="DotCache M0",
        kind="local_mechanism",
        family="affine",
        asymmetry_support="k/v independent",
        local_status="implemented",
        benchmark_harness="llama_compare",
        notes="Exact compressed-domain affine baseline.",
    ),
    "dotcache_policy_adaptive": KvQuantBaselineSpec(
        key="dotcache_policy_adaptive",
        display_name="DotCache adaptive layer/page policy",
        kind="local_mechanism",
        family="adaptive",
        asymmetry_support="k/v independent",
        local_status="implemented",
        benchmark_harness="llama_compare",
        notes="Hierarchical policy path with coarse layer sensitivity and seal-time page choice.",
    ),
    "dotcache_v_only_m1": KvQuantBaselineSpec(
        key="dotcache_v_only_m1",
        display_name="DotCache V-only M1",
        kind="local_mechanism",
        family="lut",
        asymmetry_support="values only",
        local_status="implemented",
        benchmark_harness="llama_compare",
        notes="Value-side LUT comparison lane.",
    ),
    "dotcache_fixed_m2": KvQuantBaselineSpec(
        key="dotcache_fixed_m2",
        display_name="DotCache fixed segmented M2",
        kind="local_mechanism",
        family="sketch",
        asymmetry_support="keys only",
        local_status="implemented",
        benchmark_harness="llama_compare",
        notes="Key-side segmented sketch comparison lane.",
    ),
    "dotcache_t3": KvQuantBaselineSpec(
        key="dotcache_t3",
        display_name="DotCache Turbo3-style T3",
        kind="local_mechanism",
        family="turbo3",
        asymmetry_support="k/v independent",
        local_status="implemented",
        benchmark_harness="llama_compare",
        notes="Local Turbo3-form mechanism comparison lane.",
    ),
    "turboquant": KvQuantBaselineSpec(
        key="turboquant",
        display_name="TurboQuant",
        kind="external_runner",
        family="turbo",
        asymmetry_support="implementation-defined",
        local_status="external",
        benchmark_harness="external_runner",
        notes="Tracked through the TurboQuant comparison plan and CUDA-side external runs.",
    ),
    "kvquant": KvQuantBaselineSpec(
        key="kvquant",
        display_name="KVQuant",
        kind="paper_reference_only",
        family="affine",
        asymmetry_support="limited",
        local_status="paper_only",
        benchmark_harness=None,
        notes="Paper/reference row only on this box.",
    ),
    "kivi": KvQuantBaselineSpec(
        key="kivi",
        display_name="KIVI",
        kind="paper_reference_only",
        family="affine",
        asymmetry_support="keys vs values",
        local_status="paper_only",
        benchmark_harness=None,
        notes="Paper/reference row only on this box.",
    ),
    "qjl": KvQuantBaselineSpec(
        key="qjl",
        display_name="QJL",
        kind="paper_reference_only",
        family="projection",
        asymmetry_support="keys biased",
        local_status="paper_only",
        benchmark_harness=None,
        notes="Paper/reference row only on this box.",
    ),
    "polarquant": KvQuantBaselineSpec(
        key="polarquant",
        display_name="PolarQuant",
        kind="paper_reference_only",
        family="transformed",
        asymmetry_support="implementation-defined",
        local_status="paper_only",
        benchmark_harness=None,
        notes="Paper/reference row only on this box.",
    ),
    "innerq": KvQuantBaselineSpec(
        key="innerq",
        display_name="InnerQ",
        kind="paper_reference_only",
        family="inner-dimension",
        asymmetry_support="implementation-defined",
        local_status="paper_only",
        benchmark_harness=None,
        notes="Paper/reference row only on this box.",
    ),
    "qserve": KvQuantBaselineSpec(
        key="qserve",
        display_name="QServe",
        kind="paper_reference_only",
        family="widening_pipeline",
        asymmetry_support="n/a",
        local_status="paper_only",
        benchmark_harness=None,
        notes="Storage-only low-bit serving reference row.",
    ),
    "pqcache": KvQuantBaselineSpec(
        key="pqcache",
        display_name="PQCache",
        kind="paper_reference_only",
        family="retrieval_first",
        asymmetry_support="n/a",
        local_status="paper_only",
        benchmark_harness=None,
        notes="Rematerialization-style comparison row.",
    ),
    "xquant": KvQuantBaselineSpec(
        key="xquant",
        display_name="XQuant",
        kind="paper_reference_only",
        family="rematerialization",
        asymmetry_support="n/a",
        local_status="paper_only",
        benchmark_harness=None,
        notes="Paper/reference row only on this box.",
    ),
}


def list_kv_quant_baselines() -> tuple[KvQuantBaselineSpec, ...]:
    return tuple(_KV_QUANT_REGISTRY.values())


def get_kv_quant_baseline(key: str) -> KvQuantBaselineSpec:
    try:
        return _KV_QUANT_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"unknown kv quant baseline key: {key}") from exc
