from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping


EvaluationSplit = Literal["calibration", "held_out"]
EvaluationLane = Literal["systems", "quality", "diagnostic"]
PromptFamily = Literal["synthetic_exact_length", "held_out_natural_text", "standardized_long_context"]
HarnessTruthType = Literal["reference_trace", "paged_runtime"]

_PAGE_MODE_FIELDS: tuple[str, ...] = ("m0", "m1", "m2", "m3", "m4", "t3")


def _coerce_non_negative_int(value: Any, *, field_name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return parsed


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def page_format_histogram_from_result(result: Mapping[str, Any]) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for kind_prefix in ("k", "v"):
        kind_label = kind_prefix.upper()
        for mode in _PAGE_MODE_FIELDS:
            key = f"{kind_prefix}_{mode}_pages"
            if key not in result:
                continue
            histogram[f"{kind_label}:{mode.upper()}"] = max(int(result.get(key, 0) or 0), 0)
    return histogram


def _pick_context_length(result: Mapping[str, Any]) -> int | None:
    for key in ("context_length_effective", "context_length", "prompt_length", "sequence_length", "prefix_length"):
        value = result.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _pick_float(result: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in result:
            continue
        value = result.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


@dataclass(slots=True)
class EvaluationMetadata:
    model_id: str
    model_family: str
    backend: str
    device: str
    torch_dtype: str
    split: EvaluationSplit
    lane: EvaluationLane
    prompt_family: PromptFamily
    dataset_name: str
    prompt_count: int
    batch_size: int
    truth_type: HarnessTruthType
    effective_budget_rule: str
    context_length: int | None = None
    decode_steps: int | None = None
    eval_steps: int | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        if not self.model_family:
            raise ValueError("model_family is required")
        if not self.backend:
            raise ValueError("backend is required")
        if not self.device:
            raise ValueError("device is required")
        if not self.torch_dtype:
            raise ValueError("torch_dtype is required")
        if self.split not in ("calibration", "held_out"):
            raise ValueError("split must be calibration or held_out")
        if self.lane not in ("systems", "quality", "diagnostic"):
            raise ValueError("lane must be systems, quality, or diagnostic")
        if self.prompt_family not in ("synthetic_exact_length", "held_out_natural_text", "standardized_long_context"):
            raise ValueError("prompt_family is invalid")
        if not self.dataset_name:
            raise ValueError("dataset_name is required")
        self.prompt_count = _coerce_positive_int(self.prompt_count, field_name="prompt_count")
        self.batch_size = _coerce_positive_int(self.batch_size, field_name="batch_size")
        if not self.truth_type:
            raise ValueError("truth_type is required")
        if self.truth_type not in ("reference_trace", "paged_runtime"):
            raise ValueError("truth_type must be reference_trace or paged_runtime")
        if not self.effective_budget_rule:
            raise ValueError("effective_budget_rule is required")
        if self.context_length is not None:
            self.context_length = _coerce_positive_int(self.context_length, field_name="context_length")
        if self.decode_steps is not None:
            self.decode_steps = _coerce_non_negative_int(self.decode_steps, field_name="decode_steps")
        if self.eval_steps is not None:
            self.eval_steps = _coerce_non_negative_int(self.eval_steps, field_name="eval_steps")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationRecord:
    metadata: EvaluationMetadata
    metrics: dict[str, Any]
    source_result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "metrics": dict(self.metrics),
            "source_result": dict(self.source_result),
        }


def derive_standard_metrics(result: Mapping[str, Any], metadata: EvaluationMetadata | None = None) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    systems_keys = (
        "dotcache_decode_ms_per_step",
        "ttft_ms",
        "p95_decode_ms_per_step",
        "resident_bytes",
        "prefill_ms",
        "execution_shortlist_selected_pages",
        "execution_shortlist_total_pages",
    )
    for key in systems_keys:
        if key in result:
            metrics[key] = result[key]

    effective_bytes_per_token = _pick_float(result, "effective_bytes_per_token")
    if effective_bytes_per_token is None:
        resident_bytes = _pick_float(result, "resident_bytes")
        context_length = metadata.context_length if metadata is not None else None
        if context_length is None:
            context_length = _pick_context_length(result)
        if resident_bytes is not None and context_length is not None and context_length > 0:
            effective_bytes_per_token = resident_bytes / float(context_length)
    if effective_bytes_per_token is not None:
        metrics["effective_bytes_per_token"] = float(effective_bytes_per_token)

    quality_keys = (
        "teacher_forced_loss_delta",
        "teacher_forced_perplexity_ratio",
        "teacher_forced_logit_max_abs_error",
        "teacher_forced_logit_mean_abs_error",
        "teacher_forced_logit_rmse",
        "teacher_forced_token_agreement_rate",
        "teacher_forced_target_match_rate",
    )
    for key in quality_keys:
        if key in result:
            metrics[key] = result[key]

    diagnostic_keys = (
        "shortlist_recall_exact_top_recall_mean",
        "execution_decode_shortlist_selection_ms_total",
        "execution_decode_shortlist_candidate_approx_scoring_ms_total",
        "execution_decode_shortlist_candidate_ranking_ms_total",
        "execution_decode_shortlist_materialization_ms_total",
        "execution_decode_backend_call_non_backend_ms_total",
        "replay_output_max_abs_error",
    )
    for key in diagnostic_keys:
        if key in result:
            metrics[key] = result[key]

    if "decode_backend_trace" in result:
        metrics["decode_backend_trace"] = result["decode_backend_trace"]

    histogram = page_format_histogram_from_result(result)
    if histogram:
        metrics["page_format_histogram"] = histogram

    return metrics


def build_evaluation_record(
    metadata: EvaluationMetadata,
    result: Mapping[str, Any],
    *,
    include_source_result: bool = True,
) -> EvaluationRecord:
    metrics = derive_standard_metrics(result, metadata=metadata)
    source_result = dict(result) if include_source_result else {}
    return EvaluationRecord(metadata=metadata, metrics=metrics, source_result=source_result)
