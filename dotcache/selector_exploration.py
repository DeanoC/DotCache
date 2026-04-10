from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .selector_baselines import (
    _apply_candidate_logit_offset,
    CandidateSafeLinearSelectorModel,
    CandidateSafeRouterModel,
    CandidateTargetLinearSelectorModel,
    CandidateTargetRouterModel,
    LinearSelectorModel,
    SelectorCandidateExample,
    SelectorEvaluationSummary,
    SelectorExample,
    SelectorPrediction,
    adjust_linear_selector_model_logits,
    build_selector_class_error_weights,
    build_selector_example_weights,
    calibrate_selector_logit_offset,
    candidate_feature_names_from_examples,
    discover_selector_split_dirs,
    evaluate_selector_model,
    load_selector_split_examples,
    normalize_selector_categorical_token,
    save_page_selector_artifact,
    selector_feature_vector_from_row,
    selector_candidate_feature_vector_from_row,
    selector_feature_names_from_examples,
    selector_prompt_length_from_row,
    split_selector_examples,
    train_calibrated_runtime_linear_selector,
    train_candidate_safe_linear_selector,
    train_candidate_target_linear_selector,
    train_linear_selector,
    train_runtime_linear_selector,
    train_static_rule_selector,
)

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency
    torch = None
    F = None

try:  # pragma: no cover - optional dependency
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:  # pragma: no cover - optional dependency
    GradientBoostingClassifier = None


DEFAULT_APPLE_LOCAL_SUITE_ROOT = (
    "benchmarks/results/qwen35_selector_qwen35_0p8b_local_20260406/suite"
)
DEFAULT_REPORT_AXES = (
    "min_family_safe_prediction_rate",
    "min_family_target_accuracy",
    "mean_predicted_total_bytes",
    "mean_safe_bytes_regret",
)
DEFAULT_DISTILLATION_TEACHER_WEIGHT = 0.5
DEFAULT_DISTILLATION_TEMPERATURE = 1.0
DEFAULT_BINARY_THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70)
DEFAULT_LINEAR_OFFSETS = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
DEFAULT_FALLBACK_CANDIDATE = "M3/affine/4/float16"


class _PredictsSelector(Protocol):
    def predict(self, example: SelectorExample) -> str | None:
        ...


@dataclass(slots=True)
class SelectorExplorationStrategyDefinition:
    strategy_id: str
    strategy_kind: str
    runtime_compatible: bool
    artifact_capable: bool
    supported_feature_set_ids: tuple[str, ...]
    supported_calibration_modes: tuple[str, ...]
    fit: Callable[..., "FittedSelectorExplorationStrategy"]


@dataclass(slots=True)
class FittedSelectorExplorationStrategy:
    strategy_id: str
    strategy_kind: str
    feature_set_id: str
    calibration_mode: str
    runtime_compatible: bool
    artifact_capable: bool
    feature_names: tuple[str, ...]
    predict_by_trace_fn: Callable[[Sequence[SelectorExample], Sequence[SelectorCandidateExample]], dict[str, str | None]]
    save_model_fn: Callable[[Path], str | None]
    model_summary: dict[str, Any] = field(default_factory=dict)

    def predict_by_trace(
        self,
        examples: Sequence[SelectorExample],
        candidate_examples: Sequence[SelectorCandidateExample],
    ) -> dict[str, str | None]:
        return self.predict_by_trace_fn(examples, candidate_examples)

    def save_model(self, path: Path) -> str | None:
        return self.save_model_fn(path)


@dataclass(slots=True)
class SelectorBreakdownRow:
    group_name: str
    group_value: str
    example_count: int
    target_accuracy: float
    safe_prediction_rate: float
    mean_predicted_total_bytes: float | None
    mean_safe_bytes_regret: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SelectorExplorationStrategyResult:
    strategy_id: str
    base_strategy_id: str
    strategy_kind: str
    feature_set_id: str
    calibration_mode: str
    runtime_compatible: bool
    artifact_capable: bool
    status: str
    aggregate_metrics: dict[str, Any]
    per_split_metrics: list[dict[str, Any]]
    family_breakdown: list[dict[str, Any]]
    variant_breakdown: list[dict[str, Any]]
    prediction_path: str | None
    artifact_path: str | None
    research_model_path: str | None
    serving_smoke: dict[str, Any] | None
    promotable: bool
    pareto_optimal: bool
    model_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _PredictedSelectorModel:
    predicted_by_trace: dict[str, str | None]

    def predict(self, example: SelectorExample) -> str | None:
        return self.predicted_by_trace.get(example.trace_path)


@dataclass(slots=True)
class CandidateTargetMlpModel:
    weight_1: np.ndarray
    bias_1: np.ndarray
    weight_2: np.ndarray
    bias_2: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: tuple[str, ...]

    def predict_probability_for_row(self, row: dict[str, Any]) -> float:
        if torch is None:
            raise RuntimeError("torch is required to score CandidateTargetMlpModel")
        features = _candidate_row_features(row, feature_names=self.feature_names)
        standardized = (features - self.feature_mean) / self.feature_std
        hidden = np.maximum(standardized @ self.weight_1 + self.bias_1, 0.0)
        logit = float(hidden @ self.weight_2 + float(self.bias_2))
        return float(1.0 / (1.0 + np.exp(-logit)))


@dataclass(slots=True)
class CandidateTargetGbdtModel:
    estimator: Any
    feature_names: tuple[str, ...]

    def predict_probability_for_row(self, row: dict[str, Any]) -> float:
        features = _candidate_row_features(row, feature_names=self.feature_names).reshape(1, -1)
        probabilities = self.estimator.predict_proba(features)
        return float(probabilities[0, 1])


def list_selector_exploration_strategies() -> dict[str, SelectorExplorationStrategyDefinition]:
    return {
        "static_rule": SelectorExplorationStrategyDefinition(
            strategy_id="static_rule",
            strategy_kind="row_multiclass",
            runtime_compatible=False,
            artifact_capable=False,
            supported_feature_set_ids=("runtime_safe", "research_extended"),
            supported_calibration_modes=("global",),
            fit=_fit_static_rule_strategy,
        ),
        "linear_softmax": SelectorExplorationStrategyDefinition(
            strategy_id="linear_softmax",
            strategy_kind="row_multiclass",
            runtime_compatible=True,
            artifact_capable=True,
            supported_feature_set_ids=("runtime_safe", "research_extended"),
            supported_calibration_modes=("global",),
            fit=_fit_linear_softmax_strategy,
        ),
        "linear_softmax_compression_weighted": SelectorExplorationStrategyDefinition(
            strategy_id="linear_softmax_compression_weighted",
            strategy_kind="row_multiclass",
            runtime_compatible=True,
            artifact_capable=True,
            supported_feature_set_ids=("runtime_safe", "research_extended"),
            supported_calibration_modes=("global",),
            fit=_fit_weighted_linear_strategy,
        ),
        "linear_softmax_compression_calibrated": SelectorExplorationStrategyDefinition(
            strategy_id="linear_softmax_compression_calibrated",
            strategy_kind="row_multiclass",
            runtime_compatible=True,
            artifact_capable=True,
            supported_feature_set_ids=("runtime_safe",),
            supported_calibration_modes=("global",),
            fit=_fit_calibrated_linear_strategy,
        ),
        "linear_softmax_compression_equal_tradeoff": SelectorExplorationStrategyDefinition(
            strategy_id="linear_softmax_compression_equal_tradeoff",
            strategy_kind="row_multiclass",
            runtime_compatible=True,
            artifact_capable=True,
            supported_feature_set_ids=("runtime_safe",),
            supported_calibration_modes=("global",),
            fit=_fit_equal_tradeoff_linear_strategy,
        ),
        "linear_softmax_distilled_mlp_teacher": SelectorExplorationStrategyDefinition(
            strategy_id="linear_softmax_distilled_mlp_teacher",
            strategy_kind="row_multiclass",
            runtime_compatible=True,
            artifact_capable=True,
            supported_feature_set_ids=("runtime_safe",),
            supported_calibration_modes=("global",),
            fit=_fit_distilled_linear_mlp_teacher_strategy,
        ),
        "candidate_safe_router": SelectorExplorationStrategyDefinition(
            strategy_id="candidate_safe_router",
            strategy_kind="candidate_safe",
            runtime_compatible=True,
            artifact_capable=True,
            supported_feature_set_ids=("runtime_safe", "research_extended"),
            supported_calibration_modes=("global", "per_prompt_family"),
            fit=_fit_candidate_safe_strategy,
        ),
        "candidate_target_linear": SelectorExplorationStrategyDefinition(
            strategy_id="candidate_target_linear",
            strategy_kind="candidate_target",
            runtime_compatible=True,
            artifact_capable=True,
            supported_feature_set_ids=("runtime_safe", "research_extended"),
            supported_calibration_modes=("global", "per_prompt_family"),
            fit=_fit_candidate_target_linear_strategy,
        ),
        "candidate_target_mlp": SelectorExplorationStrategyDefinition(
            strategy_id="candidate_target_mlp",
            strategy_kind="candidate_target",
            runtime_compatible=False,
            artifact_capable=False,
            supported_feature_set_ids=("runtime_safe", "research_extended"),
            supported_calibration_modes=("global", "per_prompt_family"),
            fit=_fit_candidate_target_mlp_strategy,
        ),
        "candidate_target_gbdt": SelectorExplorationStrategyDefinition(
            strategy_id="candidate_target_gbdt",
            strategy_kind="candidate_target",
            runtime_compatible=False,
            artifact_capable=False,
            supported_feature_set_ids=("runtime_safe", "research_extended"),
            supported_calibration_modes=("global", "per_prompt_family"),
            fit=_fit_candidate_target_gbdt_strategy,
        ),
    }


def run_selector_exploration_lab(
    *,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    resolved_config = resolve_selector_exploration_config(config)
    split_dirs = discover_selector_split_dirs(resolved_config["suite_root"])
    strategies = list_selector_exploration_strategies()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    strategy_results: list[SelectorExplorationStrategyResult] = []

    full_suite_examples = _collect_full_suite_examples(split_dirs)
    for strategy_entry in resolved_config["strategies"]:
        base_strategy_id = str(strategy_entry["strategy_id"])
        strategy_id = str(strategy_entry.get("result_id", base_strategy_id))
        definition = strategies.get(base_strategy_id)
        if definition is None:
            raise ValueError(f"unknown selector exploration strategy: {base_strategy_id}")
        feature_set_id = str(strategy_entry.get("feature_set_id", resolved_config["feature_set_id"]))
        calibration_mode = str(strategy_entry.get("calibration_mode", "global"))
        if feature_set_id not in definition.supported_feature_set_ids:
            raise ValueError(f"strategy {base_strategy_id} does not support feature_set_id={feature_set_id}")
        if calibration_mode not in definition.supported_calibration_modes:
            raise ValueError(f"strategy {base_strategy_id} does not support calibration_mode={calibration_mode}")

        strategy_output_dir = output_root / "strategies" / strategy_id
        strategy_output_dir.mkdir(parents=True, exist_ok=True)
        split_metrics: list[dict[str, Any]] = []
        family_breakdown_rows: list[dict[str, Any]] = []
        variant_breakdown_rows: list[dict[str, Any]] = []
        prediction_rows: list[dict[str, Any]] = []
        strategy_status = "ok"
        model_summary: dict[str, Any] = {}

        try:
            for split_dir in split_dirs:
                split_payload = load_selector_split_examples(split_dir=split_dir)
                split_summary = dict(split_payload.get("split_summary") or {})
                split_name = str(split_summary.get("split_name") or Path(split_dir).name)
                fitted = definition.fit(
                    train_examples=split_payload["train_examples"],
                    train_candidate_examples=split_payload["train_candidate_examples"],
                    feature_set_id=feature_set_id,
                    calibration_mode=calibration_mode,
                    config=resolved_config,
                    strategy_config=strategy_entry.get("params", {}),
                )
                model_summary = dict(fitted.model_summary)
                predicted_by_trace = fitted.predict_by_trace(
                    split_payload["test_examples"],
                    split_payload["test_candidate_examples"],
                )
                split_evaluation = _evaluate_strategy_predictions(
                    split_payload["test_examples"],
                    predicted_by_trace=predicted_by_trace,
                    split_name=split_name,
                )
                split_metrics.append(split_evaluation["summary"])
                family_breakdown_rows.extend(split_evaluation["family_breakdown"])
                variant_breakdown_rows.extend(split_evaluation["variant_breakdown"])
                prediction_rows.extend(split_evaluation["predictions"])
        except _StrategyUnavailableError as exc:
            strategy_status = "dependency_unavailable"
            model_summary = {"message": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            strategy_status = "error"
            model_summary = {"message": str(exc)}

        prediction_path = None
        artifact_path = None
        research_model_path = None
        serving_smoke = None
        runtime_compatible = bool(definition.runtime_compatible and feature_set_id == "runtime_safe")
        artifact_capable = bool(definition.artifact_capable and feature_set_id == "runtime_safe")
        promotable = False

        if strategy_status == "ok":
            prediction_path = str(_write_strategy_predictions(strategy_output_dir / "predictions.jsonl", prediction_rows))
            aggregated_family_breakdown = _aggregate_breakdown_rows(family_breakdown_rows, group_name="prompt_family")
            aggregated_variant_breakdown = _aggregate_breakdown_rows(variant_breakdown_rows, group_name="prompt_variant")
            aggregate_metrics = _aggregate_strategy_metrics(split_metrics, aggregated_family_breakdown, aggregated_variant_breakdown)
            fitted_full = definition.fit(
                train_examples=full_suite_examples["examples"],
                train_candidate_examples=full_suite_examples["candidate_examples"],
                feature_set_id=feature_set_id,
                calibration_mode=calibration_mode,
                config=resolved_config,
                strategy_config=strategy_entry.get("params", {}),
            )
            model_summary = dict(fitted_full.model_summary)
            if artifact_capable:
                artifact_candidate_path = strategy_output_dir / "selector_artifact.json"
                artifact_path = fitted_full.save_model(artifact_candidate_path)
            else:
                research_model_path = _save_research_model(
                    fitted_full=fitted_full,
                    path_root=strategy_output_dir / "research_model",
                )
        else:
            aggregate_metrics = _empty_aggregate_metrics()

        strategy_results.append(
            SelectorExplorationStrategyResult(
                strategy_id=strategy_id,
                base_strategy_id=base_strategy_id,
                strategy_kind=definition.strategy_kind,
                feature_set_id=feature_set_id,
                calibration_mode=calibration_mode,
                runtime_compatible=runtime_compatible,
                artifact_capable=artifact_capable,
                status=strategy_status,
                aggregate_metrics=aggregate_metrics,
                per_split_metrics=split_metrics,
                family_breakdown=aggregated_family_breakdown if strategy_status == "ok" else family_breakdown_rows,
                variant_breakdown=aggregated_variant_breakdown if strategy_status == "ok" else variant_breakdown_rows,
                prediction_path=prediction_path,
                artifact_path=artifact_path,
                research_model_path=research_model_path,
                serving_smoke=serving_smoke,
                promotable=promotable,
                pareto_optimal=False,
                model_summary=model_summary,
            )
        )

    _apply_pareto_membership(strategy_results, report_axes=tuple(resolved_config["report_axes"]))
    if resolved_config["serving_smoke"].get("enabled", False):
        for result in strategy_results:
            if not result.pareto_optimal or not result.runtime_compatible or result.artifact_path is None:
                continue
            smoke = run_selector_serving_smoke(
                result.strategy_id,
                artifact_path=result.artifact_path,
                config=resolved_config["serving_smoke"],
                output_root=output_root / "serving_smoke",
            )
            result.serving_smoke = smoke
            result.promotable = bool(
                result.runtime_compatible
                and result.feature_set_id == "runtime_safe"
                and result.artifact_path is not None
                and smoke.get("status") == "ok"
            )

    report_payload = {
        "config": resolved_config,
        "strategies": [result.to_dict() for result in strategy_results],
        "report_axes": list(resolved_config["report_axes"]),
    }
    json_path = output_root / "selector_exploration_results.json"
    markdown_path = output_root / "selector_exploration_report.md"
    json_path.write_text(json.dumps(report_payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_selector_exploration_markdown(report_payload) + "\n", encoding="utf-8")
    report_payload["json_path"] = str(json_path)
    report_payload["markdown_path"] = str(markdown_path)
    return report_payload


def resolve_selector_exploration_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    resolved["suite_root"] = str(resolved.get("suite_root") or DEFAULT_APPLE_LOCAL_SUITE_ROOT)
    resolved["feature_set_id"] = str(resolved.get("feature_set_id") or "runtime_safe")
    resolved["report_axes"] = list(resolved.get("report_axes") or DEFAULT_REPORT_AXES)
    resolved["strategies"] = list(resolved.get("strategies") or [])
    if not resolved["strategies"]:
        resolved["strategies"] = [
            {"strategy_id": "static_rule"},
            {"strategy_id": "linear_softmax"},
            {"strategy_id": "linear_softmax_compression_weighted"},
            {"strategy_id": "linear_softmax_compression_calibrated"},
            {"strategy_id": "linear_softmax_compression_equal_tradeoff"},
            {"strategy_id": "linear_softmax_distilled_mlp_teacher"},
            {"strategy_id": "candidate_safe_router"},
            {"strategy_id": "candidate_target_linear"},
            {"strategy_id": "candidate_target_mlp"},
            {"strategy_id": "candidate_target_gbdt"},
        ]
    resolved["weighted_selector"] = {
        "steps": int(dict(resolved.get("weighted_selector") or {}).get("steps", 400)),
        "learning_rate": float(dict(resolved.get("weighted_selector") or {}).get("learning_rate", 0.2)),
        "l2": float(dict(resolved.get("weighted_selector") or {}).get("l2", 1e-3)),
        "class_balance": float(dict(resolved.get("weighted_selector") or {}).get("class_balance", 0.5)),
        "safe_bytes_weight": float(dict(resolved.get("weighted_selector") or {}).get("safe_bytes_weight", 1.0)),
        "unsafe_error_weight": float(dict(resolved.get("weighted_selector") or {}).get("unsafe_error_weight", 0.5)),
        "reference_candidate": str(dict(resolved.get("weighted_selector") or {}).get("reference_candidate", DEFAULT_FALLBACK_CANDIDATE)),
        "target_candidate": str(dict(resolved.get("weighted_selector") or {}).get("target_candidate", DEFAULT_FALLBACK_CANDIDATE)),
    }
    resolved["calibration"] = {
        "fraction": float(dict(resolved.get("calibration") or {}).get("fraction", 0.25)),
        "seed": int(dict(resolved.get("calibration") or {}).get("seed", 0)),
        "min_target_accuracy": dict(resolved.get("calibration") or {}).get("min_target_accuracy", 0.999),
        "min_safe_prediction_rate": float(dict(resolved.get("calibration") or {}).get("min_safe_prediction_rate", 1.0)),
        "linear_offsets": [float(value) for value in dict(resolved.get("calibration") or {}).get("linear_offsets", DEFAULT_LINEAR_OFFSETS)],
        "binary_thresholds": [float(value) for value in dict(resolved.get("calibration") or {}).get("binary_thresholds", DEFAULT_BINARY_THRESHOLDS)],
        "correctness_weight": float(dict(resolved.get("calibration") or {}).get("correctness_weight", 1.0)),
        "bytes_weight": float(dict(resolved.get("calibration") or {}).get("bytes_weight", 1.0)),
    }
    resolved["distillation"] = {
        "teacher_feature_set_id": str(dict(resolved.get("distillation") or {}).get("teacher_feature_set_id", "research_extended")),
        "teacher_hidden_dim": int(dict(resolved.get("distillation") or {}).get("teacher_hidden_dim", 16)),
        "teacher_epochs": int(dict(resolved.get("distillation") or {}).get("teacher_epochs", 120)),
        "teacher_learning_rate": float(dict(resolved.get("distillation") or {}).get("teacher_learning_rate", 1e-2)),
        "teacher_seed": int(dict(resolved.get("distillation") or {}).get("teacher_seed", 0)),
        "teacher_weight": float(dict(resolved.get("distillation") or {}).get("teacher_weight", DEFAULT_DISTILLATION_TEACHER_WEIGHT)),
        "teacher_temperature": float(dict(resolved.get("distillation") or {}).get("teacher_temperature", DEFAULT_DISTILLATION_TEMPERATURE)),
    }
    resolved["dense_control"] = {
        "enabled": bool(dict(resolved.get("dense_control") or {}).get("enabled", False)),
        "report_path": (
            None
            if dict(resolved.get("dense_control") or {}).get("report_path") in (None, "")
            else str(dict(resolved.get("dense_control") or {}).get("report_path"))
        ),
        "correct_example_weight": float(dict(resolved.get("dense_control") or {}).get("correct_example_weight", 1.0)),
        "incorrect_example_weight": float(dict(resolved.get("dense_control") or {}).get("incorrect_example_weight", 1.0)),
    }
    resolved["serving_smoke"] = {
        "enabled": bool(dict(resolved.get("serving_smoke") or {}).get("enabled", False)),
        "command_template": list(
            dict(resolved.get("serving_smoke") or {}).get(
                "command_template",
                [
                    "bash",
                    "scripts/run_qwen35_0p8b_task_selector_compare.sh",
                    "{output_dir}",
                    "{artifact_path}",
                    "--profiles",
                    "exact",
                    "quality",
                    "--warmup-runs",
                    "0",
                    "--measured-runs",
                    "1",
                    "--max-new-tokens-retrieval",
                    "16",
                    "--max-new-tokens-reasoning",
                    "16",
                    "--max-new-tokens-instruction",
                    "8",
                ],
            )
        ),
    }
    return resolved


def render_selector_exploration_markdown(payload: dict[str, Any]) -> str:
    strategies = list(payload.get("strategies", []))
    lines = [
        "# Selector Exploration Lab",
        "",
        "## Aggregate",
        "",
        "| strategy_id | status | kind | feature_set | calibration_mode | pareto | promotable | min_family_safe_prediction_rate | min_family_target_accuracy | mean_predicted_total_bytes | mean_safe_bytes_regret |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for strategy in strategies:
        metrics = dict(strategy.get("aggregate_metrics", {}))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(strategy.get("strategy_id")),
                    str(strategy.get("status")),
                    str(strategy.get("strategy_kind")),
                    str(strategy.get("feature_set_id")),
                    str(strategy.get("calibration_mode")),
                    "yes" if bool(strategy.get("pareto_optimal")) else "no",
                    "yes" if bool(strategy.get("promotable")) else "no",
                    _format_metric(metrics.get("min_family_safe_prediction_rate")),
                    _format_metric(metrics.get("min_family_target_accuracy")),
                    _format_metric(metrics.get("mean_predicted_total_bytes"), digits=1),
                    _format_metric(metrics.get("mean_safe_bytes_regret"), digits=1),
                ]
            )
            + " |"
        )

    lines.extend(["", "## By Split", "", "| strategy_id | split | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes | mean_safe_bytes_regret |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for strategy in strategies:
        for split_row in strategy.get("per_split_metrics", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(strategy.get("strategy_id")),
                        str(split_row.get("split_name")),
                        _format_metric(split_row.get("target_accuracy")),
                        _format_metric(split_row.get("safe_prediction_rate")),
                        _format_metric(split_row.get("mean_predicted_total_bytes"), digits=1),
                        _format_metric(split_row.get("mean_safe_bytes_regret"), digits=1),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## By Prompt Family", "", "| strategy_id | prompt_family | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |", "| --- | --- | ---: | ---: | ---: |"])
    for strategy in strategies:
        for row in strategy.get("family_breakdown", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(strategy.get("strategy_id")),
                        str(row.get("group_value")),
                        _format_metric(row.get("target_accuracy")),
                        _format_metric(row.get("safe_prediction_rate")),
                        _format_metric(row.get("mean_predicted_total_bytes"), digits=1),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## By Prompt Variant", "", "| strategy_id | prompt_variant | target_accuracy | safe_prediction_rate | mean_predicted_total_bytes |", "| --- | --- | ---: | ---: | ---: |"])
    for strategy in strategies:
        for row in strategy.get("variant_breakdown", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(strategy.get("strategy_id")),
                        str(row.get("group_value")),
                        _format_metric(row.get("target_accuracy")),
                        _format_metric(row.get("safe_prediction_rate")),
                        _format_metric(row.get("mean_predicted_total_bytes"), digits=1),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def run_selector_serving_smoke(
    strategy_id: str,
    *,
    artifact_path: str,
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    output_dir = output_root / strategy_id
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(token).format(output_dir=str(output_dir), artifact_path=str(artifact_path))
        for token in list(config.get("command_template", []))
    ]
    if not command:
        return {"status": "skipped", "message": "empty command_template"}
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    json_report_path = output_dir / "task_selector_compare.json"
    markdown_report_path = output_dir / "task_selector_compare.md"
    report_payload = None
    if json_report_path.exists():
        report_payload = json.loads(json_report_path.read_text(encoding="utf-8"))
    return {
        "status": "ok" if completed.returncode == 0 and json_report_path.exists() else "error",
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "json_report_path": str(json_report_path) if json_report_path.exists() else None,
        "markdown_report_path": str(markdown_report_path) if markdown_report_path.exists() else None,
        "report": report_payload,
    }


class _StrategyUnavailableError(RuntimeError):
    pass


def _collect_full_suite_examples(split_dirs: Sequence[str | Path]) -> dict[str, Any]:
    examples_by_trace: dict[str, SelectorExample] = {}
    candidate_by_key: dict[tuple[str, str], SelectorCandidateExample] = {}
    for split_dir in split_dirs:
        payload = load_selector_split_examples(split_dir=split_dir)
        for collection_name in ("train_examples", "test_examples"):
            for example in payload[collection_name]:
                examples_by_trace.setdefault(example.trace_path, example)
        for collection_name in ("train_candidate_examples", "test_candidate_examples"):
            for example in payload[collection_name]:
                candidate_by_key.setdefault((example.trace_path, example.candidate), example)
    return {
        "examples": list(examples_by_trace.values()),
        "candidate_examples": list(candidate_by_key.values()),
    }


def _evaluate_strategy_predictions(
    examples: Sequence[SelectorExample],
    *,
    predicted_by_trace: dict[str, str | None],
    split_name: str,
) -> dict[str, Any]:
    summary = evaluate_selector_model(_PredictedSelectorModel(predicted_by_trace), examples)
    example_by_trace = {example.trace_path: example for example in examples}
    family_breakdown = _build_group_breakdown(
        examples,
        predicted_by_trace=predicted_by_trace,
        group_name="prompt_family",
        value_getter=lambda example: example.prompt_family,
    )
    variant_breakdown = _build_group_breakdown(
        examples,
        predicted_by_trace=predicted_by_trace,
        group_name="prompt_variant",
        value_getter=lambda example: example.prompt_variant,
    )
    prediction_rows = []
    for prediction in summary.predictions:
        example = example_by_trace[prediction.trace_path]
        prediction_rows.append(
            {
                **prediction.to_dict(),
                "split_name": split_name,
                "prompt_family": example.prompt_family,
                "prompt_variant": example.prompt_variant,
            }
        )
    return {
        "summary": {
            "split_name": split_name,
            "example_count": int(summary.example_count),
            "target_accuracy": float(summary.target_accuracy),
            "safe_prediction_rate": float(summary.safe_prediction_rate),
            "mean_predicted_total_bytes": summary.mean_predicted_total_bytes,
            "mean_safe_bytes_regret": summary.mean_safe_bytes_regret,
        },
        "family_breakdown": [row.to_dict() for row in family_breakdown],
        "variant_breakdown": [row.to_dict() for row in variant_breakdown],
        "predictions": prediction_rows,
    }


def _build_group_breakdown(
    examples: Sequence[SelectorExample],
    *,
    predicted_by_trace: dict[str, str | None],
    group_name: str,
    value_getter: Callable[[SelectorExample], str | None],
) -> list[SelectorBreakdownRow]:
    grouped_examples: dict[str, list[SelectorExample]] = defaultdict(list)
    for example in examples:
        grouped_examples[str(value_getter(example) or "__none__")].append(example)
    rows: list[SelectorBreakdownRow] = []
    for group_value, group_examples in sorted(grouped_examples.items()):
        summary = evaluate_selector_model(_PredictedSelectorModel(predicted_by_trace), group_examples)
        rows.append(
            SelectorBreakdownRow(
                group_name=group_name,
                group_value=group_value,
                example_count=int(summary.example_count),
                target_accuracy=float(summary.target_accuracy),
                safe_prediction_rate=float(summary.safe_prediction_rate),
                mean_predicted_total_bytes=summary.mean_predicted_total_bytes,
                mean_safe_bytes_regret=summary.mean_safe_bytes_regret,
            )
        )
    return rows


def _aggregate_breakdown_rows(
    rows: Sequence[dict[str, Any]],
    *,
    group_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_value"])].append(dict(row))
    aggregated: list[dict[str, Any]] = []
    for group_value, group_rows in sorted(grouped.items()):
        predicted_bytes = [row["mean_predicted_total_bytes"] for row in group_rows if row["mean_predicted_total_bytes"] is not None]
        safe_regrets = [row["mean_safe_bytes_regret"] for row in group_rows if row["mean_safe_bytes_regret"] is not None]
        aggregated.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "example_count": int(sum(int(row["example_count"]) for row in group_rows)),
                "target_accuracy": float(np.mean(np.asarray([float(row["target_accuracy"]) for row in group_rows], dtype=np.float32))),
                "safe_prediction_rate": float(np.mean(np.asarray([float(row["safe_prediction_rate"]) for row in group_rows], dtype=np.float32))),
                "mean_predicted_total_bytes": (
                    None if not predicted_bytes else float(np.mean(np.asarray(predicted_bytes, dtype=np.float32)))
                ),
                "mean_safe_bytes_regret": (
                    None if not safe_regrets else float(np.mean(np.asarray(safe_regrets, dtype=np.float32)))
                ),
            }
        )
    return aggregated


def _aggregate_strategy_metrics(
    split_metrics: Sequence[dict[str, Any]],
    family_breakdown_rows: Sequence[dict[str, Any]],
    variant_breakdown_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not split_metrics:
        return _empty_aggregate_metrics()
    target_accuracies = np.asarray([float(row["target_accuracy"]) for row in split_metrics], dtype=np.float32)
    safe_rates = np.asarray([float(row["safe_prediction_rate"]) for row in split_metrics], dtype=np.float32)
    mean_bytes = [row["mean_predicted_total_bytes"] for row in split_metrics if row["mean_predicted_total_bytes"] is not None]
    mean_regrets = [row["mean_safe_bytes_regret"] for row in split_metrics if row["mean_safe_bytes_regret"] is not None]
    family_safe = [float(row["safe_prediction_rate"]) for row in family_breakdown_rows]
    family_acc = [float(row["target_accuracy"]) for row in family_breakdown_rows]
    variant_safe = [float(row["safe_prediction_rate"]) for row in variant_breakdown_rows]
    variant_acc = [float(row["target_accuracy"]) for row in variant_breakdown_rows]
    return {
        "mean_target_accuracy": float(np.mean(target_accuracies)),
        "mean_safe_prediction_rate": float(np.mean(safe_rates)),
        "mean_predicted_total_bytes": None if not mean_bytes else float(np.mean(np.asarray(mean_bytes, dtype=np.float32))),
        "mean_safe_bytes_regret": None if not mean_regrets else float(np.mean(np.asarray(mean_regrets, dtype=np.float32))),
        "min_family_safe_prediction_rate": None if not family_safe else float(min(family_safe)),
        "min_family_target_accuracy": None if not family_acc else float(min(family_acc)),
        "min_variant_safe_prediction_rate": None if not variant_safe else float(min(variant_safe)),
        "min_variant_target_accuracy": None if not variant_acc else float(min(variant_acc)),
    }


def _empty_aggregate_metrics() -> dict[str, Any]:
    return {
        "mean_target_accuracy": None,
        "mean_safe_prediction_rate": None,
        "mean_predicted_total_bytes": None,
        "mean_safe_bytes_regret": None,
        "min_family_safe_prediction_rate": None,
        "min_family_target_accuracy": None,
        "min_variant_safe_prediction_rate": None,
        "min_variant_target_accuracy": None,
    }


def _apply_pareto_membership(
    strategy_results: Sequence[SelectorExplorationStrategyResult],
    *,
    report_axes: Sequence[str],
) -> None:
    eligible = [result for result in strategy_results if result.status == "ok"]
    for candidate in eligible:
        dominated = False
        for other in eligible:
            if other.strategy_id == candidate.strategy_id:
                continue
            if _dominates(other.aggregate_metrics, candidate.aggregate_metrics, report_axes):
                dominated = True
                break
        candidate.pareto_optimal = not dominated


def _dominates(left: dict[str, Any], right: dict[str, Any], axes: Sequence[str]) -> bool:
    comparisons: list[bool] = []
    strict = False
    for axis in axes:
        left_value = _normalize_axis_value(axis, left.get(axis))
        right_value = _normalize_axis_value(axis, right.get(axis))
        if axis in {"mean_predicted_total_bytes", "mean_safe_bytes_regret"}:
            comparisons.append(left_value <= right_value)
            strict = strict or left_value < right_value
        else:
            comparisons.append(left_value >= right_value)
            strict = strict or left_value > right_value
    return all(comparisons) and strict


def _normalize_axis_value(axis: str, value: Any) -> float:
    if value is None:
        return float("inf") if axis in {"mean_predicted_total_bytes", "mean_safe_bytes_regret"} else float("-inf")
    return float(value)


def _write_strategy_predictions(path: Path, prediction_rows: Sequence[dict[str, Any]]) -> Path:
    lines = [json.dumps(row, sort_keys=True) for row in prediction_rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def _save_research_model(*, fitted_full: FittedSelectorExplorationStrategy, path_root: Path) -> str | None:
    path_root.parent.mkdir(parents=True, exist_ok=True)
    state_path = path_root.with_suffix(".json")
    state_path.write_text(json.dumps(fitted_full.model_summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return str(state_path)


def _format_metric(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _build_dense_control_trace_weight_multipliers(
    examples: Sequence[SelectorExample],
    *,
    dense_control_config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    if not bool(dense_control_config.get("enabled", False)):
        return {}, {"enabled": False}
    report_path = dense_control_config.get("report_path")
    if report_path in (None, ""):
        raise ValueError("dense_control.report_path is required when dense_control.enabled=true")
    dense_outcomes = _load_dense_control_group_outcomes(report_path)
    correct_weight = float(dense_control_config.get("correct_example_weight", 1.0))
    incorrect_weight = float(dense_control_config.get("incorrect_example_weight", 1.0))

    trace_weight_multipliers: dict[str, float] = {}
    matched_examples = 0
    matched_correct_examples = 0
    matched_incorrect_examples = 0
    unmatched_examples = 0
    for example in examples:
        group_key = _selector_dense_control_group_key(example)
        dense_success = dense_outcomes.get(group_key)
        if dense_success is True:
            trace_weight_multipliers[example.trace_path] = correct_weight
            matched_examples += 1
            matched_correct_examples += 1
        elif dense_success is False:
            trace_weight_multipliers[example.trace_path] = incorrect_weight
            matched_examples += 1
            matched_incorrect_examples += 1
        else:
            trace_weight_multipliers[example.trace_path] = 1.0
            unmatched_examples += 1
    return trace_weight_multipliers, {
        "enabled": True,
        "report_path": str(report_path),
        "correct_example_weight": correct_weight,
        "incorrect_example_weight": incorrect_weight,
        "matched_examples": matched_examples,
        "matched_correct_examples": matched_correct_examples,
        "matched_incorrect_examples": matched_incorrect_examples,
        "unmatched_examples": unmatched_examples,
        "dense_group_count": len(dense_outcomes),
    }


def _load_dense_control_group_outcomes(path: str | Path) -> dict[tuple[str, int | None], bool]:
    report_path = Path(path)
    if not report_path.exists():
        raise ValueError(f"dense control report does not exist: {report_path}")
    outcomes: dict[tuple[str, int | None], bool] = {}
    with report_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            selector_profile = str(row.get("selector_profile") or row.get("runtime_mode") or "").strip().lower()
            if selector_profile != "dense":
                continue
            measurement_kind = str(row.get("measurement_kind") or "").strip().lower()
            if measurement_kind not in {"aggregate", "trial"}:
                continue
            prompt_family = normalize_selector_categorical_token(row.get("task_family")) or ""
            if not prompt_family:
                continue
            prompt_length = selector_prompt_length_from_row(row)
            group_key = (prompt_family, prompt_length)
            if measurement_kind == "aggregate" or group_key not in outcomes:
                outcomes[group_key] = bool(row.get("task_success", False))
    return outcomes


def _selector_dense_control_group_key(example: SelectorExample) -> tuple[str, int | None]:
    return (
        normalize_selector_categorical_token(example.prompt_family) or "",
        example.prompt_length,
    )


def _fit_static_rule_strategy(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    feature_set_id: str,
    calibration_mode: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> FittedSelectorExplorationStrategy:
    del train_candidate_examples, calibration_mode, config, strategy_config
    model = train_static_rule_selector(train_examples)
    return FittedSelectorExplorationStrategy(
        strategy_id="static_rule",
        strategy_kind="row_multiclass",
        feature_set_id=feature_set_id,
        calibration_mode="global",
        runtime_compatible=False,
        artifact_capable=False,
        feature_names=(),
        predict_by_trace_fn=lambda examples, _candidate_examples: {
            example.trace_path: model.predict(example) for example in examples
        },
        save_model_fn=lambda _path: None,
        model_summary={"model_type": "static_rule"},
    )


def _fit_linear_softmax_strategy(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    feature_set_id: str,
    calibration_mode: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> FittedSelectorExplorationStrategy:
    del train_candidate_examples, calibration_mode
    weighted = dict(config["weighted_selector"])
    weighted.update(strategy_config)
    if feature_set_id == "runtime_safe":
        model = train_runtime_linear_selector(
            train_examples,
            steps=int(weighted["steps"]),
            learning_rate=float(weighted["learning_rate"]),
            l2=float(weighted["l2"]),
        )
    else:
        feature_names = selector_feature_names_from_examples(train_examples, feature_set_id=feature_set_id)
        model = train_linear_selector(
            train_examples,
            steps=int(weighted["steps"]),
            learning_rate=float(weighted["learning_rate"]),
            l2=float(weighted["l2"]),
            feature_names=feature_names,
        )
    return _wrap_linear_model(
        strategy_id="linear_softmax",
        model=model,
        feature_set_id=feature_set_id,
    )


def _fit_weighted_linear_strategy(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    feature_set_id: str,
    calibration_mode: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> FittedSelectorExplorationStrategy:
    del train_candidate_examples, calibration_mode
    weighted = dict(config["weighted_selector"])
    weighted.update(strategy_config)
    use_dense_control_weighting = bool(weighted.get("dense_control_weighting", False))
    trace_weight_multipliers = None
    dense_control_summary = None
    if use_dense_control_weighting:
        trace_weight_multipliers, dense_control_summary = _build_dense_control_trace_weight_multipliers(
            train_examples,
            dense_control_config=dict(config.get("dense_control") or {}),
        )
    feature_names = None if feature_set_id == "runtime_safe" else selector_feature_names_from_examples(train_examples, feature_set_id=feature_set_id)
    model = train_linear_selector(
        train_examples,
        steps=int(weighted["steps"]),
        learning_rate=float(weighted["learning_rate"]),
        l2=float(weighted["l2"]),
        feature_names=feature_names,
        class_balance=float(weighted["class_balance"]),
        safe_bytes_weight=float(weighted["safe_bytes_weight"]),
        unsafe_error_weight=float(weighted["unsafe_error_weight"]),
        reference_candidate=str(weighted["reference_candidate"]),
        trace_weight_multipliers=trace_weight_multipliers,
    )
    return _wrap_linear_model(
        strategy_id="linear_softmax_compression_weighted",
        model=model,
        feature_set_id=feature_set_id,
        model_summary={
            "dense_control_weighting": use_dense_control_weighting,
            "dense_control_summary": dense_control_summary,
        },
    )


def _fit_calibrated_linear_strategy(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    feature_set_id: str,
    calibration_mode: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> FittedSelectorExplorationStrategy:
    del train_candidate_examples, feature_set_id
    if calibration_mode != "global":
        raise ValueError("linear_softmax_compression_calibrated supports calibration_mode=global only")
    weighted = dict(config["weighted_selector"])
    weighted.update(strategy_config)
    calibration = _resolve_calibration_strategy_config(config=config, strategy_config=strategy_config)
    model, calibration_summary = train_calibrated_runtime_linear_selector(
        train_examples,
        steps=int(weighted["steps"]),
        learning_rate=float(weighted["learning_rate"]),
        l2=float(weighted["l2"]),
        class_balance=float(weighted["class_balance"]),
        safe_bytes_weight=float(weighted["safe_bytes_weight"]),
        unsafe_error_weight=float(weighted["unsafe_error_weight"]),
        reference_candidate=str(weighted["reference_candidate"]),
        calibration_fraction=float(calibration["fraction"]),
        calibration_seed=int(calibration["seed"]),
        calibration_target_candidate=str(weighted["target_candidate"]),
        calibration_offsets=tuple(calibration["linear_offsets"]),
        calibration_min_target_accuracy=calibration["min_target_accuracy"],
        calibration_min_safe_prediction_rate=float(calibration["min_safe_prediction_rate"]),
    )
    if model is None:
        model = train_runtime_linear_selector(
            train_examples,
            steps=int(weighted["steps"]),
            learning_rate=float(weighted["learning_rate"]),
            l2=float(weighted["l2"]),
            class_balance=float(weighted["class_balance"]),
            safe_bytes_weight=float(weighted["safe_bytes_weight"]),
            unsafe_error_weight=float(weighted["unsafe_error_weight"]),
            reference_candidate=str(weighted["reference_candidate"]),
        )
    fitted = _wrap_linear_model(
        strategy_id="linear_softmax_compression_calibrated",
        model=model,
        feature_set_id="runtime_safe",
    )
    fitted.model_summary["calibration"] = calibration_summary
    return fitted


def _fit_equal_tradeoff_linear_strategy(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    feature_set_id: str,
    calibration_mode: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> FittedSelectorExplorationStrategy:
    del train_candidate_examples, feature_set_id
    if calibration_mode != "global":
        raise ValueError("linear_softmax_compression_equal_tradeoff supports calibration_mode=global only")
    weighted = dict(config["weighted_selector"])
    weighted.update(strategy_config)
    calibration = _resolve_calibration_strategy_config(config=config, strategy_config=strategy_config)
    model, calibration_summary = train_calibrated_runtime_linear_selector(
        train_examples,
        steps=int(weighted["steps"]),
        learning_rate=float(weighted["learning_rate"]),
        l2=float(weighted["l2"]),
        class_balance=float(weighted["class_balance"]),
        safe_bytes_weight=float(weighted["safe_bytes_weight"]),
        unsafe_error_weight=float(weighted["unsafe_error_weight"]),
        reference_candidate=str(weighted["reference_candidate"]),
        calibration_fraction=float(calibration["fraction"]),
        calibration_seed=int(calibration["seed"]),
        calibration_target_candidate=str(weighted["target_candidate"]),
        calibration_offsets=tuple(calibration["linear_offsets"]),
        calibration_min_target_accuracy=calibration["min_target_accuracy"],
        calibration_min_safe_prediction_rate=float(calibration["min_safe_prediction_rate"]),
        calibration_objective="equal_tradeoff",
        calibration_correctness_weight=float(calibration["correctness_weight"]),
        calibration_bytes_weight=float(calibration["bytes_weight"]),
    )
    if model is None:
        model = train_runtime_linear_selector(
            train_examples,
            steps=int(weighted["steps"]),
            learning_rate=float(weighted["learning_rate"]),
            l2=float(weighted["l2"]),
            class_balance=float(weighted["class_balance"]),
            safe_bytes_weight=float(weighted["safe_bytes_weight"]),
            unsafe_error_weight=float(weighted["unsafe_error_weight"]),
            reference_candidate=str(weighted["reference_candidate"]),
        )
    fitted = _wrap_linear_model(
        strategy_id="linear_softmax_compression_equal_tradeoff",
        model=model,
        feature_set_id="runtime_safe",
    )
    fitted.model_summary["calibration"] = calibration_summary
    return fitted


def _resolve_calibration_strategy_config(
    *,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> dict[str, Any]:
    calibration = dict(config["calibration"])
    override_keys = (
        "fraction",
        "seed",
        "min_target_accuracy",
        "min_safe_prediction_rate",
        "linear_offsets",
        "binary_thresholds",
        "correctness_weight",
        "bytes_weight",
    )
    for key in override_keys:
        strategy_key = f"calibration_{key}"
        if strategy_key in strategy_config:
            calibration[key] = strategy_config[strategy_key]
    return calibration


def _fit_distilled_linear_mlp_teacher_strategy(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    feature_set_id: str,
    calibration_mode: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> FittedSelectorExplorationStrategy:
    del calibration_mode
    if torch is None or F is None:
        raise _StrategyUnavailableError("torch is not available for linear_softmax_distilled_mlp_teacher")
    if feature_set_id != "runtime_safe":
        raise ValueError("linear_softmax_distilled_mlp_teacher requires feature_set_id=runtime_safe")

    weighted = dict(config["weighted_selector"])
    distillation = dict(config["distillation"])
    distillation.update(strategy_config)

    teacher_feature_set_id = str(distillation["teacher_feature_set_id"])
    teacher_feature_names = candidate_feature_names_from_examples(
        train_candidate_examples,
        feature_set_id=teacher_feature_set_id,
    )
    teacher_model = _train_candidate_target_mlp(
        train_candidate_examples,
        feature_names=teacher_feature_names,
        hidden_dim=int(distillation["teacher_hidden_dim"]),
        epochs=int(distillation["teacher_epochs"]),
        learning_rate=float(distillation["teacher_learning_rate"]),
        seed=int(distillation["teacher_seed"]),
    )
    teacher_probabilities_by_trace = _build_candidate_teacher_probabilities_by_trace(
        train_candidate_examples,
        predict_probability_for_row=teacher_model.predict_probability_for_row,
        temperature=float(distillation["teacher_temperature"]),
    )
    teacher_predictions = _argmax_distribution_predictions(
        train_examples,
        teacher_probabilities_by_trace=teacher_probabilities_by_trace,
    )
    teacher_summary = evaluate_selector_model(_PredictedSelectorModel(teacher_predictions), train_examples)

    student_feature_names = selector_feature_names_from_examples(train_examples, feature_set_id="runtime_safe")
    student_model = _train_distilled_linear_selector(
        train_examples,
        teacher_probabilities_by_trace=teacher_probabilities_by_trace,
        feature_names=student_feature_names,
        steps=int(weighted["steps"]),
        learning_rate=float(weighted["learning_rate"]),
        l2=float(weighted["l2"]),
        class_balance=float(weighted["class_balance"]),
        safe_bytes_weight=float(weighted["safe_bytes_weight"]),
        unsafe_error_weight=float(weighted["unsafe_error_weight"]),
        reference_candidate=str(weighted["reference_candidate"]),
        teacher_weight=float(distillation["teacher_weight"]),
    )
    return _wrap_linear_model(
        strategy_id="linear_softmax_distilled_mlp_teacher",
        model=student_model,
        feature_set_id="runtime_safe",
        model_summary={
            "model_type": "linear_selector_model_distilled",
            "classes": list(student_model.classes),
            "teacher_model_type": "candidate_target_mlp",
            "teacher_feature_set_id": teacher_feature_set_id,
            "teacher_feature_names": list(teacher_feature_names),
            "teacher_hidden_dim": int(distillation["teacher_hidden_dim"]),
            "teacher_epochs": int(distillation["teacher_epochs"]),
            "teacher_learning_rate": float(distillation["teacher_learning_rate"]),
            "teacher_seed": int(distillation["teacher_seed"]),
            "teacher_weight": float(distillation["teacher_weight"]),
            "teacher_temperature": float(distillation["teacher_temperature"]),
            "teacher_train_target_accuracy": float(teacher_summary.target_accuracy),
            "teacher_train_safe_prediction_rate": float(teacher_summary.safe_prediction_rate),
        },
    )


def _fit_candidate_safe_strategy(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    feature_set_id: str,
    calibration_mode: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> FittedSelectorExplorationStrategy:
    feature_names = candidate_feature_names_from_examples(train_candidate_examples, feature_set_id=feature_set_id)
    params = dict(config["weighted_selector"])
    params.update(strategy_config)
    calibration = dict(config["calibration"])
    split_train_candidate_examples, calibration_candidate_examples, split_train_examples, calibration_examples = _split_candidate_calibration_examples(
        train_examples=train_examples,
        train_candidate_examples=train_candidate_examples,
        fraction=float(calibration["fraction"]),
        seed=int(calibration["seed"]),
    )
    base_model = train_candidate_safe_linear_selector(
        split_train_candidate_examples or train_candidate_examples,
        steps=int(params["steps"]),
        learning_rate=float(params["learning_rate"]),
        l2=float(params["l2"]),
        feature_names=feature_names,
    )
    default_threshold, family_thresholds = _calibrate_candidate_thresholds(
        scorer_kind="candidate_safe",
        model=base_model,
        examples=calibration_examples or train_examples,
        candidate_examples=calibration_candidate_examples or train_candidate_examples,
        thresholds=tuple(calibration["binary_thresholds"]),
        calibration_mode=calibration_mode,
        min_target_accuracy=calibration["min_target_accuracy"],
        min_safe_prediction_rate=float(calibration["min_safe_prediction_rate"]),
        fallback_candidate=str(params["target_candidate"]),
    )
    full_model = train_candidate_safe_linear_selector(
        train_candidate_examples,
        steps=int(params["steps"]),
        learning_rate=float(params["learning_rate"]),
        l2=float(params["l2"]),
        feature_names=feature_names,
    )
    full_model.decision_threshold = float(default_threshold)
    candidate_tokens = _candidate_tokens_from_candidate_examples(train_candidate_examples)
    return _wrap_candidate_router_strategy(
        strategy_id="candidate_safe_router",
        strategy_kind="candidate_safe",
        feature_set_id=feature_set_id,
        calibration_mode=calibration_mode,
        binary_model=full_model,
        candidate_tokens=candidate_tokens,
        fallback_candidate=str(params["target_candidate"]),
        prompt_family_thresholds=family_thresholds,
        predict_probability_for_row=full_model.predict_probability_for_row,
    )


def _fit_candidate_target_linear_strategy(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    feature_set_id: str,
    calibration_mode: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> FittedSelectorExplorationStrategy:
    feature_names = candidate_feature_names_from_examples(train_candidate_examples, feature_set_id=feature_set_id)
    params = dict(config["weighted_selector"])
    params.update(strategy_config)
    candidate_target_params = {
        "loss_kind": str(params.get("loss_kind", "binary")),
        "class_balance": float(params.get("candidate_class_balance", params.get("class_balance", 0.0))),
        "reference_candidate": str(params.get("reference_candidate", params.get("target_candidate", DEFAULT_FALLBACK_CANDIDATE))),
        "non_reference_target_weight": float(params.get("non_reference_target_weight", 0.0)),
        "compression_target_weight": float(params.get("compression_target_weight", 0.0)),
        "reference_false_positive_weight": float(params.get("reference_false_positive_weight", 0.0)),
        "reference_logit_offset": float(params.get("reference_logit_offset", 0.0)),
    }
    calibration = dict(config["calibration"])
    split_train_candidate_examples, calibration_candidate_examples, split_train_examples, calibration_examples = _split_candidate_calibration_examples(
        train_examples=train_examples,
        train_candidate_examples=train_candidate_examples,
        fraction=float(calibration["fraction"]),
        seed=int(calibration["seed"]),
    )
    base_model = train_candidate_target_linear_selector(
        split_train_candidate_examples or train_candidate_examples,
        steps=int(params["steps"]),
        learning_rate=float(params["learning_rate"]),
        l2=float(params["l2"]),
        feature_names=feature_names,
        loss_kind=str(candidate_target_params["loss_kind"]),
        class_balance=float(candidate_target_params["class_balance"]),
        reference_candidate=str(candidate_target_params["reference_candidate"]),
        non_reference_target_weight=float(candidate_target_params["non_reference_target_weight"]),
        compression_target_weight=float(candidate_target_params["compression_target_weight"]),
        reference_false_positive_weight=float(candidate_target_params["reference_false_positive_weight"]),
    )
    default_threshold, family_thresholds = _calibrate_candidate_thresholds(
        scorer_kind="candidate_target",
        model=base_model,
        examples=calibration_examples or train_examples,
        candidate_examples=calibration_candidate_examples or train_candidate_examples,
        thresholds=tuple(calibration["binary_thresholds"]),
        calibration_mode=calibration_mode,
        min_target_accuracy=calibration["min_target_accuracy"],
        min_safe_prediction_rate=float(calibration["min_safe_prediction_rate"]),
        fallback_candidate=str(params["target_candidate"]),
    )
    full_model = train_candidate_target_linear_selector(
        train_candidate_examples,
        steps=int(params["steps"]),
        learning_rate=float(params["learning_rate"]),
        l2=float(params["l2"]),
        feature_names=feature_names,
        loss_kind=str(candidate_target_params["loss_kind"]),
        class_balance=float(candidate_target_params["class_balance"]),
        reference_candidate=str(candidate_target_params["reference_candidate"]),
        non_reference_target_weight=float(candidate_target_params["non_reference_target_weight"]),
        compression_target_weight=float(candidate_target_params["compression_target_weight"]),
        reference_false_positive_weight=float(candidate_target_params["reference_false_positive_weight"]),
    )
    full_model.decision_threshold = float(default_threshold)
    candidate_tokens = _candidate_tokens_from_candidate_examples(train_candidate_examples)
    candidate_logit_offsets = {
        str(candidate_target_params["reference_candidate"]): float(candidate_target_params["reference_logit_offset"])
    }
    return _wrap_candidate_router_strategy(
        strategy_id="candidate_target_linear",
        strategy_kind="candidate_target",
        feature_set_id=feature_set_id,
        calibration_mode=calibration_mode,
        binary_model=full_model,
        candidate_tokens=candidate_tokens,
        fallback_candidate=str(params["target_candidate"]),
        prompt_family_thresholds=family_thresholds,
        predict_probability_for_row=full_model.predict_probability_for_row,
        candidate_logit_offsets=candidate_logit_offsets,
        model_summary={
            "feature_names": list(feature_names),
            "loss_kind": str(candidate_target_params["loss_kind"]),
            "candidate_class_balance": float(candidate_target_params["class_balance"]),
            "reference_candidate": str(candidate_target_params["reference_candidate"]),
            "non_reference_target_weight": float(candidate_target_params["non_reference_target_weight"]),
            "compression_target_weight": float(candidate_target_params["compression_target_weight"]),
            "reference_false_positive_weight": float(candidate_target_params["reference_false_positive_weight"]),
            "reference_logit_offset": float(candidate_target_params["reference_logit_offset"]),
        },
    )


def _fit_candidate_target_mlp_strategy(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    feature_set_id: str,
    calibration_mode: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> FittedSelectorExplorationStrategy:
    if torch is None or F is None:
        raise _StrategyUnavailableError("torch is not available for candidate_target_mlp")
    feature_names = candidate_feature_names_from_examples(train_candidate_examples, feature_set_id=feature_set_id)
    params = {"hidden_dim": 16, "epochs": 200, "learning_rate": 1e-2, "seed": 0}
    params.update(strategy_config)
    calibration = dict(config["calibration"])
    split_train_candidate_examples, calibration_candidate_examples, split_train_examples, calibration_examples = _split_candidate_calibration_examples(
        train_examples=train_examples,
        train_candidate_examples=train_candidate_examples,
        fraction=float(calibration["fraction"]),
        seed=int(calibration["seed"]),
    )
    base_model = _train_candidate_target_mlp(
        split_train_candidate_examples or train_candidate_examples,
        feature_names=feature_names,
        hidden_dim=int(params["hidden_dim"]),
        epochs=int(params["epochs"]),
        learning_rate=float(params["learning_rate"]),
        seed=int(params["seed"]),
    )
    default_threshold, family_thresholds = _calibrate_candidate_thresholds(
        scorer_kind="candidate_target",
        model=base_model,
        examples=calibration_examples or train_examples,
        candidate_examples=calibration_candidate_examples or train_candidate_examples,
        thresholds=tuple(calibration["binary_thresholds"]),
        calibration_mode=calibration_mode,
        min_target_accuracy=calibration["min_target_accuracy"],
        min_safe_prediction_rate=float(calibration["min_safe_prediction_rate"]),
        fallback_candidate=DEFAULT_FALLBACK_CANDIDATE,
    )
    full_model = _train_candidate_target_mlp(
        train_candidate_examples,
        feature_names=feature_names,
        hidden_dim=int(params["hidden_dim"]),
        epochs=int(params["epochs"]),
        learning_rate=float(params["learning_rate"]),
        seed=int(params["seed"]),
    )
    candidate_tokens = _candidate_tokens_from_candidate_examples(train_candidate_examples)
    return _wrap_candidate_router_strategy(
        strategy_id="candidate_target_mlp",
        strategy_kind="candidate_target",
        feature_set_id=feature_set_id,
        calibration_mode=calibration_mode,
        binary_model=full_model,
        candidate_tokens=candidate_tokens,
        fallback_candidate=DEFAULT_FALLBACK_CANDIDATE,
        prompt_family_thresholds=family_thresholds,
        predict_probability_for_row=full_model.predict_probability_for_row,
        model_summary={
            "research_model_type": "candidate_target_mlp",
            "feature_names": list(feature_names),
            "hidden_dim": int(params["hidden_dim"]),
            "decision_threshold": float(default_threshold),
            "prompt_family_thresholds": dict(family_thresholds),
        },
    )


def _fit_candidate_target_gbdt_strategy(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    feature_set_id: str,
    calibration_mode: str,
    config: dict[str, Any],
    strategy_config: dict[str, Any],
) -> FittedSelectorExplorationStrategy:
    if GradientBoostingClassifier is None:
        raise _StrategyUnavailableError("scikit-learn is not available for candidate_target_gbdt")
    feature_names = candidate_feature_names_from_examples(train_candidate_examples, feature_set_id=feature_set_id)
    params = {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1, "random_state": 0}
    params.update(strategy_config)
    calibration = dict(config["calibration"])
    split_train_candidate_examples, calibration_candidate_examples, split_train_examples, calibration_examples = _split_candidate_calibration_examples(
        train_examples=train_examples,
        train_candidate_examples=train_candidate_examples,
        fraction=float(calibration["fraction"]),
        seed=int(calibration["seed"]),
    )
    base_model = _train_candidate_target_gbdt(
        split_train_candidate_examples or train_candidate_examples,
        feature_names=feature_names,
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        random_state=int(params["random_state"]),
    )
    default_threshold, family_thresholds = _calibrate_candidate_thresholds(
        scorer_kind="candidate_target",
        model=base_model,
        examples=calibration_examples or train_examples,
        candidate_examples=calibration_candidate_examples or train_candidate_examples,
        thresholds=tuple(calibration["binary_thresholds"]),
        calibration_mode=calibration_mode,
        min_target_accuracy=calibration["min_target_accuracy"],
        min_safe_prediction_rate=float(calibration["min_safe_prediction_rate"]),
        fallback_candidate=DEFAULT_FALLBACK_CANDIDATE,
    )
    full_model = _train_candidate_target_gbdt(
        train_candidate_examples,
        feature_names=feature_names,
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        random_state=int(params["random_state"]),
    )
    candidate_tokens = _candidate_tokens_from_candidate_examples(train_candidate_examples)
    return _wrap_candidate_router_strategy(
        strategy_id="candidate_target_gbdt",
        strategy_kind="candidate_target",
        feature_set_id=feature_set_id,
        calibration_mode=calibration_mode,
        binary_model=full_model,
        candidate_tokens=candidate_tokens,
        fallback_candidate=DEFAULT_FALLBACK_CANDIDATE,
        prompt_family_thresholds=family_thresholds,
        predict_probability_for_row=full_model.predict_probability_for_row,
        model_summary={
            "research_model_type": "candidate_target_gbdt",
            "feature_names": list(feature_names),
            "decision_threshold": float(default_threshold),
            "prompt_family_thresholds": dict(family_thresholds),
        },
    )


def _wrap_linear_model(
    *,
    strategy_id: str,
    model: LinearSelectorModel,
    feature_set_id: str,
    model_summary: dict[str, Any] | None = None,
) -> FittedSelectorExplorationStrategy:
    runtime_compatible = feature_set_id == "runtime_safe"
    resolved_summary = {"model_type": "linear_selector_model", "classes": list(model.classes)}
    if model_summary is not None:
        resolved_summary.update(model_summary)
    return FittedSelectorExplorationStrategy(
        strategy_id=strategy_id,
        strategy_kind="row_multiclass",
        feature_set_id=feature_set_id,
        calibration_mode="global",
        runtime_compatible=runtime_compatible,
        artifact_capable=runtime_compatible,
        feature_names=tuple(model.feature_names),
        predict_by_trace_fn=lambda examples, _candidate_examples: {
            example.trace_path: model.predict(example) for example in examples
        },
        save_model_fn=(
            (lambda path: _save_artifact_model(model, path))
            if runtime_compatible
            else (lambda _path: None)
        ),
        model_summary=resolved_summary,
    )


def _wrap_candidate_router_strategy(
    *,
    strategy_id: str,
    strategy_kind: str,
    feature_set_id: str,
    calibration_mode: str,
    binary_model: Any,
    candidate_tokens: Sequence[str],
    fallback_candidate: str,
    prompt_family_thresholds: dict[str, float],
    predict_probability_for_row: Callable[[dict[str, Any]], float],
    candidate_logit_offsets: dict[str, float] | None = None,
    model_summary: dict[str, Any] | None = None,
) -> FittedSelectorExplorationStrategy:
    candidate_token_tuple = tuple(str(token) for token in candidate_tokens)
    default_threshold = float(getattr(binary_model, "decision_threshold", 0.5))
    resolved_candidate_logit_offsets = (
        {}
        if candidate_logit_offsets is None
        else {str(key): float(value) for key, value in candidate_logit_offsets.items()}
    )
    runtime_compatible = bool(
        feature_set_id == "runtime_safe"
        and isinstance(binary_model, (CandidateSafeLinearSelectorModel, CandidateTargetLinearSelectorModel))
    )
    artifact_capable = runtime_compatible

    def _predict_by_trace(
        _examples: Sequence[SelectorExample],
        candidate_examples: Sequence[SelectorCandidateExample],
    ) -> dict[str, str | None]:
        return _route_candidate_examples(
            candidate_examples,
            strategy_kind=strategy_kind,
            predict_probability_for_row=predict_probability_for_row,
            fallback_candidate=fallback_candidate,
            default_threshold=default_threshold,
            prompt_family_thresholds=prompt_family_thresholds,
            candidate_logit_offsets=resolved_candidate_logit_offsets,
        )

    def _save(path: Path) -> str | None:
        if not artifact_capable:
            return None
        if isinstance(binary_model, CandidateSafeLinearSelectorModel):
            artifact = CandidateSafeRouterModel(
                safe_model=binary_model,
                candidate_tokens=candidate_token_tuple,
                fallback_candidate=fallback_candidate,
                prompt_family_thresholds=prompt_family_thresholds,
            )
        else:
            artifact = CandidateTargetRouterModel(
                target_model=binary_model,
                candidate_tokens=candidate_token_tuple,
                fallback_candidate=fallback_candidate,
                prompt_family_thresholds=prompt_family_thresholds,
                candidate_logit_offsets=resolved_candidate_logit_offsets,
            )
        return _save_artifact_model(artifact, path)

    resolved_summary = {
        "model_type": "candidate_router",
        "candidate_tokens": list(candidate_token_tuple),
        "fallback_candidate": str(fallback_candidate),
        "decision_threshold": default_threshold,
        "prompt_family_thresholds": dict(prompt_family_thresholds),
        "candidate_logit_offsets": dict(resolved_candidate_logit_offsets),
    }
    if model_summary is not None:
        resolved_summary.update(model_summary)

    return FittedSelectorExplorationStrategy(
        strategy_id=strategy_id,
        strategy_kind=strategy_kind,
        feature_set_id=feature_set_id,
        calibration_mode=calibration_mode,
        runtime_compatible=runtime_compatible,
        artifact_capable=artifact_capable,
        feature_names=tuple(getattr(binary_model, "feature_names", ())),
        predict_by_trace_fn=_predict_by_trace,
        save_model_fn=_save,
        model_summary=resolved_summary,
    )


def _save_artifact_model(model: Any, path: Path) -> str:
    save_page_selector_artifact(model, path)
    return str(path)


def _split_candidate_calibration_examples(
    *,
    train_examples: Sequence[SelectorExample],
    train_candidate_examples: Sequence[SelectorCandidateExample],
    fraction: float,
    seed: int,
) -> tuple[list[SelectorCandidateExample], list[SelectorCandidateExample], list[SelectorExample], list[SelectorExample]]:
    if len(train_examples) < 2 or float(fraction) <= 0.0:
        return list(train_candidate_examples), [], list(train_examples), []
    split = split_selector_examples(train_examples, test_fraction=float(fraction), seed=int(seed))
    if not split.train_indices or not split.test_indices:
        return list(train_candidate_examples), [], list(train_examples), []
    train_subset = [train_examples[index] for index in split.train_indices]
    calibration_subset = [train_examples[index] for index in split.test_indices]
    train_trace_paths = {example.trace_path for example in train_subset}
    calibration_trace_paths = {example.trace_path for example in calibration_subset}
    candidate_train_subset = [example for example in train_candidate_examples if example.trace_path in train_trace_paths]
    candidate_calibration_subset = [example for example in train_candidate_examples if example.trace_path in calibration_trace_paths]
    return candidate_train_subset, candidate_calibration_subset, train_subset, calibration_subset


def _candidate_tokens_from_candidate_examples(examples: Sequence[SelectorCandidateExample]) -> tuple[str, ...]:
    tokens = sorted(
        {
            str(example.candidate)
            for example in examples
        }
    )
    return tuple(tokens)


def _route_candidate_examples(
    candidate_examples: Sequence[SelectorCandidateExample],
    *,
    strategy_kind: str,
    predict_probability_for_row: Callable[[dict[str, Any]], float],
    fallback_candidate: str,
    default_threshold: float,
    prompt_family_thresholds: dict[str, float],
    candidate_logit_offsets: dict[str, float] | None = None,
) -> dict[str, str | None]:
    resolved_candidate_logit_offsets = (
        {}
        if candidate_logit_offsets is None
        else {str(key): float(value) for key, value in candidate_logit_offsets.items()}
    )
    grouped: dict[str, list[SelectorCandidateExample]] = defaultdict(list)
    for example in candidate_examples:
        grouped[example.trace_path].append(example)
    predictions: dict[str, str | None] = {}
    for trace_path, group in grouped.items():
        ordered = sorted(group, key=lambda example: (example.candidate_total_bytes, example.candidate))
        normalized_family = normalize_selector_categorical_token(ordered[0].prompt_family) or ""
        threshold = float(prompt_family_thresholds.get(normalized_family, default_threshold))
        scored = [
            (
                example,
                _apply_candidate_logit_offset(
                    float(predict_probability_for_row(example.row)),
                    resolved_candidate_logit_offsets.get(str(example.candidate), 0.0),
                ),
            )
            for example in ordered
        ]
        if strategy_kind == "candidate_safe":
            viable = [item for item in scored if item[1] >= threshold]
            if viable:
                viable.sort(key=lambda item: (int(item[0].candidate_total_bytes), -float(item[1]), str(item[0].candidate)))
                predictions[trace_path] = str(viable[0][0].candidate)
            else:
                predictions[trace_path] = str(fallback_candidate)
            continue
        viable = [item for item in scored if item[1] >= threshold]
        if viable:
            viable.sort(key=lambda item: (-float(item[1]), int(item[0].candidate_total_bytes), str(item[0].candidate)))
            predictions[trace_path] = str(viable[0][0].candidate)
        else:
            predictions[trace_path] = str(fallback_candidate)
    return predictions


def _calibrate_candidate_thresholds(
    *,
    scorer_kind: str,
    model: Any,
    examples: Sequence[SelectorExample],
    candidate_examples: Sequence[SelectorCandidateExample],
    thresholds: Sequence[float],
    calibration_mode: str,
    min_target_accuracy: float | None,
    min_safe_prediction_rate: float,
    fallback_candidate: str,
) -> tuple[float, dict[str, float]]:
    if not examples or not candidate_examples or not thresholds:
        return float(getattr(model, "decision_threshold", 0.5)), {}
    if calibration_mode == "global":
        best = _select_best_candidate_threshold(
            scorer_kind=scorer_kind,
            model=model,
            examples=examples,
            candidate_examples=candidate_examples,
            thresholds=thresholds,
            min_target_accuracy=min_target_accuracy,
            min_safe_prediction_rate=min_safe_prediction_rate,
            fallback_candidate=fallback_candidate,
        )
        return float(best), {}

    family_thresholds: dict[str, float] = {}
    for normalized_family in sorted(
        {
            normalize_selector_categorical_token(example.prompt_family) or ""
            for example in examples
        }
    ):
        family_examples = [
            example
            for example in examples
            if (normalize_selector_categorical_token(example.prompt_family) or "") == normalized_family
        ]
        family_trace_paths = {example.trace_path for example in family_examples}
        family_candidate_examples = [
            example for example in candidate_examples if example.trace_path in family_trace_paths
        ]
        if not family_examples or not family_candidate_examples:
            continue
        family_thresholds[normalized_family] = _select_best_candidate_threshold(
            scorer_kind=scorer_kind,
            model=model,
            examples=family_examples,
            candidate_examples=family_candidate_examples,
            thresholds=thresholds,
            min_target_accuracy=min_target_accuracy,
            min_safe_prediction_rate=min_safe_prediction_rate,
            fallback_candidate=fallback_candidate,
        )
    return float(getattr(model, "decision_threshold", 0.5)), family_thresholds


def _select_best_candidate_threshold(
    *,
    scorer_kind: str,
    model: Any,
    examples: Sequence[SelectorExample],
    candidate_examples: Sequence[SelectorCandidateExample],
    thresholds: Sequence[float],
    min_target_accuracy: float | None,
    min_safe_prediction_rate: float,
    fallback_candidate: str,
) -> float:
    evaluations: list[tuple[float, SelectorEvaluationSummary]] = []
    feasible: list[tuple[float, SelectorEvaluationSummary]] = []
    for threshold in thresholds:
        predicted_by_trace = _route_candidate_examples(
            candidate_examples,
            strategy_kind=scorer_kind,
            predict_probability_for_row=model.predict_probability_for_row,
            fallback_candidate=fallback_candidate,
            default_threshold=float(threshold),
            prompt_family_thresholds={},
        )
        summary = evaluate_selector_model(_PredictedSelectorModel(predicted_by_trace), examples)
        evaluations.append((float(threshold), summary))
        meets_accuracy = min_target_accuracy is None or float(summary.target_accuracy) >= float(min_target_accuracy)
        meets_safety = float(summary.safe_prediction_rate) >= float(min_safe_prediction_rate)
        if meets_accuracy and meets_safety:
            feasible.append((float(threshold), summary))
    candidates = feasible if feasible else evaluations
    best_threshold, _ = min(
        candidates,
        key=lambda item: (
            float("inf") if item[1].mean_predicted_total_bytes is None else float(item[1].mean_predicted_total_bytes),
            -float(item[1].target_accuracy),
            -float(item[1].safe_prediction_rate),
            float(item[0]),
        ),
    )
    return float(best_threshold)


def _candidate_row_features(row: dict[str, Any], *, feature_names: Sequence[str]) -> np.ndarray:
    return selector_candidate_feature_vector_from_row(row, feature_names=feature_names)


def _build_candidate_teacher_probabilities_by_trace(
    candidate_examples: Sequence[SelectorCandidateExample],
    *,
    predict_probability_for_row: Callable[[dict[str, Any]], float],
    temperature: float,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[SelectorCandidateExample]] = defaultdict(list)
    for example in candidate_examples:
        grouped[example.trace_path].append(example)

    resolved_temperature = max(float(temperature), 1e-3)
    probabilities_by_trace: dict[str, dict[str, float]] = {}
    for trace_path, group in grouped.items():
        scores: dict[str, float] = {}
        for example in group:
            probability = float(np.clip(predict_probability_for_row(example.row), 1e-6, 1.0 - 1e-6))
            logit = float(np.log(probability) - np.log1p(-probability))
            score = float(np.exp(logit / resolved_temperature))
            scores[str(example.candidate)] = score
        score_sum = float(sum(scores.values()))
        if score_sum <= 0.0:
            ordered = sorted(group, key=lambda item: (int(item.candidate_total_bytes), str(item.candidate)))
            if not ordered:
                probabilities_by_trace[trace_path] = {}
            else:
                probabilities_by_trace[trace_path] = {str(ordered[0].candidate): 1.0}
            continue
        probabilities_by_trace[trace_path] = {
            candidate: float(score / score_sum)
            for candidate, score in scores.items()
        }
    return probabilities_by_trace


def _argmax_distribution_predictions(
    examples: Sequence[SelectorExample],
    *,
    teacher_probabilities_by_trace: dict[str, dict[str, float]],
) -> dict[str, str | None]:
    predictions: dict[str, str | None] = {}
    for example in examples:
        candidate_distribution = dict(teacher_probabilities_by_trace.get(example.trace_path) or {})
        if not candidate_distribution:
            predictions[example.trace_path] = example.target_candidate
            continue
        predictions[example.trace_path] = max(
            candidate_distribution.items(),
            key=lambda item: (
                float(item[1]),
                -float(example.candidate_map.get(str(item[0]), {}).get("total_bytes", float("inf"))),
                str(item[0]),
            ),
        )[0]
    return predictions


def _train_distilled_linear_selector(
    examples: Sequence[SelectorExample],
    *,
    teacher_probabilities_by_trace: dict[str, dict[str, float]],
    feature_names: Sequence[str],
    steps: int,
    learning_rate: float,
    l2: float,
    class_balance: float,
    safe_bytes_weight: float,
    unsafe_error_weight: float,
    reference_candidate: str,
    teacher_weight: float,
) -> LinearSelectorModel:
    target_examples = [example for example in examples if example.target_present and example.target_candidate is not None]
    resolved_feature_names = tuple(str(value) for value in feature_names)
    classes = tuple(
        sorted(
            {
                str(example.target_candidate)
                for example in target_examples
                if example.target_candidate is not None
            }
            | {
                str(candidate)
                for distribution in teacher_probabilities_by_trace.values()
                for candidate in distribution.keys()
            }
        )
    )
    if not target_examples or not classes:
        feature_dim = len(resolved_feature_names)
        return LinearSelectorModel(
            classes=(),
            weight=np.zeros((feature_dim, 0), dtype=np.float32),
            bias=np.zeros((0,), dtype=np.float32),
            feature_mean=np.zeros((feature_dim,), dtype=np.float32),
            feature_std=np.ones((feature_dim,), dtype=np.float32),
            feature_names=resolved_feature_names,
        )

    class_to_index = {candidate: index for index, candidate in enumerate(classes)}
    x = np.stack(
        [
            selector_feature_vector_from_row(example.row, feature_names=resolved_feature_names)
            for example in target_examples
        ],
        axis=0,
    ).astype(np.float32)
    y = np.array([class_to_index[str(example.target_candidate)] for example in target_examples], dtype=np.int32)
    hard_targets = np.eye(len(classes), dtype=np.float32)[y]
    teacher_targets = np.zeros_like(hard_targets)
    for row_index, example in enumerate(target_examples):
        teacher_distribution = dict(teacher_probabilities_by_trace.get(example.trace_path) or {})
        total_mass = 0.0
        for candidate, probability in teacher_distribution.items():
            class_index = class_to_index.get(str(candidate))
            if class_index is None:
                continue
            clipped = max(float(probability), 0.0)
            teacher_targets[row_index, class_index] += clipped
            total_mass += clipped
        if total_mass <= 0.0:
            teacher_targets[row_index] = hard_targets[row_index]
        else:
            teacher_targets[row_index] /= float(total_mass)

    example_weights = build_selector_example_weights(
        target_examples,
        classes=classes,
        class_balance=class_balance,
        safe_bytes_weight=safe_bytes_weight,
        reference_candidate=reference_candidate,
    )
    class_error_weights = build_selector_class_error_weights(
        target_examples,
        classes=classes,
        unsafe_error_weight=unsafe_error_weight,
    )
    example_weight_sum = float(np.sum(example_weights, dtype=np.float32))
    if example_weight_sum <= 0.0:
        example_weights = np.ones((len(target_examples),), dtype=np.float32)
        example_weight_sum = float(len(target_examples))

    feature_mean = np.mean(x, axis=0, dtype=np.float32)
    feature_std = np.std(x, axis=0, dtype=np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    x_std = (x - feature_mean) / feature_std

    weight = np.zeros((x_std.shape[1], len(classes)), dtype=np.float32)
    bias = np.zeros((len(classes),), dtype=np.float32)
    resolved_teacher_weight = min(max(float(teacher_weight), 0.0), 1.0)
    resolved_oracle_weight = max(1.0 - resolved_teacher_weight, 0.0)
    if resolved_teacher_weight <= 0.0 and resolved_oracle_weight <= 0.0:
        resolved_oracle_weight = 1.0

    for _ in range(int(steps)):
        logits = x_std @ weight + bias
        probs = _softmax_rows(logits)
        hard_error = (probs - hard_targets) * class_error_weights
        teacher_error = probs - teacher_targets
        combined_error = (
            resolved_oracle_weight * hard_error
            + resolved_teacher_weight * teacher_error
        )
        weighted_error = combined_error * example_weights[:, None]
        grad_weight = (x_std.T @ weighted_error) / max(example_weight_sum, 1.0) + float(l2) * weight
        grad_bias = np.sum(weighted_error, axis=0, dtype=np.float32) / max(example_weight_sum, 1.0)
        weight -= float(learning_rate) * grad_weight.astype(np.float32)
        bias -= float(learning_rate) * grad_bias.astype(np.float32)

    return LinearSelectorModel(
        classes=classes,
        weight=weight,
        bias=bias,
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=resolved_feature_names,
    )


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    stabilized = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(stabilized).astype(np.float32, copy=False)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _train_candidate_target_mlp(
    examples: Sequence[SelectorCandidateExample],
    *,
    feature_names: Sequence[str],
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> CandidateTargetMlpModel:
    if torch is None or F is None:
        raise _StrategyUnavailableError("torch is not available for candidate_target_mlp")
    if not examples:
        feature_dim = len(tuple(feature_names))
        return CandidateTargetMlpModel(
            weight_1=np.zeros((feature_dim, hidden_dim), dtype=np.float32),
            bias_1=np.zeros((hidden_dim,), dtype=np.float32),
            weight_2=np.zeros((hidden_dim,), dtype=np.float32),
            bias_2=0.0,
            feature_mean=np.zeros((feature_dim,), dtype=np.float32),
            feature_std=np.ones((feature_dim,), dtype=np.float32),
            feature_names=tuple(str(value) for value in feature_names),
        )
    rng = np.random.default_rng(int(seed))
    x = np.stack([_candidate_row_features(example.row, feature_names=feature_names) for example in examples], axis=0).astype(np.float32)
    y = np.asarray([1.0 if bool(example.row.get("candidate_is_target", False)) else 0.0 for example in examples], dtype=np.float32)
    feature_mean = np.mean(x, axis=0, dtype=np.float32)
    feature_std = np.std(x, axis=0, dtype=np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    x_std = (x - feature_mean) / feature_std

    x_tensor = torch.tensor(x_std, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    weight_1 = torch.tensor(rng.normal(scale=0.05, size=(x_std.shape[1], hidden_dim)), dtype=torch.float32, requires_grad=True)
    bias_1 = torch.zeros((hidden_dim,), dtype=torch.float32, requires_grad=True)
    weight_2 = torch.tensor(rng.normal(scale=0.05, size=(hidden_dim,)), dtype=torch.float32, requires_grad=True)
    bias_2 = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weight_1, bias_1, weight_2, bias_2], lr=float(learning_rate))
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        hidden = torch.relu(x_tensor @ weight_1 + bias_1)
        logits = hidden @ weight_2 + bias_2
        loss = F.binary_cross_entropy_with_logits(logits, y_tensor)
        loss.backward()
        optimizer.step()
    return CandidateTargetMlpModel(
        weight_1=weight_1.detach().cpu().numpy().astype(np.float32),
        bias_1=bias_1.detach().cpu().numpy().astype(np.float32),
        weight_2=weight_2.detach().cpu().numpy().astype(np.float32),
        bias_2=float(bias_2.detach().cpu().item()),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        feature_names=tuple(str(value) for value in feature_names),
    )


def _train_candidate_target_gbdt(
    examples: Sequence[SelectorCandidateExample],
    *,
    feature_names: Sequence[str],
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    random_state: int,
) -> CandidateTargetGbdtModel:
    if GradientBoostingClassifier is None:
        raise _StrategyUnavailableError("scikit-learn is not available for candidate_target_gbdt")
    x = np.stack([_candidate_row_features(example.row, feature_names=feature_names) for example in examples], axis=0).astype(np.float32)
    y = np.asarray([1 if bool(example.row.get("candidate_is_target", False)) else 0 for example in examples], dtype=np.int32)
    estimator = GradientBoostingClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        random_state=int(random_state),
    )
    estimator.fit(x, y)
    return CandidateTargetGbdtModel(
        estimator=estimator,
        feature_names=tuple(str(value) for value in feature_names),
    )
