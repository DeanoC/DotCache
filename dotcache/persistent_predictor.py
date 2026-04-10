from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np


PERSISTENT_PREDICTOR_FEATURE_NAMES: tuple[str, ...] = (
    "selected_fraction",
    "selected_block_count",
    "selected_token_count",
    "full_block_count",
    "full_token_count",
    "beta_upper",
    "delta_upper",
    "residual_mass_upper",
    "residual_value_upper",
    "remaining_block_count",
    "remaining_token_count",
    "history_snapshot_count",
    "history_prev_attention_nonzero_count",
    "step_index",
    "layer_id",
    "kv_head_id",
    "num_pages",
    "total_tokens",
    "tokens_per_page",
    "head_dim",
    "shaped_prev_attention_max",
    "shaped_prev_attention_nonzero_count",
    "persistent_runtime_recent_block_count",
    "persistent_runtime_mandatory_recent_block_count",
    "persistent_runtime_optional_top_k",
    "persistent_runtime_optional_upper_bound_quota",
    "persistent_runtime_optional_far_quota",
    "persistent_runtime_optional_mid_quota",
    "persistent_runtime_optional_near_quota",
    "persistent_runtime_optional_far_anchor_quota",
    "persistent_runtime_optional_far_anchor_priority_margin",
    "persistent_runtime_optional_diversity_weight",
    "persistent_runtime_optional_diversity_radius",
    "persistent_runtime_optional_diversity_min_history_count",
    "persistent_runtime_key_centroid_count",
    "persistent_runtime_probe_refine_top_k",
    "persistent_runtime_probe_sample_count",
    "persistent_runtime_region_residual_caps",
    "persistent_runtime_residual_cluster_count",
)


_PERSISTENT_SHORTLIST_CONFIG_KEY_TO_RUNTIME_FIELD: dict[str, str] = {
    "persistent_runtime_recent_block_count": "full_attention_recent_block_count",
    "persistent_runtime_mandatory_recent_block_count": "full_attention_mandatory_recent_block_count",
    "persistent_runtime_optional_top_k": "full_attention_optional_top_k",
    "persistent_runtime_optional_upper_bound_quota": "full_attention_optional_upper_bound_quota",
    "persistent_runtime_optional_far_quota": "full_attention_optional_far_quota",
    "persistent_runtime_optional_mid_quota": "full_attention_optional_mid_quota",
    "persistent_runtime_optional_near_quota": "full_attention_optional_near_quota",
    "persistent_runtime_optional_far_anchor_quota": "full_attention_optional_far_anchor_quota",
    "persistent_runtime_optional_far_anchor_priority_margin": "full_attention_optional_far_anchor_priority_margin",
    "persistent_runtime_optional_diversity_weight": "full_attention_optional_diversity_weight",
    "persistent_runtime_optional_diversity_radius": "full_attention_optional_diversity_radius",
    "persistent_runtime_optional_diversity_min_history_count": "full_attention_optional_diversity_min_history_count",
    "persistent_runtime_key_centroid_count": "full_attention_key_centroid_count",
    "persistent_runtime_probe_refine_top_k": "full_attention_probe_refine_top_k",
    "persistent_runtime_probe_sample_count": "full_attention_probe_sample_count",
    "persistent_runtime_region_residual_caps": "full_attention_region_residual_caps",
    "persistent_runtime_residual_cluster_count": "full_attention_residual_cluster_count",
}

_PERSISTENT_SHORTLIST_BOOL_FIELDS = frozenset(
    {
        "persistent_runtime_region_residual_caps",
    }
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _threshold_sweep(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = _binary_metrics(probabilities, labels, threshold=0.5)
    best_score = float(best_metrics["f1"])
    for threshold in np.linspace(0.05, 0.95, num=19, dtype=np.float32):
        metrics = _binary_metrics(probabilities, labels, threshold=float(threshold))
        score = float(metrics["f1"])
        if score > best_score or (score == best_score and float(threshold) < best_threshold):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def _binary_metrics(probabilities: np.ndarray, labels: np.ndarray, *, threshold: float) -> dict[str, float]:
    predicted = probabilities >= float(threshold)
    truth = labels >= 0.5
    tp = int(np.logical_and(predicted, truth).sum())
    tn = int(np.logical_and(~predicted, ~truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-8))
    accuracy = float((tp + tn) / max(labels.shape[0], 1))
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "safe_prediction_rate": float(predicted.mean()) if predicted.size else 0.0,
    }


def _normalized_feature_matrix(
    records: Sequence[dict[str, Any]],
    *,
    feature_names: Sequence[str],
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    for record in records:
        selected_fraction = float(record.get("selected_token_count", 0.0)) / max(float(record.get("full_token_count", 1.0)), 1.0)
        row = []
        for feature_name in feature_names:
            if feature_name == "selected_fraction":
                row.append(selected_fraction)
            else:
                value = record.get(feature_name, 0.0)
                row.append(0.0 if value is None else float(value))
        rows.append(row)
    x = np.asarray(rows, dtype=np.float32)
    if x.size == 0:
        x = np.zeros((0, len(feature_names)), dtype=np.float32)
    if feature_mean is None:
        feature_mean = np.mean(x, axis=0, dtype=np.float32) if x.shape[0] else np.zeros((len(feature_names),), dtype=np.float32)
    if feature_std is None:
        feature_std = np.std(x, axis=0, dtype=np.float32) if x.shape[0] else np.ones((len(feature_names),), dtype=np.float32)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)
    x_std = (x - feature_mean) / feature_std
    return x_std, feature_mean.astype(np.float32), feature_std.astype(np.float32)


def _label_vector(records: Sequence[dict[str, Any]], *, abs_threshold: float) -> np.ndarray:
    return np.asarray(
        [1.0 if float(record.get("max_abs_error", float("inf"))) <= float(abs_threshold) else 0.0 for record in records],
        dtype=np.float32,
    )


def _group_hash_fraction(group_key: str) -> float:
    digest = hashlib.sha1(group_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / float(16**8 - 1)


def split_predictor_records(
    records: Sequence[dict[str, Any]],
    *,
    test_fraction: float = 0.2,
) -> dict[str, list[dict[str, Any]]]:
    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    for record in records:
        group_key = str(record.get("snapshot_path", ""))
        target = test_records if _group_hash_fraction(group_key) < float(test_fraction) else train_records
        target.append(dict(record))
    if not test_records and train_records:
        test_records.append(train_records.pop())
    if not train_records and test_records:
        train_records.append(test_records.pop())
    return {"train_records": train_records, "test_records": test_records}


def _record_group_key(record: dict[str, Any]) -> str:
    return str(record.get("snapshot_path", ""))


def _record_config_key(record: dict[str, Any]) -> str:
    return json.dumps(
        {
            key: record.get(key)
            for key in (
                "persistent_runtime_recent_block_count",
                "persistent_runtime_mandatory_recent_block_count",
                "persistent_runtime_optional_top_k",
                "persistent_runtime_optional_upper_bound_quota",
                "persistent_runtime_optional_far_quota",
                "persistent_runtime_optional_mid_quota",
                "persistent_runtime_optional_near_quota",
                "persistent_runtime_optional_far_anchor_quota",
                "persistent_runtime_optional_far_anchor_priority_margin",
                "persistent_runtime_optional_diversity_weight",
                "persistent_runtime_optional_diversity_radius",
                "persistent_runtime_optional_diversity_min_history_count",
                "persistent_runtime_key_centroid_count",
                "persistent_runtime_probe_refine_top_k",
                "persistent_runtime_probe_sample_count",
                "persistent_runtime_region_residual_caps",
                "persistent_runtime_residual_cluster_count",
            )
        },
        sort_keys=True,
    )


def _record_quality_key(record: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(record.get("max_abs_error", float("inf"))),
        float(record.get("selected_token_count", float("inf"))),
        float(record.get("beta_upper", float("inf"))),
    )


def _safe_then_cheapest_quality_key(record: dict[str, Any], *, abs_threshold: float) -> tuple[float, float, float, float]:
    max_abs_error = float(record.get("max_abs_error", float("inf")))
    return (
        0.0 if max_abs_error <= float(abs_threshold) else 1.0,
        float(record.get("selected_token_count", float("inf"))),
        float(record.get("beta_upper", float("inf"))),
        max_abs_error,
    )


def _group_records_by_snapshot(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_record_group_key(record), []).append(dict(record))
    for group_key, group_records in grouped.items():
        deduped: dict[str, dict[str, Any]] = {}
        for record in group_records:
            deduped[_record_config_key(record)] = dict(record)
        grouped[group_key] = list(deduped.values())
    return grouped


def _pairwise_examples_from_records(
    records: Sequence[dict[str, Any]],
    *,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    grouped = _group_records_by_snapshot(records)
    pair_features: list[np.ndarray] = []
    labels: list[float] = []
    for group_records in grouped.values():
        if len(group_records) < 2:
            continue
        x_std, _, _ = _normalized_feature_matrix(group_records, feature_names=feature_names)
        for i in range(len(group_records)):
            for j in range(i + 1, len(group_records)):
                left = group_records[i]
                right = group_records[j]
                left_key = _record_quality_key(left)
                right_key = _record_quality_key(right)
                if left_key == right_key:
                    continue
                if left_key < right_key:
                    better_idx, worse_idx = i, j
                else:
                    better_idx, worse_idx = j, i
                pair_features.append((x_std[better_idx] - x_std[worse_idx]).astype(np.float32))
                labels.append(1.0)
                pair_features.append((x_std[worse_idx] - x_std[better_idx]).astype(np.float32))
                labels.append(0.0)
    if not pair_features:
        return (
            np.zeros((0, len(feature_names)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return np.stack(pair_features, axis=0).astype(np.float32), np.asarray(labels, dtype=np.float32)


@dataclass(slots=True)
class PersistentResidualPredictorModel:
    weight: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: tuple[str, ...]
    target_abs_threshold: float
    decision_threshold: float

    def predict_probability_from_record(self, record: dict[str, Any]) -> float:
        x_std, _, _ = _normalized_feature_matrix(
            [record],
            feature_names=self.feature_names,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )
        if x_std.shape[0] == 0:
            return 0.0
        logits = x_std @ self.weight + float(self.bias)
        return float(_sigmoid(logits)[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "persistent_residual_predictor_model",
            "weight": self.weight.astype(np.float32).tolist(),
            "bias": float(self.bias),
            "feature_mean": self.feature_mean.astype(np.float32).tolist(),
            "feature_std": self.feature_std.astype(np.float32).tolist(),
            "feature_names": list(self.feature_names),
            "target_abs_threshold": float(self.target_abs_threshold),
            "decision_threshold": float(self.decision_threshold),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersistentResidualPredictorModel":
        return cls(
            weight=np.asarray(payload["weight"], dtype=np.float32),
            bias=float(payload["bias"]),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(payload["feature_std"], dtype=np.float32),
            feature_names=tuple(str(item) for item in payload["feature_names"]),
            target_abs_threshold=float(payload["target_abs_threshold"]),
            decision_threshold=float(payload["decision_threshold"]),
        )


def save_persistent_residual_predictor_model(
    model: PersistentResidualPredictorModel,
    path: str | Path,
) -> None:
    Path(path).write_text(json.dumps(model.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_persistent_residual_predictor_model(path: str | Path) -> PersistentResidualPredictorModel:
    return PersistentResidualPredictorModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def train_persistent_residual_predictor(
    records: Sequence[dict[str, Any]],
    *,
    abs_threshold: float,
    feature_names: Sequence[str] | None = None,
    steps: int = 400,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
) -> PersistentResidualPredictorModel:
    resolved_feature_names = tuple(feature_names) if feature_names is not None else PERSISTENT_PREDICTOR_FEATURE_NAMES
    x_std, feature_mean, feature_std = _normalized_feature_matrix(records, feature_names=resolved_feature_names)
    y = _label_vector(records, abs_threshold=float(abs_threshold))
    weight = np.zeros((x_std.shape[1],), dtype=np.float32)
    bias = 0.0
    if x_std.shape[0] > 0:
        positive_fraction = float(y.mean()) if y.size else 0.0
        positive_weight = 1.0 / max(positive_fraction, 1e-3)
        negative_weight = 1.0 / max(1.0 - positive_fraction, 1e-3)
        sample_weight = np.where(y >= 0.5, positive_weight, negative_weight).astype(np.float32)
        sample_weight = sample_weight / max(sample_weight.mean(), 1e-6)
        for _ in range(int(steps)):
            logits = x_std @ weight + bias
            probabilities = _sigmoid(logits)
            error = (probabilities - y) * sample_weight
            grad_weight = (x_std.T @ error) / max(x_std.shape[0], 1) + float(l2) * weight
            grad_bias = float(np.mean(error, dtype=np.float32))
            weight -= float(learning_rate) * grad_weight.astype(np.float32)
            bias -= float(learning_rate) * grad_bias
    probabilities = _sigmoid(x_std @ weight + bias) if x_std.shape[0] else np.zeros((0,), dtype=np.float32)
    decision_threshold, _ = _threshold_sweep(probabilities, y)
    return PersistentResidualPredictorModel(
        weight=weight.astype(np.float32),
        bias=float(bias),
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=resolved_feature_names,
        target_abs_threshold=float(abs_threshold),
        decision_threshold=float(decision_threshold),
    )


def evaluate_persistent_residual_predictor(
    model: PersistentResidualPredictorModel,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    x_std, _, _ = _normalized_feature_matrix(
        records,
        feature_names=model.feature_names,
        feature_mean=model.feature_mean,
        feature_std=model.feature_std,
    )
    labels = _label_vector(records, abs_threshold=model.target_abs_threshold)
    probabilities = _sigmoid(x_std @ model.weight + float(model.bias)) if x_std.shape[0] else np.zeros((0,), dtype=np.float32)
    metrics = _binary_metrics(probabilities, labels, threshold=float(model.decision_threshold))
    metrics["example_count"] = int(len(records))
    metrics["positive_rate"] = float(labels.mean()) if labels.size else 0.0
    metrics["mean_probability"] = float(probabilities.mean()) if probabilities.size else 0.0
    metrics["max_probability"] = float(probabilities.max()) if probabilities.size else 0.0
    metrics["min_probability"] = float(probabilities.min()) if probabilities.size else 0.0
    return metrics


@dataclass(slots=True)
class PersistentPairwiseRankerModel:
    weight: np.ndarray
    bias: float
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_names: tuple[str, ...]

    def score_record(self, record: dict[str, Any]) -> float:
        x_std, _, _ = _normalized_feature_matrix(
            [record],
            feature_names=self.feature_names,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )
        if x_std.shape[0] == 0:
            return 0.0
        return float((x_std @ self.weight + float(self.bias))[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "persistent_pairwise_ranker_model",
            "weight": self.weight.astype(np.float32).tolist(),
            "bias": float(self.bias),
            "feature_mean": self.feature_mean.astype(np.float32).tolist(),
            "feature_std": self.feature_std.astype(np.float32).tolist(),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersistentPairwiseRankerModel":
        return cls(
            weight=np.asarray(payload["weight"], dtype=np.float32),
            bias=float(payload["bias"]),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(payload["feature_std"], dtype=np.float32),
            feature_names=tuple(str(item) for item in payload["feature_names"]),
        )


def save_persistent_pairwise_ranker_model(
    model: PersistentPairwiseRankerModel,
    path: str | Path,
) -> None:
    Path(path).write_text(json.dumps(model.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_persistent_pairwise_ranker_model(path: str | Path) -> PersistentPairwiseRankerModel:
    return PersistentPairwiseRankerModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def train_persistent_pairwise_ranker(
    records: Sequence[dict[str, Any]],
    *,
    feature_names: Sequence[str] | None = None,
    steps: int = 400,
    learning_rate: float = 0.2,
    l2: float = 1e-3,
) -> PersistentPairwiseRankerModel:
    resolved_feature_names = tuple(feature_names) if feature_names is not None else PERSISTENT_PREDICTOR_FEATURE_NAMES
    _, feature_mean, feature_std = _normalized_feature_matrix(records, feature_names=resolved_feature_names)
    pair_x, pair_y = _pairwise_examples_from_records(records, feature_names=resolved_feature_names)
    weight = np.zeros((len(resolved_feature_names),), dtype=np.float32)
    bias = 0.0
    if pair_x.shape[0] > 0:
        for _ in range(int(steps)):
            logits = pair_x @ weight + bias
            probabilities = _sigmoid(logits)
            error = probabilities - pair_y
            grad_weight = (pair_x.T @ error) / max(pair_x.shape[0], 1) + float(l2) * weight
            grad_bias = float(np.mean(error, dtype=np.float32))
            weight -= float(learning_rate) * grad_weight.astype(np.float32)
            bias -= float(learning_rate) * grad_bias
    return PersistentPairwiseRankerModel(
        weight=weight.astype(np.float32),
        bias=float(bias),
        feature_mean=feature_mean,
        feature_std=feature_std,
        feature_names=resolved_feature_names,
    )


def evaluate_persistent_pairwise_ranker(
    model: PersistentPairwiseRankerModel,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    grouped = _group_records_by_snapshot(records)
    pair_total = 0
    pair_correct = 0
    snapshot_total = 0
    snapshot_top1_correct = 0
    for group_records in grouped.values():
        if len(group_records) < 2:
            continue
        scored_records = [
            (record, model.score_record(record), _record_quality_key(record))
            for record in group_records
        ]
        for i in range(len(scored_records)):
            for j in range(i + 1, len(scored_records)):
                left_record, left_score, left_quality = scored_records[i]
                right_record, right_score, right_quality = scored_records[j]
                if left_quality == right_quality:
                    continue
                oracle_left_better = left_quality < right_quality
                predicted_left_better = left_score > right_score
                pair_total += 1
                if oracle_left_better == predicted_left_better:
                    pair_correct += 1
        snapshot_total += 1
        predicted_best = max(scored_records, key=lambda item: item[1])[0]
        oracle_best = min(group_records, key=_record_quality_key)
        if _record_config_key(predicted_best) == _record_config_key(oracle_best):
            snapshot_top1_correct += 1
    return {
        "snapshot_group_count": int(snapshot_total),
        "pair_count": int(pair_total),
        "pair_accuracy": float(pair_correct / max(pair_total, 1)),
        "top1_accuracy": float(snapshot_top1_correct / max(snapshot_total, 1)),
    }


def evaluate_safe_then_cheapest_policy(
    model: PersistentResidualPredictorModel,
    records: Sequence[dict[str, Any]],
    *,
    abs_threshold: float | None = None,
) -> dict[str, Any]:
    resolved_abs_threshold = float(model.target_abs_threshold if abs_threshold is None else abs_threshold)
    grouped = _group_records_by_snapshot(records)
    snapshot_total = 0
    top1_correct = 0
    chosen_safe_count = 0
    oracle_safe_count = 0
    predicted_safe_nonempty_count = 0
    total_selected_tokens = 0.0
    oracle_selected_tokens = 0.0

    for group_records in grouped.values():
        if not group_records:
            continue
        snapshot_total += 1
        scored_records = [
            (record, model.predict_probability_from_record(record))
            for record in group_records
        ]
        predicted_safe_records = [
            record
            for record, probability in scored_records
            if probability >= float(model.decision_threshold)
        ]
        if predicted_safe_records:
            predicted_safe_nonempty_count += 1
            chosen_record = min(
                predicted_safe_records,
                key=lambda record: (
                    float(record.get("selected_token_count", float("inf"))),
                    float(record.get("beta_upper", float("inf"))),
                    float(record.get("max_abs_error", float("inf"))),
                ),
            )
        else:
            chosen_record = max(
                scored_records,
                key=lambda item: (
                    item[1],
                    -float(item[0].get("selected_token_count", float("inf"))),
                    -float(item[0].get("beta_upper", float("inf"))),
                ),
            )[0]

        oracle_record = min(
            group_records,
            key=lambda record: _safe_then_cheapest_quality_key(record, abs_threshold=resolved_abs_threshold),
        )
        chosen_safe = float(chosen_record.get("max_abs_error", float("inf"))) <= resolved_abs_threshold
        oracle_safe = float(oracle_record.get("max_abs_error", float("inf"))) <= resolved_abs_threshold
        if chosen_safe:
            chosen_safe_count += 1
        if oracle_safe:
            oracle_safe_count += 1
        total_selected_tokens += float(chosen_record.get("selected_token_count", 0.0))
        oracle_selected_tokens += float(oracle_record.get("selected_token_count", 0.0))
        if _record_config_key(chosen_record) == _record_config_key(oracle_record):
            top1_correct += 1

    return {
        "snapshot_group_count": int(snapshot_total),
        "top1_accuracy": float(top1_correct / max(snapshot_total, 1)),
        "chosen_safe_rate": float(chosen_safe_count / max(snapshot_total, 1)),
        "oracle_safe_rate": float(oracle_safe_count / max(snapshot_total, 1)),
        "predicted_safe_nonempty_rate": float(predicted_safe_nonempty_count / max(snapshot_total, 1)),
        "avg_selected_token_count": float(total_selected_tokens / max(snapshot_total, 1)),
        "avg_oracle_selected_token_count": float(oracle_selected_tokens / max(snapshot_total, 1)),
    }


def recommend_safe_then_cheapest_configs(
    model: PersistentResidualPredictorModel,
    records: Sequence[dict[str, Any]],
    *,
    abs_threshold: float | None = None,
) -> dict[str, Any]:
    resolved_abs_threshold = float(model.target_abs_threshold if abs_threshold is None else abs_threshold)
    grouped = _group_records_by_snapshot(records)
    recommendations: list[dict[str, Any]] = []
    for snapshot_path, group_records in sorted(grouped.items()):
        if not group_records:
            continue
        scored_records = [
            (record, model.predict_probability_from_record(record))
            for record in group_records
        ]
        predicted_safe_records = [
            (record, probability)
            for record, probability in scored_records
            if probability >= float(model.decision_threshold)
        ]
        if predicted_safe_records:
            chosen_record, chosen_probability = min(
                predicted_safe_records,
                key=lambda item: (
                    float(item[0].get("selected_token_count", float("inf"))),
                    float(item[0].get("beta_upper", float("inf"))),
                    float(item[0].get("max_abs_error", float("inf"))),
                ),
            )
        else:
            chosen_record, chosen_probability = max(
                scored_records,
                key=lambda item: (
                    item[1],
                    -float(item[0].get("selected_token_count", float("inf"))),
                    -float(item[0].get("beta_upper", float("inf"))),
                ),
            )
        oracle_record = min(
            group_records,
            key=lambda record: _safe_then_cheapest_quality_key(record, abs_threshold=resolved_abs_threshold),
        )
        recommendations.append({
            "snapshot_path": snapshot_path,
            "candidate_count": int(len(group_records)),
            "predicted_safe_count": int(len(predicted_safe_records)),
            "chosen_config_key": _record_config_key(chosen_record),
            "chosen_source_compare_json": str(chosen_record.get("source_compare_json", "")),
            "chosen_probability": float(chosen_probability),
            "chosen_max_abs_error": float(chosen_record.get("max_abs_error", float("inf"))),
            "chosen_selected_token_count": float(chosen_record.get("selected_token_count", 0.0)),
            "chosen_is_safe": bool(float(chosen_record.get("max_abs_error", float("inf"))) <= resolved_abs_threshold),
            "oracle_config_key": _record_config_key(oracle_record),
            "oracle_source_compare_json": str(oracle_record.get("source_compare_json", "")),
            "oracle_max_abs_error": float(oracle_record.get("max_abs_error", float("inf"))),
            "oracle_selected_token_count": float(oracle_record.get("selected_token_count", 0.0)),
            "oracle_is_safe": bool(float(oracle_record.get("max_abs_error", float("inf"))) <= resolved_abs_threshold),
            "matched_oracle": bool(_record_config_key(chosen_record) == _record_config_key(oracle_record)),
        })
    return {
        "summary": evaluate_safe_then_cheapest_policy(model, records, abs_threshold=resolved_abs_threshold),
        "recommendations": recommendations,
    }


_SNAPSHOT_RE = re.compile(r"layer(?P<layer_id>\d+)_kv(?P<kv_head_id>\d+)_step(?P<step_index>[+-]?\d+)")


def _snapshot_metadata(snapshot_path: str) -> dict[str, Any]:
    resolved = Path(snapshot_path)
    match = _SNAPSHOT_RE.search(resolved.name)
    layer_id = int(match.group("layer_id")) if match else -1
    kv_head_id = int(match.group("kv_head_id")) if match else -1
    step_index = int(match.group("step_index")) if match else -1
    prompt_family = resolved.parent.name
    return {
        "snapshot_path": snapshot_path,
        "prompt_family": prompt_family,
        "layer_id": layer_id,
        "kv_head_id": kv_head_id,
        "step_index": step_index,
        "step_bucket": (
            "bootstrap"
            if step_index <= 0
            else "mid"
            if step_index <= 3
            else "late"
        ),
    }


def _policy_bucket(snapshot_path: str, *, group_by: Sequence[str]) -> dict[str, Any]:
    metadata = _snapshot_metadata(snapshot_path)
    return {field: metadata[field] for field in group_by}


def persistent_shortlist_step_bucket(step_index: int) -> str:
    resolved_step_index = int(step_index)
    if resolved_step_index <= 0:
        return "bootstrap"
    if resolved_step_index <= 3:
        return "mid"
    return "late"


def load_persistent_shortlist_policy(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def parse_persistent_shortlist_config_key(config_key: str) -> dict[str, Any]:
    parsed = json.loads(str(config_key))
    overrides: dict[str, Any] = {}
    for source_key, runtime_key in _PERSISTENT_SHORTLIST_CONFIG_KEY_TO_RUNTIME_FIELD.items():
        if source_key not in parsed or parsed[source_key] is None:
            continue
        value = parsed[source_key]
        if source_key in _PERSISTENT_SHORTLIST_BOOL_FIELDS:
            overrides[runtime_key] = bool(value)
        else:
            overrides[runtime_key] = value
    return overrides


def resolve_persistent_shortlist_policy_choice(
    policy_payload: dict[str, Any],
    *,
    layer_id: int,
    kv_head_ids: Sequence[int],
    prompt_family: str,
    step_index: int,
) -> dict[str, Any] | None:
    group_by = tuple(str(field) for field in policy_payload.get("group_by", ()))
    policy_groups = {
        json.dumps(group.get("bucket", {}), sort_keys=True): group
        for group in policy_payload.get("groups", [])
    }
    step_bucket = persistent_shortlist_step_bucket(int(step_index))
    resolved_prompt_family = str(prompt_family)
    candidate_groups: list[dict[str, Any]] = []
    resolved_kv_head_ids = sorted({int(kv_head_id) for kv_head_id in kv_head_ids})
    kv_head_iterable: Sequence[int | None] = resolved_kv_head_ids if "kv_head_id" in group_by else [None]

    for kv_head_id in kv_head_iterable:
        bucket: dict[str, Any] = {}
        for field in group_by:
            if field == "layer_id":
                bucket[field] = int(layer_id)
            elif field == "kv_head_id":
                bucket[field] = int(kv_head_id) if kv_head_id is not None else -1
            elif field == "prompt_family":
                bucket[field] = resolved_prompt_family
            elif field == "step_bucket":
                bucket[field] = step_bucket
            elif field == "step_index":
                bucket[field] = int(step_index)
        group = policy_groups.get(json.dumps(bucket, sort_keys=True))
        if group is not None:
            candidate_groups.append(
                {
                    "bucket": bucket,
                    "group": group,
                }
            )

    if not candidate_groups:
        return None

    config_votes: dict[str, dict[str, Any]] = {}
    for candidate in candidate_groups:
        ranked_configs = list(candidate["group"].get("ranked_configs", []))
        if not ranked_configs:
            continue
        top_ranked = ranked_configs[0]
        config_key = str(top_ranked.get("config_key", ""))
        vote_entry = config_votes.setdefault(
            config_key,
            {
                "config_key": config_key,
                "source_compare_json": str(top_ranked.get("source_compare_json", "")),
                "bucket_match_count": 0,
                "vote_count": 0,
                "selected_token_total": 0.0,
                "max_abs_error_total": 0.0,
                "matched_oracle_total": 0.0,
                "chosen_safe_total": 0.0,
                "buckets": [],
            },
        )
        vote_entry["bucket_match_count"] += 1
        vote_entry["vote_count"] += int(top_ranked.get("vote_count", 0))
        vote_entry["selected_token_total"] += (
            float(top_ranked.get("avg_selected_token_count", 0.0)) * max(int(top_ranked.get("vote_count", 0)), 1)
        )
        vote_entry["max_abs_error_total"] += (
            float(top_ranked.get("avg_max_abs_error", 0.0)) * max(int(top_ranked.get("vote_count", 0)), 1)
        )
        vote_entry["matched_oracle_total"] += (
            float(top_ranked.get("matched_oracle_rate", 0.0)) * max(int(top_ranked.get("vote_count", 0)), 1)
        )
        vote_entry["chosen_safe_total"] += (
            float(top_ranked.get("chosen_safe_rate", 0.0)) * max(int(top_ranked.get("vote_count", 0)), 1)
        )
        vote_entry["buckets"].append(dict(candidate["bucket"]))

    if not config_votes:
        return None

    best_vote = min(
        config_votes.values(),
        key=lambda vote: (
            -int(vote["bucket_match_count"]),
            -int(vote["vote_count"]),
            float(vote["selected_token_total"]) / max(int(vote["vote_count"]), 1),
            float(vote["max_abs_error_total"]) / max(int(vote["vote_count"]), 1),
            str(vote["source_compare_json"]),
        ),
    )
    total_votes = max(int(best_vote["vote_count"]), 1)
    return {
        "group_by": list(group_by),
        "config_key": str(best_vote["config_key"]),
        "source_compare_json": str(best_vote["source_compare_json"]),
        "config_overrides": parse_persistent_shortlist_config_key(str(best_vote["config_key"])),
        "bucket_match_count": int(best_vote["bucket_match_count"]),
        "matched_oracle_rate": float(best_vote["matched_oracle_total"]) / total_votes,
        "chosen_safe_rate": float(best_vote["chosen_safe_total"]) / total_votes,
        "avg_selected_token_count": float(best_vote["selected_token_total"]) / total_votes,
        "avg_max_abs_error": float(best_vote["max_abs_error_total"]) / total_votes,
        "matched_buckets": list(best_vote["buckets"]),
        "step_bucket": step_bucket,
    }


def build_persistent_shortlist_policy(
    recommendations_payload: dict[str, Any],
    *,
    group_by: Sequence[str] = ("layer_id", "step_bucket"),
) -> dict[str, Any]:
    resolved_group_by = tuple(str(field) for field in group_by)
    grouped: dict[str, dict[str, Any]] = {}
    for item in recommendations_payload.get("recommendations", []):
        bucket = _policy_bucket(str(item.get("snapshot_path", "")), group_by=resolved_group_by)
        bucket_key = json.dumps(bucket, sort_keys=True)
        group_entry = grouped.setdefault(
            bucket_key,
            {
                "bucket": bucket,
                "config_stats": {},
                "snapshot_count": 0,
            },
        )
        group_entry["snapshot_count"] += 1
        config_stats = group_entry["config_stats"].setdefault(
            str(item.get("chosen_config_key", "")),
            {
                "config_key": str(item.get("chosen_config_key", "")),
                "source_compare_json": str(item.get("chosen_source_compare_json", "")),
                "vote_count": 0,
                "matched_oracle_count": 0,
                "safe_count": 0,
                "selected_token_total": 0.0,
                "max_abs_error_total": 0.0,
            },
        )
        config_stats["vote_count"] += 1
        config_stats["matched_oracle_count"] += int(bool(item.get("matched_oracle", False)))
        config_stats["safe_count"] += int(bool(item.get("chosen_is_safe", False)))
        config_stats["selected_token_total"] += float(item.get("chosen_selected_token_count", 0.0))
        config_stats["max_abs_error_total"] += float(item.get("chosen_max_abs_error", 0.0))

    groups = []
    for group_entry in grouped.values():
        ranked_configs = sorted(
            group_entry["config_stats"].values(),
            key=lambda stats: (
                -int(stats["vote_count"]),
                float(stats["selected_token_total"]) / max(int(stats["vote_count"]), 1),
                float(stats["max_abs_error_total"]) / max(int(stats["vote_count"]), 1),
                str(stats["source_compare_json"]),
            ),
        )
        serialized_configs = []
        for stats in ranked_configs:
            vote_count = max(int(stats["vote_count"]), 1)
            serialized_configs.append(
                {
                    "config_key": str(stats["config_key"]),
                    "source_compare_json": str(stats["source_compare_json"]),
                    "vote_count": int(stats["vote_count"]),
                    "matched_oracle_rate": float(stats["matched_oracle_count"]) / vote_count,
                    "chosen_safe_rate": float(stats["safe_count"]) / vote_count,
                    "avg_selected_token_count": float(stats["selected_token_total"]) / vote_count,
                    "avg_max_abs_error": float(stats["max_abs_error_total"]) / vote_count,
                }
            )
        groups.append(
            {
                "bucket": dict(group_entry["bucket"]),
                "snapshot_count": int(group_entry["snapshot_count"]),
                "ranked_configs": serialized_configs,
            }
        )

    groups.sort(key=lambda item: tuple(item["bucket"].get(field) for field in resolved_group_by))
    return {
        "group_by": list(resolved_group_by),
        "group_count": int(len(groups)),
        "groups": groups,
    }


def evaluate_persistent_shortlist_policy(
    policy_payload: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    abs_threshold: float = 0.05,
) -> dict[str, Any]:
    group_by = tuple(str(field) for field in policy_payload.get("group_by", []))
    policy_groups = {
        json.dumps(group.get("bucket", {}), sort_keys=True): group
        for group in policy_payload.get("groups", [])
    }
    grouped_records = _group_records_by_snapshot(records)
    evaluations: list[dict[str, Any]] = []
    match_count = 0
    safe_count = 0
    selected_token_total = 0.0
    oracle_selected_token_total = 0.0
    fallback_count = 0
    missing_bucket_count = 0

    for snapshot_path, group_records in sorted(grouped_records.items()):
        bucket = _policy_bucket(snapshot_path, group_by=group_by)
        bucket_key = json.dumps(bucket, sort_keys=True)
        policy_group = policy_groups.get(bucket_key)
        if policy_group is None:
            missing_bucket_count += 1
            ranked_configs = []
        else:
            ranked_configs = list(policy_group.get("ranked_configs", []))

        record_by_config = {
            _record_config_key(record): record
            for record in group_records
        }
        chosen_record = None
        chosen_rank = -1
        for index, ranked in enumerate(ranked_configs):
            candidate = record_by_config.get(str(ranked.get("config_key", "")))
            if candidate is not None:
                chosen_record = candidate
                chosen_rank = index
                break
        if chosen_record is None:
            fallback_count += 1
            chosen_record = min(
                group_records,
                key=lambda record: (
                    float(record.get("selected_token_count", float("inf"))),
                    float(record.get("beta_upper", float("inf"))),
                    float(record.get("max_abs_error", float("inf"))),
                ),
            )

        oracle_record = min(
            group_records,
            key=lambda record: _safe_then_cheapest_quality_key(record, abs_threshold=float(abs_threshold)),
        )
        chosen_is_safe = float(chosen_record.get("max_abs_error", float("inf"))) <= float(abs_threshold)
        if chosen_is_safe:
            safe_count += 1
        selected_token_total += float(chosen_record.get("selected_token_count", 0.0))
        oracle_selected_token_total += float(oracle_record.get("selected_token_count", 0.0))
        matched = _record_config_key(chosen_record) == _record_config_key(oracle_record)
        if matched:
            match_count += 1
        evaluations.append(
            {
                "snapshot_path": snapshot_path,
                "bucket": bucket,
                "policy_bucket_found": policy_group is not None,
                "fallback_used": chosen_rank < 0,
                "chosen_rank": int(chosen_rank),
                "chosen_config_key": _record_config_key(chosen_record),
                "chosen_source_compare_json": str(chosen_record.get("source_compare_json", "")),
                "chosen_max_abs_error": float(chosen_record.get("max_abs_error", float("inf"))),
                "chosen_selected_token_count": float(chosen_record.get("selected_token_count", 0.0)),
                "chosen_is_safe": bool(chosen_is_safe),
                "oracle_config_key": _record_config_key(oracle_record),
                "oracle_source_compare_json": str(oracle_record.get("source_compare_json", "")),
                "oracle_max_abs_error": float(oracle_record.get("max_abs_error", float("inf"))),
                "oracle_selected_token_count": float(oracle_record.get("selected_token_count", 0.0)),
                "matched_oracle": bool(matched),
            }
        )

    snapshot_count = max(len(evaluations), 1)
    return {
        "summary": {
            "snapshot_group_count": int(len(evaluations)),
            "top1_accuracy": float(match_count / snapshot_count),
            "chosen_safe_rate": float(safe_count / snapshot_count),
            "avg_selected_token_count": float(selected_token_total / snapshot_count),
            "avg_oracle_selected_token_count": float(oracle_selected_token_total / snapshot_count),
            "fallback_rate": float(fallback_count / snapshot_count),
            "missing_bucket_rate": float(missing_bucket_count / snapshot_count),
        },
        "evaluations": evaluations,
    }
