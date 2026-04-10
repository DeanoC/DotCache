from __future__ import annotations

import json
from pathlib import Path

from dotcache.persistent_predictor import (
    PERSISTENT_PREDICTOR_FEATURE_NAMES,
    evaluate_persistent_pairwise_ranker,
    evaluate_persistent_residual_predictor,
    evaluate_safe_then_cheapest_policy,
    recommend_safe_then_cheapest_configs,
    load_persistent_pairwise_ranker_model,
    load_persistent_residual_predictor_model,
    save_persistent_pairwise_ranker_model,
    save_persistent_residual_predictor_model,
    split_predictor_records,
    train_persistent_pairwise_ranker,
    train_persistent_residual_predictor,
)


def _record(snapshot_path: str, *, max_abs_error: float, step_index: int, beta_upper: float, delta_upper: float) -> dict[str, object]:
    return {
        "snapshot_path": snapshot_path,
        "selected_token_count": 128,
        "full_token_count": 256,
        "selected_block_count": 8,
        "full_block_count": 16,
        "beta_upper": beta_upper,
        "delta_upper": delta_upper,
        "residual_mass_upper": beta_upper * 10.0,
        "residual_value_upper": delta_upper * 10.0,
        "remaining_block_count": 8,
        "remaining_token_count": 128,
        "history_snapshot_count": step_index,
        "history_prev_attention_nonzero_count": 4 + step_index,
        "step_index": step_index,
        "layer_id": 15,
        "kv_head_id": 0,
        "num_pages": 32,
        "total_tokens": 256,
        "tokens_per_page": 16,
        "head_dim": 128,
        "shaped_prev_attention_max": 0.9,
        "shaped_prev_attention_nonzero_count": 5,
        "persistent_runtime_recent_block_count": 64,
        "persistent_runtime_mandatory_recent_block_count": 16,
        "persistent_runtime_optional_top_k": 128,
        "persistent_runtime_optional_upper_bound_quota": 16,
        "persistent_runtime_optional_far_quota": 32,
        "persistent_runtime_optional_mid_quota": 48,
        "persistent_runtime_optional_near_quota": 32,
        "persistent_runtime_optional_far_anchor_quota": 4,
        "persistent_runtime_optional_far_anchor_priority_margin": 0.25,
        "persistent_runtime_optional_diversity_weight": 0.5,
        "persistent_runtime_optional_diversity_radius": 4,
        "persistent_runtime_optional_diversity_min_history_count": 1,
        "persistent_runtime_key_centroid_count": 2,
        "persistent_runtime_probe_refine_top_k": 0,
        "persistent_runtime_probe_sample_count": 4,
        "persistent_runtime_region_residual_caps": 0,
        "persistent_runtime_residual_cluster_count": 0,
        "max_abs_error": max_abs_error,
    }


def test_split_predictor_records_keeps_snapshot_groups_together() -> None:
    records = [
        _record("a.npz", max_abs_error=0.01, step_index=0, beta_upper=0.1, delta_upper=0.1),
        _record("a.npz", max_abs_error=0.02, step_index=1, beta_upper=0.2, delta_upper=0.2),
        _record("b.npz", max_abs_error=0.30, step_index=0, beta_upper=0.8, delta_upper=0.8),
    ]
    split = split_predictor_records(records, test_fraction=0.5)
    train_paths = {str(record["snapshot_path"]) for record in split["train_records"]}
    test_paths = {str(record["snapshot_path"]) for record in split["test_records"]}
    assert train_paths.isdisjoint(test_paths)


def test_train_and_round_trip_persistent_predictor(tmp_path: Path) -> None:
    records = [
        _record("safe_0.npz", max_abs_error=0.01, step_index=0, beta_upper=0.05, delta_upper=0.05),
        _record("safe_1.npz", max_abs_error=0.02, step_index=1, beta_upper=0.08, delta_upper=0.07),
        _record("unsafe_0.npz", max_abs_error=0.40, step_index=0, beta_upper=0.80, delta_upper=0.70),
        _record("unsafe_1.npz", max_abs_error=0.35, step_index=1, beta_upper=0.75, delta_upper=0.65),
    ]
    model = train_persistent_residual_predictor(
        records,
        abs_threshold=0.1,
        feature_names=PERSISTENT_PREDICTOR_FEATURE_NAMES,
        steps=200,
        learning_rate=0.3,
        l2=1e-3,
    )
    metrics = evaluate_persistent_residual_predictor(model, records)
    assert metrics["accuracy"] >= 0.75
    path = tmp_path / "persistent_residual_predictor_model.json"
    save_persistent_residual_predictor_model(model, path)
    loaded = load_persistent_residual_predictor_model(path)
    loaded_metrics = evaluate_persistent_residual_predictor(loaded, records)
    assert loaded.feature_names == model.feature_names
    assert loaded.target_abs_threshold == model.target_abs_threshold
    assert loaded_metrics["accuracy"] == metrics["accuracy"]


def test_train_and_round_trip_pairwise_ranker(tmp_path: Path) -> None:
    records = [
        _record("snap_a.npz", max_abs_error=0.02, step_index=0, beta_upper=0.05, delta_upper=0.04)
        | {"selected_token_count": 112, "persistent_runtime_optional_top_k": 96},
        _record("snap_a.npz", max_abs_error=0.08, step_index=0, beta_upper=0.12, delta_upper=0.10)
        | {"selected_token_count": 144, "persistent_runtime_optional_top_k": 128},
        _record("snap_a.npz", max_abs_error=0.20, step_index=0, beta_upper=0.35, delta_upper=0.28)
        | {"selected_token_count": 176, "persistent_runtime_optional_top_k": 160},
        _record("snap_b.npz", max_abs_error=0.03, step_index=1, beta_upper=0.06, delta_upper=0.05)
        | {"selected_token_count": 120, "persistent_runtime_optional_top_k": 96},
        _record("snap_b.npz", max_abs_error=0.09, step_index=1, beta_upper=0.14, delta_upper=0.12)
        | {"selected_token_count": 152, "persistent_runtime_optional_top_k": 128},
        _record("snap_b.npz", max_abs_error=0.25, step_index=1, beta_upper=0.40, delta_upper=0.32)
        | {"selected_token_count": 192, "persistent_runtime_optional_top_k": 160},
    ]
    model = train_persistent_pairwise_ranker(
        records,
        feature_names=PERSISTENT_PREDICTOR_FEATURE_NAMES,
        steps=300,
        learning_rate=0.2,
        l2=1e-3,
    )
    metrics = evaluate_persistent_pairwise_ranker(model, records)
    assert metrics["pair_accuracy"] >= 0.8
    assert metrics["top1_accuracy"] >= 0.5

    path = tmp_path / "persistent_pairwise_ranker_model.json"
    save_persistent_pairwise_ranker_model(model, path)
    loaded = load_persistent_pairwise_ranker_model(path)
    loaded_metrics = evaluate_persistent_pairwise_ranker(loaded, records)
    assert loaded.feature_names == model.feature_names
    assert loaded_metrics["pair_accuracy"] == metrics["pair_accuracy"]


def test_safe_then_cheapest_policy_prefers_safe_low_token_record() -> None:
    records = [
        _record("snap_a.npz", max_abs_error=0.03, step_index=0, beta_upper=0.05, delta_upper=0.04)
        | {"selected_token_count": 112, "persistent_runtime_optional_top_k": 96},
        _record("snap_a.npz", max_abs_error=0.04, step_index=0, beta_upper=0.09, delta_upper=0.07)
        | {"selected_token_count": 144, "persistent_runtime_optional_top_k": 128},
        _record("snap_a.npz", max_abs_error=0.20, step_index=0, beta_upper=0.35, delta_upper=0.28)
        | {"selected_token_count": 96, "persistent_runtime_optional_top_k": 64},
        _record("snap_b.npz", max_abs_error=0.02, step_index=1, beta_upper=0.04, delta_upper=0.03)
        | {"selected_token_count": 120, "persistent_runtime_optional_top_k": 96},
        _record("snap_b.npz", max_abs_error=0.06, step_index=1, beta_upper=0.11, delta_upper=0.09)
        | {"selected_token_count": 96, "persistent_runtime_optional_top_k": 64},
    ]
    model = train_persistent_residual_predictor(
        records,
        abs_threshold=0.05,
        feature_names=PERSISTENT_PREDICTOR_FEATURE_NAMES,
        steps=300,
        learning_rate=0.3,
        l2=1e-3,
    )
    metrics = evaluate_safe_then_cheapest_policy(model, records, abs_threshold=0.05)
    assert metrics["top1_accuracy"] >= 0.5
    assert metrics["chosen_safe_rate"] >= 0.5
    assert metrics["avg_selected_token_count"] <= metrics["avg_oracle_selected_token_count"] + 32.0


def test_recommend_safe_then_cheapest_configs_emits_snapshot_choices() -> None:
    records = [
        _record("snap_a.npz", max_abs_error=0.03, step_index=0, beta_upper=0.05, delta_upper=0.04)
        | {"selected_token_count": 112, "persistent_runtime_optional_top_k": 96, "source_compare_json": "a.json"},
        _record("snap_a.npz", max_abs_error=0.20, step_index=0, beta_upper=0.35, delta_upper=0.28)
        | {"selected_token_count": 96, "persistent_runtime_optional_top_k": 64, "source_compare_json": "b.json"},
        _record("snap_b.npz", max_abs_error=0.02, step_index=1, beta_upper=0.04, delta_upper=0.03)
        | {"selected_token_count": 120, "persistent_runtime_optional_top_k": 96, "source_compare_json": "a.json"},
        _record("snap_b.npz", max_abs_error=0.06, step_index=1, beta_upper=0.11, delta_upper=0.09)
        | {"selected_token_count": 96, "persistent_runtime_optional_top_k": 64, "source_compare_json": "b.json"},
    ]
    model = train_persistent_residual_predictor(
        records,
        abs_threshold=0.05,
        feature_names=PERSISTENT_PREDICTOR_FEATURE_NAMES,
        steps=300,
        learning_rate=0.3,
        l2=1e-3,
    )
    payload = recommend_safe_then_cheapest_configs(model, records, abs_threshold=0.05)
    assert payload["summary"]["snapshot_group_count"] == 2
    assert len(payload["recommendations"]) == 2
    assert all("chosen_source_compare_json" in item for item in payload["recommendations"])
