from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_full_attention_snapshot_compare import (
    _build_sequence_metrics,
    _resolve_policy_driven_config,
    _build_summary,
    _resolve_snapshot_records,
)
from dotcache.integrations.qwen35 import PersistentServingConfig


def test_resolve_snapshot_records_collects_manifest_and_explicit_paths(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    first.write_bytes(b"")
    second.write_bytes(b"")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_records": [
                    {"paged_attention_snapshot_path": str(first)},
                    {"paged_attention_snapshot_path": str(second)},
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = _resolve_snapshot_records(
        snapshot_paths=[str(first)],
        snapshot_glob=None,
        manifest_path=str(manifest),
    )

    assert [Path(record["snapshot_path"]) for record in resolved] == [first.resolve(), second.resolve()]


def test_build_summary_aggregates_snapshot_compare_records() -> None:
    summary = _build_summary(
        [
            {
                "selected_block_count": 4,
                "selected_token_count": 64,
                "full_block_count": 8,
                "full_token_count": 128,
                "max_abs_error": 0.01,
                "max_rel_error": 0.1,
            },
            {
                "selected_block_count": 2,
                "selected_token_count": 32,
                "full_block_count": 8,
                "full_token_count": 128,
                "max_abs_error": 0.02,
                "max_rel_error": 0.2,
            },
        ]
    )

    assert summary["snapshot_count"] == 2
    assert summary["max_abs_error"] == 0.02
    assert summary["max_rel_error"] == 0.2
    assert summary["avg_selected_block_count"] == 3.0
    assert summary["avg_selected_token_count"] == 48.0
    assert summary["avg_selected_fraction"] == 0.375


def test_build_sequence_metrics_groups_by_case_layer_head_and_step() -> None:
    sequence_summary = _build_sequence_metrics(
        [
            {
                "case_tag": "readme",
                "layer_id": 3,
                "kv_head_id": 0,
                "step_index": 0,
                "max_abs_error": 0.2,
                "selected_token_count": 64,
                "full_token_count": 128,
                "history_snapshot_count": 0,
            },
            {
                "case_tag": "readme",
                "layer_id": 3,
                "kv_head_id": 0,
                "step_index": 1,
                "max_abs_error": 0.1,
                "selected_token_count": 64,
                "full_token_count": 128,
                "history_snapshot_count": 1,
            },
            {
                "case_tag": "paper",
                "layer_id": 3,
                "kv_head_id": 0,
                "step_index": 0,
                "max_abs_error": 0.3,
                "selected_token_count": 32,
                "full_token_count": 128,
                "history_snapshot_count": 0,
            },
            {
                "case_tag": "paper",
                "layer_id": 3,
                "kv_head_id": 0,
                "step_index": 1,
                "max_abs_error": 0.15,
                "selected_token_count": 32,
                "full_token_count": 128,
                "history_snapshot_count": 1,
            },
        ]
    )

    assert sequence_summary["sequence_count"] == 2
    assert sequence_summary["terminal"]["max_abs_error"] == 0.15
    assert sequence_summary["by_step_index"]["0"]["count"] == 2
    assert sequence_summary["by_step_index"]["1"]["avg_history_snapshot_count"] == 1.0


def test_resolve_policy_driven_config_applies_matching_bucket() -> None:
    base_config = PersistentServingConfig(
        enable_priority=True,
        full_attention_optional_top_k=128,
        full_attention_optional_far_anchor_quota=0,
        full_attention_optional_far_anchor_priority_margin=0.0,
    )
    policy_payload = {
        "group_by": ["layer_id", "kv_head_id", "prompt_family", "step_bucket"],
        "group_count": 2,
        "groups": [
            {
                "bucket": {
                    "layer_id": 3,
                    "kv_head_id": 1,
                    "prompt_family": "paper",
                    "step_bucket": "bootstrap",
                },
                "snapshot_count": 1,
                "ranked_configs": [
                    {
                        "config_key": json.dumps(
                            {"persistent_runtime_optional_top_k": 64},
                            sort_keys=True,
                        ),
                        "source_compare_json": "compare.json",
                        "vote_count": 3,
                        "matched_oracle_rate": 1.0,
                        "chosen_safe_rate": 1.0,
                        "avg_selected_token_count": 2300.0,
                        "avg_max_abs_error": 0.03,
                    }
                ],
            },
            {
                "bucket": {
                    "layer_id": 3,
                    "kv_head_id": 1,
                    "prompt_family": "paper",
                    "step_bucket": "bootstrap",
                },
                "snapshot_count": 1,
                "ranked_configs": [
                    {
                        "config_key": json.dumps(
                            {
                                "persistent_runtime_recent_block_count": 64,
                                "persistent_runtime_mandatory_recent_block_count": 16,
                                "persistent_runtime_optional_top_k": 128,
                                "persistent_runtime_optional_upper_bound_quota": 16,
                                "persistent_runtime_optional_far_quota": 32,
                                "persistent_runtime_optional_mid_quota": 48,
                                "persistent_runtime_optional_near_quota": 32,
                                "persistent_runtime_optional_far_anchor_quota": 4,
                                "persistent_runtime_optional_far_anchor_priority_margin": 0.25,
                                "persistent_runtime_optional_diversity_weight": 0.0,
                                "persistent_runtime_optional_diversity_radius": 0,
                                "persistent_runtime_optional_diversity_min_history_count": 1,
                                "persistent_runtime_key_centroid_count": None,
                                "persistent_runtime_probe_refine_top_k": None,
                                "persistent_runtime_probe_sample_count": None,
                                "persistent_runtime_region_residual_caps": None,
                                "persistent_runtime_residual_cluster_count": None,
                            },
                            sort_keys=True,
                        ),
                        "source_compare_json": "compare.json",
                        "vote_count": 3,
                        "matched_oracle_rate": 1.0,
                        "chosen_safe_rate": 1.0,
                        "avg_selected_token_count": 2300.0,
                        "avg_max_abs_error": 0.03,
                    }
                ],
            }
        ],
    }
    effective_config, choice = _resolve_policy_driven_config(
        base_config=base_config,
        shortlist_policy_payload=policy_payload,
        case_tag="paper",
        layer_id=3,
        kv_head_id=1,
        step_index=0,
    )
    assert choice is not None
    assert effective_config.full_attention_optional_far_anchor_quota == 4
    assert effective_config.full_attention_optional_far_anchor_priority_margin == 0.25


def test_resolve_policy_driven_config_respects_shortlist_policy_step_gate() -> None:
    base_config = PersistentServingConfig(
        enable_priority=True,
        full_attention_optional_top_k=128,
        full_attention_shortlist_policy_min_step_index=1,
    )
    policy_payload = {
        "group_by": ["layer_id", "kv_head_id", "prompt_family", "step_bucket"],
        "group_count": 1,
        "groups": [
            {
                "bucket": {
                    "layer_id": 3,
                    "kv_head_id": 1,
                    "prompt_family": "paper",
                    "step_bucket": "mid",
                },
                "snapshot_count": 1,
                "ranked_configs": [
                    {
                        "config_key": json.dumps(
                            {"persistent_runtime_optional_top_k": 96},
                            sort_keys=True,
                        ),
                        "source_compare_json": "compare.json",
                        "vote_count": 3,
                        "matched_oracle_rate": 1.0,
                        "chosen_safe_rate": 1.0,
                        "avg_selected_token_count": 2300.0,
                        "avg_max_abs_error": 0.03,
                    }
                ],
            }
        ],
    }
    bootstrap_config, bootstrap_choice = _resolve_policy_driven_config(
        base_config=base_config,
        shortlist_policy_payload=policy_payload,
        case_tag="paper",
        layer_id=3,
        kv_head_id=1,
        step_index=0,
    )
    mid_config, mid_choice = _resolve_policy_driven_config(
        base_config=base_config,
        shortlist_policy_payload=policy_payload,
        case_tag="paper",
        layer_id=3,
        kv_head_id=1,
        step_index=1,
    )
    assert bootstrap_choice is None
    assert bootstrap_config.full_attention_optional_top_k == 128
    assert mid_choice is not None
    assert mid_config.full_attention_optional_top_k == 96


def test_resolve_policy_driven_config_assist_mode_preserves_diversity() -> None:
    base_config = PersistentServingConfig(
        enable_priority=True,
        full_attention_shortlist_policy_mode="assist",
        full_attention_optional_top_k=128,
        full_attention_optional_diversity_weight=0.5,
        full_attention_optional_diversity_radius=4,
    )
    policy_payload = {
        "group_by": ["layer_id", "kv_head_id", "prompt_family", "step_bucket"],
        "group_count": 1,
        "groups": [
            {
                "bucket": {
                    "layer_id": 3,
                    "kv_head_id": 1,
                    "prompt_family": "paper",
                    "step_bucket": "mid",
                },
                "snapshot_count": 1,
                "ranked_configs": [
                    {
                        "config_key": json.dumps(
                            {
                                "persistent_runtime_optional_top_k": 96,
                                "persistent_runtime_optional_diversity_weight": 0.0,
                                "persistent_runtime_optional_diversity_radius": 0,
                            },
                            sort_keys=True,
                        ),
                        "source_compare_json": "compare.json",
                        "vote_count": 3,
                        "matched_oracle_rate": 1.0,
                        "chosen_safe_rate": 1.0,
                        "avg_selected_token_count": 2300.0,
                        "avg_max_abs_error": 0.03,
                    }
                ],
            }
        ],
    }
    effective_config, choice = _resolve_policy_driven_config(
        base_config=base_config,
        shortlist_policy_payload=policy_payload,
        case_tag="paper",
        layer_id=3,
        kv_head_id=1,
        step_index=1,
    )
    assert choice is not None
    assert effective_config.full_attention_optional_top_k == 96
    assert effective_config.full_attention_optional_diversity_weight == 0.5
    assert effective_config.full_attention_optional_diversity_radius == 4
