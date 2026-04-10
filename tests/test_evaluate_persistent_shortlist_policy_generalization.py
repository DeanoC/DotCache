from __future__ import annotations

from scripts.evaluate_persistent_shortlist_policy_generalization import (
    evaluate_persistent_shortlist_policy_generalization,
)


def _record(snapshot_path: str, *, max_abs_error: float, selected_token_count: int, top_k: int) -> dict[str, object]:
    return {
        "snapshot_path": snapshot_path,
        "selected_token_count": selected_token_count,
        "max_abs_error": max_abs_error,
        "beta_upper": 0.1,
        "source_compare_json": "compare.json",
        "persistent_runtime_recent_block_count": 64,
        "persistent_runtime_mandatory_recent_block_count": 16,
        "persistent_runtime_optional_top_k": top_k,
        "persistent_runtime_optional_upper_bound_quota": 16,
        "persistent_runtime_optional_far_quota": 32,
        "persistent_runtime_optional_mid_quota": 48,
        "persistent_runtime_optional_near_quota": 32,
        "persistent_runtime_optional_far_anchor_quota": 4,
        "persistent_runtime_optional_far_anchor_priority_margin": 0.25,
        "persistent_runtime_optional_diversity_weight": 0.5,
        "persistent_runtime_optional_diversity_radius": 4,
        "persistent_runtime_optional_diversity_min_history_count": 1,
        "persistent_runtime_key_centroid_count": None,
        "persistent_runtime_probe_refine_top_k": None,
        "persistent_runtime_probe_sample_count": None,
        "persistent_runtime_region_residual_caps": None,
        "persistent_runtime_residual_cluster_count": None,
    }


def test_generalization_eval_distinguishes_prompt_family_dependent_policy() -> None:
    recommendations = {
        "recommendations": [
            {
                "snapshot_path": "/tmp/family_a/layer03_kv00_step+00.npz",
                "chosen_config_key": '{"persistent_runtime_optional_top_k": 96}',
                "chosen_source_compare_json": "a.json",
                "chosen_selected_token_count": 112.0,
                "chosen_max_abs_error": 0.02,
                "chosen_is_safe": True,
                "matched_oracle": True,
            },
            {
                "snapshot_path": "/tmp/family_b/layer03_kv00_step+00.npz",
                "chosen_config_key": '{"persistent_runtime_optional_top_k": 96}',
                "chosen_source_compare_json": "a.json",
                "chosen_selected_token_count": 120.0,
                "chosen_max_abs_error": 0.03,
                "chosen_is_safe": True,
                "matched_oracle": True,
            },
        ]
    }
    records = [
        _record("/tmp/family_a/layer03_kv00_step+00.npz", max_abs_error=0.02, selected_token_count=112, top_k=96),
        _record("/tmp/family_a/layer03_kv00_step+00.npz", max_abs_error=0.20, selected_token_count=96, top_k=64),
        _record("/tmp/family_b/layer03_kv00_step+00.npz", max_abs_error=0.03, selected_token_count=120, top_k=96),
        _record("/tmp/family_b/layer03_kv00_step+00.npz", max_abs_error=0.25, selected_token_count=88, top_k=64),
    ]
    payload = evaluate_persistent_shortlist_policy_generalization(
        recommendations,
        records,
        group_bys=[
            ["layer_id", "kv_head_id", "step_bucket"],
            ["layer_id", "kv_head_id", "prompt_family", "step_bucket"],
        ],
        abs_threshold=0.05,
    )
    assert payload["prompt_families"] == ["family_a", "family_b"]
    by_key = {result["group_key"]: result for result in payload["results"]}
    runtime_friendly = by_key["layer_id__kv_head_id__step_bucket"]
    prompt_specific = by_key["layer_id__kv_head_id__prompt_family__step_bucket"]
    assert runtime_friendly["aggregate_summary"]["missing_bucket_rate"] == 0.0
    assert prompt_specific["aggregate_summary"]["missing_bucket_rate"] == 1.0
    assert prompt_specific["aggregate_summary"]["fallback_rate"] == 1.0
