from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_real_mixed_probe import (  # noqa: E402
    _summarize_records,
    real_mixed_probe_dotcache_config,
    real_mixed_probe_serving_config,
)


def test_real_mixed_probe_configs_match_canonical_stage9_settings() -> None:
    dotcache_config = real_mixed_probe_dotcache_config()
    serving_config = real_mixed_probe_serving_config(policy_path=None)

    assert dotcache_config.bits_k == 8
    assert dotcache_config.bits_v == 4
    assert dotcache_config.tokens_per_page == 16
    assert dotcache_config.group_size == 32
    assert serving_config.enable_early_exit is True
    assert serving_config.enable_full_attention_mixed_mode_execution is True
    assert serving_config.enable_compression is True
    assert serving_config.full_attention_mixed_mode_execution_strategy == "direct_m0"
    assert serving_config.full_attention_mixed_mode_execution_max_k_comp_error == 0.20
    assert serving_config.full_attention_check_interval == 16
    assert serving_config.full_attention_streaming_order_mode == "priority_value_hybrid"
    assert serving_config.full_attention_streaming_priority_value_upper_weight == 0.25
    assert serving_config.full_attention_key_centroid_count_by_layer == {19: 8, 23: 16}
    assert serving_config.full_attention_mixed_mode_detailed_timing is False
    assert serving_config.full_attention_mixed_mode_execution_max_k_comp_error_by_layer is None


def test_real_mixed_probe_summary_aggregates_query_prep_and_gather() -> None:
    summary = _summarize_records(
        [
            {
                "hand_tuned_decode_ms_per_step": 10.0,
                "bias_decode_ms_per_step": 8.0,
                "bias_matches_hand_tuned_exact": True,
                "hand_tuned_direct_m0_query_prep_ms_total": 1.0,
                "bias_direct_m0_query_prep_ms_total": 2.0,
                "hand_tuned_direct_m0_gather_ms_total": 3.0,
                "bias_direct_m0_gather_ms_total": 4.0,
                "hand_tuned_direct_m0_score_ms_total": 5.0,
                "bias_direct_m0_score_ms_total": 6.0,
                "hand_tuned_exact_m3_score_ms_total": 7.0,
                "bias_exact_m3_score_ms_total": 8.0,
                "hand_tuned_aux_exact_m3_score_ms_total": 0.5,
                "bias_aux_exact_m3_score_ms_total": 1.5,
                "hand_tuned_final_mix_ms_total": 9.0,
                "bias_final_mix_ms_total": 10.0,
                "hand_tuned_final_mix_logits_ms_total": 1.5,
                "bias_final_mix_logits_ms_total": 2.5,
                "hand_tuned_final_mix_softmax_ms_total": 3.5,
                "bias_final_mix_softmax_ms_total": 4.5,
                "hand_tuned_final_mix_value_ms_total": 5.5,
                "bias_final_mix_value_ms_total": 6.5,
                "hand_tuned_executed_m0_block_count_total": 11,
                "bias_executed_m0_block_count_total": 12,
                "hand_tuned_executed_m3_block_count_total": 13,
                "bias_executed_m3_block_count_total": 14,
                "hand_tuned_executed_exact_key_m3_block_count_total": 17,
                "bias_executed_exact_key_m3_block_count_total": 18,
            },
            {
                "hand_tuned_decode_ms_per_step": 14.0,
                "bias_decode_ms_per_step": 12.0,
                "bias_matches_hand_tuned_exact": False,
                "hand_tuned_direct_m0_query_prep_ms_total": 5.0,
                "bias_direct_m0_query_prep_ms_total": 6.0,
                "hand_tuned_direct_m0_gather_ms_total": 7.0,
                "bias_direct_m0_gather_ms_total": 8.0,
                "hand_tuned_direct_m0_score_ms_total": 9.0,
                "bias_direct_m0_score_ms_total": 10.0,
                "hand_tuned_exact_m3_score_ms_total": 11.0,
                "bias_exact_m3_score_ms_total": 12.0,
                "hand_tuned_aux_exact_m3_score_ms_total": 2.5,
                "bias_aux_exact_m3_score_ms_total": 3.5,
                "hand_tuned_final_mix_ms_total": 13.0,
                "bias_final_mix_ms_total": 14.0,
                "hand_tuned_final_mix_logits_ms_total": 7.5,
                "bias_final_mix_logits_ms_total": 8.5,
                "hand_tuned_final_mix_softmax_ms_total": 9.5,
                "bias_final_mix_softmax_ms_total": 10.5,
                "hand_tuned_final_mix_value_ms_total": 11.5,
                "bias_final_mix_value_ms_total": 12.5,
                "hand_tuned_executed_m0_block_count_total": 15,
                "bias_executed_m0_block_count_total": 16,
                "hand_tuned_executed_m3_block_count_total": 17,
                "bias_executed_m3_block_count_total": 18,
                "hand_tuned_executed_exact_key_m3_block_count_total": 21,
                "bias_executed_exact_key_m3_block_count_total": 22,
            },
        ]
    )

    assert summary["case_count"] == 2
    assert summary["hand_tuned_avg_ms_per_step"] == 12.0
    assert summary["bias_avg_ms_per_step"] == 10.0
    assert summary["bias_vs_hand_exact_match_rate"] == 0.5
    assert summary["bias_beats_hand_tuned_latency_rate"] == 1.0
    assert summary["hand_tuned_direct_m0_query_prep_ms_per_case"] == 3.0
    assert summary["bias_direct_m0_query_prep_ms_per_case"] == 4.0
    assert summary["hand_tuned_direct_m0_gather_ms_per_case"] == 5.0
    assert summary["bias_direct_m0_gather_ms_per_case"] == 6.0
    assert summary["hand_tuned_direct_m0_score_ms_per_case"] == 7.0
    assert summary["bias_direct_m0_score_ms_per_case"] == 8.0
    assert summary["hand_tuned_exact_m3_score_ms_per_case"] == 9.0
    assert summary["bias_exact_m3_score_ms_per_case"] == 10.0
    assert summary["hand_tuned_aux_exact_m3_score_ms_per_case"] == 1.5
    assert summary["bias_aux_exact_m3_score_ms_per_case"] == 2.5
    assert summary["hand_tuned_final_mix_ms_per_case"] == 11.0
    assert summary["bias_final_mix_ms_per_case"] == 12.0
    assert summary["hand_tuned_final_mix_logits_ms_per_case"] == 4.5
    assert summary["bias_final_mix_logits_ms_per_case"] == 5.5
    assert summary["hand_tuned_final_mix_softmax_ms_per_case"] == 6.5
    assert summary["bias_final_mix_softmax_ms_per_case"] == 7.5
    assert summary["hand_tuned_final_mix_value_ms_per_case"] == 8.5
    assert summary["bias_final_mix_value_ms_per_case"] == 9.5
    assert summary["hand_tuned_executed_m0_blocks_per_case"] == 13.0
    assert summary["bias_executed_m0_blocks_per_case"] == 14.0
    assert summary["hand_tuned_executed_m3_blocks_per_case"] == 15.0
    assert summary["bias_executed_m3_blocks_per_case"] == 16.0
    assert summary["hand_tuned_executed_exact_key_m3_blocks_per_case"] == 19.0
    assert summary["bias_executed_exact_key_m3_blocks_per_case"] == 20.0
