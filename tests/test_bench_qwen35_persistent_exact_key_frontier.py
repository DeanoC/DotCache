from __future__ import annotations

from benchmarks.bench_qwen35_persistent_exact_key_frontier import (
    _discover_candidate_layers,
    _summarize_records,
    _summarize_sweep,
)


def test_discover_candidate_layers_ignores_zero_counts() -> None:
    records = [
        {"executed_exact_key_m3_by_layer": {15: 8.0, 23: 0.0}},
        {"executed_exact_key_m3_by_layer": {15: 4.0, 19: 2.0}},
    ]

    assert _discover_candidate_layers(records) == [15, 19]


def test_summarize_records_averages_layer_maps() -> None:
    records = [
        {
            "decode_ms_per_step": 10.0,
            "executed_exact_key_m3_by_layer": {15: 8.0},
            "executed_m0_by_layer": {15: 100.0},
            "direct_m0_gather_ms_by_layer": {15: 3.0},
            "direct_m0_score_ms_by_layer": {15: 4.0},
            "exact_m3_score_ms_by_layer": {15: 1.0},
        },
        {
            "decode_ms_per_step": 14.0,
            "executed_exact_key_m3_by_layer": {15: 4.0, 19: 2.0},
            "executed_m0_by_layer": {15: 90.0, 19: 10.0},
            "direct_m0_gather_ms_by_layer": {15: 5.0, 19: 1.0},
            "direct_m0_score_ms_by_layer": {15: 7.0, 19: 2.0},
            "exact_m3_score_ms_by_layer": {15: 2.0, 19: 0.5},
        },
    ]

    summary = _summarize_records(records)

    assert summary["case_count"] == 2
    assert summary["bias_avg_ms_per_step"] == 12.0
    assert summary["executed_exact_key_m3_blocks_per_case"] == 7.0
    assert summary["executed_exact_key_m3_by_layer_per_case"] == {"15": 6.0, "19": 1.0}
    assert summary["executed_m0_by_layer_per_case"] == {"15": 95.0, "19": 5.0}
    assert summary["direct_m0_gather_ms_per_case_by_layer"] == {"15": 4.0, "19": 0.5}
    assert summary["direct_m0_score_ms_per_case_by_layer"] == {"15": 5.5, "19": 1.0}
    assert summary["exact_m3_score_ms_per_case_by_layer"] == {"15": 1.5, "19": 0.25}


def test_summarize_sweep_compares_against_baseline_ids() -> None:
    baseline_records = [
        {"case_tag": "a", "generated_ids": [1, 2], "decode_ms_per_step": 10.0, "executed_exact_key_m3_by_layer": {15: 8.0}},
        {"case_tag": "b", "generated_ids": [3, 4], "decode_ms_per_step": 14.0, "executed_exact_key_m3_by_layer": {15: 4.0}},
    ]
    sweep_records = [
        {"case_tag": "a", "generated_ids": [1, 2], "decode_ms_per_step": 11.0, "executed_exact_key_m3_by_layer": {15: 0.0}},
        {"case_tag": "b", "generated_ids": [3, 9], "decode_ms_per_step": 15.0, "executed_exact_key_m3_by_layer": {15: 2.0}},
    ]

    summary = _summarize_sweep(baseline_records=baseline_records, sweep_records=sweep_records)

    assert summary["bias_avg_ms_per_step"] == 13.0
    assert summary["delta_vs_baseline_ms_per_step"] == 1.0
    assert summary["bias_vs_baseline_exact_match_rate"] == 0.5
    assert summary["executed_exact_key_m3_blocks_per_case"] == 1.0
