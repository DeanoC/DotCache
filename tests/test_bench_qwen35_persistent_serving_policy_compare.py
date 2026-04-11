from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_qwen35_persistent_serving_policy_compare import (
    _persistent_base_config,
    _resolve_prompt_records,
    _summarize_records,
)


def test_resolve_prompt_records_reads_manifest_and_explicit_files(tmp_path) -> None:
    prompt_a = tmp_path / "a.md"
    prompt_b = tmp_path / "b.md"
    prompt_a.write_text("alpha", encoding="utf-8")
    prompt_b.write_text("beta", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "case_tag": "alpha_case",
                        "prompt_file_path": str(prompt_a),
                        "prompt_length": 128,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    records = _resolve_prompt_records(
        manifest_path=str(manifest),
        prompt_files=[str(prompt_b)],
        prompt_file_target_length=64,
    )

    assert [record["case_tag"] for record in records] == ["alpha_case", "b"]
    assert [record["prompt_length"] for record in records] == [128, 64]


def test_summarize_records_aggregates_serving_matches_and_latency() -> None:
    summary = _summarize_records(
        [
            {
                "dense_decode_ms_per_step": 10.0,
                "hand_tuned_decode_ms_per_step": 8.0,
                "bias_decode_ms_per_step": 7.0,
                "hand_tuned_matches_dense_exact": True,
                "bias_matches_dense_exact": True,
                "bias_matches_hand_tuned_exact": True,
                "hand_tuned_policy_resolve_ms_total": 1.0,
                "bias_policy_resolve_ms_total": 2.0,
                "hand_tuned_score_ms_total": 3.0,
                "bias_score_ms_total": 4.0,
                "hand_tuned_selection_ms_total": 5.0,
                "bias_selection_ms_total": 6.0,
                "hand_tuned_optional_selection_ms_total": 2.0,
                "bias_optional_selection_ms_total": 3.0,
                "hand_tuned_diverse_selection_ms_total": 1.0,
                "bias_diverse_selection_ms_total": 2.0,
                "hand_tuned_compression_selection_ms_total": 0.5,
                "bias_compression_selection_ms_total": 1.5,
                "hand_tuned_policy_bias_ms_total": 0.0,
                "bias_policy_bias_ms_total": 1.5,
                "hand_tuned_direct_m0_assembly_ms_total": 10.0,
                "bias_direct_m0_assembly_ms_total": 20.0,
                "hand_tuned_direct_m0_score_ms_total": 30.0,
                "bias_direct_m0_score_ms_total": 40.0,
                "hand_tuned_exact_m3_score_ms_total": 50.0,
                "bias_exact_m3_score_ms_total": 60.0,
                "hand_tuned_final_mix_ms_total": 70.0,
                "bias_final_mix_ms_total": 80.0,
            },
            {
                "dense_decode_ms_per_step": 12.0,
                "hand_tuned_decode_ms_per_step": 9.0,
                "bias_decode_ms_per_step": 11.0,
                "hand_tuned_matches_dense_exact": False,
                "bias_matches_dense_exact": True,
                "bias_matches_hand_tuned_exact": False,
                "hand_tuned_policy_resolve_ms_total": 3.0,
                "bias_policy_resolve_ms_total": 4.0,
                "hand_tuned_score_ms_total": 5.0,
                "bias_score_ms_total": 6.0,
                "hand_tuned_selection_ms_total": 7.0,
                "bias_selection_ms_total": 8.0,
                "hand_tuned_optional_selection_ms_total": 4.0,
                "bias_optional_selection_ms_total": 5.0,
                "hand_tuned_diverse_selection_ms_total": 2.0,
                "bias_diverse_selection_ms_total": 3.0,
                "hand_tuned_compression_selection_ms_total": 1.5,
                "bias_compression_selection_ms_total": 2.5,
                "hand_tuned_policy_bias_ms_total": 0.0,
                "bias_policy_bias_ms_total": 2.5,
                "hand_tuned_direct_m0_assembly_ms_total": 12.0,
                "bias_direct_m0_assembly_ms_total": 22.0,
                "hand_tuned_direct_m0_score_ms_total": 32.0,
                "bias_direct_m0_score_ms_total": 42.0,
                "hand_tuned_exact_m3_score_ms_total": 52.0,
                "bias_exact_m3_score_ms_total": 62.0,
                "hand_tuned_final_mix_ms_total": 72.0,
                "bias_final_mix_ms_total": 82.0,
            },
        ]
    )

    assert summary["case_count"] == 2
    assert summary["dense_avg_ms_per_step"] == 11.0
    assert summary["hand_tuned_avg_ms_per_step"] == 8.5
    assert summary["bias_avg_ms_per_step"] == 9.0
    assert summary["hand_vs_dense_exact_match_rate"] == 0.5
    assert summary["bias_vs_dense_exact_match_rate"] == 1.0
    assert summary["bias_vs_hand_exact_match_rate"] == 0.5
    assert summary["bias_beats_hand_tuned_latency_rate"] == 0.5
    assert summary["hand_tuned_policy_resolve_ms_per_case"] == 2.0
    assert summary["bias_policy_resolve_ms_per_case"] == 3.0
    assert summary["hand_tuned_score_ms_per_case"] == 4.0
    assert summary["bias_score_ms_per_case"] == 5.0
    assert summary["hand_tuned_selection_ms_per_case"] == 6.0
    assert summary["bias_selection_ms_per_case"] == 7.0
    assert summary["hand_tuned_optional_selection_ms_per_case"] == 3.0
    assert summary["bias_optional_selection_ms_per_case"] == 4.0
    assert summary["hand_tuned_diverse_selection_ms_per_case"] == 1.5
    assert summary["bias_diverse_selection_ms_per_case"] == 2.5
    assert summary["hand_tuned_compression_selection_ms_per_case"] == 1.0
    assert summary["bias_compression_selection_ms_per_case"] == 2.0
    assert summary["hand_tuned_policy_bias_ms_per_case"] == 0.0
    assert summary["bias_policy_bias_ms_per_case"] == 2.0
    assert summary["hand_tuned_direct_m0_assembly_ms_per_case"] == 11.0
    assert summary["bias_direct_m0_assembly_ms_per_case"] == 21.0
    assert summary["hand_tuned_direct_m0_score_ms_per_case"] == 31.0
    assert summary["bias_direct_m0_score_ms_per_case"] == 41.0
    assert summary["hand_tuned_exact_m3_score_ms_per_case"] == 51.0
    assert summary["bias_exact_m3_score_ms_per_case"] == 61.0
    assert summary["hand_tuned_final_mix_ms_per_case"] == 71.0
    assert summary["bias_final_mix_ms_per_case"] == 81.0


def test_persistent_base_config_enables_compression_for_mixed_execution() -> None:
    config = _persistent_base_config(enable_mixed_execution=True, mixed_execution_strategy="direct_m0")

    assert config.enable_full_attention_mixed_mode_execution is True
    assert config.enable_compression is True
    assert config.full_attention_mixed_mode_execution_strategy == "direct_m0"
