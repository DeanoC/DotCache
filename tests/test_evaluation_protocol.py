from __future__ import annotations

from dotcache.evaluation_protocol import EvaluationMetadata, build_evaluation_record, page_format_histogram_from_result


def test_page_format_histogram_from_result_collects_k_and_v_modes() -> None:
    result = {
        "k_m0_pages": 10,
        "k_m3_pages": 2,
        "v_m0_pages": 7,
        "v_m1_pages": 1,
    }

    histogram = page_format_histogram_from_result(result)

    assert histogram == {
        "K:M0": 10,
        "K:M3": 2,
        "V:M0": 7,
        "V:M1": 1,
    }


def test_build_evaluation_record_derives_metrics_from_existing_result() -> None:
    metadata = EvaluationMetadata(
        model_id="Qwen/Qwen3.5-0.8B",
        model_family="qwen35",
        backend="torch_cuda",
        device="cuda",
        torch_dtype="float16",
        split="held_out",
        lane="systems",
        prompt_family="synthetic_exact_length",
        dataset_name="needle_pack_v1",
        prompt_count=4,
        batch_size=1,
        truth_type="paged_runtime",
        effective_budget_rule="resident_bytes_plus_metadata_over_context_tokens",
        context_length=8192,
        decode_steps=8,
    )
    result = {
        "dotcache_decode_ms_per_step": 1.25,
        "resident_bytes": 32768,
        "execution_shortlist_selected_pages": 81,
        "execution_shortlist_total_pages": 2992,
        "execution_decode_shortlist_selection_ms_total": 0.8,
        "execution_decode_shortlist_candidate_approx_scoring_ms_total": 0.2,
        "execution_decode_shortlist_materialization_ms_total": 0.1,
        "execution_decode_backend_call_non_backend_ms_total": 0.3,
        "k_m0_pages": 12,
        "v_m3_pages": 4,
    }

    record = build_evaluation_record(metadata, result).to_dict()

    assert record["metadata"]["model_id"] == "Qwen/Qwen3.5-0.8B"
    assert record["metrics"]["dotcache_decode_ms_per_step"] == 1.25
    assert record["metrics"]["resident_bytes"] == 32768
    assert record["metrics"]["effective_bytes_per_token"] == 4.0
    assert record["metrics"]["page_format_histogram"] == {"K:M0": 12, "V:M3": 4}
    assert record["source_result"]["execution_decode_shortlist_materialization_ms_total"] == 0.1
