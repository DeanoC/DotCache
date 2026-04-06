from __future__ import annotations

import json

import numpy as np

from dotcache.config import DotCacheConfig
from dotcache.page_oracle import (
    OracleDatasetSplitSummary,
    OracleDatasetSplitSuiteSpec,
    OracleThresholds,
    PageTraceRecord,
    build_oracle_label_records,
    build_selector_training_rows,
    build_selector_candidate_training_rows,
    load_oracle_dataset_split_manifest,
    load_page_trace,
    load_oracle_label_records,
    load_page_trace_manifest,
    materialize_oracle_dataset_split,
    materialize_oracle_dataset_split_suite,
    merge_page_trace_manifests,
    run_oracle_labeling,
    run_oracle_batch_replay,
    run_oracle_replay,
    save_oracle_labels,
    save_oracle_dataset_split_manifest,
    save_selector_candidate_training_rows,
    save_selector_training_rows,
    save_page_trace,
    save_page_trace_manifest,
    select_page_trace_paths,
    upsert_oracle_dataset_split_manifest_entry,
)
from dotcache.planner import parse_page_mode_token


def test_page_trace_round_trip_and_oracle_replay(tmp_path) -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(16, 32)).astype(np.float32)
    query = rng.normal(size=(32,)).astype(np.float32)
    record = PageTraceRecord(
        source="unit-test",
        kind="K",
        layer_id=3,
        kv_head_id=1,
        token_start=64,
        token_age=256,
        values=values,
        query=query,
    )

    trace_path = tmp_path / "page_trace.npz"
    save_page_trace(record, trace_path)
    loaded = load_page_trace(trace_path)

    assert loaded.kind == "K"
    assert loaded.layer_id == 3
    assert loaded.kv_head_id == 1
    assert loaded.token_start == 64
    assert loaded.token_age == 256
    assert np.allclose(loaded.values, values)
    assert np.allclose(loaded.query, query)

    replay = run_oracle_replay(
        loaded,
        base_config=DotCacheConfig(head_dim=32, group_size=16, tokens_per_page=16),
        candidates=(
            parse_page_mode_token("M0/affine/4"),
            parse_page_mode_token("M3/affine/4/float16"),
        ),
        thresholds=OracleThresholds(
            max_mean_abs_error_ratio=0.50,
            max_max_abs_error_ratio=4.0,
            max_token_p95_error_ratio=2.0,
            max_score_max_abs_error=1.0,
            min_score_topk_agreement=0.50,
        ),
    )

    assert replay.cheapest_safe_candidate is not None
    assert len(replay.candidates) == 2
    exact_candidate = next(candidate for candidate in replay.candidates if candidate.mode == "M3")
    assert exact_candidate.safe is True
    assert exact_candidate.mean_abs_error < 1e-3
    assert exact_candidate.total_bytes >= 16 * 32


def test_oracle_batch_replay_samples_manifest_and_summarizes_candidates(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    trace_paths: list[str] = []
    for index, (stage, kind, token_start) in enumerate(
        [
            ("prefill", "K", 0),
            ("prefill", "V", 0),
            ("decode", "K", 128),
            ("decode", "V", 128),
        ]
    ):
        values = np.full((4, 8), fill_value=float(index + 1), dtype=np.float32)
        record = PageTraceRecord(
            source="unit-test",
            kind=kind,  # type: ignore[arg-type]
            layer_id=1,
            kv_head_id=0,
            token_start=token_start,
            token_age=8 - index,
            values=values,
            query=np.ones((8,), dtype=np.float32) if stage == "decode" else None,
            notes=[f"stage={stage}"],
        )
        trace_path = tmp_path / f"{stage}_{kind}_{index}.npz"
        save_page_trace(record, trace_path)
        trace_paths.append(str(trace_path))

    manifest_payload = {
        "output_dir": str(tmp_path),
        "page_trace_count": len(trace_paths),
        "page_trace_paths": trace_paths,
        "tokens_per_page": 4,
    }
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    loaded_manifest = load_page_trace_manifest(manifest_path)
    selected_paths = select_page_trace_paths(
        loaded_manifest,
        max_per_stage_kind=1,
        seed=0,
        kinds=("K", "V"),
        stages=("prefill", "decode"),
    )

    assert len(selected_paths) == 4

    replay = run_oracle_batch_replay(
        manifest_path,
        group_size=4,
        max_per_stage_kind=1,
        seed=0,
        candidates=(
            parse_page_mode_token("M0/affine/4"),
            parse_page_mode_token("M3/affine/4/float16"),
        ),
        thresholds=OracleThresholds(
            max_mean_abs_error_ratio=0.50,
            max_max_abs_error_ratio=4.0,
            max_token_p95_error_ratio=2.0,
            max_score_max_abs_error=100.0,
            min_score_topk_agreement=0.50,
        ),
    )

    assert replay.selected_trace_count == 4
    assert replay.selected_trace_counts_by_stage == {"decode": 2, "prefill": 2}
    assert replay.selected_trace_counts_by_kind == {"K": 2, "V": 2}
    assert replay.cheapest_safe_candidate_histogram == {"M0/affine/4": 4}
    assert "candidate | selected | safe_count | safe_rate" in replay.summary_table_markdown
    assert replay.candidate_stats["M0/affine/4"]["selected_count"] == 4

    labels = build_oracle_label_records(replay)
    assert len(labels) == 4
    assert labels[0].candidate_labels
    assert labels[0].query_present is (labels[0].stage == "decode")
    assert "prompt_family" in labels[0].to_dict()

    selector_rows = build_selector_training_rows(labels)
    assert len(selector_rows) == 4
    assert selector_rows[0].target_present is True
    assert selector_rows[0].safe_candidate_count >= 1


def test_oracle_labeling_writes_jsonl_bundle(tmp_path) -> None:
    trace_paths: list[str] = []
    for index, kind in enumerate(("K", "V")):
        record = PageTraceRecord(
            source="unit-test",
            kind=kind,  # type: ignore[arg-type]
            layer_id=2,
            kv_head_id=0,
            token_start=index * 4,
            token_age=4,
            values=np.full((4, 8), fill_value=float(index + 1), dtype=np.float32),
            query=np.ones((8,), dtype=np.float32) if kind == "K" else None,
            notes=["stage=decode"],
        )
        trace_path = tmp_path / f"trace_{kind}.npz"
        save_page_trace(record, trace_path)
        trace_paths.append(str(trace_path))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "output_dir": str(tmp_path),
                "page_trace_count": len(trace_paths),
                "page_trace_paths": trace_paths,
                "tokens_per_page": 4,
            }
        ),
        encoding="utf-8",
    )

    labeling = run_oracle_labeling(
        manifest_path,
        group_size=4,
        candidates=(
            parse_page_mode_token("M0/affine/4"),
            parse_page_mode_token("M3/affine/4/float16"),
        ),
        thresholds=OracleThresholds(
            max_mean_abs_error_ratio=0.50,
            max_max_abs_error_ratio=4.0,
            max_token_p95_error_ratio=2.0,
            max_score_max_abs_error=100.0,
            min_score_topk_agreement=0.50,
        ),
    )

    labels_path = tmp_path / "labels.jsonl"
    summary_path = tmp_path / "summary.json"
    save_oracle_labels(labeling, labels_path=labels_path, summary_path=summary_path)

    lines = labels_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first_label = json.loads(lines[0])
    assert first_label["trace_path"].endswith(".npz")
    assert "candidate_labels" in first_label
    assert "safe_candidates" in first_label
    assert "query_present" in first_label
    assert "prompt_family" in first_label

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["label_count"] == 2
    assert "summary_table_markdown" in summary_payload

    selector_rows = build_selector_training_rows(labeling.labels)
    selector_path = tmp_path / "selector_dataset.jsonl"
    save_selector_training_rows(selector_rows, selector_path)

    selector_lines = selector_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(selector_lines) == 2
    first_selector_row = json.loads(selector_lines[0])
    assert "target_candidate" in first_selector_row
    assert "trace_rms" in first_selector_row
    assert "query_present" in first_selector_row
    assert "prompt_family" in first_selector_row


def test_merge_page_trace_manifests_combines_counts_and_paths(tmp_path) -> None:
    manifest_a = {
        "output_dir": str(tmp_path / "a"),
        "page_trace_count": 2,
        "page_trace_paths": ["/tmp/a0.npz", "/tmp/a1.npz"],
        "page_trace_counts_by_kind": {"K": 1, "V": 1},
        "page_trace_counts_by_stage": {"prefill": 2},
        "page_trace_counts_by_layer": {"3": 2},
        "tokens_per_page": 2,
        "kinds": ["K", "V"],
        "source": "a",
    }
    manifest_b = {
        "output_dir": str(tmp_path / "b"),
        "page_trace_count": 1,
        "page_trace_paths": ["/tmp/b0.npz"],
        "page_trace_counts_by_kind": {"K": 1},
        "page_trace_counts_by_stage": {"decode": 1},
        "page_trace_counts_by_layer": {"7": 1},
        "tokens_per_page": 2,
        "kinds": ["K"],
        "source": "b",
    }
    manifest_path = tmp_path / "merged.json"
    merged = merge_page_trace_manifests((manifest_a, manifest_b), output_dir=tmp_path, source="merged")
    save_page_trace_manifest(merged, manifest_path)
    loaded = load_page_trace_manifest(manifest_path)

    assert loaded["page_trace_count"] == 3
    assert loaded["page_trace_counts_by_kind"] == {"K": 2, "V": 1}
    assert loaded["page_trace_counts_by_stage"] == {"decode": 1, "prefill": 2}
    assert loaded["page_trace_counts_by_layer"] == {"3": 2, "7": 1}
    assert loaded["tokens_per_page"] == 2
    assert loaded["member_manifest_count"] == 2


def test_prompt_family_and_variant_are_inferred_from_variant_run_paths(tmp_path) -> None:
    record = PageTraceRecord(
        source="unit-test",
        kind="K",
        layer_id=3,
        kv_head_id=0,
        token_start=0,
        token_age=0,
        values=np.ones((2, 4), dtype=np.float32),
        notes=["stage=prefill"],
    )
    trace_dir = tmp_path / "family-retrieval_variant-transcript_prompt032_decode02"
    trace_path = trace_dir / "trace.npz"
    save_page_trace(record, trace_path)

    manifest = {
        "output_dir": str(trace_dir),
        "page_trace_count": 1,
        "page_trace_paths": [str(trace_path)],
        "tokens_per_page": 2,
    }
    replay = run_oracle_batch_replay(
        manifest,
        group_size=4,
        candidates=(parse_page_mode_token("M0/affine/4"),),
        thresholds=OracleThresholds(
            max_mean_abs_error_ratio=10.0,
            max_max_abs_error_ratio=10.0,
            max_token_p95_error_ratio=10.0,
        ),
    )

    labels = build_oracle_label_records(replay)

    assert labels[0].prompt_family == "retrieval"
    assert labels[0].prompt_variant == "transcript"


def test_materialize_oracle_dataset_split_freezes_aligned_train_test_bundles(tmp_path) -> None:
    labels = [
        {
            "trace_path": str(tmp_path / "trace_cache.npz"),
            "stage": "prefill",
            "prompt_family": "cache",
            "prompt_variant": "locality",
            "source": "unit-test",
            "kind": "K",
            "layer_id": 3,
            "kv_head_id": 0,
            "token_start": 0,
            "token_age": 8,
            "token_count": 2,
            "head_dim": 8,
            "query_present": False,
            "cheapest_safe_candidate": "M0/affine/4",
            "safe_candidates": ["M0/affine/4"],
            "best_safe_total_bytes": 64,
            "candidate_labels": [
                {
                    "candidate": "M0/affine/4",
                    "mode": "M0",
                    "bits": 4,
                    "quant_scheme": "affine",
                    "payload_bytes": 48,
                    "metadata_bytes": 16,
                    "total_bytes": 64,
                    "safe": True,
                }
            ],
            "trace_stats": {"rms": 1.0},
            "notes": ["stage=prefill"],
        },
        {
            "trace_path": str(tmp_path / "trace_reasoning.npz"),
            "stage": "decode",
            "prompt_family": "reasoning",
            "prompt_variant": "logic",
            "source": "unit-test",
            "kind": "V",
            "layer_id": 23,
            "kv_head_id": 0,
            "token_start": 2,
            "token_age": 0,
            "token_count": 2,
            "head_dim": 8,
            "query_present": True,
            "cheapest_safe_candidate": "M3/affine/4/float16",
            "safe_candidates": ["M3/affine/4/float16"],
            "best_safe_total_bytes": 96,
            "candidate_labels": [
                {
                    "candidate": "M3/affine/4/float16",
                    "mode": "M3",
                    "bits": 4,
                    "quant_scheme": "affine",
                    "payload_bytes": 64,
                    "metadata_bytes": 32,
                    "total_bytes": 96,
                    "safe": True,
                }
            ],
            "trace_stats": {"rms": 2.0},
            "notes": ["stage=decode"],
        },
    ]
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in labels) + "\n", encoding="utf-8")

    loaded_labels = load_oracle_label_records(labels_path)
    selector_rows = build_selector_training_rows(loaded_labels)
    selector_candidate_rows = build_selector_candidate_training_rows(loaded_labels)
    selector_dataset_path = tmp_path / "selector_dataset.jsonl"
    selector_candidate_dataset_path = tmp_path / "selector_candidate_dataset.jsonl"
    save_selector_training_rows(selector_rows, selector_dataset_path)
    save_selector_candidate_training_rows(selector_candidate_rows, selector_candidate_dataset_path)

    output_dir = tmp_path / "split"
    summary = materialize_oracle_dataset_split(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
        selector_candidate_dataset_path=selector_candidate_dataset_path,
        output_dir=output_dir,
        holdout_prompt_families=("reasoning",),
        split_name="holdout_reasoning",
    )

    assert summary.split_name == "holdout_reasoning"
    assert summary.train_label_count == 1
    assert summary.test_label_count == 1
    assert summary.test_prompt_family_histogram == {"reasoning": 1}
    assert summary.train_prompt_family_histogram == {"cache": 1}

    train_labels = [json.loads(line) for line in (output_dir / "train" / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    test_labels = [json.loads(line) for line in (output_dir / "test" / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    test_selector_rows = [json.loads(line) for line in (output_dir / "test" / "selector_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    test_candidate_rows = [json.loads(line) for line in (output_dir / "test" / "selector_candidate_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    assert train_labels[0]["prompt_family"] == "cache"
    assert test_labels[0]["prompt_family"] == "reasoning"
    assert test_selector_rows[0]["trace_path"] == test_labels[0]["trace_path"]
    assert test_candidate_rows[0]["trace_path"] == test_labels[0]["trace_path"]
    assert json.loads((output_dir / "split_summary.json").read_text(encoding="utf-8"))["split_name"] == "holdout_reasoning"


def test_oracle_dataset_split_manifest_upserts_entries_in_order(tmp_path) -> None:
    manifest_path = tmp_path / "split_manifest.json"
    save_oracle_dataset_split_manifest({"manifest_version": 1, "split_count": 0, "splits": []}, manifest_path)

    split_a = OracleDatasetSplitSummary(
        split_name="split_a",
        holdout_prompt_families=["reasoning"],
        holdout_prompt_variants=[],
        holdout_layers=[],
        train_trace_paths=["a_train"],
        test_trace_paths=["a_test"],
        train_label_count=10,
        test_label_count=2,
        train_selector_row_count=10,
        test_selector_row_count=2,
        train_selector_candidate_row_count=20,
        test_selector_candidate_row_count=4,
        train_prompt_family_histogram={"cache": 10},
        test_prompt_family_histogram={"reasoning": 2},
        train_prompt_variant_histogram={"locality": 10},
        test_prompt_variant_histogram={"logic": 2},
        train_layer_histogram={"3": 10},
        test_layer_histogram={"23": 2},
    )
    split_b = OracleDatasetSplitSummary(
        split_name="split_b",
        holdout_prompt_families=[],
        holdout_prompt_variants=["logic"],
        holdout_layers=[],
        train_trace_paths=["b_train"],
        test_trace_paths=["b_test"],
        train_label_count=11,
        test_label_count=3,
        train_selector_row_count=11,
        test_selector_row_count=3,
        train_selector_candidate_row_count=22,
        test_selector_candidate_row_count=6,
        train_prompt_family_histogram={"cache": 11},
        test_prompt_family_histogram={"reasoning": 3},
        train_prompt_variant_histogram={"locality": 11},
        test_prompt_variant_histogram={"logic": 3},
        train_layer_histogram={"3": 11},
        test_layer_histogram={"23": 3},
    )

    upsert_oracle_dataset_split_manifest_entry(
        manifest_path,
        split_dir=tmp_path / "split_a",
        summary=split_a,
        annotations={"tier": "family"},
    )
    upsert_oracle_dataset_split_manifest_entry(
        manifest_path,
        split_dir=tmp_path / "split_b",
        summary=split_b,
        annotations={"tier": "variant"},
    )
    upsert_oracle_dataset_split_manifest_entry(
        manifest_path,
        split_dir=tmp_path / "split_a",
        summary=split_a,
        annotations={"tier": "family", "rev": "2"},
    )

    manifest = load_oracle_dataset_split_manifest(manifest_path)

    assert manifest["split_count"] == 2
    assert [entry["split_name"] for entry in manifest["splits"]] == ["split_a", "split_b"]
    assert manifest["splits"][0]["annotations"] == {"rev": "2", "tier": "family"}


def test_materialize_oracle_dataset_split_suite_preserves_declared_order(tmp_path) -> None:
    labels = [
        {
            "trace_path": str(tmp_path / "trace_cache.npz"),
            "stage": "prefill",
            "prompt_family": "cache",
            "prompt_variant": "locality",
            "source": "unit-test",
            "kind": "K",
            "layer_id": 3,
            "kv_head_id": 0,
            "token_start": 0,
            "token_age": 8,
            "token_count": 2,
            "head_dim": 8,
            "query_present": False,
            "cheapest_safe_candidate": "M0/affine/4",
            "safe_candidates": ["M0/affine/4"],
            "best_safe_total_bytes": 64,
            "candidate_labels": [{"candidate": "M0/affine/4", "mode": "M0", "bits": 4, "quant_scheme": "affine", "payload_bytes": 48, "metadata_bytes": 16, "total_bytes": 64, "safe": True}],
            "trace_stats": {"rms": 1.0},
            "notes": ["stage=prefill"],
        },
        {
            "trace_path": str(tmp_path / "trace_reasoning.npz"),
            "stage": "decode",
            "prompt_family": "reasoning",
            "prompt_variant": "logic",
            "source": "unit-test",
            "kind": "V",
            "layer_id": 23,
            "kv_head_id": 0,
            "token_start": 2,
            "token_age": 0,
            "token_count": 2,
            "head_dim": 8,
            "query_present": True,
            "cheapest_safe_candidate": "M3/affine/4/float16",
            "safe_candidates": ["M3/affine/4/float16"],
            "best_safe_total_bytes": 96,
            "candidate_labels": [{"candidate": "M3/affine/4/float16", "mode": "M3", "bits": 4, "quant_scheme": "affine", "payload_bytes": 64, "metadata_bytes": 32, "total_bytes": 96, "safe": True}],
            "trace_stats": {"rms": 2.0},
            "notes": ["stage=decode"],
        },
    ]
    labels_path = tmp_path / "labels.jsonl"
    labels_path.write_text("\n".join(json.dumps(record, sort_keys=True) for record in labels) + "\n", encoding="utf-8")
    loaded_labels = load_oracle_label_records(labels_path)
    selector_rows = build_selector_training_rows(loaded_labels)
    selector_candidate_rows = build_selector_candidate_training_rows(loaded_labels)
    selector_dataset_path = tmp_path / "selector_dataset.jsonl"
    selector_candidate_dataset_path = tmp_path / "selector_candidate_dataset.jsonl"
    save_selector_training_rows(selector_rows, selector_dataset_path)
    save_selector_candidate_training_rows(selector_candidate_rows, selector_candidate_dataset_path)

    suite_result = materialize_oracle_dataset_split_suite(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
        selector_candidate_dataset_path=selector_candidate_dataset_path,
        output_root=tmp_path / "suite",
        suite_name="unit_suite",
        manifest_path=tmp_path / "suite" / "split_manifest.json",
        suite_specs=[
            OracleDatasetSplitSuiteSpec(
                split_name="reasoning_family",
                output_subdir="family_reasoning",
                holdout_prompt_families=["reasoning"],
                annotations={"tier": "family"},
            ),
            OracleDatasetSplitSuiteSpec(
                split_name="logic_variant",
                output_subdir="variant_logic",
                holdout_prompt_variants=["logic"],
                annotations={"tier": "variant"},
            ),
        ],
    )

    assert suite_result.split_count == 2
    assert suite_result.split_names == ["reasoning_family", "logic_variant"]
    manifest = load_oracle_dataset_split_manifest(tmp_path / "suite" / "split_manifest.json")
    assert [entry["split_name"] for entry in manifest["splits"]] == ["reasoning_family", "logic_variant"]
    assert manifest["splits"][0]["annotations"] == {"tier": "family"}
    assert json.loads((tmp_path / "suite" / "split_suite_summary.json").read_text(encoding="utf-8"))["suite_name"] == "unit_suite"
