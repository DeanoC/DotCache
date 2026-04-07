from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dotcache.selector_baselines import (
    CandidateSafeRouterModel,
    CandidateTargetLinearSelectorModel,
    CandidateTargetRouterModel,
    RUNTIME_SELECTOR_FEATURE_NAMES,
    LinearSelectorModel,
    SelectorExample,
    adjust_linear_selector_model_logits,
    evaluate_candidate_safe_router_model,
    evaluate_candidate_target_router_model,
    build_selector_class_error_weights,
    build_selector_example_weights,
    calibrate_selector_logit_offset,
    candidate_feature_names_from_examples,
    discover_selector_split_dirs,
    load_selector_candidate_examples,
    load_linear_selector_model,
    load_selector_examples,
    load_selector_split_examples,
    normalize_selector_categorical_token,
    render_selector_fixed_split_batch_markdown,
    run_selector_baseline_bakeoff,
    run_selector_fixed_split_batch_bakeoff,
    save_linear_selector_model,
    run_selector_fixed_split_bakeoff,
    run_selector_leave_prompt_family_layer_out_bakeoff,
    run_selector_leave_prompt_family_out_bakeoff,
    run_selector_leave_prompt_variant_out_bakeoff,
    run_selector_leave_layer_out_bakeoff,
    run_selector_multiseed_bakeoff,
    save_page_selector_artifact,
    load_page_selector_artifact,
    split_selector_examples,
    selector_feature_names_from_examples,
    train_candidate_safe_linear_selector,
    train_candidate_safe_router,
    train_candidate_target_router,
    train_linear_selector,
    train_calibrated_runtime_linear_selector,
    train_runtime_linear_selector,
)


def _write_example_bundle(tmp_path) -> tuple[str, str, str]:
    labels_path = tmp_path / "labels.jsonl"
    selector_dataset_path = tmp_path / "selector_dataset.jsonl"
    selector_candidate_dataset_path = tmp_path / "selector_candidate_dataset.jsonl"
    label_lines: list[str] = []
    selector_lines: list[str] = []
    selector_candidate_lines: list[str] = []
    for stage, query_present, target_candidate, safe_candidates in (
        ("prefill", False, "M0/affine/4", {"M0/affine/4": 740, "M3/affine/4/float16": 1448}),
        ("decode", True, "M3/affine/4/float16", {"M3/affine/4/float16": 1450}),
    ):
        for kind in ("K", "V"):
            for replica in range(2):
                trace_path = str(tmp_path / f"{stage}_{kind}_{replica}.npz")
                prompt_family = "reasoning" if stage == "decode" else "cache"
                prompt_variant = "logic" if stage == "decode" else "locality"
                m0_total_bytes = 740 if kind == "K" else 744
                m3_total_bytes = 1448 if kind == "K" else 1450
                candidate_byte_map = {
                    "M0/affine/4": m0_total_bytes,
                    "M3/affine/4/float16": m3_total_bytes,
                }
                candidate_labels = [
                    {
                        "candidate": "M0/affine/4",
                        "safe": "M0/affine/4" in safe_candidates,
                        "total_bytes": m0_total_bytes,
                    },
                    {
                        "candidate": "M3/affine/4/float16",
                        "safe": "M3/affine/4/float16" in safe_candidates,
                        "total_bytes": m3_total_bytes,
                    },
                ]
                label_lines.append(
                    json.dumps(
                        {
                            "trace_path": trace_path,
                            "stage": stage,
                            "prompt_family": prompt_family,
                            "prompt_variant": prompt_variant,
                            "source": "unit-test",
                            "kind": kind,
                            "layer_id": 3 if stage == "prefill" else 23,
                            "kv_head_id": replica,
                            "token_start": replica * 16,
                            "token_age": 24 if stage == "prefill" else 0,
                            "token_count": 2,
                            "head_dim": 256,
                            "query_present": query_present,
                            "cheapest_safe_candidate": target_candidate,
                            "safe_candidates": list(safe_candidates.keys()),
                            "best_safe_total_bytes": min(candidate_byte_map[candidate] for candidate in safe_candidates.keys()),
                            "candidate_labels": candidate_labels,
                            "trace_stats": {
                                "rms": 0.5 if stage == "prefill" else 1.2,
                                "abs_max": 2.0 if stage == "prefill" else 6.0,
                                "channel_range_mean": 0.3 if kind == "V" else 0.9,
                                "outlier_fraction": 0.0 if stage == "prefill" else 0.01,
                            },
                            "notes": [f"stage={stage}"],
                        },
                        sort_keys=True,
                    )
                )
                selector_lines.append(
                    json.dumps(
                        {
                            "trace_path": trace_path,
                            "source": "unit-test",
                            "stage": stage,
                            "prompt_family": prompt_family,
                            "prompt_variant": prompt_variant,
                            "kind": kind,
                            "layer_id": 3 if stage == "prefill" else 23,
                            "layer_fraction": 0.13 if stage == "prefill" else 1.0,
                            "kv_head_id": replica,
                            "kv_head_fraction": float(replica),
                            "token_start": replica * 16,
                            "token_age": 24 if stage == "prefill" else 0,
                            "token_count": 2,
                            "head_dim": 256,
                            "query_present": query_present,
                            "safe_candidate_count": len(safe_candidates),
                            "best_safe_total_bytes": min(candidate_byte_map[candidate] for candidate in safe_candidates.keys()),
                            "target_candidate": target_candidate,
                            "target_present": True,
                            "trace_rms": 0.5 if stage == "prefill" else 1.2,
                            "trace_abs_max": 2.0 if stage == "prefill" else 6.0,
                            "trace_channel_range_mean": 0.3 if kind == "V" else 0.9,
                            "trace_outlier_fraction": 0.0 if stage == "prefill" else 0.01,
                            "age_per_token": 12.0 if stage == "prefill" else 0.0,
                        },
                        sort_keys=True,
                    )
                )
                for candidate_label in candidate_labels:
                    selector_candidate_lines.append(
                        json.dumps(
                            {
                                "trace_path": trace_path,
                                "source": "unit-test",
                                "stage": stage,
                                "prompt_family": prompt_family,
                                "prompt_variant": prompt_variant,
                                "kind": kind,
                                "layer_id": 3 if stage == "prefill" else 23,
                                "layer_fraction": 0.13 if stage == "prefill" else 1.0,
                                "kv_head_id": replica,
                                "kv_head_fraction": float(replica),
                                "token_start": replica * 16,
                                "token_age": 24 if stage == "prefill" else 0,
                                "token_count": 2,
                                "head_dim": 256,
                                "query_present": query_present,
                                "safe_candidate_count": len(safe_candidates),
                                "best_safe_total_bytes": min(candidate_byte_map[candidate] for candidate in safe_candidates.keys()),
                                "target_candidate": target_candidate,
                                "target_present": True,
                                "trace_rms": 0.5 if stage == "prefill" else 1.2,
                                "trace_abs_max": 2.0 if stage == "prefill" else 6.0,
                                "trace_channel_range_mean": 0.3 if kind == "V" else 0.9,
                                "trace_outlier_fraction": 0.0 if stage == "prefill" else 0.01,
                                "age_per_token": 12.0 if stage == "prefill" else 0.0,
                                "candidate": candidate_label["candidate"],
                                "candidate_mode": "M0" if candidate_label["candidate"] == "M0/affine/4" else "M3",
                                "candidate_bits": 4,
                                "candidate_quant_scheme": "affine",
                                "candidate_total_bytes": candidate_label["total_bytes"],
                                "candidate_payload_bytes": 256 if candidate_label["candidate"] == "M0/affine/4" else 1024,
                                "candidate_metadata_bytes": candidate_label["total_bytes"] - (256 if candidate_label["candidate"] == "M0/affine/4" else 1024),
                                "candidate_has_escape_dtype": candidate_label["candidate"] == "M3/affine/4/float16",
                                "candidate_safe": candidate_label["safe"],
                                "candidate_is_target": candidate_label["candidate"] == target_candidate,
                                "candidate_bytes_over_best_safe": candidate_label["total_bytes"] - min(candidate_byte_map[candidate] for candidate in safe_candidates.keys()),
                            },
                            sort_keys=True,
                        )
                    )
    labels_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    selector_dataset_path.write_text("\n".join(selector_lines) + "\n", encoding="utf-8")
    selector_candidate_dataset_path.write_text("\n".join(selector_candidate_lines) + "\n", encoding="utf-8")
    return str(labels_path), str(selector_dataset_path), str(selector_candidate_dataset_path)


def test_selector_baseline_bakeoff_reaches_perfect_accuracy_on_separable_bundle(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    candidate_examples = load_selector_candidate_examples(
        selector_candidate_dataset_path=selector_candidate_dataset_path,
    )

    assert len(examples) == 8
    assert len(candidate_examples) == 16

    payload = run_selector_baseline_bakeoff(
        examples,
        candidate_examples=candidate_examples,
        test_fraction=0.5,
        seed=0,
        linear_steps=600,
        linear_learning_rate=0.3,
        linear_l2=1e-4,
    )

    assert payload["split"]["train_count"] == 4
    assert payload["split"]["test_count"] == 4
    assert "static_rule" in payload["results"]
    assert "linear_softmax" in payload["results"]
    assert "candidate_linear_safe" in payload["results"]

    static_result = payload["results"]["static_rule"]
    linear_result = payload["results"]["linear_softmax"]
    candidate_result = payload["results"]["candidate_linear_safe"]
    assert static_result["target_accuracy"] == 1.0
    assert static_result["safe_prediction_rate"] == 1.0
    assert static_result["mean_safe_bytes_regret"] == 0.0
    assert linear_result["target_accuracy"] == 1.0
    assert linear_result["safe_prediction_rate"] == 1.0
    assert linear_result["mean_safe_bytes_regret"] == 0.0
    assert candidate_result["target_accuracy"] == 1.0
    assert candidate_result["safe_prediction_rate"] == 1.0
    assert candidate_result["mean_safe_bytes_regret"] == 0.0
    assert "baseline | examples | target_accuracy" in payload["summary_markdown"]


def test_adjust_linear_selector_model_logits_updates_only_target_bias() -> None:
    model = LinearSelectorModel(
        classes=("M0/affine/4", "M3/affine/4/float16"),
        weight=np.zeros((len(RUNTIME_SELECTOR_FEATURE_NAMES), 2), dtype=np.float32),
        bias=np.asarray([0.0, 1.0], dtype=np.float32),
        feature_mean=np.zeros((len(RUNTIME_SELECTOR_FEATURE_NAMES),), dtype=np.float32),
        feature_std=np.ones((len(RUNTIME_SELECTOR_FEATURE_NAMES),), dtype=np.float32),
        feature_names=tuple(RUNTIME_SELECTOR_FEATURE_NAMES),
    )
    adjusted = adjust_linear_selector_model_logits(model, candidate_logit_offsets={"M3/affine/4/float16": -0.5})

    assert adjusted.classes == ("M0/affine/4", "M3/affine/4/float16")
    assert float(adjusted.bias[0]) == 0.0
    assert float(adjusted.bias[1]) == 0.5


def test_build_selector_example_weights_upweights_compression_friendly_rows(tmp_path) -> None:
    labels_path, selector_dataset_path, _ = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )

    weights = build_selector_example_weights(
        examples,
        safe_bytes_weight=2.0,
        reference_candidate="M3/affine/4/float16",
    )

    assert weights.shape == (8,)
    assert all(float(weight) > 1.0 for weight in weights[:4])
    assert all(float(weight) == 1.0 for weight in weights[4:])


def test_build_selector_class_error_weights_upweights_unsafe_non_target_candidates(tmp_path) -> None:
    labels_path, selector_dataset_path, _ = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )

    weights = build_selector_class_error_weights(
        examples,
        classes=("M0/affine/4", "M3/affine/4/float16"),
        unsafe_error_weight=3.0,
    )

    assert weights.shape == (8, 2)
    assert float(weights[0, 1]) == 1.0
    assert float(weights[4, 0]) == 4.0
    assert float(weights[4, 1]) == 1.0


def test_calibrate_selector_logit_offset_finds_more_compressive_feasible_offset() -> None:
    candidate_map = {
        "M0/affine/4": {"candidate": "M0/affine/4", "safe": True, "total_bytes": 740},
        "M3/affine/4/float16": {"candidate": "M3/affine/4/float16", "safe": True, "total_bytes": 1448},
    }

    def make_example(trace_path: str, *, trace_rms: float, target_candidate: str) -> SelectorExample:
        best_safe_total_bytes = 740 if target_candidate == "M0/affine/4" else 1448
        safe_candidates = ["M0/affine/4", "M3/affine/4/float16"] if target_candidate == "M0/affine/4" else ["M3/affine/4/float16"]
        return SelectorExample(
            trace_path=trace_path,
            row={
                "trace_path": trace_path,
                "source": "unit-test",
                "stage": "prefill",
                "prompt_family": "cache",
                "prompt_variant": "locality",
                "kind": "K",
                "layer_id": 3,
                "layer_fraction": 0.13,
                "kv_head_id": 0,
                "kv_head_fraction": 0.0,
                "token_start": 0,
                "token_age": 24,
                "token_count": 2,
                "head_dim": 256,
                "query_present": False,
                "safe_candidate_count": len(safe_candidates),
                "best_safe_total_bytes": best_safe_total_bytes,
                "target_candidate": target_candidate,
                "target_present": True,
                "trace_rms": trace_rms,
                "trace_abs_max": 2.0,
                "trace_channel_range_mean": 0.9,
                "trace_outlier_fraction": 0.0,
                "age_per_token": 12.0,
            },
            label={"safe_candidates": safe_candidates},
            candidate_map=dict(candidate_map),
        )

    examples = [
        make_example("row0", trace_rms=0.8, target_candidate="M0/affine/4"),
        make_example("row1", trace_rms=0.4, target_candidate="M0/affine/4"),
        make_example("row2", trace_rms=0.9, target_candidate="M3/affine/4/float16"),
    ]
    feature_index = RUNTIME_SELECTOR_FEATURE_NAMES.index("trace_rms")
    weight = np.zeros((len(RUNTIME_SELECTOR_FEATURE_NAMES), 2), dtype=np.float32)
    weight[feature_index, 0] = -1.0
    weight[feature_index, 1] = 1.0
    model = LinearSelectorModel(
        classes=("M0/affine/4", "M3/affine/4/float16"),
        weight=weight,
        bias=np.asarray([0.7, -0.7], dtype=np.float32),
        feature_mean=np.zeros((len(RUNTIME_SELECTOR_FEATURE_NAMES),), dtype=np.float32),
        feature_std=np.ones((len(RUNTIME_SELECTOR_FEATURE_NAMES),), dtype=np.float32),
        feature_names=tuple(RUNTIME_SELECTOR_FEATURE_NAMES),
    )

    calibration = calibrate_selector_logit_offset(
        model,
        examples,
        target_candidate="M3/affine/4/float16",
        offsets=[0.0, -0.3, -1.0],
        min_target_accuracy=1.0,
        min_safe_prediction_rate=1.0,
    )

    assert calibration["used_feasible_subset"] is True
    assert calibration["best"]["logit_offset"] == -0.3
    assert calibration["best"]["target_accuracy"] == 1.0
    assert calibration["best"]["mean_predicted_total_bytes"] < calibration["evaluations"][0]["mean_predicted_total_bytes"]


def test_selector_split_falls_back_when_stratified_groups_are_singletons(tmp_path) -> None:
    labels_path, selector_dataset_path, _ = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    unique_examples = examples[:4]

    split = split_selector_examples(unique_examples, test_fraction=0.5, seed=0)

    assert len(split.train_indices) == 2
    assert len(split.test_indices) == 2


def test_selector_multiseed_and_leave_layer_out_bakeoffs_produce_aggregate_summaries(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    candidate_examples = load_selector_candidate_examples(
        selector_candidate_dataset_path=selector_candidate_dataset_path,
    )

    multiseed_payload = run_selector_multiseed_bakeoff(
        examples,
        candidate_examples=candidate_examples,
        seeds=(0, 1),
        test_fraction=0.5,
        linear_steps=600,
        linear_learning_rate=0.3,
        linear_l2=1e-4,
    )
    leave_layer_payload = run_selector_leave_layer_out_bakeoff(
        examples,
        candidate_examples=candidate_examples,
        linear_steps=600,
        linear_learning_rate=0.3,
        linear_l2=1e-4,
    )

    assert multiseed_payload["evaluation_mode"] == "multiseed"
    assert len(multiseed_payload["folds"]) == 2
    assert multiseed_payload["aggregate_results"]["linear_softmax"]["fold_count"] == 2
    assert "mean_target_accuracy" in multiseed_payload["aggregate_results"]["candidate_linear_safe"]
    assert "baseline | folds | mean_target_accuracy" in multiseed_payload["summary_markdown"]

    assert leave_layer_payload["evaluation_mode"] == "leave_layer_out"
    assert len(leave_layer_payload["folds"]) == 2
    assert leave_layer_payload["aggregate_results"]["static_rule"]["fold_count"] == 2
    assert "baseline | folds | mean_target_accuracy" in leave_layer_payload["summary_markdown"]


def test_selector_fixed_split_bakeoff_uses_predeclared_bundle(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    split_dir = tmp_path / "split"
    train_dir = split_dir / "train"
    test_dir = split_dir / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    labels = [json.loads(line) for line in Path(labels_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    selector_rows = [json.loads(line) for line in Path(selector_dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    selector_candidate_rows = [json.loads(line) for line in Path(selector_candidate_dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]

    train_trace_paths = {labels[0]["trace_path"], labels[1]["trace_path"], labels[4]["trace_path"], labels[5]["trace_path"]}
    test_trace_paths = {labels[2]["trace_path"], labels[3]["trace_path"], labels[6]["trace_path"], labels[7]["trace_path"]}
    for target_dir, trace_paths in ((train_dir, train_trace_paths), (test_dir, test_trace_paths)):
        (target_dir / "labels.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in labels if row["trace_path"] in trace_paths) + "\n",
            encoding="utf-8",
        )
        (target_dir / "selector_dataset.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in selector_rows if row["trace_path"] in trace_paths) + "\n",
            encoding="utf-8",
        )
        (target_dir / "selector_candidate_dataset.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in selector_candidate_rows if row["trace_path"] in trace_paths) + "\n",
            encoding="utf-8",
        )
    (split_dir / "split_summary.json").write_text(json.dumps({"split_name": "unit_fixed"}, sort_keys=True) + "\n", encoding="utf-8")

    payload = load_selector_split_examples(split_dir=split_dir)
    result = run_selector_fixed_split_bakeoff(
        train_examples=payload["train_examples"],
        test_examples=payload["test_examples"],
        train_candidate_examples=payload["train_candidate_examples"],
        test_candidate_examples=payload["test_candidate_examples"],
        linear_steps=600,
        linear_learning_rate=0.3,
        linear_l2=1e-4,
        weighted_selector_config={
            "class_balance": 0.5,
            "safe_bytes_weight": 1.0,
            "reference_candidate": "M3/affine/4/float16",
            "calibration_fraction": 0.5,
            "calibration_seed": 0,
            "calibration_target_candidate": "M3/affine/4/float16",
            "calibration_offsets": [0.0, -0.3, -1.0],
            "calibration_min_target_accuracy": 1.0,
            "calibration_min_safe_prediction_rate": 1.0,
        },
        split_metadata=payload["split_summary"],
    )

    assert result["split"]["split_type"] == "fixed"
    assert result["split"]["train_count"] == 4
    assert result["split"]["test_count"] == 4
    assert result["split"]["split_metadata"]["split_name"] == "unit_fixed"
    assert result["results"]["linear_softmax"]["target_accuracy"] == 1.0
    assert result["results"]["linear_softmax_compression_weighted"]["target_accuracy"] == 1.0
    assert result["results"]["linear_softmax_compression_calibrated"]["safe_prediction_rate"] == 1.0
    assert result["results"]["candidate_linear_safe"]["safe_prediction_rate"] == 1.0


def test_selector_fixed_split_batch_bakeoff_compares_multiple_frozen_splits(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    labels = [json.loads(line) for line in Path(labels_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    selector_rows = [json.loads(line) for line in Path(selector_dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    selector_candidate_rows = [json.loads(line) for line in Path(selector_candidate_dataset_path).read_text(encoding="utf-8").splitlines() if line.strip()]

    split_root = tmp_path / "split_root"
    split_specs = [
        ("reasoning_holdout", {"reasoning"}, {"cache"}),
        ("cache_holdout", {"cache"}, {"reasoning"}),
    ]
    for split_name, test_families, train_families in split_specs:
        split_dir = split_root / split_name
        train_dir = split_dir / "train"
        test_dir = split_dir / "test"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        for target_dir, families in ((train_dir, train_families), (test_dir, test_families)):
            trace_paths = {row["trace_path"] for row in labels if row["prompt_family"] in families}
            (target_dir / "labels.jsonl").write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in labels if row["trace_path"] in trace_paths) + "\n",
                encoding="utf-8",
            )
            (target_dir / "selector_dataset.jsonl").write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in selector_rows if row["trace_path"] in trace_paths) + "\n",
                encoding="utf-8",
            )
            (target_dir / "selector_candidate_dataset.jsonl").write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in selector_candidate_rows if row["trace_path"] in trace_paths) + "\n",
                encoding="utf-8",
            )
        (split_dir / "split_summary.json").write_text(json.dumps({"split_name": split_name}, sort_keys=True) + "\n", encoding="utf-8")

    discovered = discover_selector_split_dirs(split_root)
    assert [path.name for path in discovered] == ["cache_holdout", "reasoning_holdout"]

    payload = run_selector_fixed_split_batch_bakeoff(
        split_dirs=discovered,
        linear_steps=600,
        linear_learning_rate=0.3,
        linear_l2=1e-4,
        weighted_selector_config={
            "class_balance": 0.5,
            "safe_bytes_weight": 1.0,
            "reference_candidate": "M3/affine/4/float16",
            "calibration_fraction": 0.5,
            "calibration_seed": 0,
            "calibration_target_candidate": "M3/affine/4/float16",
            "calibration_offsets": [0.0, -0.3, -1.0],
            "calibration_min_target_accuracy": 1.0,
            "calibration_min_safe_prediction_rate": 1.0,
        },
    )

    assert payload["split_count"] == 2
    assert [split["split_name"] for split in payload["splits"]] == ["cache_holdout", "reasoning_holdout"]
    assert payload["aggregate_results"]["linear_softmax"]["fold_count"] == 2
    assert payload["aggregate_results"]["linear_softmax_compression_weighted"]["fold_count"] == 2
    assert payload["aggregate_results"]["linear_softmax_compression_calibrated"]["fold_count"] == 1
    markdown = render_selector_fixed_split_batch_markdown(payload["splits"])
    assert "split | baseline | test_examples" in markdown
    assert "mean_predicted_total_bytes" in markdown
    assert "cache_holdout" in markdown
    assert "reasoning_holdout" in markdown


def test_selector_leave_prompt_family_out_bakeoff_produces_family_folds(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    candidate_examples = load_selector_candidate_examples(
        selector_candidate_dataset_path=selector_candidate_dataset_path,
    )

    payload = run_selector_leave_prompt_family_out_bakeoff(
        examples,
        candidate_examples=candidate_examples,
        linear_steps=600,
        linear_learning_rate=0.3,
        linear_l2=1e-4,
    )

    assert payload["evaluation_mode"] == "leave_prompt_family_out"
    assert payload["held_out_prompt_families"] == ["cache", "reasoning"]
    assert len(payload["folds"]) == 2
    assert payload["aggregate_results"]["candidate_linear_safe"]["fold_count"] == 2
    assert "baseline | folds | mean_target_accuracy" in payload["summary_markdown"]


def test_selector_leave_prompt_variant_out_bakeoff_produces_variant_folds(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    candidate_examples = load_selector_candidate_examples(
        selector_candidate_dataset_path=selector_candidate_dataset_path,
    )

    payload = run_selector_leave_prompt_variant_out_bakeoff(
        examples,
        candidate_examples=candidate_examples,
        linear_steps=600,
        linear_learning_rate=0.3,
        linear_l2=1e-4,
    )

    assert payload["evaluation_mode"] == "leave_prompt_variant_out"
    assert payload["held_out_prompt_variants"] == ["locality", "logic"]
    assert len(payload["folds"]) == 2
    assert payload["aggregate_results"]["linear_softmax"]["fold_count"] == 2
    assert "baseline | folds | mean_target_accuracy" in payload["summary_markdown"]


def test_selector_leave_prompt_family_layer_out_bakeoff_produces_combined_folds(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    candidate_examples = load_selector_candidate_examples(
        selector_candidate_dataset_path=selector_candidate_dataset_path,
    )

    payload = run_selector_leave_prompt_family_layer_out_bakeoff(
        examples,
        candidate_examples=candidate_examples,
        linear_steps=600,
        linear_learning_rate=0.3,
        linear_l2=1e-4,
    )

    assert payload["evaluation_mode"] == "leave_prompt_family_layer_out"
    assert payload["held_out_prompt_family_layers"] == [
        {"held_out_prompt_family": "cache", "held_out_layer": 3},
        {"held_out_prompt_family": "reasoning", "held_out_layer": 23},
    ]
    assert len(payload["folds"]) == 2
    assert payload["folds"][0]["fold_name"].startswith("prompt_family_")
    assert payload["aggregate_results"]["static_rule"]["fold_count"] == 2
    assert "baseline | folds | mean_target_accuracy" in payload["summary_markdown"]


def test_learned_selector_models_include_prompt_variant_features(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    candidate_examples = load_selector_candidate_examples(
        selector_candidate_dataset_path=selector_candidate_dataset_path,
    )

    linear_model = train_linear_selector(examples, steps=100, learning_rate=0.1, l2=1e-4)
    candidate_model = train_candidate_safe_linear_selector(candidate_examples, steps=100, learning_rate=0.1, l2=1e-4)

    assert "family_cache" in linear_model.feature_names
    assert "family_reasoning" in linear_model.feature_names
    assert "variant_locality" in linear_model.feature_names
    assert "variant_logic" in linear_model.feature_names
    assert "variant_locality" in candidate_model.feature_names
    assert "variant_logic" in candidate_model.feature_names


def test_runtime_linear_selector_artifact_round_trips(tmp_path) -> None:
    labels_path, selector_dataset_path, _ = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )

    model = train_runtime_linear_selector(examples, steps=100, learning_rate=0.1, l2=1e-4)
    target_path = tmp_path / "linear_selector_model.json"
    save_linear_selector_model(model, target_path)
    loaded = load_linear_selector_model(target_path)

    assert loaded.classes == model.classes
    assert tuple(RUNTIME_SELECTOR_FEATURE_NAMES) == loaded.feature_names[: len(RUNTIME_SELECTOR_FEATURE_NAMES)]
    assert "query_present" in loaded.feature_names
    assert "page_distance" in loaded.feature_names
    assert "log_page_distance" in loaded.feature_names
    assert "page_distance_ge_4" in loaded.feature_names
    assert "safe_candidate_count" not in loaded.feature_names
    assert "family_cache" in loaded.feature_names
    assert "family_reasoning" in loaded.feature_names
    assert "variant_locality" in loaded.feature_names
    assert "variant_logic" in loaded.feature_names


def test_candidate_safe_router_round_trips_and_evaluates(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    _ = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    candidate_examples = load_selector_candidate_examples(
        selector_candidate_dataset_path=selector_candidate_dataset_path,
    )

    router = train_candidate_safe_router(candidate_examples, steps=100, learning_rate=0.1, l2=1e-4)
    target_path = tmp_path / "candidate_safe_router_model.json"
    save_page_selector_artifact(router, target_path)
    loaded = load_page_selector_artifact(target_path)

    assert isinstance(loaded, CandidateSafeRouterModel)
    assert loaded.candidate_tokens == ("M0/affine/4", "M3/affine/4/float16")
    summary = evaluate_candidate_safe_router_model(loaded, candidate_examples)
    assert summary.target_accuracy == 1.0
    assert summary.safe_prediction_rate == 1.0


def test_candidate_target_router_round_trips_and_evaluates(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    candidate_examples = load_selector_candidate_examples(
        selector_candidate_dataset_path=selector_candidate_dataset_path,
    )

    router = train_candidate_target_router(
        candidate_examples,
        steps=600,
        learning_rate=0.3,
        l2=1e-4,
        feature_names=candidate_feature_names_from_examples(candidate_examples, feature_set_id="runtime_safe"),
        prompt_family_thresholds={normalize_selector_categorical_token("reasoning") or "": 0.4},
    )
    target_path = tmp_path / "candidate_target_router_model.json"
    save_page_selector_artifact(router, target_path)
    loaded = load_page_selector_artifact(target_path)

    assert isinstance(loaded, CandidateTargetRouterModel)
    assert loaded.candidate_tokens == ("M0/affine/4", "M3/affine/4/float16")
    assert loaded.prompt_family_thresholds == {"reasoning": 0.4}
    summary = evaluate_candidate_target_router_model(loaded, candidate_examples)
    assert summary.target_accuracy == 1.0
    assert summary.safe_prediction_rate == 1.0


def test_candidate_target_router_logit_offset_can_prefer_m0_on_ties(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    _ = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    candidate_examples = load_selector_candidate_examples(
        selector_candidate_dataset_path=selector_candidate_dataset_path,
    )
    feature_names = candidate_feature_names_from_examples(candidate_examples, feature_set_id="runtime_safe")
    weight = np.zeros((len(feature_names),), dtype=np.float32)
    weight[feature_names.index("candidate_mode_m3")] = 2.0
    model = CandidateTargetLinearSelectorModel(
        weight=weight,
        bias=0.0,
        feature_mean=np.zeros((len(feature_names),), dtype=np.float32),
        feature_std=np.ones((len(feature_names),), dtype=np.float32),
        feature_names=feature_names,
    )
    row = {
        key: value
        for key, value in candidate_examples[0].row.items()
        if not key.startswith("candidate_")
    }
    baseline_router = CandidateTargetRouterModel(
        target_model=model,
        candidate_tokens=("M0/affine/4", "M3/affine/4/float16"),
        fallback_candidate="M3/affine/4/float16",
    )
    penalized_router = CandidateTargetRouterModel(
        target_model=model,
        candidate_tokens=("M0/affine/4", "M3/affine/4/float16"),
        fallback_candidate="M3/affine/4/float16",
        candidate_logit_offsets={"M3/affine/4/float16": -3.0},
    )

    assert baseline_router.predict_row(row) == "M3/affine/4/float16"
    assert penalized_router.predict_row(row) == "M0/affine/4"


def test_selector_feature_set_builders_distinguish_runtime_and_research(tmp_path) -> None:
    labels_path, selector_dataset_path, selector_candidate_dataset_path = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )
    candidate_examples = load_selector_candidate_examples(
        selector_candidate_dataset_path=selector_candidate_dataset_path,
    )

    runtime_row_features = selector_feature_names_from_examples(examples, feature_set_id="runtime_safe")
    research_row_features = selector_feature_names_from_examples(examples, feature_set_id="research_extended")
    runtime_candidate_features = candidate_feature_names_from_examples(candidate_examples, feature_set_id="runtime_safe")
    research_candidate_features = candidate_feature_names_from_examples(candidate_examples, feature_set_id="research_extended")

    assert "safe_candidate_count" not in runtime_row_features
    assert "safe_candidate_count" in research_row_features
    assert "log_best_safe_total_bytes" not in research_row_features
    assert "compression_gain_vs_m3" not in research_row_features
    assert "safe_candidate_count" not in runtime_candidate_features
    assert "candidate_bytes_over_best_safe" not in runtime_candidate_features
    assert "safe_candidate_count" in research_candidate_features
    assert "candidate_bytes_over_best_safe" not in research_candidate_features
    assert "log_best_safe_total_bytes" not in research_candidate_features


def test_train_calibrated_runtime_linear_selector_returns_global_calibration(tmp_path) -> None:
    labels_path, selector_dataset_path, _ = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )

    model, calibration = train_calibrated_runtime_linear_selector(
        examples,
        steps=100,
        learning_rate=0.1,
        l2=1e-4,
        class_balance=0.5,
        safe_bytes_weight=1.0,
        unsafe_error_weight=0.5,
        calibration_fraction=0.5,
        calibration_offsets=(-0.5, 0.0, 0.5),
        calibration_target_candidate="M3/affine/4/float16",
        calibration_min_target_accuracy=0.0,
        calibration_min_safe_prediction_rate=0.0,
    )

    assert model is not None
    assert calibration is not None
    assert calibration["target_candidate"] == "M3/affine/4/float16"
    assert float(calibration["best"]["logit_offset"]) in {-0.5, 0.0, 0.5}


def test_train_calibrated_runtime_linear_selector_supports_equal_tradeoff_objective(tmp_path) -> None:
    labels_path, selector_dataset_path, _ = _write_example_bundle(tmp_path)
    examples = load_selector_examples(
        labels_path=labels_path,
        selector_dataset_path=selector_dataset_path,
    )

    model, calibration = train_calibrated_runtime_linear_selector(
        examples,
        steps=100,
        learning_rate=0.1,
        l2=1e-4,
        class_balance=0.5,
        safe_bytes_weight=1.0,
        unsafe_error_weight=0.5,
        calibration_fraction=0.5,
        calibration_offsets=(-0.5, 0.0, 0.5),
        calibration_target_candidate="M3/affine/4/float16",
        calibration_objective="equal_tradeoff",
        calibration_correctness_weight=1.0,
        calibration_bytes_weight=1.0,
    )

    assert model is not None
    assert calibration is not None
    assert calibration["calibration_objective"] == "equal_tradeoff"
    assert float(calibration["correctness_weight"]) == 0.5
    assert float(calibration["bytes_weight"]) == 0.5
    assert float(calibration["best"]["logit_offset"]) in {-0.5, 0.0, 0.5}
