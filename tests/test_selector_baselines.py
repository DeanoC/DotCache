from __future__ import annotations

import json
from pathlib import Path

from dotcache.selector_baselines import (
    discover_selector_split_dirs,
    load_selector_candidate_examples,
    load_selector_examples,
    load_selector_split_examples,
    render_selector_fixed_split_batch_markdown,
    run_selector_baseline_bakeoff,
    run_selector_fixed_split_batch_bakeoff,
    run_selector_fixed_split_bakeoff,
    run_selector_leave_prompt_family_layer_out_bakeoff,
    run_selector_leave_prompt_family_out_bakeoff,
    run_selector_leave_prompt_variant_out_bakeoff,
    run_selector_leave_layer_out_bakeoff,
    run_selector_multiseed_bakeoff,
    split_selector_examples,
    train_candidate_safe_linear_selector,
    train_linear_selector,
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
        split_metadata=payload["split_summary"],
    )

    assert result["split"]["split_type"] == "fixed"
    assert result["split"]["train_count"] == 4
    assert result["split"]["test_count"] == 4
    assert result["split"]["split_metadata"]["split_name"] == "unit_fixed"
    assert result["results"]["linear_softmax"]["target_accuracy"] == 1.0
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
    )

    assert payload["split_count"] == 2
    assert [split["split_name"] for split in payload["splits"]] == ["cache_holdout", "reasoning_holdout"]
    assert payload["aggregate_results"]["linear_softmax"]["fold_count"] == 2
    markdown = render_selector_fixed_split_batch_markdown(payload["splits"])
    assert "split | baseline | test_examples" in markdown
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
