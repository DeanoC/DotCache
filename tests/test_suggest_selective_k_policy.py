from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_module():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    module_path = scripts_dir / "suggest_selective_k_policy.py"
    spec = importlib.util.spec_from_file_location("suggest_selective_k_policy", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(*, layer_id: int, kv_head_id: int, top1: bool, topk: float, rank: float, kl: float, rmse: float) -> dict[str, object]:
    return {
        "variant_id": "stub",
        "layer_id": layer_id,
        "step_index": 0,
        "q_head_id": kv_head_id,
        "kv_head_id": kv_head_id,
        "token_count": 128,
        "score_max_abs": 1.0,
        "score_mean_abs": 0.5,
        "score_rmse": 0.5,
        "score_cosine": rank,
        "score_rank_corr": rank,
        "score_top1_match": top1,
        "score_topk_overlap": topk,
        "attn_kl": kl,
        "attn_entropy_exact": 1.0,
        "attn_entropy_test": 1.0,
        "attn_entropy_delta": 0.0,
        "top_src_exact": 0,
        "top_src_test": 0 if top1 else 1,
        "top_src_age_bucket": "mid",
        "output_max_abs": rmse,
        "output_mean_abs": rmse,
        "output_rmse": rmse,
    }


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        family="qwen2",
        model_id="Qwen/Qwen2.5-Test",
        device="cuda",
        backend="torch_cuda",
        torch_dtype="float16",
        group_size=32,
        bits_k=4,
        bits_v=4,
        quant_scheme_k="affine",
        quant_scheme_v="affine",
        tokens_per_page=256,
        decode_steps=3,
        top_k=8,
        prompt_unit="x",
        prompt_length=2048,
        candidate_layers=2,
        max_kv_groups_per_layer=2,
        max_combo_size=2,
        target_recovery=0.65,
        budget_recovery_margin=0.9,
        top_policies=6,
        format="json",
    )


def test_policy_suggester_prefers_minimal_policy_that_clears_target(monkeypatch) -> None:
    module = _load_module()

    d5_rows = [
        _row(layer_id=0, kv_head_id=0, top1=False, topk=0.4, rank=0.4, kl=0.8, rmse=0.8),
        _row(layer_id=0, kv_head_id=1, top1=False, topk=0.45, rank=0.45, kl=0.7, rmse=0.7),
        _row(layer_id=1, kv_head_id=0, top1=True, topk=0.95, rank=0.95, kl=0.05, rmse=0.05),
        _row(layer_id=1, kv_head_id=1, top1=True, topk=0.95, rank=0.95, kl=0.05, rmse=0.05),
    ]
    d6_rows = [
        _row(layer_id=0, kv_head_id=0, top1=True, topk=1.0, rank=1.0, kl=0.0, rmse=0.0),
        _row(layer_id=0, kv_head_id=1, top1=True, topk=1.0, rank=1.0, kl=0.0, rmse=0.0),
        _row(layer_id=1, kv_head_id=0, top1=True, topk=1.0, rank=1.0, kl=0.0, rmse=0.0),
        _row(layer_id=1, kv_head_id=1, top1=True, topk=1.0, rank=1.0, kl=0.0, rmse=0.0),
    ]
    baseline = module._summary_from_rows(d5_rows, variant_id="D5")
    exact = module._summary_from_rows(d6_rows, variant_id="D6")
    layer0_rows = module._compose_rescue_rows(
        d5_rows=d5_rows,
        d6_rows=d6_rows,
        selector=lambda row: int(row["layer_id"]) == 0,
    )
    layer1_rows = module._compose_rescue_rows(
        d5_rows=d5_rows,
        d6_rows=d6_rows,
        selector=lambda row: int(row["layer_id"]) == 1,
    )
    probe_result = {
        "variant_summaries": [baseline, exact],
        "layer_rescue_summaries": [
            {
                **module._summary_from_rows(layer0_rows, variant_id="rescue_L0"),
                "rescue_layer_id": 0,
            },
            {
                **module._summary_from_rows(layer1_rows, variant_id="rescue_L1"),
                "rescue_layer_id": 1,
            },
        ],
        "rows_by_variant": {"D5": d5_rows, "D6": d6_rows},
    }
    monkeypatch.setattr(module, "probe_attention_score_fidelity", lambda args: probe_result)

    result = module.build_policy_suggestions(_args())

    assert result["recommended_policy"] is not None
    assert result["recommended_policy"]["label"] == "layer:0=M3"
    assert abs(result["recommended_policy"]["estimated_k_exact_fraction"] - 0.5) < 1e-9


def test_policy_suggester_prefers_kv_group_when_it_matches_layer_recovery(monkeypatch) -> None:
    module = _load_module()

    d5_rows = [
        _row(layer_id=0, kv_head_id=0, top1=False, topk=0.35, rank=0.35, kl=0.9, rmse=0.9),
        _row(layer_id=0, kv_head_id=1, top1=True, topk=0.98, rank=0.98, kl=0.02, rmse=0.02),
        _row(layer_id=1, kv_head_id=0, top1=True, topk=0.98, rank=0.98, kl=0.02, rmse=0.02),
        _row(layer_id=1, kv_head_id=1, top1=True, topk=0.98, rank=0.98, kl=0.02, rmse=0.02),
    ]
    d6_rows = [
        _row(layer_id=0, kv_head_id=0, top1=True, topk=1.0, rank=1.0, kl=0.0, rmse=0.0),
        _row(layer_id=0, kv_head_id=1, top1=True, topk=1.0, rank=1.0, kl=0.0, rmse=0.0),
        _row(layer_id=1, kv_head_id=0, top1=True, topk=1.0, rank=1.0, kl=0.0, rmse=0.0),
        _row(layer_id=1, kv_head_id=1, top1=True, topk=1.0, rank=1.0, kl=0.0, rmse=0.0),
    ]
    baseline = module._summary_from_rows(d5_rows, variant_id="D5")
    exact = module._summary_from_rows(d6_rows, variant_id="D6")
    layer0_rows = module._compose_rescue_rows(
        d5_rows=d5_rows,
        d6_rows=d6_rows,
        selector=lambda row: int(row["layer_id"]) == 0,
    )
    layer1_rows = module._compose_rescue_rows(
        d5_rows=d5_rows,
        d6_rows=d6_rows,
        selector=lambda row: int(row["layer_id"]) == 1,
    )
    probe_result = {
        "variant_summaries": [baseline, exact],
        "layer_rescue_summaries": [
            {
                **module._summary_from_rows(layer0_rows, variant_id="rescue_L0"),
                "rescue_layer_id": 0,
            },
            {
                **module._summary_from_rows(layer1_rows, variant_id="rescue_L1"),
                "rescue_layer_id": 1,
            },
        ],
        "rows_by_variant": {"D5": d5_rows, "D6": d6_rows},
    }
    monkeypatch.setattr(module, "probe_attention_score_fidelity", lambda args: probe_result)

    result = module.build_policy_suggestions(_args())

    assert result["recommended_policy"] is not None
    assert result["recommended_policy"]["label"] == "layer:0:kv:0=M3"
    assert abs(result["recommended_policy"]["estimated_k_exact_fraction"] - 0.25) < 1e-9
