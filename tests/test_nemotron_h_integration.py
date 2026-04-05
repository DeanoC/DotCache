from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

import dotcache.integrations.nemotron_h as nemotron_h_mod
from dotcache.integrations.nemotron_h import (
    NemotronHTextModelAdapter,
    inspect_nemotron_h_native_config_compatibility,
    nemotron_h_attention_layers,
    nemotron_h_block_summary,
    nemotron_h_layer_types,
    partition_nemotron_h_hybrid_state,
    parse_nemotron_h_hybrid_pattern,
    summarize_nemotron_h_dotcache_fit,
)
from dotcache.model_registry import get_model_spec


def test_parse_nemotron_h_hybrid_pattern_supports_mamba_attention_and_mlp() -> None:
    assert parse_nemotron_h_hybrid_pattern("M-*") == ("mamba", "mlp", "attention")


def test_nemotron_h_layer_types_prefers_explicit_layers_block_type() -> None:
    config = SimpleNamespace(layers_block_type=["mamba", "attention", "mlp"])
    assert nemotron_h_layer_types(config) == ("mamba", "attention", "mlp")
    assert nemotron_h_attention_layers(config) == (1,)


def test_nemotron_h_block_summary_reads_published_pattern_shape() -> None:
    config = SimpleNamespace(
        model_type="nemotron_h",
        num_hidden_layers=42,
        hybrid_override_pattern="M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-",
        attention_head_dim=128,
        num_attention_heads=40,
        num_key_value_heads=8,
        mamba_num_heads=96,
        mamba_head_dim=80,
        ssm_state_size=128,
        chunk_size=256,
        max_position_embeddings=262144,
    )
    summary = nemotron_h_block_summary(config)
    assert summary["hybrid_layer_count"] == 42
    assert summary["hybrid_attention_layer_count"] == 4
    assert summary["hybrid_mamba_layer_count"] == 21
    assert summary["hybrid_mlp_layer_count"] == 17
    assert summary["attention_layers"] == [12, 17, 24, 32]
    assert summary["head_dim"] == 128


def test_inspect_nemotron_h_native_config_compatibility_reports_current_parser_failure() -> None:
    with patch.object(
        nemotron_h_mod.AutoConfig,
        "from_pretrained",
        side_effect=KeyError("-"),
    ):
        result = inspect_nemotron_h_native_config_compatibility("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
    assert result["native_autoconfig_ok"] is False
    assert result["native_autoconfig_error_type"] == "KeyError"
    assert "-" in result["native_autoconfig_error_message"]


def test_partition_nemotron_h_hybrid_state_separates_attention_mamba_and_mlp_layers() -> None:
    config = SimpleNamespace(
        layers_block_type=["mamba", "mlp", "attention"],
        model_type="nemotron_h",
        num_hidden_layers=3,
    )
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                conv_states=torch.zeros((1, 2, 4), dtype=torch.float32),
                ssm_states=torch.zeros((1, 2, 3), dtype=torch.float32),
            ),
            SimpleNamespace(),
            SimpleNamespace(
                keys=torch.zeros((1, 2, 5, 8), dtype=torch.float32),
                values=torch.zeros((1, 2, 5, 8), dtype=torch.float32),
            ),
        ]
    )
    partition = partition_nemotron_h_hybrid_state(cache, config)
    summary = partition.to_summary()
    assert [record.layer_id for record in partition.fixed_resident_layers] == [0]
    assert [record.layer_id for record in partition.no_cache_layers] == [1]
    assert [record.layer_id for record in partition.token_growing_layers] == [2]
    assert summary["hybrid_mamba_conv_state_bytes"] > 0
    assert summary["hybrid_mamba_recurrent_state_bytes"] > 0
    assert summary["hybrid_attention_kv_bytes"] > 0


def test_nemotron_h_adapter_exposes_block_and_fit_summaries() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            layers_block_type=["mamba", "attention", "mlp", "attention"],
            model_type="nemotron_h",
            num_hidden_layers=4,
            head_dim=128,
            num_attention_heads=40,
            num_key_value_heads=8,
            mamba_num_heads=96,
            mamba_head_dim=80,
            ssm_state_size=128,
            chunk_size=256,
            max_position_embeddings=262144,
        )
    )
    adapter = NemotronHTextModelAdapter(model=model)
    assert adapter.hybrid_block_summary()["hybrid_attention_layer_count"] == 2
    assert adapter.hybrid_fit_summary() == summarize_nemotron_h_dotcache_fit(model)


def test_nemotron_h_adapter_tracks_native_runtime_state_growth() -> None:
    model = SimpleNamespace(
        config=SimpleNamespace(
            layers_block_type=["mamba", "attention", "mlp"],
            model_type="nemotron_h",
            num_hidden_layers=3,
            head_dim=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            mamba_num_heads=4,
            mamba_head_dim=8,
            ssm_state_size=3,
            chunk_size=4,
            max_position_embeddings=1024,
        )
    )
    adapter = NemotronHTextModelAdapter(model=model)
    prefill_cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                conv_states=torch.zeros((1, 2, 4), dtype=torch.float32),
                ssm_states=torch.zeros((1, 2, 3), dtype=torch.float32),
            ),
            SimpleNamespace(
                keys=torch.zeros((1, 2, 5, 8), dtype=torch.float32),
                values=torch.zeros((1, 2, 5, 8), dtype=torch.float32),
            ),
            SimpleNamespace(),
        ]
    )
    summary = adapter.summarize_dotcache_native_hybrid_state(prefill_cache)
    assert summary["hybrid_dotcache_runtime_ready"] is True
    assert summary["native_hybrid_fixed_resident_preserved"] is True
    assert summary["native_hybrid_token_growing_growth_bytes"] == 0

    grown_cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                conv_states=torch.zeros((1, 2, 4), dtype=torch.float32),
                ssm_states=torch.zeros((1, 2, 3), dtype=torch.float32),
            ),
            SimpleNamespace(
                keys=torch.zeros((1, 2, 7, 8), dtype=torch.float32),
                values=torch.zeros((1, 2, 7, 8), dtype=torch.float32),
            ),
            SimpleNamespace(),
        ]
    )
    updated = adapter.summarize_dotcache_native_hybrid_state(grown_cache)
    assert updated["native_hybrid_fixed_resident_preserved"] is True
    assert updated["native_hybrid_token_growing_growth_bytes"] > 0
    assert updated["hybrid_runtime_state_kind"] == "nemotron_h_attention_subset"


def test_model_registry_exposes_nemotron_h_probe_lane() -> None:
    spec = get_model_spec("nemotron3_nano_4b_hf")
    assert spec.family == "nemotron_h"
    assert spec.dotcache_ready is False
