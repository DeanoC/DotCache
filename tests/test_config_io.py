from pathlib import Path

import pytest

from dotcache.config_io import load_dotcache_config, load_flat_yaml, load_layer_profile


def test_load_flat_yaml_parses_simple_scalars(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "head_dim: 256",
                "tokens_per_page: 128",
                "payload_layout_k: group_major",
                "escape_dtype: float16",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_flat_yaml(config_path)

    assert loaded["head_dim"] == 256
    assert loaded["tokens_per_page"] == 128
    assert loaded["payload_layout_k"] == "group_major"
    assert loaded["escape_dtype"] == "float16"


def test_load_dotcache_config_builds_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "head_dim: 256",
                "group_size: 32",
                "bits_k: 4",
                "bits_v: 4",
                "tokens_per_page: 256",
                "payload_layout_k: group_major",
                "payload_layout_v: group_major",
                "default_mode_k: M0",
                "default_mode_v: M0",
                "quant_scheme_k: affine",
                "quant_scheme_v: affine",
            ]
        ),
        encoding="utf-8",
    )

    config = load_dotcache_config(config_path)

    assert config.head_dim == 256
    assert config.tokens_per_page == 256
    assert config.quant_scheme_k == "affine"


def test_load_dotcache_config_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("head_dim: 128\nmade_up_field: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown DotCacheConfig fields"):
        load_dotcache_config(config_path)


def test_load_layer_profile_parses_lists_and_nested_metadata(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "\n".join(
            [
                "model_id: TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "source:",
                "  benchmark: inspect_policy_prefill",
                "  backend: torch_mps",
                "  notes: >",
                "    first line",
                "    second line",
                "key_policy_tier: balanced",
                "value_policy_tier: aggressive",
                "key_layer_sensitivity:",
                "  - layer:3=strict",
                "  - layer:4=strict",
                "value_layer_sensitivity: []",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_layer_profile(profile_path)

    assert loaded["model_id"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert loaded["source"]["benchmark"] == "inspect_policy_prefill"
    assert loaded["source"]["backend"] == "torch_mps"
    assert loaded["source"]["notes"] == "first line second line"
    assert loaded["key_policy_tier"] == "balanced"
    assert loaded["value_policy_tier"] == "aggressive"
    assert loaded["key_layer_sensitivity"] == ["layer:3=strict", "layer:4=strict"]
    assert loaded["value_layer_sensitivity"] == []
