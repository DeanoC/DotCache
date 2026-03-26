from pathlib import Path

import pytest

from dotcache.config_io import load_dotcache_config, load_flat_yaml


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
