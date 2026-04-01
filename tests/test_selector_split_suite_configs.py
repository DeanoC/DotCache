from __future__ import annotations

import json
from pathlib import Path


def test_selector_split_suite_configs_are_well_formed() -> None:
    config_dir = Path(__file__).resolve().parent.parent / "configs" / "selector_split_suites"
    config_paths = sorted(config_dir.glob("*.json"))

    assert config_paths, "expected checked-in selector split suite configs"

    for config_path in config_paths:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        assert isinstance(payload.get("suite_name"), str)
        splits = payload.get("splits")
        assert isinstance(splits, list)
        assert splits, f"{config_path} should define at least one split"

        split_names = [str(split["split_name"]) for split in splits]
        assert len(split_names) == len(set(split_names)), f"{config_path} has duplicate split_name values"

        for split in splits:
            assert isinstance(split.get("split_name"), str)
            assert isinstance(split.get("annotations", {}), dict)
            families = split.get("holdout_prompt_families", [])
            variants = split.get("holdout_prompt_variants", [])
            layers = split.get("holdout_layers", [])
            assert isinstance(families, list)
            assert isinstance(variants, list)
            assert isinstance(layers, list)
            assert families or variants or layers, f"{config_path} split {split['split_name']} must hold out something"
