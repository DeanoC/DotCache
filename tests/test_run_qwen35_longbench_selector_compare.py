from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_qwen35_longbench_selector_compare.py"
SPEC = importlib.util.spec_from_file_location("run_qwen35_longbench_selector_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_apply_prompt_shard_uses_round_robin_partitioning() -> None:
    specs = [{"prompt_id": f"prompt_{index}"} for index in range(7)]

    shard = MODULE._apply_prompt_shard(specs, shard_count=3, shard_index=1)

    assert [row["prompt_id"] for row in shard] == ["prompt_1", "prompt_4"]


@pytest.mark.parametrize(
    ("shard_count", "shard_index"),
    [
        (0, 0),
        (2, -1),
        (2, 2),
    ],
)
def test_apply_prompt_shard_rejects_invalid_shard_config(shard_count: int, shard_index: int) -> None:
    with pytest.raises(SystemExit):
        MODULE._apply_prompt_shard([{"prompt_id": "only"}], shard_count=shard_count, shard_index=shard_index)
