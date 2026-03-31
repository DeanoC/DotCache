from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_qwen35_cuda_passkey_probe import _load_prompt_specs, _retag_passkey_payload, parse_args


def test_retag_passkey_payload_adds_passkey_aliases() -> None:
    payload = {
        "benchmark": "qwen35_attention_subset_dotcache_needle",
        "benchmark_task": "needle_in_a_haystack",
        "prompt_mode": "needle_in_a_haystack",
        "needle_key": "five-digit passkey",
        "needle_value": "58142",
        "needle_answer_correct": True,
        "needle_answer_exact_match": True,
        "needle_generated_text": "58142",
        "needle_generated_first_line": "58142",
    }
    prompt_spec = {
        "prompt_id": "ops_digits",
        "passkey_key": "five-digit passkey",
        "passkey_value": "58142",
        "passkey_position_fraction": 0.2,
    }

    record = _retag_passkey_payload(payload, prompt_spec=prompt_spec)

    assert record["benchmark_task"] == "passkey_retrieval"
    assert record["prompt_mode"] == "passkey_retrieval"
    assert record["passkey_answer_correct"] is True
    assert record["passkey_answer_exact_match"] is True
    assert record["passkey_generated_text"] == "58142"


def test_prompt_pack_loader_accepts_passkey_fields(tmp_path: Path) -> None:
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(
        '[{"prompt_id":"digits","passkey_key":"archive pin","passkey_value":"90317","passkey_position_fraction":0.4}]',
        encoding="utf-8",
    )

    args = parse_args([])
    args.prompt_pack = str(pack_path)
    specs = _load_prompt_specs(args)

    assert specs == [
        {
            "prompt_id": "digits",
            "passkey_key": "archive pin",
            "passkey_value": "90317",
            "passkey_position_fraction": 0.4,
        }
    ]
