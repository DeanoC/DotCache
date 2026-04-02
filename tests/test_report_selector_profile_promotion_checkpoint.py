from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "report_selector_profile_promotion_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("report_selector_profile_promotion_checkpoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_report_captures_qwen_and_llama_calls() -> None:
    payload, markdown = MODULE.build_report(
        qwen_task_payload={
            "rows": [
                {
                    "task_name": "retrieval_passkey",
                    "prompt_length": 1024,
                    "quality_decode_ms_per_step": 120.0,
                    "systems_decode_ms_per_step": 40.0,
                    "systems_vs_quality_speedup": 3.0,
                    "quality_success": 1.0,
                    "systems_success": 1.0,
                }
            ]
        },
        qwen_quality_payload={
            "comparisons": [
                {
                    "prompt_length": 1024,
                    "systems_vs_quality_decode_speedup": 1.5,
                    "variants": {
                        "quality": {
                            "decode_ms_per_step": 80.0,
                            "teacher_forced_token_agreement_rate": 1.0,
                            "teacher_forced_logit_rmse": 0.4,
                            "m3_fraction": 0.95,
                        },
                        "systems": {
                            "decode_ms_per_step": 40.0,
                            "teacher_forced_token_agreement_rate": 1.0,
                            "teacher_forced_logit_rmse": 0.3,
                            "m3_fraction": 0.99,
                        },
                    },
                }
            ]
        },
        llama_task_payload={
            "rows": [
                {
                    "task_name": "retrieval_passkey",
                    "prompt_length": 1024,
                    "quality_decode_ms_per_step": 60.0,
                    "systems_decode_ms_per_step": 60.0,
                    "systems_vs_quality_speedup": 1.0,
                    "quality_success": 1.0,
                    "systems_success": 1.0,
                }
            ]
        },
    )
    assert payload["promotion_calls"]["qwen"]["default_profile"] == "systems"
    assert payload["promotion_calls"]["llama"]["default_profile"] == "quality_or_systems_equivalent"
    assert "Promote systems as default" in markdown
    assert "Llama 3.2 3B" in markdown
