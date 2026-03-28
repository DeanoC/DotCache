from argparse import Namespace
from pathlib import Path

from benchmarks.bench_gguf_external import _llama_cli_command, _parse_timings


def test_parse_timings_extracts_prompt_and_decode_metrics() -> None:
    text = """
llama_print_timings:        prompt eval time =    1914.63 ms / 15 tokens ( 127.64 ms per token, 7.83 tokens per second)
llama_print_timings:               eval time =    4228.67 ms / 33 runs   ( 128.14 ms per token, 7.80 tokens per second)
llama_print_timings:              total time =    6143.30 ms / 48 tokens
"""
    metrics = _parse_timings(text)
    assert metrics["prompt_eval_time_ms"] == 1914.63
    assert metrics["prompt_eval_time_count"] == 15
    assert metrics["eval_time_ms"] == 4228.67
    assert metrics["eval_time_count"] == 33
    assert metrics["decode_ms_per_step"] == 4228.67 / 33
    assert metrics["total_time_ms"] == 6143.30


def test_parse_timings_handles_missing_optional_rates() -> None:
    text = "llama_print_timings:              total time =    6143.30 ms / 48 tokens"
    metrics = _parse_timings(text)
    assert metrics["total_time_ms"] == 6143.30
    assert metrics["total_time_count"] == 48
    assert "decode_ms_per_step" not in metrics


def test_llama_cli_command_uses_hf_file_when_requested(tmp_path: Path) -> None:
    args = Namespace(
        llama_cli="llama-cli",
        model_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        hf_file="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        gguf_models_dir=str(tmp_path),
        max_new_tokens=2,
        threads=None,
        n_gpu_layers=None,
        context_size=None,
    )
    command = _llama_cli_command(args, prompt_text="hello")
    assert command[:5] == [
        "llama-cli",
        "-hf",
        "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "-hff",
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    ]


def test_llama_cli_command_prefers_local_workspace_gguf_when_present(tmp_path: Path) -> None:
    local_dir = tmp_path / "llama32_3b"
    local_dir.mkdir()
    local_file = local_dir / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    local_file.write_bytes(b"gguf")

    args = Namespace(
        llama_cli="llama-cli",
        model_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        hf_file="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        gguf_models_dir=str(tmp_path),
        max_new_tokens=2,
        threads=None,
        n_gpu_layers=None,
        context_size=None,
    )
    command = _llama_cli_command(args, prompt_text="hello")
    assert command[:3] == ["llama-cli", "-m", str(local_file)]
