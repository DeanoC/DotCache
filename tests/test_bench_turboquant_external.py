from argparse import Namespace

from benchmarks.bench_turboquant_external import (
    _build_llama_cli_command,
    _parse_ppl,
    _parse_timings,
    _CONFIGS,
)


def test_parse_turboquant_timings_extracts_decode_metrics() -> None:
    text = """
llama_print_timings:        prompt eval time =    1914.63 ms / 15 tokens ( 127.64 ms per token, 7.83 tokens per second)
llama_print_timings:               eval time =    4228.67 ms / 33 runs   ( 128.14 ms per token, 7.80 tokens per second)
"""
    metrics = _parse_timings(text)
    assert metrics["prompt_eval_time_tokens_per_second"] == 7.83
    assert metrics["eval_time_tokens_per_second"] == 7.80
    assert metrics["decode_ms_per_step"] == 4228.67 / 33


def test_parse_ppl_extracts_value() -> None:
    text = "llama_perf_context_print: PPL = 5.8375"
    assert _parse_ppl(text) == 5.8375


def test_build_llama_cli_command_sets_layer_adaptive_env() -> None:
    args = Namespace(
        llama_cli="llama-cli",
        model_id="/tmp/model.gguf",
        hf_file=None,
        gguf_models_dir="/tmp",
        max_new_tokens=2,
        threads=None,
        n_gpu_layers=99,
        context_size=4096,
    )
    command, env = _build_llama_cli_command(
        args,
        config=_CONFIGS["turbo3_la1"],
        prompt_text="hello",
    )
    assert command[:9] == [
        "llama-cli",
        "-m",
        "/tmp/model.gguf",
        "-ctk",
        "turbo3",
        "-ctv",
        "turbo3",
        "-fa",
        "on",
    ]
    assert env["TURBO_LAYER_ADAPTIVE"] == "1"


def test_build_llama_cli_command_unsets_layer_adaptive_for_uniform() -> None:
    args = Namespace(
        llama_cli="llama-cli",
        model_id="/tmp/model.gguf",
        hf_file=None,
        gguf_models_dir="/tmp",
        max_new_tokens=2,
        threads=None,
        n_gpu_layers=None,
        context_size=None,
    )
    _, env = _build_llama_cli_command(
        args,
        config=_CONFIGS["q8_0"],
        prompt_text="hello",
    )
    assert "TURBO_LAYER_ADAPTIVE" not in env
