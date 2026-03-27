from benchmarks.bench_gguf_external import _parse_timings


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
