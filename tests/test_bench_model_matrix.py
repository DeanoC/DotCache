from dotcache.model_registry import get_model_spec
from benchmarks.bench_model_matrix import _matrix_record


def test_matrix_record_for_llama32_emits_runnable_command_with_continue_on_error() -> None:
    spec = get_model_spec("llama32_3b_hf")
    record = _matrix_record(
        spec,
        backend="torch_mps",
        device="mps",
        torch_dtype="float16",
        tokens_per_page=256,
        max_new_tokens=4,
        prompt_lengths_override=[],
        continue_on_error=True,
    )
    command = record["command"]
    assert isinstance(command, list)
    assert "--model-id" in command
    assert "meta-llama/Llama-3.2-3B-Instruct" in command
    assert "--continue-on-error" in command
    assert record["status"] == "runnable"


def test_matrix_record_can_disable_continue_on_error() -> None:
    spec = get_model_spec("llama32_3b_hf")
    record = _matrix_record(
        spec,
        backend="torch_mps",
        device="mps",
        torch_dtype="float16",
        tokens_per_page=256,
        max_new_tokens=4,
        prompt_lengths_override=[1024],
        continue_on_error=False,
    )
    command = record["command"]
    assert isinstance(command, list)
    assert "--continue-on-error" not in command
    assert record["planned_prompt_lengths"] == (1024,)


def test_matrix_record_for_llama32_gguf_emits_external_runner_command() -> None:
    spec = get_model_spec("llama32_3b_gguf")
    record = _matrix_record(
        spec,
        backend="auto",
        device=None,
        torch_dtype="float16",
        tokens_per_page=256,
        max_new_tokens=4,
        prompt_lengths_override=[],
        continue_on_error=True,
    )
    command = record["command"]
    assert isinstance(command, list)
    assert "bench_gguf_external.py" in " ".join(command)
    assert "--tokenizer-model-id" in command
    assert "meta-llama/Llama-3.2-3B-Instruct" in command
    assert record["status"] == "runnable"
