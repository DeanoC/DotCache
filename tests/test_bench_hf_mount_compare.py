from pathlib import Path

from benchmarks.bench_hf_mount_compare import _benchmark_command, _default_mount_point


class _Args:
    def __init__(
        self,
        *,
        benchmark_kind: str,
        backend: str = "torch_mps",
        device: str | None = "mps",
        torch_dtype: str = "float16",
        tokens_per_page: int = 256,
        max_new_tokens: int = 4,
        target_prompt_lengths: list[int] | None = None,
        continue_on_error: bool = True,
    ) -> None:
        self.benchmark_kind = benchmark_kind
        self.backend = backend
        self.device = device
        self.torch_dtype = torch_dtype
        self.tokens_per_page = tokens_per_page
        self.max_new_tokens = max_new_tokens
        self.target_prompt_lengths = [1024, 2048] if target_prompt_lengths is None else target_prompt_lengths
        self.continue_on_error = continue_on_error


def test_default_mount_point_sanitizes_repo_id() -> None:
    mount_point = _default_mount_point("Qwen/Qwen2.5-3B-Instruct")
    assert mount_point == Path(".cache") / "hf-mount" / "Qwen__Qwen2.5-3B-Instruct"


def test_benchmark_command_targets_llama_compare_script() -> None:
    args = _Args(benchmark_kind="llama_compare")
    mount_path = Path("/tmp/mounted-model")
    command = _benchmark_command(args, mount_path)
    joined = " ".join(command)
    assert "bench_llama_compare.py" in joined
    model_id_index = command.index("--model-id") + 1
    assert Path(command[model_id_index]).name == mount_path.name
    assert "--continue-on-error" in command


def test_benchmark_command_targets_qwen2_compare_script() -> None:
    args = _Args(benchmark_kind="qwen2_compare", device=None, continue_on_error=False)
    mount_path = Path("/tmp/mounted-qwen")
    command = _benchmark_command(args, mount_path)
    joined = " ".join(command)
    assert "bench_qwen2_compare.py" in joined
    model_id_index = command.index("--model-id") + 1
    assert Path(command[model_id_index]).name == mount_path.name
    assert "--continue-on-error" not in command
