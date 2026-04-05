import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_gemma4_text_uses_dtype_kwarg(monkeypatch, capsys) -> None:
    module = _load_module("probe_gemma4_text", "scripts/probe_gemma4_text.py")
    captured_load_kwargs: dict[str, object] = {}

    class _FakeModel:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.hf_device_map = None

        def eval(self) -> None:
            return None

        def parameters(self):
            yield torch.zeros(1)

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    class _FakeTokenizer:
        def __call__(self, prompt, return_tensors=None):
            class _Inputs(dict):
                def to(self, device):
                    return self

            return _Inputs({"input_ids": torch.tensor([[1, 2]], dtype=torch.long)})

        def decode(self, ids, skip_special_tokens=True):
            return "decoded"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            model_id="google/gemma-4-E2B",
            device_map="cpu",
            torch_dtype="float32",
            attn_implementation=None,
            max_new_tokens=2,
            prompt="hello",
            run_dotcache=False,
            dotcache_profile="balanced",
            tokens_per_page=4,
            bits_k=4,
            bits_v=4,
            group_size=32,
            dotcache_backend="auto",
            output_path=None,
        ),
    )
    monkeypatch.setattr(
        module,
        "resolve_gemma4_text_runtime",
        lambda *, device, torch_dtype: ("cpu", "float32"),
    )
    monkeypatch.setattr(
        module.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: type("FakeConfig", (), {"model_type": "gemma4"})()),
    )
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: _FakeTokenizer()),
    )

    def _fake_model_from_pretrained(*args, **kwargs):
        captured_load_kwargs.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(
        module.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_fake_model_from_pretrained),
    )
    monkeypatch.setattr(module, "resolve_hf_auth_kwargs", lambda: {})

    module.main()
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "ok"
    assert "dtype" in captured_load_kwargs
    assert captured_load_kwargs["dtype"] == torch.float32
    assert "torch_dtype" not in captured_load_kwargs
    assert output["runtime_torch_dtype"] == "float32"


def test_probe_gemma4_text_dense_uses_resolved_mps_runtime_and_output_path(monkeypatch, capsys, tmp_path) -> None:
    module = _load_module("probe_gemma4_text_dense_mps", "scripts/probe_gemma4_text.py")
    captured_load_kwargs: dict[str, object] = {}
    output_path = tmp_path / "dense_probe.json"

    class _FakeModel:
        def __init__(self) -> None:
            self.device = torch.device("mps")
            self.hf_device_map = None

        def eval(self) -> None:
            return None

        def parameters(self):
            yield torch.zeros(1, device="cpu")

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]], dtype=torch.long)

    class _FakeTokenizer:
        def __call__(self, prompt, return_tensors=None):
            class _Inputs(dict):
                def to(self, device):
                    return self

            return _Inputs({"input_ids": torch.tensor([[1, 2]], dtype=torch.long)})

        def decode(self, ids, skip_special_tokens=True):
            return "decoded"

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            model_id="google/gemma-4-E2B",
            device_map="mps",
            torch_dtype="bfloat16",
            attn_implementation=None,
            max_new_tokens=1,
            prompt="hello",
            run_dotcache=False,
            dotcache_profile="balanced",
            tokens_per_page=4,
            bits_k=4,
            bits_v=4,
            group_size=32,
            dotcache_backend="auto",
            output_path=str(output_path),
        ),
    )
    monkeypatch.setattr(
        module,
        "resolve_gemma4_text_runtime",
        lambda *, device, torch_dtype: ("mps", "float16"),
    )
    monkeypatch.setattr(
        module.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: type("FakeConfig", (), {"model_type": "gemma4"})()),
    )
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: _FakeTokenizer()),
    )

    def _fake_model_from_pretrained(*args, **kwargs):
        captured_load_kwargs.update(kwargs)
        return _FakeModel()

    monkeypatch.setattr(
        module.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(_fake_model_from_pretrained),
    )
    monkeypatch.setattr(module, "resolve_hf_auth_kwargs", lambda: {})

    module.main()
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "ok"
    assert output["runtime_torch_dtype"] == "float16"
    assert captured_load_kwargs["dtype"] == torch.float16
    assert output_path.exists()
    assert json.loads(output_path.read_text()) == output


def test_probe_gemma4_text_dotcache_auto_device_uses_resolved_mps_runtime(monkeypatch, capsys) -> None:
    module = _load_module("probe_gemma4_text_dotcache", "scripts/probe_gemma4_text.py")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            model_id="google/gemma-4-E2B",
            device_map="auto",
            torch_dtype="bfloat16",
            attn_implementation=None,
            max_new_tokens=2,
            prompt="hello",
            run_dotcache=True,
            dotcache_profile="balanced",
            tokens_per_page=4,
            bits_k=4,
            bits_v=4,
            group_size=32,
            dotcache_backend="torch_mps",
            output_path=None,
        ),
    )
    monkeypatch.setattr(
        module.AutoConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(module, "resolve_hf_auth_kwargs", lambda: {})
    monkeypatch.setattr(
        module,
        "resolve_gemma4_text_runtime",
        lambda *, device, torch_dtype: ("mps", "float16"),
    )
    monkeypatch.setattr(
        module,
        "gemma4_text_recommended_dotcache_config",
        lambda *args, **kwargs: "fake-config",
    )

    class _FakeHarness:
        def generate_greedy(self, prompt=None, max_new_tokens=0):
            return {
                "dense_generated_ids": [1, 2],
                "dotcache_generated_ids": [1, 2],
                "greedy_token_agreement_rate": 1.0,
                "teacher_forced_logit_max_abs_error": 0.0,
                "teacher_forced_logit_max_rel_error": 0.0,
                "resident_bytes": 1,
                "kv_resident_bytes": 1,
                "decode_ms_per_step": 1.0,
                "m0_pages": 1,
                "m3_pages": 0,
                "dense_text": "dense",
                "dotcache_text": "dotcache",
            }

    def _fake_from_pretrained(model_id, config, **kwargs):
        captured["model_id"] = model_id
        captured["config"] = config
        captured["kwargs"] = dict(kwargs)
        return _FakeHarness()

    monkeypatch.setattr(
        module.Gemma4TextHarness,
        "from_pretrained",
        staticmethod(_fake_from_pretrained),
    )

    module.main()
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "ok"
    assert output["mode"] == "dotcache"
    assert captured["kwargs"]["device"] == "mps"
    assert captured["kwargs"]["torch_dtype"] == "float16"
    assert output["runtime_device"] == "mps"
    assert output["runtime_torch_dtype"] == "float16"
