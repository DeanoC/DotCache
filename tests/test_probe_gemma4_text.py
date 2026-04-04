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


def test_probe_gemma4_text_uses_torch_dtype_kwarg(monkeypatch, capsys) -> None:
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
        ),
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
    assert "torch_dtype" in captured_load_kwargs
    assert captured_load_kwargs["torch_dtype"] == torch.float32
    assert "dtype" not in captured_load_kwargs
