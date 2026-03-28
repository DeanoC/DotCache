import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from dotcache.backends import mps_available
from dotcache.config import DotCacheConfig
from dotcache.integrations import llama as llama_integration
from dotcache.integrations.llama import (
    LlamaDotCacheModelAdapter,
    _prewarm_torch_decode_layers,
    resolve_hf_auth_kwargs,
    run_llama_loss_harness,
    run_llama_generation_harness,
    run_llama_replay_harness,
)

LlamaConfig = transformers.LlamaConfig
LlamaForCausalLM = transformers.LlamaForCausalLM

requires_mps = pytest.mark.skipif(not mps_available(), reason="torch_mps is unavailable")


def _tiny_llama_model(*, device: str = "cpu"):
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config)
    model.to(device)
    model.eval()
    dotcache_config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    adapter = LlamaDotCacheModelAdapter(model, dotcache_config, backend="torch_mps" if device == "mps" else "cpu_ref")
    return model, adapter


def test_llama_replay_harness_runs_on_tiny_random_model() -> None:
    model, adapter = _tiny_llama_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    result = run_llama_replay_harness(model, adapter, input_ids=input_ids, decode_steps=3)

    assert result["decode_steps"] == 3
    assert np.isfinite(result["replay_context_max_abs_error"])
    assert np.isfinite(result["teacher_forced_logit_max_abs_error"])


def test_llama_generation_harness_runs_on_tiny_random_model() -> None:
    model, adapter = _tiny_llama_model()
    input_ids = torch.tensor([[7, 8, 9, 10, 11, 12]], dtype=torch.long)

    result = run_llama_generation_harness(model, adapter, input_ids=input_ids, max_new_tokens=4)

    assert len(result["dense_generated_ids"]) == 4
    assert len(result["dotcache_generated_ids"]) == 4
    assert np.isfinite(result["decode_ms_per_step"])
    assert np.isfinite(result["teacher_forced_logit_max_abs_error"])


def test_llama_adapter_can_reconfigure_dotcache_mode() -> None:
    model, adapter = _tiny_llama_model()
    assert adapter.dotcache_config.default_mode_k == "M0"
    assert adapter.dotcache_config.default_mode_v == "M0"

    reconfigured = DotCacheConfig(
        head_dim=32,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=4,
        default_mode_k="T3",
        default_mode_v="T3",
        quant_scheme_k="turbo3",
        quant_scheme_v="turbo3",
    )
    adapter.reconfigure(reconfigured)

    assert adapter.dotcache_config.default_mode_k == "T3"
    assert adapter.dotcache_config.default_mode_v == "T3"
    assert adapter.model_kv_cache.config.default_mode_k == "T3"
    assert adapter.model_kv_cache.config.default_mode_v == "T3"


def test_resolve_hf_auth_kwargs_prefers_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    assert resolve_hf_auth_kwargs() == {}

    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", "fallback-token")
    assert resolve_hf_auth_kwargs() == {"token": "fallback-token"}

    monkeypatch.setenv("HF_TOKEN", "primary-token")
    assert resolve_hf_auth_kwargs() == {"token": "primary-token"}


def test_llama_harness_from_pretrained_forwards_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, tuple[str, dict[str, object]]] = {}

    def _fake_model_from_pretrained(model_id: str, **kwargs):
        captured["model"] = (model_id, dict(kwargs))
        config = LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=128,
            max_position_embeddings=64,
        )
        return LlamaForCausalLM(config)

    class _FakeTokenizer:
        pad_token_id = None
        eos_token_id = 7

    def _fake_tokenizer_from_pretrained(model_id: str, **kwargs):
        captured["tokenizer"] = (model_id, dict(kwargs))
        return _FakeTokenizer()

    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setattr(
        llama_integration,
        "AutoModelForCausalLM",
        type("FakeAutoModelForCausalLM", (), {"from_pretrained": staticmethod(_fake_model_from_pretrained)}),
    )
    monkeypatch.setattr(
        llama_integration,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(_fake_tokenizer_from_pretrained)}),
    )

    harness = llama_integration.LlamaDotCacheHarness.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4),
        backend="cpu_ref",
        device="cpu",
    )

    assert captured["model"][0] == "meta-llama/Llama-3.2-3B-Instruct"
    assert captured["model"][1]["token"] == "hf-secret"
    assert captured["tokenizer"][0] == "meta-llama/Llama-3.2-3B-Instruct"
    assert captured["tokenizer"][1]["token"] == "hf-secret"
    assert harness.tokenizer.pad_token_id == harness.tokenizer.eos_token_id

def test_llama_generation_harness_emits_profile_on_tiny_random_model() -> None:
    model, adapter = _tiny_llama_model()
    input_ids = torch.tensor([[7, 8, 9, 10, 11, 12]], dtype=torch.long)

    result = run_llama_generation_harness(model, adapter, input_ids=input_ids, max_new_tokens=4, profile=True)

    profile = result["profile"]
    assert result["resident_bytes"] >= result["kv_resident_bytes"]
    assert result["dotcache_vs_dense_total_resident_bytes_ratio"] >= result["dotcache_vs_dense_kv_bytes_ratio"]
    assert result["resident_bytes"] == result["kv_resident_bytes"] + result["prepared_chunk_resident_bytes"]
    assert result["prepared_chunk_resident_bytes"] <= result["prepared_chunk_cache_budget_bytes"]
    assert profile["device_type"] == "cpu"
    assert profile["prefill_cache_ingest"]["host_to_device_bytes"] >= 0
    assert profile["prefill_cache_ingest"]["trace"]["prepare_calls"] >= 0
    assert profile["dotcache_decode"]["model_forward_ms_total"] >= 0.0
    assert profile["dotcache_decode"]["trace"]["score_calls"] >= 0
    assert profile["dotcache_decode"]["trace"]["mix_calls"] >= 0
    assert len(profile["dotcache_decode"]["per_layer"]) == model.config.num_hidden_layers
    assert result["decode_score_ms_per_step"] >= 0.0
    assert result["decode_mix_ms_per_step"] >= 0.0


def test_llama_loss_harness_runs_on_tiny_random_model() -> None:
    model, adapter = _tiny_llama_model()
    input_ids = torch.tensor([[7, 8, 9, 10, 11, 12, 13, 14]], dtype=torch.long)

    result = run_llama_loss_harness(model, adapter, input_ids=input_ids, prefix_length=4, eval_steps=4)

    assert result["sequence_length"] == 8
    assert result["prefix_length"] == 4
    assert result["eval_steps"] == 4
    assert np.isfinite(result["dense_teacher_forced_loss"])
    assert np.isfinite(result["dotcache_teacher_forced_loss"])
    assert np.isfinite(result["teacher_forced_loss_delta"])
    assert 0.0 <= result["teacher_forced_token_agreement_rate"] <= 1.0


def test_prewarm_torch_decode_layers_warms_each_populated_cuda_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeKVCache:
        def __init__(self) -> None:
            self._torch_device_type = "cuda"
            self.calls: list[int] = []

        def layer_sequence_length(self, layer_id: int) -> int:
            return 4 if layer_id != 1 else 0

        def decode_layer_torch(self, layer_id, query_step, q_head_to_kv_head, *, query_scale, trace):
            self.calls.append(layer_id)
            assert query_step.shape == (4, 32)
            assert query_step.device.type == "cpu"
            assert query_scale == 1.0
            assert trace is None
            return query_step

    class _FakeAdapter:
        def __init__(self) -> None:
            self.backend = "torch_cuda"
            self.model = type(
                "FakeModel",
                (),
                {"config": type("FakeConfig", (), {"num_attention_heads": 4, "num_hidden_layers": 3})()},
            )()
            self.dotcache_config = type("FakeDotCacheConfig", (), {"head_dim": 32})()
            self.q_head_to_kv_head = np.array([0, 0, 1, 1], dtype=np.int64)
            self.model_kv_cache = _FakeKVCache()

    adapter = _FakeAdapter()
    _prewarm_torch_decode_layers(adapter, device=torch.device("cpu"))
    assert adapter.model_kv_cache.calls == []

    original_zeros = torch.zeros
    monkeypatch.setattr(
        llama_integration.torch,
        "zeros",
        lambda shape, dtype, device: original_zeros(shape, dtype=dtype),
    )
    _prewarm_torch_decode_layers(adapter, device=torch.device("cuda"))
    assert adapter.model_kv_cache.calls == [0, 2]


@requires_mps
def test_llama_generation_harness_runs_on_mps_tiny_random_model() -> None:
    model, adapter = _tiny_llama_model(device="mps")
    input_ids = torch.tensor([[2, 4, 6, 8]], dtype=torch.long, device="mps")

    result = run_llama_generation_harness(model, adapter, input_ids=input_ids, max_new_tokens=3)

    assert len(result["dotcache_generated_ids"]) == 3
    assert result["resident_bytes"] >= 0
