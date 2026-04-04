import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from dotcache.config import DotCacheConfig
from dotcache.integrations import gemma4 as gemma4_integration
from dotcache.integrations.gemma4 import (
    Gemma4TextHarness,
    Gemma4TextModelAdapter,
    Gemma4TextModelWrapper,
    gemma4_full_attention_source_layers,
    gemma4_text_dotcache_supported,
    gemma4_text_recommended_dotcache_config,
    run_gemma4_text_generation_harness,
    run_gemma4_text_replay_harness,
)

Gemma4TextConfig = transformers.Gemma4TextConfig
Gemma4ForCausalLM = transformers.Gemma4ForCausalLM


def _tiny_gemma4_model():
    config = Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        global_head_dim=64,
        hidden_size_per_layer_input=0,
        vocab_size=128,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_kv_shared_layers=2,
        use_cache=True,
    )
    root_model = Gemma4ForCausalLM(config)
    root_model.eval()
    model = Gemma4TextModelWrapper(root_model)
    dotcache_config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    adapter = Gemma4TextModelAdapter(model, dotcache_config, backend="cpu_ref")
    return model, adapter


def test_gemma4_replay_harness_runs_on_tiny_random_model() -> None:
    model, adapter = _tiny_gemma4_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    result = run_gemma4_text_replay_harness(model, adapter, input_ids=input_ids, decode_steps=3)

    assert result["decode_steps"] == 3
    assert np.isfinite(result["replay_context_max_abs_error"])
    assert np.isfinite(result["teacher_forced_logit_max_abs_error"])


def test_gemma4_generation_harness_runs_on_tiny_random_model() -> None:
    model, adapter = _tiny_gemma4_model()
    input_ids = torch.tensor([[7, 8, 9, 10, 11, 12]], dtype=torch.long)

    result = run_gemma4_text_generation_harness(model, adapter, input_ids=input_ids, max_new_tokens=4)

    assert len(result["dense_generated_ids"]) == 4
    assert len(result["dotcache_generated_ids"]) == 4
    assert np.isfinite(result["decode_ms_per_step"])
    assert np.isfinite(result["teacher_forced_logit_max_abs_error"])


def test_gemma4_support_check_accepts_mixed_head_dims() -> None:
    config = Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        global_head_dim=64,
        hidden_size_per_layer_input=0,
        vocab_size=128,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        num_kv_shared_layers=0,
        use_cache=True,
    )

    assert gemma4_text_dotcache_supported(config) is True


def test_gemma4_full_attention_source_layers_ignores_shared_tail() -> None:
    config = Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        global_head_dim=64,
        hidden_size_per_layer_input=0,
        vocab_size=128,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_kv_shared_layers=2,
        use_cache=True,
    )

    assert gemma4_full_attention_source_layers(config) == (1, 3)


def test_gemma4_recommended_dotcache_config_balanced_keeps_values_exact_and_global_keys_exact() -> None:
    config = Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        global_head_dim=64,
        hidden_size_per_layer_input=0,
        vocab_size=128,
        max_position_embeddings=64,
        sliding_window=8,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        num_kv_shared_layers=2,
        use_cache=True,
    )

    tuned = gemma4_text_recommended_dotcache_config(
        config,
        bits_k=4,
        bits_v=4,
        tokens_per_page=8,
        group_size=16,
        profile="balanced",
    )

    assert tuned.head_dim == 32
    assert tuned.tokens_per_page == 8
    assert tuned.group_size == 16
    assert tuned.default_mode_k == "M0"
    assert tuned.default_mode_v == "M3"
    assert tuned.key_mode_overrides == ("layer:1=M3", "layer:3=M3")
    assert tuned.value_mode_overrides == ()


def test_gemma4_harness_from_pretrained_forwards_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, tuple[str, dict[str, object]]] = {}

    def _fake_model_from_pretrained(model_id: str, **kwargs):
        captured["model"] = (model_id, dict(kwargs))
        config = Gemma4TextConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=32,
            global_head_dim=32,
            hidden_size_per_layer_input=0,
            vocab_size=128,
            max_position_embeddings=64,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
            num_kv_shared_layers=0,
            use_cache=True,
        )
        return Gemma4ForCausalLM(config)

    class _FakeTokenizer:
        pad_token_id = None
        eos_token_id = 7

    def _fake_tokenizer_from_pretrained(model_id: str, **kwargs):
        captured["tokenizer"] = (model_id, dict(kwargs))
        return _FakeTokenizer()

    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.setattr(
        gemma4_integration,
        "AutoModelForCausalLM",
        type("FakeAutoModelForCausalLM", (), {"from_pretrained": staticmethod(_fake_model_from_pretrained)}),
    )
    monkeypatch.setattr(
        gemma4_integration,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(_fake_tokenizer_from_pretrained)}),
    )

    harness = Gemma4TextHarness.from_pretrained(
        "google/gemma-4-E2B",
        DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4),
        backend="cpu_ref",
        device="cpu",
        torch_dtype="float32",
    )

    assert captured["model"][0] == "google/gemma-4-E2B"
    assert captured["model"][1]["token"] == "hf-secret"
    assert captured["tokenizer"][0] == "google/gemma-4-E2B"
    assert captured["tokenizer"][1]["token"] == "hf-secret"
    assert harness.tokenizer.pad_token_id == harness.tokenizer.eos_token_id
