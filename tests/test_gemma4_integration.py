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
    gemma4_text_tuned_knobs_for_workload,
    gemma4_text_tuned_preset_for_workload,
    gemma4_text_tuned_profile_for_workload,
    gemma4_text_tuned_value_layers_for_workload,
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


def test_gemma4_recommended_dotcache_config_balanced_layer0_adds_first_sliding_key_layer() -> None:
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

    tuned = gemma4_text_recommended_dotcache_config(config, profile="balanced_layer0")

    assert tuned.default_mode_k == "M0"
    assert tuned.default_mode_v == "M3"
    assert tuned.key_mode_overrides == ("layer:0=M3", "layer:1=M3", "layer:3=M3")


def test_gemma4_recommended_dotcache_config_balanced_layer0_8_adds_both_sliding_key_layers() -> None:
    config = Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=12,
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

    tuned = gemma4_text_recommended_dotcache_config(config, profile="balanced_layer0_8")

    assert tuned.default_mode_k == "M0"
    assert tuned.default_mode_v == "M3"
    assert tuned.key_mode_overrides == (
        "layer:0=M3",
        "layer:1=M3",
        "layer:3=M3",
        "layer:5=M3",
        "layer:7=M3",
        "layer:8=M3",
        "layer:9=M3",
    )


def test_gemma4_tuned_profile_for_workload_matches_current_matrix_heuristic() -> None:
    assert gemma4_text_tuned_profile_for_workload(prompt_length=512, decode_budget=8) == "balanced_layer0"
    assert gemma4_text_tuned_profile_for_workload(prompt_length=1024, decode_budget=8) == "balanced_layer0_8"
    assert gemma4_text_tuned_profile_for_workload(prompt_length=1024, decode_budget=24) == "balanced"
    assert gemma4_text_tuned_profile_for_workload(prompt_length=1024, decode_budget=32) == "balanced"
    assert gemma4_text_tuned_profile_for_workload(prompt_length=2048, decode_budget=16) == "balanced_layer0_8"
    assert gemma4_text_tuned_profile_for_workload(prompt_length=4096, decode_budget=16) == "balanced"
    assert gemma4_text_tuned_profile_for_workload(prompt_length=4096, decode_budget=24) == "balanced_layer0_8"
    assert gemma4_text_tuned_profile_for_workload(prompt_length=4096, decode_budget=32) == "balanced_layer0"


def test_gemma4_tuned_preset_for_workload_reads_measured_cutoff_table() -> None:
    preset = gemma4_text_tuned_preset_for_workload(prompt_length=2048, decode_budget=24)

    assert preset.profile == "balanced_layer0_8"
    assert preset.bits_k == 4
    assert preset.group_size == 16
    assert preset.tokens_per_page == 4
    assert preset.exact_value_layers is None

    preset = gemma4_text_tuned_preset_for_workload(prompt_length=4096, decode_budget=32)

    assert preset.profile == "balanced_layer0"
    assert preset.bits_k == 4
    assert preset.group_size == 16
    assert preset.tokens_per_page == 8
    assert preset.exact_value_layers == (0, 4, 8, 9, 14)


def test_gemma4_tuned_knobs_for_workload_match_current_knob_sweep() -> None:
    assert gemma4_text_tuned_knobs_for_workload(prompt_length=1024, decode_budget=24) == (4, 32, 4)
    assert gemma4_text_tuned_knobs_for_workload(prompt_length=2048, decode_budget=24) == (4, 16, 4)
    assert gemma4_text_tuned_knobs_for_workload(prompt_length=4096, decode_budget=24) == (4, 16, 8)
    assert gemma4_text_tuned_knobs_for_workload(prompt_length=4096, decode_budget=32) == (4, 16, 8)


def test_gemma4_tuned_value_layers_for_workload_match_current_value_sweep() -> None:
    assert gemma4_text_tuned_value_layers_for_workload(prompt_length=2048, decode_budget=24) is None
    assert gemma4_text_tuned_value_layers_for_workload(prompt_length=4096, decode_budget=24) == (0, 4, 8, 9, 14)
    assert gemma4_text_tuned_value_layers_for_workload(prompt_length=4096, decode_budget=32) == (0, 4, 8, 9, 14)


def test_gemma4_recommended_dotcache_config_adaptive_routes_to_tuned_profile() -> None:
    config = Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=12,
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
        profile="adaptive",
        prompt_length=1024,
        decode_budget=8,
    )

    assert tuned.default_mode_k == "M0"
    assert tuned.default_mode_v == "M3"
    assert tuned.key_mode_overrides == (
        "layer:0=M3",
        "layer:1=M3",
        "layer:3=M3",
        "layer:5=M3",
        "layer:7=M3",
        "layer:8=M3",
        "layer:9=M3",
    )


def test_gemma4_recommended_dotcache_config_adaptive_knobs_override_defaults_for_measured_workloads() -> None:
    config = Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=12,
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
        profile="adaptive",
        prompt_length=4096,
        decode_budget=24,
        adaptive_knobs=True,
    )

    assert tuned.bits_k == 4
    assert tuned.group_size == 16
    assert tuned.tokens_per_page == 8


def test_gemma4_recommended_dotcache_config_adaptive_values_enable_measured_4k_value_policy() -> None:
    config = Gemma4TextConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=12,
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
        profile="adaptive",
        prompt_length=4096,
        decode_budget=24,
        adaptive_knobs=True,
        adaptive_values=True,
    )

    assert tuned.default_mode_v == "M0"
    assert tuned.value_mode_overrides == (
        "layer:0=M3",
        "layer:4=M3",
        "layer:8=M3",
        "layer:9=M3",
        "layer:14=M3",
    )


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


def test_gemma4_prefill_ingest_preserves_absolute_context_for_truncated_sliding_layers() -> None:
    model, adapter = _tiny_gemma4_model()
    seq_len = 8
    context_length = 12
    head_dims = [32, 64, 32, 64]
    prefill_layers = [
        (
            np.zeros((1, seq_len if layer_idx % 2 == 0 else context_length, head_dims[layer_idx]), dtype=np.float32),
            np.zeros((1, seq_len if layer_idx % 2 == 0 else context_length, head_dims[layer_idx]), dtype=np.float32),
        )
        for layer_idx in range(4)
    ]

    adapter.load_prefill_cache_arrays(prefill_layers, context_length=context_length)

    sliding_source_cache = adapter.model_kv_cache._source_caches[0]
    full_source_cache = adapter.model_kv_cache._source_caches[1]
    assert adapter.model_kv_cache.layer_sequence_length(0) == context_length
    assert adapter.model_kv_cache.layer_sequence_length(1) == context_length
    assert sliding_source_cache._state(0, 0).session.key_pages[0].header.token_start == context_length - seq_len
    assert full_source_cache._state(1, 0).session.key_pages[0].header.token_start == 0
