from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from dotcache.config import DotCacheConfig
from dotcache.integrations.qwen35 import (
    Qwen35AttentionSubsetDotCacheHarness,
    Qwen35AttentionSubsetDotCacheModelAdapter,
    Qwen35AttentionSubsetHarness,
    Qwen35AttentionSubsetModelAdapter,
    Qwen35DeltaNetStateRecord,
    Qwen35DeltaNetStateHarness,
    Qwen35DeltaNetStateModelAdapter,
    build_qwen35_deltanet_state_sample,
    Qwen35TextHarness,
    Qwen35TextModelAdapter,
    inspect_qwen35_deltanet_state,
    inspect_qwen35_hybrid_state,
    load_qwen35_text_only_from_pretrained,
    parse_qwen35_deltanet_statecache_int_overrides,
    resolve_qwen35_deltanet_statecache_recurrent_group_size_policy,
    resolve_qwen35_deltanet_statecache_recurrent_mode_policy,
    resolve_qwen35_deltanet_statecache_readout_mode_policy,
    resolve_qwen35_deltanet_statecache_readout_policy,
    run_qwen35_attention_subset_dotcache_harness,
    run_qwen35_attention_subset_dotcache_loss_harness,
    run_qwen35_attention_subset_dotcache_serving_scorer_diagnostic_harness,
    run_qwen35_attention_subset_dotcache_serving_harness,
    run_qwen35_attention_subset_dotcache_serving_recall_analysis_harness,
    run_qwen35_attention_subset_dotcache_serving_quality_harness,
    run_qwen35_hybrid_combined_localization_harness,
    run_qwen35_attention_subset_statecache_dotcache_harness,
    run_qwen35_attention_subset_replay_harness,
    run_qwen35_deltanet_state_ablation_harness,
    run_qwen35_deltanet_statecache_localization_harness,
    run_qwen35_deltanet_statecache_readout_harness,
    run_qwen35_deltanet_statecache_serving_harness,
    run_qwen35_deltanet_statecache_loss_harness,
    run_qwen35_text_generation_harness,
    run_qwen35_text_loss_harness,
    summarize_qwen35_dotcache_fit,
    _advance_attention_subset_cache_placeholder,
    _extract_attention_subset_prefill_tensors,
    _configure_qwen35_linear_attention_runtime,
    _replace_attention_subset_cache_with_placeholders,
    _decode_input_id_sequence,
    _qwen35_mps_serving_shortlist_heuristic,
)
from dotcache.model_kv_cache import ModelPagedKVCache


class _TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None

    def __call__(self, text: str, *, return_tensors: str | None = None, add_special_tokens: bool = True):
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        payload = [3 + (ord(char) % 17) for char in text]
        token_ids.extend(payload or [self.eos_token_id])
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return {"input_ids": token_ids, "attention_mask": [1] * len(token_ids)}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        filtered = []
        for token_id in token_ids:
            if skip_special_tokens and int(token_id) in {self.bos_token_id, self.eos_token_id, 0}:
                continue
            filtered.append(str(int(token_id)))
        return " ".join(filtered)


class _FakeLayerCacheRecord:
    def __init__(self, *, keys=None, values=None):
        self.keys = keys
        self.values = values


class _FakeLayerStructuredCache:
    def __init__(self, *, layers, conv_states=None, recurrent_states=None):
        self.layers = list(layers)
        self.conv_states = list(conv_states or [])
        self.recurrent_states = list(recurrent_states or [])


def _tiny_qwen35_model() -> Qwen3_5ForConditionalGeneration:
    config = Qwen3_5Config(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "vocab_size": 128,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "depth": 1,
            "num_heads": 4,
        },
    )
    return Qwen3_5ForConditionalGeneration(config).eval()


def _tiny_deltanet_qwen35_model() -> Qwen3_5ForConditionalGeneration:
    config = Qwen3_5Config(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 128,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "depth": 1,
            "num_heads": 2,
        },
    )
    return Qwen3_5ForConditionalGeneration(config).eval()


@pytest.mark.parametrize(
    ("prompt_length", "expected_band", "expected_overrides"),
    [
        (512, "baseline", {}),
        (1024, "late", {12: 2, 18: 2, 21: 2}),
        (2048, "late", {12: 2, 18: 2, 21: 2}),
        (4096, "full", {5: 2, 8: 2, 12: 2, 18: 2, 21: 2}),
        (8192, "early", {5: 2, 8: 2}),
    ],
)
def test_resolve_qwen35_deltanet_statecache_readout_policy_890m_bands(
    prompt_length: int,
    expected_band: str,
    expected_overrides: dict[int, int],
) -> None:
    overrides, band = resolve_qwen35_deltanet_statecache_readout_policy(
        prompt_length=prompt_length,
        policy="890m_context_banded_v1",
    )
    assert overrides == expected_overrides
    assert band == expected_band


@pytest.mark.parametrize(
    ("prompt_length", "expected_band", "expected_overrides"),
    [
        (2048, "baseline", {}),
        (3072, "midband_outliers", {4: "M3", 20: "M3"}),
        (4096, "midband_outliers", {4: "M3", 20: "M3"}),
        (6144, "baseline", {}),
    ],
)
def test_resolve_qwen35_deltanet_statecache_readout_mode_policy_890m_bands(
    prompt_length: int,
    expected_band: str,
    expected_overrides: dict[int, str],
) -> None:
    overrides, band = resolve_qwen35_deltanet_statecache_readout_mode_policy(
        prompt_length=prompt_length,
        policy="890m_m3_outlier_pair_midband_v1",
    )
    assert overrides == expected_overrides
    assert band == expected_band


@pytest.mark.parametrize(
    ("prompt_length", "expected_band", "expected_overrides"),
    [
        (2048, "baseline", {}),
        (3072, "midband_outliers", {4: "M3", 20: "M3"}),
        (4096, "midband_outliers", {4: "M3", 20: "M3"}),
        (6144, "baseline", {}),
    ],
)
def test_resolve_qwen35_deltanet_statecache_recurrent_mode_policy_890m_bands(
    prompt_length: int,
    expected_band: str,
    expected_overrides: dict[int, str],
) -> None:
    overrides, band = resolve_qwen35_deltanet_statecache_recurrent_mode_policy(
        prompt_length=prompt_length,
        policy="890m_m3_outlier_pair_midband_v1",
    )
    assert overrides == expected_overrides
    assert band == expected_band


@pytest.mark.parametrize(
    ("prompt_length", "decode_steps", "expected_band", "expected_overrides"),
    [
        (4096, 8, "baseline", {}),
        (4096, 16, "baseline", {}),
        (6144, 8, "baseline", {}),
        (6144, 16, "long_horizon", {18: 8, 20: 8}),
    ],
)
def test_resolve_qwen35_deltanet_statecache_recurrent_group_size_policy_890m_bands(
    prompt_length: int,
    decode_steps: int,
    expected_band: str,
    expected_overrides: dict[int, int],
) -> None:
    overrides, band = resolve_qwen35_deltanet_statecache_recurrent_group_size_policy(
        prompt_length=prompt_length,
        decode_steps=decode_steps,
        policy="890m_long_horizon_group_escape_v1",
    )
    assert overrides == expected_overrides
    assert band == expected_band


def test_qwen35_rocm_fast_path_wrapper_downcasts_float32_qkv(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeLinearAttn:
        def __init__(self) -> None:
            self.chunk_seen_dtypes: list[torch.dtype] = []
            self.recur_seen_dtypes: list[torch.dtype] = []

            def _chunk(q, k, v, *args, **kwargs):
                self.chunk_seen_dtypes.append(q.dtype)
                return q + k + v, q

            def _recur(q, k, v, *args, **kwargs):
                self.recur_seen_dtypes.append(q.dtype)
                return q + k + v, q

            self.chunk_gated_delta_rule = _chunk
            self.recurrent_gated_delta_rule = _recur

    class _FakeLayer:
        def __init__(self) -> None:
            self.linear_attn = _FakeLinearAttn()

    class _FakeLanguageModel:
        def __init__(self) -> None:
            self.layers = [_FakeLayer()]

    class _FakeRootModel:
        def __init__(self) -> None:
            self.language_model = _FakeLanguageModel()

        def parameters(self):
            yield _FakeParam()

    class _FakeTextConfig:
        layer_types = ["linear_attention"]

    class _FakeConfig:
        def __init__(self) -> None:
            self.text_config = _FakeTextConfig()

    class _FakeParam:
        device = torch.device("cuda")

    class _FakeModel:
        def __init__(self) -> None:
            self.model = _FakeRootModel()
            self.config = _FakeConfig()

    fake_model = _FakeModel()
    monkeypatch.setattr(torch.version, "hip", "7.1", raising=False)

    _configure_qwen35_linear_attention_runtime(fake_model)

    linear_attn = fake_model.model.language_model.layers[0].linear_attn
    q = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    k = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    v = torch.randn(1, 2, 3, 4, dtype=torch.float32)

    out_chunk, state_chunk = linear_attn.chunk_gated_delta_rule(q, k, v)
    out_recur, state_recur = linear_attn.recurrent_gated_delta_rule(q, k, v)

    assert linear_attn.chunk_seen_dtypes == [torch.float16]
    assert linear_attn.recur_seen_dtypes == [torch.float16]
    assert out_chunk.dtype == torch.float32
    assert out_recur.dtype == torch.float32
    assert state_chunk.dtype == torch.float16
    assert state_recur.dtype == torch.float16


def test_qwen35_adapter_rejects_dotcache_mode() -> None:
    adapter = Qwen35TextModelAdapter(model=_tiny_qwen35_model())
    with pytest.raises(ValueError, match="dense mode"):
        adapter.set_mode("dotcache")


def test_qwen35_harness_rejects_multimodal_inputs() -> None:
    model = _tiny_qwen35_model()
    harness = Qwen35TextHarness(model=model, tokenizer=_TinyTokenizer(), adapter=Qwen35TextModelAdapter(model=model))
    with pytest.raises(ValueError, match="text-only"):
        harness.tokenize_prompt("hello", multimodal_inputs={"image": b"not-supported"})


def test_qwen35_generation_harness_runs_on_tiny_hybrid_model() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35TextModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello qwen", return_tensors="pt")
    result = run_qwen35_text_generation_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=2,
        tokenizer=tokenizer,
    )
    assert result["text_only"] is True
    assert result["dotcache_ready"] is False
    assert result["hybrid_family"] == "qwen3_5"
    assert result["hybrid_linear_attention_layer_count"] == 3
    assert result["hybrid_full_attention_layer_count"] == 1
    assert result["prompt_length"] == int(encoded["input_ids"].shape[1])
    assert len(result["dense_generated_ids"]) == 2


def test_qwen35_loss_harness_runs_on_tiny_hybrid_model() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35TextModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("abcdefgh", return_tensors="pt")
    result = run_qwen35_text_loss_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        prefix_length=4,
        eval_steps=3,
        tokenizer=tokenizer,
    )
    assert result["text_only"] is True
    assert result["dotcache_ready"] is False
    assert result["dense_teacher_forced_loss"] >= 0.0
    assert result["dense_teacher_forced_perplexity"] >= 1.0
    assert 0.0 <= result["dense_teacher_forced_target_match_rate"] <= 1.0


def test_qwen35_from_pretrained_uses_native_loader_and_sets_pad_token(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _tiny_qwen35_model()
    load_calls: list[tuple[str, dict[str, object]]] = []

    class _FakeTokenizer(_TinyTokenizer):
        pass

    fake_tokenizer = _FakeTokenizer()

    def _fake_model_loader(model_id: str, **kwargs):
        load_calls.append((model_id, kwargs))
        return fake_model

    def _fake_tokenizer_loader(model_id: str, **kwargs):
        load_calls.append((f"tokenizer:{model_id}", kwargs))
        return fake_tokenizer

    monkeypatch.setattr("dotcache.integrations.qwen35.Qwen3_5ForConditionalGeneration.from_pretrained", _fake_model_loader)
    monkeypatch.setattr("dotcache.integrations.qwen35.AutoTokenizer.from_pretrained", _fake_tokenizer_loader)

    model, tokenizer = load_qwen35_text_only_from_pretrained("Qwen/Qwen3.5-0.8B", device="cpu", torch_dtype="float32")
    assert model is fake_model
    assert tokenizer is fake_tokenizer
    assert fake_tokenizer.pad_token_id == fake_tokenizer.eos_token_id
    assert load_calls[0][0] == "Qwen/Qwen3.5-0.8B"
    assert load_calls[1][0] == "tokenizer:Qwen/Qwen3.5-0.8B"


def test_qwen35_harness_exposes_block_summary() -> None:
    model = _tiny_qwen35_model()
    harness = Qwen35TextHarness(model=model, tokenizer=_TinyTokenizer(), adapter=Qwen35TextModelAdapter(model=model))
    summary = harness.adapter.hybrid_block_summary()
    assert summary["hybrid_layer_count"] == 4
    assert summary["vision_config_present"] is True


def test_qwen35_hybrid_fit_summary_marks_attention_subset_and_hybrid_need() -> None:
    summary = summarize_qwen35_dotcache_fit(_tiny_qwen35_model())
    assert summary["attention_candidate_layer_ids"] == [3]
    assert summary["hybrid_only_layer_ids"] == [0, 1, 2]
    assert summary["requires_hybrid_state_abstraction"] is True
    assert summary["suggested_next_step"] == "attention_subset_only_then_generalize_state"


def test_qwen35_hybrid_state_inspection_reports_split_state_bytes() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35TextModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello hybrid", return_tensors="pt")
    result = inspect_qwen35_hybrid_state(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
    )
    assert result["text_only"] is True
    assert result["dotcache_ready"] is False
    assert result["attention_candidate_layer_count"] == 1
    assert result["hybrid_only_layer_count"] == 3
    assert result["hybrid_state_total_bytes"] > 0
    assert result["hybrid_linear_recurrent_state_bytes"] > 0
    assert result["hybrid_attention_kv_bytes"] > 0
    assert result["hybrid_fixed_resident_bytes"] == (
        result["hybrid_linear_conv_state_bytes"] + result["hybrid_linear_recurrent_state_bytes"]
    )
    assert result["hybrid_token_growing_bytes"] == result["hybrid_attention_kv_bytes"]
    assert len(result["hybrid_state_layers"]) == 4
    assert result["hybrid_state_layers"][0]["layer_type"] == "linear_attention"
    assert result["hybrid_state_layers"][3]["layer_type"] == "full_attention"
    assert result["hybrid_state_layers"][0]["state_growth_family"] == "fixed_resident"
    assert result["hybrid_state_layers"][3]["state_growth_family"] == "token_growing"
    assert result["hybrid_fixed_resident_layer_ids"] == [0, 1, 2]
    assert result["hybrid_token_growing_layer_ids"] == [3]
    assert result["hybrid_state_partition_ready"] is True


def test_qwen35_hybrid_state_inspection_reports_decode_growth() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35TextModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello hybrid growth", return_tensors="pt")
    result = inspect_qwen35_hybrid_state(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
    )
    assert result["decode_steps"] == 2
    assert result["dense_decode_ms_per_step"] >= 0.0
    assert result["hybrid_prefill_state_total_bytes"] > 0
    assert result["hybrid_final_state_total_bytes"] >= result["hybrid_prefill_state_total_bytes"]
    assert result["hybrid_state_growth_bytes"] >= 0
    assert result["hybrid_attention_kv_growth_bytes"] >= 0
    assert result["hybrid_fixed_resident_growth_bytes"] == 0
    assert result["hybrid_token_growing_growth_bytes"] == result["hybrid_attention_kv_growth_bytes"]
    assert len(result["hybrid_state_growth_layers"]) == 4
    assert any(layer["layer_type"] == "linear_attention" for layer in result["hybrid_state_growth_layers"])
    assert any(layer["layer_type"] == "full_attention" for layer in result["hybrid_state_growth_layers"])
    assert any(layer["state_growth_family"] == "fixed_resident" for layer in result["hybrid_state_growth_layers"])
    assert any(layer["state_growth_family"] == "token_growing" for layer in result["hybrid_state_growth_layers"])


def test_qwen35_adapter_partitions_native_hybrid_state() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35TextModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello hybrid partition", return_tensors="pt")
    outputs = model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        use_cache=True,
    )
    partition = adapter.partition_hybrid_state(outputs.past_key_values)
    assert partition.fixed_resident_layer_ids == [0, 1, 2]
    assert partition.token_growing_layer_ids == [3]
    assert len(partition.fixed_resident_layers) == 3
    assert len(partition.token_growing_layers) == 1
    assert all(layer.state_growth_family == "fixed_resident" for layer in partition.fixed_resident_layers)
    assert all(layer.state_growth_family == "token_growing" for layer in partition.token_growing_layers)
    assert any(layer.conv_state is not None for layer in partition.fixed_resident_layers)
    assert any(layer.recurrent_state is not None for layer in partition.fixed_resident_layers)
    assert partition.token_growing_layers[0].key_cache is not None
    assert partition.token_growing_layers[0].value_cache is not None
    summary = partition.to_summary(model_or_config=model)
    assert summary["hybrid_fixed_resident_layer_count"] == 3
    assert summary["hybrid_token_growing_layer_count"] == 1
    assert summary["hybrid_fixed_resident_layer_ids"] == [0, 1, 2]
    assert summary["hybrid_token_growing_layer_ids"] == [3]


def test_qwen35_adapter_partitions_layer_structured_hybrid_state() -> None:
    model = _tiny_qwen35_model()
    key_tensor = torch.ones((1, 1, 3, 16), dtype=torch.float32)
    value_tensor = torch.full((1, 1, 3, 16), 2.0, dtype=torch.float32)
    cache = _FakeLayerStructuredCache(
        layers=[
            _FakeLayerCacheRecord(),
            _FakeLayerCacheRecord(),
            _FakeLayerCacheRecord(),
            _FakeLayerCacheRecord(keys=key_tensor, values=value_tensor),
        ],
        conv_states=[torch.zeros((1, 16), dtype=torch.float32) for _ in range(3)],
        recurrent_states=[torch.zeros((1, 16), dtype=torch.float32) for _ in range(3)],
    )
    partition = Qwen35TextModelAdapter(model=model).partition_hybrid_state(cache)
    assert partition.fixed_resident_layer_ids == [0, 1, 2]
    assert partition.token_growing_layer_ids == [3]
    assert torch.equal(partition.token_growing_layers[0].key_cache, key_tensor)
    assert torch.equal(partition.token_growing_layers[0].value_cache, value_tensor)


def test_qwen35_attention_subset_prefill_helpers_support_layer_structured_cache() -> None:
    key_tensor = torch.randn((1, 1, 4, 8), dtype=torch.float32)
    value_tensor = torch.randn((1, 1, 4, 8), dtype=torch.float32)
    cache = _FakeLayerStructuredCache(
        layers=[
            _FakeLayerCacheRecord(),
            _FakeLayerCacheRecord(keys=key_tensor.clone(), values=value_tensor.clone()),
        ]
    )
    extracted = _extract_attention_subset_prefill_tensors(cache, [1])
    assert torch.equal(extracted[1][0], key_tensor)
    assert torch.equal(extracted[1][1], value_tensor)

    _replace_attention_subset_cache_with_placeholders(cache, [1])
    assert cache.layers[1].keys.shape[2] == 0
    assert cache.layers[1].values.shape[2] == 0

    cache.layers[1].keys = key_tensor[:, :, :1, :].clone()
    cache.layers[1].values = value_tensor[:, :, :1, :].clone()
    _advance_attention_subset_cache_placeholder(cache, 1)
    assert cache.layers[1].keys.shape[2] == 2
    assert cache.layers[1].values.shape[2] == 2


def test_qwen35_deltanet_state_adapter_wraps_only_linear_layers() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    assert adapter.deltanet_layer_ids() == [0, 1, 2]


def test_qwen35_deltanet_state_inspection_reports_only_linear_layers() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    result = inspect_qwen35_deltanet_state(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=1,
    )
    assert result["deltanet_state_ready"] is True
    assert result["deltanet_total_state_bytes"] > 0
    assert result["deltanet_conv_state_bytes"] > 0
    assert result["deltanet_recurrent_state_bytes"] > 0
    assert len(result["deltanet_state_layers"]) == 3
    assert all(record["layer_type"] == "linear_attention" for record in result["deltanet_state_layers"])
    assert "recurrent_state" in result["deltanet_state_layers"][0]["state_shapes"]
    assert len(result["deltanet_state_layers"][0]["state_delta_norms"]) == 1


def test_qwen35_deltanet_state_harness_tokenizes_and_runs() -> None:
    model = _tiny_deltanet_qwen35_model()
    tokenizer = _TinyTokenizer()
    harness = Qwen35DeltaNetStateHarness(
        model=model,
        tokenizer=tokenizer,
        adapter=Qwen35DeltaNetStateModelAdapter(model=model),
    )
    input_ids, attention_mask = harness.tokenize_prompt("x")
    result = harness.inspect_deltanet_state(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
    )
    assert result["deltanet_state_ready"] is True
    assert len(result["deltanet_state_layers"]) == 3


def test_qwen35_deltanet_state_ablation_reports_stages() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    result = run_qwen35_deltanet_state_ablation_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=1,
        group_size=16,
        bits=(4,),
    )
    assert result["deltanet_state_ablation_ready"] is True
    assert result["deltanet_dominant_failure_stage"] in {
        "dense_baseline",
        "readout_only_m0",
        "post_update_m0",
        "pre_update_m0",
        "full_state_path_m0",
    }
    stages = {(entry["stage_name"], entry["bits"]) for entry in result["deltanet_ablation_results"]}
    assert ("dense_baseline", None) in stages
    assert ("escape_m3", None) in stages
    dense_stage = next(entry for entry in result["deltanet_ablation_results"] if entry["stage_name"] == "dense_baseline")
    escape_stage = next(entry for entry in result["deltanet_ablation_results"] if entry["stage_name"] == "escape_m3")
    assert dense_stage["output_max_abs_error"] <= 1e-6
    assert escape_stage["output_max_abs_error"] <= 1e-4

def test_qwen35_build_deltanet_state_sample_from_records() -> None:
    pre_state = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    post_state = pre_state + 1.0
    record = Qwen35DeltaNetStateRecord(
        step_index=0,
        layer_id=2,
        token_index=17,
        hidden_states=torch.zeros((1, 1, 8), dtype=torch.float32),
        output_states=torch.zeros((1, 1, 8), dtype=torch.float32),
        pre_conv_state=None,
        post_conv_state=None,
        pre_recurrent_state=pre_state,
        post_recurrent_state=post_state,
    )
    sample = build_qwen35_deltanet_state_sample([[record]], prompt_length=16, layer_id=2, state_kind="recurrent")
    assert sample["state_kind"] == "recurrent"
    assert sample["layer_id"] == 2
    assert sample["prompt_length"] == 16
    assert sample["token_indices"] == [17]
    assert sample["initial_state"].shape == (1, 3, 8)
    assert sample["update_deltas"].shape == (1, 1, 3, 8)


def test_qwen35_deltanet_statecache_readout_reports_compressed_recurrent_bytes() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=1,
        group_size=16,
        bits=8,
    )
    assert result["deltanet_statecache_ready"] is True
    assert result["deltanet_statecache_scope"] == "recurrent_only"
    assert result["deltanet_statecache_stage_name"] == "readout_only_m0"
    assert result["deltanet_statecache_bits"] == 8
    assert result["deltanet_statecache_conv_state_bytes"] == result["deltanet_conv_state_bytes"]
    assert result["deltanet_recurrent_state_bytes"] > result["deltanet_statecache_recurrent_state_bytes"]
    assert result["deltanet_dense_fixed_resident_bytes"] > result["deltanet_statecache_fixed_resident_bytes"]
    assert result["deltanet_statecache_output_max_abs_error"] >= 0.0
    assert result["deltanet_statecache_effective_recurrent_compression_ratio"] > 1.0
    assert len(result["deltanet_dense_generated_ids"]) == 1
    assert len(result["deltanet_statecache_generated_ids"]) == 1
    assert 0.0 <= result["deltanet_statecache_greedy_token_agreement_rate"] <= 1.0


def test_qwen35_deltanet_statecache_readout_supports_conv_only_scope() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=1,
        group_size=16,
        bits=8,
        statecache_scope="conv_only",
        conv_bits=8,
    )
    assert result["deltanet_statecache_scope"] == "conv_only"
    assert result["deltanet_conv_state_bytes"] >= result["deltanet_statecache_conv_state_bytes"]
    assert result["deltanet_statecache_recurrent_state_bytes"] == result["deltanet_recurrent_state_bytes"]
    assert result["deltanet_statecache_per_layer_conv_bytes"]


def test_qwen35_deltanet_statecache_readout_supports_conv_plus_recurrent_scope() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=1,
        group_size=16,
        bits=8,
        statecache_scope="conv_plus_recurrent",
        conv_bits=8,
    )
    assert result["deltanet_statecache_scope"] == "conv_plus_recurrent"
    assert result["deltanet_conv_state_bytes"] >= result["deltanet_statecache_conv_state_bytes"]
    assert result["deltanet_recurrent_state_bytes"] >= result["deltanet_statecache_recurrent_state_bytes"]
    assert result["deltanet_statecache_per_layer_conv_bytes"]
    assert result["deltanet_statecache_per_layer_recurrent_bytes"]


def test_qwen35_deltanet_statecache_readout_accepts_per_layer_renorm_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
        group_size=16,
        bits=8,
        statecache_scope="conv_plus_recurrent",
        conv_bits=8,
        state_stage="post_update_m0",
        recurrent_renorm_interval_overrides={0: 2},
        conv_renorm_interval_overrides={0: 2},
    )
    assert result["deltanet_statecache_recurrent_renorm_interval_overrides"] == {"0": 2}
    assert result["deltanet_statecache_conv_renorm_interval_overrides"] == {"0": 2}


def test_qwen35_deltanet_statecache_readout_accepts_post_update_recurrent_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        post_update_recurrent_renorm_interval_overrides={0: 2},
        post_update_recurrent_mode_overrides={0: "M3"},
    )
    assert result["deltanet_statecache_recurrent_renorm_interval_overrides"] == {}
    assert result["deltanet_statecache_post_update_recurrent_renorm_interval_overrides"] == {"0": 2}
    assert result["deltanet_statecache_recurrent_mode_overrides"] == {}
    assert result["deltanet_statecache_post_update_recurrent_mode_overrides"] == {"0": "M3"}
    assert result["deltanet_statecache_per_layer_recurrent_mode"]["0"] == "M3"


def test_qwen35_deltanet_statecache_readout_accepts_readout_recurrent_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        readout_recurrent_renorm_interval_overrides={0: 2},
        readout_recurrent_mode_overrides={0: "M3"},
    )
    assert result["deltanet_statecache_readout_recurrent_renorm_interval_overrides"] == {"0": 2}
    assert result["deltanet_statecache_readout_recurrent_mode_overrides"] == {"0": "M3"}
    assert result["deltanet_statecache_post_update_recurrent_renorm_interval_overrides"] == {}
    assert result["deltanet_statecache_post_update_recurrent_mode_overrides"] == {}
    assert result["deltanet_statecache_per_layer_recurrent_mode"]["0"] == "M3"


def test_qwen35_deltanet_statecache_readout_accepts_readout_recurrent_policy() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    input_ids = torch.ones((1, 1024), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        readout_recurrent_policy="890m_context_banded_v1",
    )
    assert result["deltanet_statecache_readout_recurrent_policy"] == "890m_context_banded_v1"
    assert result["deltanet_statecache_readout_recurrent_policy_band"] == "late"
    assert result["deltanet_statecache_readout_recurrent_renorm_interval_overrides"] == {
        "12": 2,
        "18": 2,
        "21": 2,
    }
    assert result["deltanet_statecache_post_update_recurrent_renorm_interval_overrides"] == {}


def test_qwen35_deltanet_statecache_readout_accepts_recurrent_mode_policy() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    input_ids = torch.ones((1, 3072), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        recurrent_mode_policy="890m_m3_outlier_pair_midband_v1",
    )
    assert result["deltanet_statecache_recurrent_mode_policy"] == "890m_m3_outlier_pair_midband_v1"
    assert result["deltanet_statecache_recurrent_mode_policy_band"] == "midband_outliers"
    assert result["deltanet_statecache_recurrent_mode_overrides"] == {}


def test_qwen35_deltanet_statecache_serving_accepts_recurrent_group_size_policy() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    input_ids = torch.ones((1, 6144), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    result = run_qwen35_deltanet_statecache_serving_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=16,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        recurrent_group_size_policy="890m_long_horizon_group_escape_v1",
    )
    assert result["deltanet_statecache_recurrent_group_size_policy"] == "890m_long_horizon_group_escape_v1"
    assert result["deltanet_statecache_recurrent_group_size_policy_band"] == "long_horizon"
    assert result["deltanet_statecache_recurrent_layer_group_size_overrides"] == {"18": 8, "20": 8}


def test_qwen35_deltanet_statecache_serving_policy_survives_empty_group_override_map() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    input_ids = torch.ones((1, 6144), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    result = run_qwen35_deltanet_statecache_serving_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=16,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        recurrent_group_size_policy="890m_long_horizon_group_escape_v1",
        recurrent_layer_group_size_overrides={},
    )
    assert result["deltanet_statecache_recurrent_group_size_policy_band"] == "long_horizon"
    assert result["deltanet_statecache_recurrent_layer_group_size_overrides"] == {"18": 8, "20": 8}


def test_qwen35_deltanet_statecache_readout_accepts_readout_recurrent_mode_policy() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    input_ids = torch.ones((1, 3072), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        readout_recurrent_mode_policy="890m_m3_outlier_pair_midband_v1",
    )
    assert result["deltanet_statecache_readout_recurrent_mode_policy"] == "890m_m3_outlier_pair_midband_v1"
    assert result["deltanet_statecache_readout_recurrent_mode_policy_band"] == "midband_outliers"
    assert result["deltanet_statecache_readout_recurrent_mode_overrides"] == {"4": "M3", "20": "M3"}
    assert result["deltanet_statecache_post_update_recurrent_mode_overrides"] == {}


def test_qwen35_deltanet_statecache_readout_rejects_policy_plus_manual_readout_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    with pytest.raises(ValueError, match="cannot be combined"):
        run_qwen35_deltanet_statecache_readout_harness(
            model,
            adapter,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            tokenizer=tokenizer,
            decode_steps=1,
            group_size=16,
            bits=8,
            state_stage="post_update_m0",
            readout_recurrent_policy="890m_context_banded_v1",
            readout_recurrent_renorm_interval_overrides={0: 2},
        )


def test_qwen35_deltanet_statecache_readout_rejects_mode_policy_plus_manual_mode_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    with pytest.raises(ValueError, match="cannot be combined"):
        run_qwen35_deltanet_statecache_readout_harness(
            model,
            adapter,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            tokenizer=tokenizer,
            decode_steps=1,
            group_size=16,
            bits=8,
            state_stage="post_update_m0",
            readout_recurrent_mode_policy="890m_m3_outlier_pair_midband_v1",
            readout_recurrent_mode_overrides={0: "M3"},
        )


def test_qwen35_deltanet_statecache_readout_rejects_recurrent_mode_policy_plus_manual_mode_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    with pytest.raises(ValueError, match="cannot be combined"):
        run_qwen35_deltanet_statecache_readout_harness(
            model,
            adapter,
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            tokenizer=tokenizer,
            decode_steps=1,
            group_size=16,
            bits=8,
            state_stage="post_update_m0",
            recurrent_mode_policy="890m_m3_outlier_pair_midband_v1",
            recurrent_mode_overrides={0: "M3"},
        )


def test_qwen35_deltanet_statecache_readout_uses_fresh_prefill_for_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("x", return_tensors="pt")
    import dotcache.integrations.qwen35 as qwen35_mod

    original_prefill = qwen35_mod._run_dense_prefill
    prefill_call_count = 0

    def _counting_prefill(*args, **kwargs):
        nonlocal prefill_call_count
        prefill_call_count += 1
        return original_prefill(*args, **kwargs)

    monkeypatch.setattr(qwen35_mod, "_run_dense_prefill", _counting_prefill)

    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=1,
        group_size=16,
        bits=8,
    )

    assert result["deltanet_statecache_ready"] is True
    assert prefill_call_count == 2


def test_qwen35_deltanet_statecache_loss_reports_teacher_forced_metrics() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_loss_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
    )
    assert result["deltanet_statecache_ready"] is True
    assert result["deltanet_statecache_teacher_forced_loss"] >= 0.0
    assert result["deltanet_statecache_teacher_forced_perplexity"] >= 1.0
    assert 0.0 <= result["deltanet_statecache_teacher_forced_target_match_rate"] <= 1.0
    assert result["deltanet_recurrent_state_bytes"] > result["deltanet_statecache_recurrent_state_bytes"]


def test_qwen35_deltanet_statecache_post_update_stage_reports_stage_metadata() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_loss_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        renorm_interval=2,
    )
    assert result["deltanet_statecache_ready"] is True
    assert result["deltanet_statecache_stage_name"] == "post_update_m0"
    assert result["deltanet_statecache_renorm_interval"] == 2
    assert result["deltanet_recurrent_state_bytes"] > result["deltanet_statecache_recurrent_state_bytes"]

def test_qwen35_deltanet_statecache_readout_accepts_layer_bit_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache", return_tensors="pt")

    baseline = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=1,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
    )
    selective = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=1,
        group_size=16,
        bits=8,
        layer_bits_overrides={0: 4},
        state_stage="post_update_m0",
    )

    assert selective["deltanet_statecache_layer_bits"]["0"] == 4
    assert selective["deltanet_statecache_layer_bits"]["1"] == 8
    assert selective["deltanet_statecache_per_layer_recurrent_bytes"]["0"] < baseline["deltanet_statecache_per_layer_recurrent_bytes"]["0"]


def test_qwen35_deltanet_statecache_loss_accepts_layer_bit_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_loss_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        layer_bits_overrides={0: 4},
        state_stage="post_update_m0",
    )
    assert result["deltanet_statecache_layer_bits"]["0"] == 4
    assert result["deltanet_statecache_layer_bits"]["1"] == 8


def test_qwen35_deltanet_statecache_loss_accepts_recurrent_mode_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_loss_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        recurrent_mode_overrides={0: "M3"},
    )
    assert result["deltanet_statecache_recurrent_mode_overrides"] == {"0": "M3"}
    assert result["deltanet_statecache_recurrent_state_bytes"] > 0


def test_qwen35_deltanet_statecache_loss_accepts_per_layer_renorm_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_loss_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        recurrent_renorm_interval_overrides={0: 2},
        conv_renorm_interval_overrides={0: 2},
    )
    assert result["deltanet_statecache_recurrent_renorm_interval_overrides"] == {"0": 2}
    assert result["deltanet_statecache_conv_renorm_interval_overrides"] == {"0": 2}


def test_qwen35_deltanet_statecache_loss_accepts_post_update_recurrent_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_loss_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        post_update_recurrent_renorm_interval_overrides={0: 2},
        post_update_recurrent_mode_overrides={0: "M3"},
    )
    assert result["deltanet_statecache_recurrent_renorm_interval_overrides"] == {}
    assert result["deltanet_statecache_post_update_recurrent_renorm_interval_overrides"] == {"0": 2}
    assert result["deltanet_statecache_recurrent_mode_overrides"] == {}
    assert result["deltanet_statecache_post_update_recurrent_mode_overrides"] == {"0": "M3"}


def test_qwen35_deltanet_statecache_localization_reports_first_failure_hints() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache localization", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_localization_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
    )
    assert result["deltanet_statecache_ready"] is True
    assert result["runtime_mode"] == "dense_deltanet_statecache_localization"
    assert len(result["deltanet_statecache_per_step_logit_max_abs_error"]) == 3
    assert "0" in result["deltanet_statecache_result"]["per_layer_output_max_abs_error"]
    assert "deltanet_statecache_first_recurrent_failure_layer" in result
    assert "deltanet_statecache_first_conv_failure_layer" in result
    assert "deltanet_statecache_first_combined_failure_layer" in result
    assert "deltanet_statecache_recurrent_result" in result
    assert "deltanet_statecache_conv_result" in result
    assert "deltanet_statecache_combined_result" in result


def test_qwen35_deltanet_statecache_localization_accepts_per_layer_renorm_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache localization", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_localization_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        recurrent_renorm_interval_overrides={0: 2},
        conv_renorm_interval_overrides={0: 2},
    )
    assert result["deltanet_statecache_recurrent_renorm_interval_overrides"] == {"0": 2}
    assert result["deltanet_statecache_conv_renorm_interval_overrides"] == {"0": 2}


def test_qwen35_deltanet_statecache_localization_accepts_post_update_recurrent_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache localization", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_localization_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        post_update_recurrent_renorm_interval_overrides={0: 2},
        post_update_recurrent_mode_overrides={0: "M3"},
    )
    assert result["deltanet_statecache_recurrent_renorm_interval_overrides"] == {}
    assert result["deltanet_statecache_post_update_recurrent_renorm_interval_overrides"] == {"0": 2}
    assert result["deltanet_statecache_post_update_recurrent_mode_overrides"] == {"0": "M3"}


def test_qwen35_deltanet_statecache_localization_accepts_recurrent_mode_policy() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    input_ids = torch.ones((1, 4099), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    result = run_qwen35_deltanet_statecache_localization_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prefix_length=4096,
        eval_steps=3,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        recurrent_mode_policy="890m_m3_outlier_pair_midband_v1",
    )
    assert result["deltanet_statecache_recurrent_mode_policy"] == "890m_m3_outlier_pair_midband_v1"
    assert result["deltanet_statecache_recurrent_mode_policy_band"] == "midband_outliers"
    assert result["deltanet_statecache_recurrent_mode_overrides"] == {"4": "M3", "20": "M3"}
    assert "deltanet_statecache_recurrent_state_max_abs_error_by_layer" in result
    assert "deltanet_statecache_recurrent_output_max_abs_error_by_layer" in result


def test_qwen35_deltanet_statecache_localization_accepts_readout_recurrent_mode_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache localization", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_localization_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        readout_recurrent_mode_overrides={0: "M3"},
    )
    assert result["deltanet_statecache_recurrent_mode_overrides"] == {}
    assert result["deltanet_statecache_readout_recurrent_mode_overrides"] == {"0": "M3"}
    assert result["deltanet_statecache_post_update_recurrent_mode_overrides"] == {}


def test_qwen35_deltanet_statecache_localization_accepts_readout_recurrent_renorm_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache localization", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_localization_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        readout_recurrent_renorm_interval_overrides={0: 2},
    )
    assert result["deltanet_statecache_recurrent_renorm_interval_overrides"] == {}
    assert result["deltanet_statecache_readout_recurrent_renorm_interval_overrides"] == {"0": 2}
    assert result["deltanet_statecache_post_update_recurrent_renorm_interval_overrides"] == {}


def test_qwen35_deltanet_statecache_localization_reports_quantization_telemetry() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache localization", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_localization_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        quantization_telemetry_layer_ids={0},
    )
    assert result["deltanet_statecache_quantization_telemetry_tracked_layers"] == [0]
    assert len(result["deltanet_statecache_quantization_telemetry_records"]) >= 1
    summary = result["deltanet_statecache_quantization_telemetry_summary"]
    assert "recurrent:0:prefill_post_update" in summary
    assert summary["recurrent:0:prefill_post_update"]["event_count"] >= 1
    assert "top_groups_by_mean_abs_error" in summary["recurrent:0:prefill_post_update"]


def test_qwen35_deltanet_statecache_readout_reports_per_layer_recurrent_modes() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_readout_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=1,
        group_size=16,
        bits=8,
        recurrent_mode_overrides={0: "M3"},
    )
    assert result["deltanet_statecache_recurrent_mode_overrides"] == {"0": "M3"}
    assert result["deltanet_statecache_per_layer_recurrent_mode"]["0"] == "M3"


def test_qwen35_deltanet_statecache_serving_reports_serving_runtime_mode() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache serving", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_serving_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
        group_size=16,
        bits=8,
    )
    assert result["runtime_mode"] == "statecache_serving_only"
    assert result["deltanet_statecache_ready"] is True
    assert result["deltanet_recurrent_state_bytes"] > result["deltanet_statecache_recurrent_state_bytes"]
    assert result["deltanet_dense_fixed_resident_bytes"] > result["deltanet_statecache_fixed_resident_bytes"]
    assert len(result["deltanet_statecache_generated_ids"]) == 2
    assert len(result["deltanet_statecache_per_step_decode_ms"]) == 2


def test_qwen35_deltanet_statecache_serving_accepts_per_layer_renorm_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache serving", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_serving_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        recurrent_renorm_interval_overrides={0: 2},
    )
    assert result["deltanet_statecache_recurrent_renorm_interval_overrides"] == {"0": 2}


def test_qwen35_deltanet_statecache_serving_accepts_readout_recurrent_overrides() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello state cache serving", return_tensors="pt")
    result = run_qwen35_deltanet_statecache_serving_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        readout_recurrent_renorm_interval_overrides={0: 2},
        readout_recurrent_mode_overrides={0: "M3"},
    )
    assert result["deltanet_statecache_readout_recurrent_renorm_interval_overrides"] == {"0": 2}
    assert result["deltanet_statecache_readout_recurrent_mode_overrides"] == {"0": "M3"}
    assert result["deltanet_statecache_post_update_recurrent_renorm_interval_overrides"] == {}
    assert result["deltanet_statecache_post_update_recurrent_mode_overrides"] == {}


def test_qwen35_deltanet_statecache_serving_accepts_readout_recurrent_policy() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    input_ids = torch.ones((1, 4096), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    result = run_qwen35_deltanet_statecache_serving_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        readout_recurrent_policy="890m_context_banded_v1",
    )
    assert result["deltanet_statecache_readout_recurrent_policy"] == "890m_context_banded_v1"
    assert result["deltanet_statecache_readout_recurrent_policy_band"] == "full"
    assert result["deltanet_statecache_readout_recurrent_renorm_interval_overrides"] == {
        "5": 2,
        "8": 2,
        "12": 2,
        "18": 2,
        "21": 2,
    }


def test_qwen35_deltanet_statecache_serving_accepts_recurrent_mode_policy() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    input_ids = torch.ones((1, 4096), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    result = run_qwen35_deltanet_statecache_serving_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        recurrent_mode_policy="890m_m3_outlier_pair_midband_v1",
    )
    assert result["deltanet_statecache_recurrent_mode_policy"] == "890m_m3_outlier_pair_midband_v1"
    assert result["deltanet_statecache_recurrent_mode_policy_band"] == "midband_outliers"
    assert result["deltanet_statecache_recurrent_mode_overrides"] == {}


def test_qwen35_deltanet_statecache_serving_accepts_readout_recurrent_mode_policy() -> None:
    model = _tiny_deltanet_qwen35_model()
    adapter = Qwen35DeltaNetStateModelAdapter(model=model)
    input_ids = torch.ones((1, 4096), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    result = run_qwen35_deltanet_statecache_serving_harness(
        model,
        adapter,
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        readout_recurrent_mode_policy="890m_m3_outlier_pair_midband_v1",
    )
    assert result["deltanet_statecache_readout_recurrent_mode_policy"] == "890m_m3_outlier_pair_midband_v1"
    assert result["deltanet_statecache_readout_recurrent_mode_policy_band"] == "midband_outliers"
    assert result["deltanet_statecache_readout_recurrent_mode_overrides"] == {"4": "M3", "20": "M3"}


def test_qwen35_attention_subset_adapter_wraps_only_full_attention_layers() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35AttentionSubsetModelAdapter(model=model)
    assert adapter.attention_subset_layer_ids() == [3]
    text_model = model.model.language_model
    assert hasattr(text_model.layers[3], "self_attn")
    assert hasattr(text_model.layers[0], "linear_attn")


def test_qwen35_attention_subset_replay_captures_only_attention_layers() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35AttentionSubsetModelAdapter(model=model)
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello subset", return_tensors="pt")
    result = run_qwen35_attention_subset_replay_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
    )
    assert result["attention_subset_layer_ids"] == [3]
    assert result["attention_subset_capture_layer_count"] == 1
    assert result["attention_subset_capture_record_count"] == 2
    assert result["attention_subset_capture_counts_by_layer"] == {"3": 2}
    assert "3" in result["attention_subset_capture_shapes_by_layer"]


def test_qwen35_attention_subset_harness_tokenizes_and_runs() -> None:
    model = _tiny_qwen35_model()
    tokenizer = _TinyTokenizer()
    harness = Qwen35AttentionSubsetHarness(
        model=model,
        tokenizer=tokenizer,
        adapter=Qwen35AttentionSubsetModelAdapter(model=model),
    )
    input_ids, attention_mask = harness.tokenize_prompt("hello")
    result = harness.run_attention_subset_replay(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
    )
    assert result["attention_subset_capture_layer_count"] == 1


def test_qwen35_attention_subset_dotcache_harness_runs_on_tiny_hybrid_model() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=DotCacheConfig(head_dim=16, group_size=16, bits_k=4, bits_v=4, tokens_per_page=2),
        backend="cpu_ref",
    )
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello subset dotcache", return_tensors="pt")
    result = run_qwen35_attention_subset_dotcache_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
    )
    assert result["attention_subset_layer_ids"] == [3]
    assert result["dotcache_attention_subset_ready"] is True
    assert result["dotcache_ready"] is False
    assert result["native_hybrid_fixed_resident_layer_ids"] == [0, 1, 2]
    assert result["native_hybrid_token_growing_layer_ids"] == [3]
    assert result["native_hybrid_fixed_resident_preserved"] is True
    assert result["native_hybrid_fixed_resident_growth_bytes"] == 0
    assert result["native_hybrid_token_growing_growth_bytes"] >= 0
    assert result["attention_subset_capture_record_count"] == 2
    assert np.isfinite(result["replay_context_max_abs_error"])
    assert np.isfinite(result["teacher_forced_logit_max_abs_error"])
    assert adapter.native_hybrid_runtime_state is not None
    assert adapter.hybrid_dotcache_runtime_state is not None
    assert adapter.native_hybrid_runtime_state.fixed_resident_layer_ids == [0, 1, 2]
    assert adapter.native_hybrid_runtime_state.token_growing_layer_ids == [3]
    assert adapter.hybrid_dotcache_runtime_state.model_past_key_values is adapter.native_hybrid_runtime_state.past_key_values
    assert result["hybrid_dotcache_runtime_ready"] is True
    assert result["hybrid_runtime_state_kind"] == "qwen35_attention_subset"
    assert result["hybrid_runtime_fixed_resident_layer_ids"] == [0, 1, 2]
    assert result["hybrid_runtime_token_growing_layer_ids"] == [3]


def test_qwen35_attention_subset_dotcache_harness_class_tokenizes_and_runs() -> None:
    model = _tiny_qwen35_model()
    tokenizer = _TinyTokenizer()
    harness = Qwen35AttentionSubsetDotCacheHarness(
        model=model,
        tokenizer=tokenizer,
        adapter=Qwen35AttentionSubsetDotCacheModelAdapter(
            model=model,
            dotcache_config=DotCacheConfig(head_dim=16, group_size=16, bits_k=4, bits_v=4, tokens_per_page=2),
            backend="cpu_ref",
        ),
    )
    input_ids, attention_mask = harness.tokenize_prompt("hello")
    result = harness.run_attention_subset_dotcache(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
    )
    assert result["attention_subset_capture_layer_count"] == 1
    assert result["dotcache_attention_subset_ready"] is True
    assert result["native_hybrid_fixed_resident_preserved"] is True
    assert harness.adapter.native_hybrid_runtime_state is not None
    assert harness.adapter.hybrid_dotcache_runtime_state is not None


def test_qwen35_attention_subset_dotcache_serving_harness_runs_on_tiny_hybrid_model() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=DotCacheConfig(head_dim=16, group_size=16, bits_k=4, bits_v=4, tokens_per_page=2),
        backend="cpu_ref",
    )
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello subset dotcache serving", return_tensors="pt")
    result = run_qwen35_attention_subset_dotcache_serving_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
        profile_backend=True,
    )
    assert result["runtime_mode"] == "dotcache_attention_subset_serving"
    assert result["dotcache_attention_subset_ready"] is True
    assert result["dotcache_ready"] is False
    assert result["hybrid_dotcache_runtime_ready"] is True
    assert result["hybrid_runtime_state_kind"] == "qwen35_attention_subset"
    assert result["hybrid_runtime_fixed_resident_layer_ids"] == [0, 1, 2]
    assert result["hybrid_runtime_token_growing_layer_ids"] == [3]
    assert result["attention_subset_layer_ids"] == [3]
    assert result["attention_subset_capture_layer_count"] == 1
    assert result["num_attention_heads"] == 4
    assert result["num_key_value_heads"] == 1
    assert result["query_heads_per_kv_head"] == 4
    assert result["head_dim"] == adapter.dotcache_config.head_dim
    assert result["group_size"] == adapter.dotcache_config.group_size
    assert result["num_groups"] == adapter.dotcache_config.num_groups
    assert result["padded_head_dim"] == adapter.dotcache_config.padded_head_dim
    assert result["tokens_per_page"] == adapter.dotcache_config.tokens_per_page
    assert "decode_path_counts" in result
    assert "decode_path_counts_by_layer" in result
    assert "dotcache_decode_runtime_ms_total_by_layer" in result
    assert "dotcache_append_runtime_ms_total_by_layer" in result
    assert "dotcache_qkv_projection_ms_total_by_layer" in result
    assert "dotcache_output_projection_ms_total_by_layer" in result
    assert "execution_shortlist_invocations" in result
    assert "execution_shortlist_applied" in result
    assert "execution_exact_refine_invocations" in result
    assert "execution_exact_refine_candidate_pages" in result
    assert "serving_shortlist_heuristic_applied" in result
    assert "execution_recent_window" in result
    assert "execution_sink_window" in result
    assert "execution_recent_window_overrides" in result
    assert "execution_recent_window_context_overrides" in result
    assert "execution_relevance_top_k" in result
    assert "execution_relevance_top_k_context_overrides" in result
    assert "execution_full_context_layers" in result
    assert "execution_disable_grouped_batching_layers" in result
    assert "execution_recent_old_bonus_window" in result
    assert "execution_recent_old_bonus_strength" in result
    assert "execution_recent_old_bonus_layers" in result
    assert "execution_secondary_relevance_mode" in result
    assert "execution_secondary_relevance_top_k" in result
    assert "execution_secondary_relevance_min_overlap" in result
    assert "execution_secondary_relevance_layers" in result
    assert "execution_recent_neighbor_rescue_top_k" in result
    assert "execution_recent_neighbor_rescue_anchor_window" in result
    assert "execution_recent_neighbor_rescue_min_anchor_pages" in result
    assert "execution_recent_neighbor_rescue_layers" in result
    assert "execution_exact_promote_top_k" in result
    assert "execution_exact_promote_min_margin_threshold" in result
    assert "execution_exact_promote_max_context" in result
    assert "execution_exact_promote_margin_threshold" in result
    assert "execution_exact_promote_layers" in result
    assert "execution_exact_promote_union_rescue_top_k" in result
    assert "execution_grouped_decode_compact" in result
    assert "execution_grouped_mix_compact" in result
    assert "execution_grouped_mix_disable_packed_cuda" in result
    assert "execution_freeze_chunk_budget_during_decode" in result
    assert "execution_builtin_selector_cache" in result
    assert "execution_builtin_selector_score_all_pages" in result
    assert "execution_builtin_selector_candidate_only" in result
    assert "execution_builtin_selector_score_all_pages_min_candidate_fraction" in result
    assert "execution_builtin_selector_score_all_pages_calls" in result
    assert "execution_builtin_selector_candidate_only_calls" in result
    assert "execution_builtin_selector_candidate_fraction_max" in result
    assert "execution_builtin_selector_cache_hits" in result
    assert "execution_builtin_selector_cache_builds" in result
    assert "execution_builtin_selector_cache_build_bytes" in result
    assert "execution_builtin_selector_cache_build_bytes_max" in result
    assert "decode_backend_trace" in result
    trace = result["decode_backend_trace"]
    assert "grouped_decode_calls" in trace
    assert "grouped_decode_output_only_calls" in trace
    assert "grouped_score_chunk_count" in trace
    assert "grouped_mix_chunk_count" in trace
    assert "grouped_logits_elements_total" in trace
    assert "grouped_weights_elements_total" in trace
    assert "grouped_output_elements_total" in trace
    assert "grouped_score_packed_cuda_calls" in trace
    assert "grouped_mix_packed_cuda_calls" in trace
    assert "per_kv_decode_calls" in trace
    assert "per_kv_score_chunk_count" in trace
    assert "per_kv_mix_chunk_count" in trace
    assert "per_kv_logits_elements_total" in trace
    assert "per_kv_weights_elements_total" in trace
    assert "per_kv_output_elements_total" in trace
    assert "per_kv_score_generic_calls" in trace
    assert "per_kv_mix_generic_calls" in trace
    assert len(result["dotcache_generated_ids"]) == 2
    assert np.isfinite(result["dotcache_decode_ms_per_step"])


def test_qwen35_attention_subset_dotcache_serving_quality_harness_reports_replay_metrics() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=DotCacheConfig(head_dim=16, group_size=16, bits_k=4, bits_v=4, tokens_per_page=2),
        backend="cpu_ref",
    )
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello subset dotcache serving quality", return_tensors="pt")
    result = run_qwen35_attention_subset_dotcache_serving_quality_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
    )
    assert result["runtime_mode"] == "dotcache_attention_subset_serving_quality"
    assert result["dotcache_attention_subset_ready"] is True
    assert result["dotcache_ready"] is False
    assert result["hybrid_dotcache_runtime_ready"] is True
    assert np.isfinite(result["replay_context_max_abs_error"])
    assert np.isfinite(result["replay_output_max_abs_error"])
    assert np.isfinite(result["teacher_forced_logit_max_abs_error"])
    assert np.isfinite(result["teacher_forced_logit_mean_abs_error"])
    assert np.isfinite(result["teacher_forced_logit_rmse"])
    assert 0.0 <= result["teacher_forced_token_agreement_rate"] <= 1.0
    assert len(result["teacher_forced_per_step_logit_max_abs_error"]) == 2
    assert "serving_shortlist_heuristic_applied" in result
    assert "execution_recent_window_overrides" in result
    assert "execution_recent_window_context_overrides" in result
    assert "execution_relevance_top_k_context_overrides" in result
    assert "execution_full_context_layers" in result
    assert "execution_disable_grouped_batching_layers" in result
    assert "execution_recent_old_bonus_window" in result
    assert "execution_recent_old_bonus_strength" in result
    assert "execution_recent_old_bonus_layers" in result
    assert "execution_secondary_relevance_mode" in result
    assert "execution_secondary_relevance_top_k" in result
    assert "execution_secondary_relevance_min_overlap" in result
    assert "execution_secondary_relevance_layers" in result
    assert "execution_recent_neighbor_rescue_top_k" in result
    assert "execution_recent_neighbor_rescue_anchor_window" in result
    assert "execution_recent_neighbor_rescue_min_anchor_pages" in result
    assert "execution_recent_neighbor_rescue_layers" in result
    assert "execution_exact_promote_top_k" in result
    assert "execution_exact_promote_min_margin_threshold" in result
    assert "execution_exact_promote_max_context" in result
    assert "execution_exact_promote_margin_threshold" in result
    assert "execution_exact_promote_layers" in result
    assert "execution_exact_promote_union_rescue_top_k" in result
    assert "execution_grouped_decode_compact" in result
    assert "execution_grouped_mix_compact" in result
    assert "execution_grouped_mix_disable_packed_cuda" in result
    assert "execution_freeze_chunk_budget_during_decode" in result
    assert "execution_builtin_selector_cache" in result
    assert "execution_builtin_selector_score_all_pages" in result
    assert "execution_builtin_selector_cache_hits" in result
    assert "execution_builtin_selector_cache_builds" in result
    assert "execution_builtin_selector_cache_build_bytes" in result
    assert "execution_builtin_selector_cache_build_bytes_max" in result
    assert "dotcache_step_runtime_breakdown" in result
    assert len(result["dotcache_step_runtime_breakdown"]) == 2
    assert "dotcache_backend_decode_ms_total_from_trace" in result
    assert "dotcache_decode_non_backend_ms_total" in result
    assert "dotcache_model_step_non_adapter_ms_total" in result
    assert "dotcache_python_allocation_tracing" in result
    assert "dotcache_python_tracemalloc_peak_bytes_max" in result
    assert "dotcache_python_tracemalloc_current_bytes_delta_total" in result
    assert "dotcache_python_allocated_blocks_delta_total" in result
    assert "dotcache_python_gc_count_delta_total" in result
    assert "execution_decode_prepare_pages_with_tail_ms_total" in result
    assert "execution_decode_m2_prefilter_ms_total" in result
    assert "execution_decode_shortlist_selection_ms_total" in result
    assert "execution_decode_shortlist_materialization_ms_total" in result
    assert "execution_decode_backend_call_non_backend_ms_total" in result
    assert "execution_decode_shortlist_candidate_approx_scoring_ms_total" in result
    assert "execution_decode_shortlist_candidate_ranking_ms_total" in result
    assert "execution_decode_shortlist_candidate_builtin_candidate_index_build_ms_total" in result
    assert "execution_decode_shortlist_candidate_builtin_sidecar_stack_ms_total" in result
    assert "execution_decode_shortlist_candidate_builtin_score_compute_ms_total" in result
    assert "execution_decode_shortlist_candidate_builtin_ranking_ms_total" in result
    assert "execution_chunk_budget_dirty_marks" in result
    assert "execution_chunk_budget_dirty_reason_counts" in result
    assert "execution_chunk_budget_override_calls" in result
    first_step = result["dotcache_step_runtime_breakdown"][0]
    assert "decode_prepare_pages_with_tail_ms_total" in first_step
    assert "decode_shortlist_materialization_ms_total" in first_step
    assert "decode_backend_call_non_backend_ms_total" in first_step
    assert "decode_non_backend_unattributed_ms_total" in first_step
    assert "decode_shortlist_candidate_approx_scoring_ms_total" in first_step
    assert "decode_shortlist_candidate_ranking_ms_total" in first_step
    assert "decode_shortlist_candidate_builtin_candidate_index_build_ms_total" in first_step
    assert "decode_shortlist_candidate_builtin_sidecar_stack_ms_total" in first_step
    assert "decode_shortlist_candidate_builtin_score_compute_ms_total" in first_step
    assert "decode_shortlist_candidate_builtin_ranking_ms_total" in first_step
    assert "decode_chunk_budget_dirty_reason_counts" in first_step
    assert "decode_chunk_budget_override_calls" in first_step
    assert "decode_builtin_selector_cache_hits" in first_step
    assert "decode_builtin_selector_cache_builds" in first_step
    assert "decode_builtin_selector_cache_build_bytes" in first_step
    assert "decode_builtin_selector_cache_build_bytes_max" in first_step
    assert "python_tracemalloc_current_bytes_delta" in first_step
    assert "python_tracemalloc_peak_bytes" in first_step
    assert "python_allocated_blocks_delta" in first_step
    assert "python_gc_count_delta" in first_step


def test_decode_input_id_sequence_flattens_decode_steps() -> None:
    decode_inputs = [
        torch.tensor([[11]], dtype=torch.long),
        torch.tensor([[22]], dtype=torch.long),
        torch.tensor([[33]], dtype=torch.long),
    ]
    assert _decode_input_id_sequence(decode_inputs) == [11, 22, 33]


def test_qwen35_attention_subset_dotcache_serving_recall_analysis_reports_shortlist_metrics() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=DotCacheConfig(
            head_dim=16,
            group_size=16,
            bits_k=4,
            bits_v=4,
            tokens_per_page=2,
            execution_recent_window=2,
            execution_sink_window=2,
            execution_relevance_top_k=1,
        ),
        backend="cpu_ref",
    )
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello subset dotcache serving recall analysis", return_tensors="pt")
    result = run_qwen35_attention_subset_dotcache_serving_recall_analysis_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
    )
    assert result["runtime_mode"] == "dotcache_attention_subset_serving_recall_analysis"
    assert result["dotcache_attention_subset_ready"] is True
    assert result["shortlist_recall_ready"] is True
    assert result["shortlist_recall_record_count"] > 0
    assert result["shortlist_recall_exact_top_budget_total"] >= result["shortlist_recall_exact_top_hits_total"]
    assert 0.0 <= result["shortlist_recall_exact_top_recall_mean"] <= 1.0
    assert 0.0 <= result["shortlist_recall_exact_top_recall_weighted"] <= 1.0
    assert isinstance(result["shortlist_recall_mean_by_layer"], dict)
    assert "serving_shortlist_heuristic_applied" in result
    assert "execution_recent_window_overrides" in result
    assert "execution_recent_window_context_overrides" in result
    assert "execution_relevance_top_k_context_overrides" in result
    assert "execution_full_context_layers" in result
    assert "execution_disable_grouped_batching_layers" in result
    assert "execution_recent_old_bonus_window" in result
    assert "execution_recent_old_bonus_strength" in result
    assert "execution_recent_old_bonus_layers" in result
    assert "execution_secondary_relevance_mode" in result
    assert "execution_secondary_relevance_top_k" in result
    assert "execution_secondary_relevance_min_overlap" in result
    assert "execution_secondary_relevance_layers" in result
    assert "execution_recent_neighbor_rescue_top_k" in result
    assert "execution_recent_neighbor_rescue_anchor_window" in result
    assert "execution_recent_neighbor_rescue_min_anchor_pages" in result
    assert "execution_recent_neighbor_rescue_layers" in result
    assert "execution_exact_promote_top_k" in result
    assert "execution_exact_promote_min_margin_threshold" in result
    assert "execution_exact_promote_max_context" in result
    assert "execution_exact_promote_margin_threshold" in result
    assert "execution_exact_promote_layers" in result
    assert "execution_exact_promote_union_rescue_top_k" in result
    assert "execution_grouped_decode_compact" in result
    assert "execution_grouped_mix_compact" in result
    assert "execution_grouped_mix_disable_packed_cuda" in result
    assert "execution_freeze_chunk_budget_during_decode" in result
    assert "execution_builtin_selector_cache" in result
    assert "execution_builtin_selector_score_all_pages" in result
    assert "execution_builtin_selector_cache_hits" in result
    assert "execution_builtin_selector_cache_builds" in result
    assert "execution_builtin_selector_cache_build_bytes" in result
    assert "execution_builtin_selector_cache_build_bytes_max" in result


def test_qwen35_attention_subset_dotcache_serving_scorer_diagnostic_reports_rank_metrics() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=DotCacheConfig(
            head_dim=16,
            group_size=16,
            bits_k=4,
            bits_v=4,
            tokens_per_page=2,
            execution_recent_window=2,
            execution_sink_window=2,
            execution_relevance_top_k=1,
        ),
        backend="cpu_ref",
    )
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello subset dotcache scorer diagnostic", return_tensors="pt")
    result = run_qwen35_attention_subset_dotcache_serving_scorer_diagnostic_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
    )
    assert result["runtime_mode"] == "dotcache_attention_subset_serving_scorer_diagnostic"
    assert result["scorer_diagnostic_ready"] is True
    assert result["scorer_diagnostic_record_count"] > 0
    assert isinstance(result["scorer_rank_correlation_mean_by_layer"], dict)
    assert isinstance(result["scorer_value_correlation_mean_by_layer"], dict)
    assert isinstance(result["scorer_approx_exact_top_recall_mean_by_layer"], dict)
    assert isinstance(result["scorer_secondary_trigger_rate_by_layer"], dict)
    assert isinstance(result["scorer_recent_neighbor_rescue_trigger_rate_by_layer"], dict)
    assert isinstance(result["scorer_missed_exact_age_buckets_by_layer"], dict)
    assert "execution_recent_old_bonus_window" in result
    assert "execution_recent_old_bonus_strength" in result
    assert "execution_recent_old_bonus_layers" in result
    assert "execution_secondary_relevance_mode" in result
    assert "execution_secondary_relevance_top_k" in result
    assert "execution_secondary_relevance_min_overlap" in result
    assert "execution_secondary_relevance_layers" in result
    assert "execution_recent_neighbor_rescue_top_k" in result
    assert "execution_recent_neighbor_rescue_anchor_window" in result
    assert "execution_recent_neighbor_rescue_min_anchor_pages" in result
    assert "execution_recent_neighbor_rescue_layers" in result
    assert "execution_exact_promote_top_k" in result
    assert "execution_exact_promote_min_margin_threshold" in result
    assert "execution_exact_promote_max_context" in result
    assert "execution_exact_promote_margin_threshold" in result
    assert "execution_exact_promote_layers" in result
    assert "execution_exact_promote_union_rescue_top_k" in result
    assert "execution_grouped_decode_compact" in result
    assert "execution_grouped_mix_compact" in result
    assert "execution_grouped_mix_disable_packed_cuda" in result
    assert "execution_freeze_chunk_budget_during_decode" in result
    assert "execution_builtin_selector_cache" in result
    assert "execution_builtin_selector_score_all_pages" in result
    assert "dotcache_step_runtime_breakdown" in result
    assert len(result["dotcache_step_runtime_breakdown"]) == 2
    assert "dotcache_backend_decode_ms_total_from_trace" in result
    assert "dotcache_decode_non_backend_ms_total" in result
    assert "dotcache_model_step_non_adapter_ms_total" in result
    assert "dotcache_python_allocation_tracing" in result
    assert "dotcache_python_tracemalloc_peak_bytes_max" in result
    assert "dotcache_python_tracemalloc_current_bytes_delta_total" in result
    assert "dotcache_python_allocated_blocks_delta_total" in result
    assert "dotcache_python_gc_count_delta_total" in result
    assert "execution_decode_prepare_pages_with_tail_ms_total" in result
    assert "execution_decode_m2_prefilter_ms_total" in result
    assert "execution_decode_shortlist_selection_ms_total" in result
    assert "execution_decode_shortlist_materialization_ms_total" in result
    assert "execution_decode_backend_call_non_backend_ms_total" in result
    assert "execution_decode_shortlist_candidate_approx_scoring_ms_total" in result
    assert "execution_decode_shortlist_candidate_ranking_ms_total" in result
    assert "execution_decode_shortlist_candidate_builtin_candidate_index_build_ms_total" in result
    assert "execution_decode_shortlist_candidate_builtin_sidecar_stack_ms_total" in result
    assert "execution_decode_shortlist_candidate_builtin_score_compute_ms_total" in result
    assert "execution_decode_shortlist_candidate_builtin_ranking_ms_total" in result
    assert "execution_chunk_budget_dirty_marks" in result
    assert "execution_chunk_budget_dirty_reason_counts" in result
    assert "execution_chunk_budget_override_calls" in result
    first_step = result["dotcache_step_runtime_breakdown"][0]
    assert "decode_prepare_pages_with_tail_ms_total" in first_step
    assert "decode_shortlist_materialization_ms_total" in first_step
    assert "decode_backend_call_non_backend_ms_total" in first_step
    assert "decode_non_backend_unattributed_ms_total" in first_step
    assert "decode_shortlist_candidate_approx_scoring_ms_total" in first_step
    assert "decode_shortlist_candidate_ranking_ms_total" in first_step
    assert "decode_shortlist_candidate_builtin_candidate_index_build_ms_total" in first_step
    assert "decode_shortlist_candidate_builtin_sidecar_stack_ms_total" in first_step
    assert "decode_shortlist_candidate_builtin_score_compute_ms_total" in first_step
    assert "decode_shortlist_candidate_builtin_ranking_ms_total" in first_step
    assert "decode_chunk_budget_dirty_reason_counts" in first_step
    assert "decode_chunk_budget_override_calls" in first_step
    assert "decode_builtin_selector_cache_hits" in first_step
    assert "decode_builtin_selector_cache_builds" in first_step
    assert "decode_builtin_selector_cache_build_bytes" in first_step
    assert "decode_builtin_selector_cache_build_bytes_max" in first_step
    assert "python_tracemalloc_current_bytes_delta" in first_step
    assert "python_tracemalloc_peak_bytes" in first_step
    assert "python_allocated_blocks_delta" in first_step
    assert "python_gc_count_delta" in first_step
    first_record = result["scorer_layer_records"][0]
    first_group = first_record["groups"][0]
    assert "context_length_page_max" in first_group
    assert "context_length_effective" in first_group
    assert "context_length_override_applied" in first_group
    assert "exact_promote_candidate_expansion_enabled" in first_group
    assert "exact_promote_candidate_expansion_disable_reason" in first_group
    assert "exact_promote_enabled" in first_group
    assert "exact_promote_disable_reason" in first_group
    assert "execution_shortlist_trace_records" in result
    assert result["execution_shortlist_trace_records"]
    first_trace = result["execution_shortlist_trace_records"][0]
    assert "kv_head_id" in first_trace
    assert "stage1_old_page_ranges" in first_trace
    assert "final_old_page_ranges" in first_trace
    assert "promote_candidate_page_ranges" in first_trace
    assert "promote_selected_page_ranges" in first_trace
    assert "promote_candidate_indices" in first_trace
    assert "promote_selected_indices" in first_trace


def test_qwen35_mps_serving_shortlist_heuristic_is_context_aware_and_non_overriding() -> None:
    base = DotCacheConfig(head_dim=256, group_size=32, bits_k=4, bits_v=4, tokens_per_page=16)
    unchanged_short, applied_short = _qwen35_mps_serving_shortlist_heuristic(base, backend="torch_mps", prompt_length=2048)
    assert applied_short is False
    assert unchanged_short == base

    updated, applied = _qwen35_mps_serving_shortlist_heuristic(base, backend="torch_mps", prompt_length=4096)
    assert applied is True
    assert updated.execution_recent_window == 1024
    assert updated.execution_sink_window == 256
    assert updated.execution_relevance_top_k == 4
    assert updated.execution_relevance_mode == "envelope"
    assert updated.execution_relevance_top_k_context_overrides == ("layer:23:min_ctx:8192=8",)

    explicit = DotCacheConfig(
        head_dim=256,
        group_size=32,
        bits_k=4,
        bits_v=4,
        tokens_per_page=16,
        execution_recent_window=512,
        execution_relevance_top_k=2,
    )
    unchanged_explicit, applied_explicit = _qwen35_mps_serving_shortlist_heuristic(
        explicit,
        backend="torch_mps",
        prompt_length=8192,
    )
    assert applied_explicit is False
    assert unchanged_explicit == explicit

    unchanged_cpu, applied_cpu = _qwen35_mps_serving_shortlist_heuristic(base, backend="cpu_ref", prompt_length=8192)
    assert applied_cpu is False
    assert unchanged_cpu == base


def test_execution_exact_promote_max_context_disables_promotion_before_candidate_expansion() -> None:
    cache = ModelPagedKVCache.__new__(ModelPagedKVCache)
    cache.config = DotCacheConfig(
        head_dim=16,
        group_size=16,
        bits_k=4,
        bits_v=4,
        tokens_per_page=2,
        execution_exact_promote_top_k=2,
        execution_exact_promote_layers=(23,),
        execution_exact_promote_max_context=16384,
    )

    assert cache._execution_exact_promote_enabled(layer_id=23, context_length=16384) is True
    assert cache._execution_exact_promote_enabled(layer_id=23, context_length=32768) is False
    assert cache._execution_exact_promote_enabled(layer_id=11, context_length=8192) is False


def test_execution_exact_promote_candidate_expansion_ignores_margin_threshold() -> None:
    cache = ModelPagedKVCache.__new__(ModelPagedKVCache)
    cache.config = DotCacheConfig(
        head_dim=16,
        group_size=16,
        bits_k=4,
        bits_v=4,
        tokens_per_page=2,
        execution_exact_promote_top_k=2,
        execution_exact_promote_layers=(23,),
        execution_exact_promote_min_margin_threshold=0.5,
    )

    assert cache._execution_exact_promote_enabled(layer_id=23, context_length=8192) is True
    enabled, reason = cache._execution_exact_promote_status(
        layer_id=23,
        context_length=8192,
        boundary_margin_normalized=0.25,
    )
    assert enabled is False
    assert reason == "below_min_margin_threshold"


def test_qwen35_attention_subset_dotcache_loss_harness_reports_teacher_forced_metrics() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=DotCacheConfig(head_dim=16, group_size=16, bits_k=4, bits_v=4, tokens_per_page=2),
        backend="cpu_ref",
    )
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello subset dotcache loss path", return_tensors="pt")
    result = run_qwen35_attention_subset_dotcache_loss_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
    )
    assert result["runtime_mode"] == "dotcache_attention_subset_loss"
    assert result["dotcache_attention_subset_ready"] is True
    assert result["dotcache_ready"] is False
    assert result["hybrid_dotcache_runtime_ready"] is True
    assert np.isfinite(result["dense_teacher_forced_loss"])
    assert np.isfinite(result["dotcache_teacher_forced_loss"])
    assert np.isfinite(result["teacher_forced_logit_max_abs_error"])
    assert 0.0 <= result["teacher_forced_token_agreement_rate"] <= 1.0
    assert 0.0 <= result["teacher_forced_target_match_rate"] <= 1.0


def test_qwen35_attention_subset_statecache_dotcache_harness_runs_on_tiny_hybrid_model() -> None:
    model = _tiny_qwen35_model()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=DotCacheConfig(head_dim=16, group_size=16, bits_k=4, bits_v=4, tokens_per_page=2),
        backend="cpu_ref",
    )
    tokenizer = _TinyTokenizer()
    encoded = tokenizer("hello subset dotcache statecache", return_tensors="pt")
    result = run_qwen35_attention_subset_statecache_dotcache_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        decode_steps=2,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
        recurrent_mode_overrides={0: "M3"},
    )
    assert result["dotcache_attention_subset_ready"] is True
    assert result["deltanet_statecache_ready"] is True
    assert result["hybrid_dotcache_statecache_ready"] is True
    assert result["hybrid_runtime_state_kind"] == "qwen35_attention_subset_statecache"
    assert result["deltanet_statecache_recurrent_mode_overrides"] == {"0": "M3"}
    assert result["deltanet_statecache_per_layer_recurrent_mode"]["0"] == "M3"
    assert result["native_hybrid_fixed_resident_preserved"] is True
    assert np.isfinite(result["teacher_forced_logit_max_abs_error"])
    assert np.isfinite(result["replay_context_max_abs_error"])


def test_qwen35_attention_subset_statecache_dotcache_harness_class_runs() -> None:
    model = _tiny_qwen35_model()
    tokenizer = _TinyTokenizer()
    harness = Qwen35AttentionSubsetDotCacheHarness(
        model=model,
        tokenizer=tokenizer,
        adapter=Qwen35AttentionSubsetDotCacheModelAdapter(
            model=model,
            dotcache_config=DotCacheConfig(head_dim=16, group_size=16, bits_k=4, bits_v=4, tokens_per_page=2),
            backend="cpu_ref",
        ),
    )
    input_ids, attention_mask = harness.tokenize_prompt("hello")
    result = harness.run_attention_subset_dotcache_statecache(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
        group_size=16,
        bits=8,
        state_stage="post_update_m0",
    )
    assert result["hybrid_dotcache_statecache_ready"] is True
    assert result["dotcache_attention_subset_ready"] is True


def test_qwen35_attention_subset_dotcache_harness_accepts_policy_aware_config() -> None:
    model = _tiny_qwen35_model()
    tokenizer = _TinyTokenizer()
    harness = Qwen35AttentionSubsetDotCacheHarness(
        model=model,
        tokenizer=tokenizer,
        adapter=Qwen35AttentionSubsetDotCacheModelAdapter(
            model=model,
            dotcache_config=DotCacheConfig(
                head_dim=16,
                group_size=16,
                bits_k=4,
                bits_v=4,
                tokens_per_page=2,
                key_policy_tier="balanced",
                value_policy_tier="balanced",
                key_layer_sensitivity=("layer:3=strict",),
                value_layer_sensitivity=("layer:3=strict",),
            ),
            backend="cpu_ref",
        ),
    )
    input_ids, attention_mask = harness.tokenize_prompt("hello subset policy path")
    result = harness.run_attention_subset_dotcache(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
    )
    assert result["dotcache_attention_subset_ready"] is True
    assert "policy_tier_counts" in result
    assert "mode_signature_counts" in result


def test_qwen35_attention_subset_prefill_ablation_runs_on_tiny_hybrid_model() -> None:
    model = _tiny_qwen35_model()
    tokenizer = _TinyTokenizer()
    harness = Qwen35AttentionSubsetHarness(
        model=model,
        tokenizer=tokenizer,
        adapter=Qwen35AttentionSubsetModelAdapter(model=model),
    )
    input_ids, attention_mask = harness.tokenize_prompt("hello ablation")
    result = harness.run_prefill_ablation(
        DotCacheConfig(head_dim=256, group_size=32, bits_k=4, bits_v=4, tokens_per_page=2),
        input_ids=input_ids,
        attention_mask=attention_mask,
        decode_steps=1,
    )
    assert result["attention_subset_prefill_ablation_ready"] is True
    assert result["attention_subset_layer_ids"] == [3]
    assert "3" in result["prefill_k_only_context_max_abs_error_by_layer"]
    assert result["prefill_dominant_kind_by_layer"]["3"] in {"K", "V", "mixed"}


def test_qwen35_hybrid_combined_localization_runs_on_tiny_hybrid_model() -> None:
    model = _tiny_qwen35_model()
    tokenizer = _TinyTokenizer()
    adapter = Qwen35AttentionSubsetDotCacheModelAdapter(
        model=model,
        dotcache_config=DotCacheConfig(head_dim=16, group_size=16, bits_k=4, bits_v=4, tokens_per_page=2),
        backend="cpu_ref",
    )
    encoded = tokenizer("hello combined localization", return_tensors="pt")
    result = run_qwen35_hybrid_combined_localization_harness(
        model,
        adapter,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        tokenizer=tokenizer,
        prefix_length=4,
        eval_steps=3,
        statecache_group_size=16,
        statecache_bits=8,
        statecache_stage="post_update_m0",
    )
    assert result["hybrid_combined_ready"] is True
    assert result["runtime_mode"] == "qwen35_hybrid_combined_localization"
    assert len(result["combined_per_step_logit_max_abs_error"]) == 3
    assert result["native_hybrid_fixed_resident_preserved"] is True
    assert "combined_first_recurrent_failure_layer" in result
    assert "combined_first_conv_failure_layer" in result
    assert "combined_deltanet_recurrent_result" in result
    assert "combined_deltanet_conv_result" in result
    assert "combined_deltanet_combined_result" in result
    assert result["combined_first_failure_family"] in {None, "attention", "recurrent", "conv", "mixed"}


def test_qwen35_statecache_cli_parse_supports_conv_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    import benchmarks.bench_qwen35_deltanet_statecache_readout as readout_bench
    import benchmarks.bench_qwen35_deltanet_statecache_loss as loss_bench
    import benchmarks.bench_qwen35_deltanet_statecache_serving as serving_bench
    import benchmarks.bench_qwen35_hybrid_failure_localize as hybrid_bench

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_deltanet_statecache_readout.py",
            "--statecache-scope",
            "conv_plus_recurrent",
            "--conv-bits",
            "4",
            "--conv-layer-bit-overrides",
            "1:8",
            "--conv-mode-override",
            "1:M3",
        ],
    )
    readout_args = readout_bench.parse_args()
    assert readout_args.statecache_scope == "conv_plus_recurrent"
    assert readout_args.conv_bits == 4
    assert readout_args.conv_layer_bit_overrides == ["1:8"]
    assert readout_args.conv_mode_override == ["1:M3"]

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_deltanet_statecache_serving.py",
            "--recurrent-mode-policy",
            "890m_m3_outlier_pair_midband_v1",
            "--recurrent-group-size-policy",
            "890m_long_horizon_group_escape_v1",
            "--recurrent-layer-group-size-override",
            "layer:18=8",
            "--paired-recurrent-group-size-policy",
            "890m_long_horizon_group_escape_v1",
            "--paired-recurrent-layer-group-size-override",
            "layer:20=8",
            "--paired-recurrent-mode-policy",
            "890m_m3_outlier_pair_midband_v1",
            "--paired-order-schedule",
            "ABBA",
            "--readout-recurrent-policy",
            "890m_context_banded_v1",
            "--readout-recurrent-mode-policy",
            "890m_m3_outlier_pair_midband_v1",
            "--warmup-in-process-repeats",
            "1",
            "--in-process-repeats",
            "3",
        ],
    )
    serving_args = serving_bench.parse_args()
    assert serving_args.recurrent_mode_policy == "890m_m3_outlier_pair_midband_v1"
    assert serving_args.recurrent_group_size_policy == "890m_long_horizon_group_escape_v1"
    assert serving_args.recurrent_layer_group_size_override == ["layer:18=8"]
    assert serving_args.paired_recurrent_group_size_policy == "890m_long_horizon_group_escape_v1"
    assert serving_args.paired_recurrent_layer_group_size_override == ["layer:20=8"]
    assert serving_args.paired_recurrent_mode_policy == "890m_m3_outlier_pair_midband_v1"
    assert serving_args.paired_order_schedule == "ABBA"
    assert serving_args.readout_recurrent_policy == "890m_context_banded_v1"
    assert serving_args.readout_recurrent_mode_policy == "890m_m3_outlier_pair_midband_v1"
    assert serving_args.readout_recurrent_renorm_interval_override == []
    assert serving_args.warmup_in_process_repeats == 1
    assert serving_args.in_process_repeats == 3

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_deltanet_statecache_localization.py",
            "--prefix-length",
            "4096",
            "--readout-recurrent-renorm-interval-override",
            "layer:18=2",
            "--readout-recurrent-mode-override",
            "layer:18=M3",
            "--recurrent-group-size-policy",
            "890m_long_horizon_group_escape_v1",
            "--recurrent-layer-group-size-override",
            "layer:18=8",
        ],
    )
    import benchmarks.bench_qwen35_deltanet_statecache_localization as localization_bench

    localization_args = localization_bench.parse_args()
    assert localization_args.readout_recurrent_renorm_interval_override == ["layer:18=2"]
    assert localization_args.readout_recurrent_mode_override == ["layer:18=M3"]
    assert localization_args.recurrent_group_size_policy == "890m_long_horizon_group_escape_v1"
    assert localization_args.recurrent_layer_group_size_override == ["layer:18=8"]


def test_parse_qwen35_deltanet_statecache_int_overrides_parses_group_size() -> None:
    parsed = parse_qwen35_deltanet_statecache_int_overrides(
        ["layer:18=8", "layer:20=16"],
        value_name="group_size",
        minimum=1,
    )
    assert parsed == {18: 8, 20: 16}


def test_qwen35_dotcache_serving_cli_parse_supports_backend_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    serving_bench_spec = importlib.util.spec_from_file_location(
        "bench_qwen35_attention_subset_dotcache_serving",
        repo_root / "benchmarks" / "bench_qwen35_attention_subset_dotcache_serving.py",
    )
    assert serving_bench_spec is not None and serving_bench_spec.loader is not None
    serving_bench = importlib.util.module_from_spec(serving_bench_spec)
    serving_bench_spec.loader.exec_module(serving_bench)

    serving_sweep_spec = importlib.util.spec_from_file_location(
        "run_qwen35_serving_sweep",
        repo_root / "scripts" / "run_qwen35_serving_sweep.py",
    )
    assert serving_sweep_spec is not None and serving_sweep_spec.loader is not None
    serving_sweep = importlib.util.module_from_spec(serving_sweep_spec)
    serving_sweep_spec.loader.exec_module(serving_sweep)

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_attention_subset_dotcache_serving.py",
            "--profile-backend",
            "--trace-python-allocations",
            "--blas-num-threads",
            "1",
            "--warmup-runs",
            "2",
            "--measured-runs",
            "3",
            "--tokens-per-page",
            "8",
            "--execution-recent-window",
            "64",
            "--execution-sink-window",
            "16",
            "--execution-recent-window-layer",
            "layer:23=96",
            "--execution-recent-window-context-layer",
            "layer:23:min_ctx:8192=1536",
            "--execution-relevance-top-k",
            "4",
            "--execution-relevance-top-k-layer",
            "layer:23=8",
            "--execution-relevance-top-k-context-layer",
            "layer:23:min_ctx:8192=8",
            "--execution-full-context-layer",
            "23",
            "--execution-disable-grouped-batching-layer",
            "23",
            "--execution-recent-old-bonus-window",
            "512",
            "--execution-recent-old-bonus-strength",
            "0.5",
            "--execution-recent-old-bonus-layer",
            "23",
            "--execution-exact-promote-top-k",
            "2",
            "--execution-exact-promote-min-margin-threshold",
            "0.0558",
            "--execution-exact-promote-max-context",
            "16384",
            "--execution-exact-promote-margin-threshold",
            "0.25",
            "--execution-exact-promote-layer",
            "23",
            "--execution-exact-promote-union-rescue-top-k",
            "2",
            "--execution-grouped-decode-compact",
            "--execution-grouped-mix-compact",
            "--execution-grouped-mix-disable-packed-cuda",
            "--execution-freeze-chunk-budget-during-decode",
            "--execution-builtin-selector-cache",
            "--execution-builtin-selector-score-all-pages",
            "--execution-builtin-selector-candidate-only",
            "--execution-builtin-selector-score-all-pages-min-candidate-fraction",
            "0.5",
            "--execution-value-escape-layer",
            "23",
            "--execution-value-escape-mode",
            "M3",
            "--execution-value-escape-old-only",
            "--execution-value-escape-top-k",
            "64",
            "--execution-value-escape-prewarm",
            "--execution-value-escape-prewarm-min-context",
            "49152",
            "--learned-page-selector-scope",
            "K",
            "--learned-page-selector-target-candidate",
            "M3/affine/4/float16",
            "--learned-page-selector-logit-offset",
            "1.5",
            "--learned-page-selector-profile",
            "manual",
            "--prepared-chunk-cache-min-page-count",
            "2",
            "--scorer-diagnostic",
            "--execution-relevance-mode",
            "envelope",
            "--execution-secondary-relevance-mode",
            "sketch",
            "--execution-secondary-relevance-top-k",
            "2",
            "--execution-secondary-relevance-min-overlap",
            "0.5",
            "--execution-secondary-relevance-layer",
            "23",
            "--execution-recent-neighbor-rescue-top-k",
            "2",
            "--execution-recent-neighbor-rescue-anchor-window",
            "1024",
            "--execution-recent-neighbor-rescue-min-anchor-pages",
            "4",
            "--execution-recent-neighbor-rescue-layer",
            "23",
            "--execution-exact-refine-top-k",
            "2",
            "--execution-exact-refine-layer",
            "23",
            "--recall-analysis",
            "--quality-check",
        ],
    )
    serving_args = serving_bench.parse_args()
    assert serving_args.profile_backend is True
    assert serving_args.trace_python_allocations is True
    assert serving_args.blas_num_threads == 1
    assert serving_args.warmup_runs == 2
    assert serving_args.measured_runs == 3
    assert serving_args.tokens_per_page == 8
    assert serving_args.prepared_chunk_cache_min_page_count == 2
    assert serving_args.execution_recent_window == 64
    assert serving_args.execution_sink_window == 16
    assert serving_args.execution_recent_window_layer == ["layer:23=96"]
    assert serving_args.execution_recent_window_context_layer == ["layer:23:min_ctx:8192=1536"]
    assert serving_args.execution_relevance_top_k == 4
    assert serving_args.execution_relevance_top_k_layer == ["layer:23=8"]
    assert serving_args.execution_relevance_top_k_context_layer == ["layer:23:min_ctx:8192=8"]
    assert serving_args.execution_full_context_layer == [23]
    assert serving_args.execution_disable_grouped_batching_layer == [23]
    assert serving_args.execution_recent_old_bonus_window == 512
    assert serving_args.execution_recent_old_bonus_strength == 0.5
    assert serving_args.execution_recent_old_bonus_layer == [23]
    assert serving_args.execution_relevance_mode == "envelope"
    assert serving_args.execution_secondary_relevance_mode == "sketch"
    assert serving_args.execution_secondary_relevance_top_k == 2
    assert serving_args.execution_secondary_relevance_min_overlap == 0.5
    assert serving_args.execution_secondary_relevance_layer == [23]
    assert serving_args.execution_recent_neighbor_rescue_top_k == 2
    assert serving_args.execution_recent_neighbor_rescue_anchor_window == 1024
    assert serving_args.execution_recent_neighbor_rescue_min_anchor_pages == 4
    assert serving_args.execution_recent_neighbor_rescue_layer == [23]
    assert serving_args.execution_exact_promote_top_k == 2
    assert serving_args.execution_exact_promote_min_margin_threshold == 0.0558
    assert serving_args.execution_exact_promote_max_context == 16384
    assert serving_args.execution_exact_promote_margin_threshold == 0.25
    assert serving_args.execution_exact_promote_layer == [23]
    assert serving_args.execution_exact_promote_union_rescue_top_k == 2
    assert serving_args.execution_grouped_decode_compact is True
    assert serving_args.execution_grouped_mix_compact is True
    assert serving_args.execution_grouped_mix_disable_packed_cuda is True
    assert serving_args.execution_freeze_chunk_budget_during_decode is True
    assert serving_args.execution_builtin_selector_cache is True
    assert serving_args.execution_builtin_selector_score_all_pages is True
    assert serving_args.execution_builtin_selector_candidate_only is True
    assert serving_args.execution_builtin_selector_score_all_pages_min_candidate_fraction == 0.5
    assert serving_args.execution_value_escape_layer == [23]
    assert serving_args.execution_value_escape_mode == "M3"
    assert serving_args.execution_value_escape_old_only is True
    assert serving_args.execution_value_escape_top_k == 64
    assert serving_args.execution_value_escape_prewarm is True
    assert serving_args.execution_value_escape_prewarm_min_context == 49152
    assert serving_args.learned_page_selector_profile == "manual"
    assert serving_args.learned_page_selector_scope == "K"
    assert serving_args.learned_page_selector_target_candidate == "M3/affine/4/float16"
    assert serving_args.learned_page_selector_logit_offset == 1.5
    assert serving_args.scorer_diagnostic is True
    assert serving_args.execution_exact_refine_top_k == 2
    assert serving_args.execution_exact_refine_layer == [23]
    assert serving_args.recall_analysis is True
    assert serving_args.quality_check is True

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_serving_sweep.py",
            "--dotcache-profile-backend",
            "--warmup-runs",
            "1",
            "--measured-runs",
            "4",
            "--learned-page-selector-scope",
            "V",
            "--learned-page-selector-target-candidate",
            "M0/affine/4",
            "--learned-page-selector-logit-offset",
            "-0.75",
            "--learned-page-selector-profile",
            "systems",
            "--contexts",
            "4096",
            "16384",
        ],
    )
    sweep_args = serving_sweep.parse_args()
    assert sweep_args.dotcache_profile_backend is True
    assert sweep_args.warmup_runs == 1
    assert sweep_args.measured_runs == 4
    assert sweep_args.learned_page_selector_profile == "systems"
    assert sweep_args.learned_page_selector_scope == "V"
    assert sweep_args.learned_page_selector_target_candidate == "M0/affine/4"
    assert sweep_args.learned_page_selector_logit_offset == -0.75
    assert sweep_args.contexts == [4096, 16384]


def test_qwen35_dotcache_serving_run_case_forwards_quality_allocation_trace() -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    serving_bench_spec = importlib.util.spec_from_file_location(
        "bench_qwen35_attention_subset_dotcache_serving",
        repo_root / "benchmarks" / "bench_qwen35_attention_subset_dotcache_serving.py",
    )
    assert serving_bench_spec is not None and serving_bench_spec.loader is not None
    serving_bench = importlib.util.module_from_spec(serving_bench_spec)
    serving_bench_spec.loader.exec_module(serving_bench)

    class _FakeHarness:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def run_attention_subset_dotcache_serving_quality(self, **kwargs):
            self.calls.append(kwargs)
            return {"ok": True}

    harness = _FakeHarness()
    serving_bench._run_case(
        harness,
        input_ids=torch.ones((1, 4), dtype=torch.long),
        attention_mask=torch.ones((1, 4), dtype=torch.long),
        max_new_tokens=2,
        base_record={
            "benchmark": "qwen35_attention_subset_dotcache_serving",
            "quality_check": True,
            "profile_backend": True,
            "trace_python_allocations": True,
        },
        continue_on_error=False,
    )
    assert len(harness.calls) == 1
    assert harness.calls[0]["trace_python_allocations"] is True
    assert harness.calls[0]["profile_backend"] is True


def test_qwen35_cuda_shortlist_probe_cli_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    probe_spec = importlib.util.spec_from_file_location(
        "run_qwen35_cuda_shortlist_probe",
        repo_root / "scripts" / "run_qwen35_cuda_shortlist_probe.py",
    )
    assert probe_spec is not None and probe_spec.loader is not None
    probe_module = importlib.util.module_from_spec(probe_spec)
    probe_spec.loader.exec_module(probe_module)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_cuda_shortlist_probe.py",
            "--contexts",
            "16384",
            "32768",
            "--cases",
            "shortlist_base",
            "shortlist_l23_ctx",
            "--timeout-seconds",
            "90",
            "--profile-backend",
            "--quality-check",
            "--quality-mode",
            "loss_tail",
            "--quality-eval-steps",
            "8",
            "--output",
            "benchmarks/results/test_probe.jsonl",
        ],
    )
    args = probe_module.parse_args()
    assert args.contexts == [16384, 32768]
    assert args.cases == ["shortlist_base", "shortlist_l23_ctx"]
    assert args.timeout_seconds == 90
    assert args.profile_backend is True
    assert args.quality_check is True
    assert args.quality_mode == "loss_tail"
    assert args.quality_eval_steps == 8
    assert args.output == "benchmarks/results/test_probe.jsonl"


def test_qwen35_layer23_ablation_matrix_cli_builds_selector_and_kv_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    script_spec = importlib.util.spec_from_file_location(
        "run_qwen35_layer23_ablation_matrix",
        repo_root / "scripts" / "run_qwen35_layer23_ablation_matrix.py",
    )
    assert script_spec is not None and script_spec.loader is not None
    script_module = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(script_module)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_layer23_ablation_matrix.py",
            "--contexts",
            "32768",
            "--selector-modes",
            "layer23_full_context",
            "--kv-modes",
            "m0_v_escape_top256",
            "--quality-check",
            "--blas-num-threads",
            "1",
        ],
    )
    args = script_module.parse_args()
    assert args.contexts == [32768]
    assert args.selector_modes == ["layer23_full_context"]
    assert args.kv_modes == ["m0_v_escape_top256"]
    assert args.quality_check is True
    assert args.blas_num_threads == 1

    command = script_module._benchmark_command(
        args,
        context=32768,
        selector_mode="layer23_full_context",
        kv_mode="m0_v_escape_top256",
    )
    assert "--execution-full-context-layer" in command
    assert "23" in command
    assert "--key-mode-override" in command
    assert "layer:23=M0" in command
    assert "--execution-value-escape-layer" in command
    assert "--execution-value-escape-mode" in command
    assert "--execution-value-escape-top-k" in command
    assert "256" in command
    assert "M3" in command
    assert "--blas-num-threads" in command
    assert "1" in command


def test_qwen35_layer23_ablation_matrix_can_disable_default_layer_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    script_spec = importlib.util.spec_from_file_location(
        "run_qwen35_layer23_ablation_matrix",
        repo_root / "scripts" / "run_qwen35_layer23_ablation_matrix.py",
    )
    assert script_spec is not None and script_spec.loader is not None
    script_module = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(script_module)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_layer23_ablation_matrix.py",
            "--model-id",
            "Qwen/Qwen3.5-4B",
            "--layer-profile",
            "none",
            "--contexts",
            "16384",
            "--selector-modes",
            "approx_shortlist",
            "--kv-modes",
            "m0_v_escape",
            "--quality-check",
        ],
    )
    args = script_module.parse_args()
    command = script_module._benchmark_command(
        args,
        context=16384,
        selector_mode="approx_shortlist",
        kv_mode="m0_v_escape",
    )
    assert "--layer-profile" not in command
    assert "Qwen/Qwen3.5-4B" in command


def test_qwen35_value_escape_layer_scan_builds_layer_specific_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    script_spec = importlib.util.spec_from_file_location(
        "run_qwen35_value_escape_layer_scan",
        repo_root / "scripts" / "run_qwen35_value_escape_layer_scan.py",
    )
    assert script_spec is not None and script_spec.loader is not None
    script_module = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(script_module)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_value_escape_layer_scan.py",
            "--model-id",
            "Qwen/Qwen3.5-4B",
            "--layer-profile",
            "none",
            "--layers",
            "19",
            "23",
            "--contexts",
            "16384",
            "--selector-modes",
            "layer_full_context",
            "--kv-modes",
            "m0_v_escape",
            "--quality-check",
        ],
    )
    args = script_module.parse_args()
    assert args.layers == [19, 23]
    assert args.contexts == [16384]
    assert args.selector_modes == ["layer_full_context"]
    assert args.kv_modes == ["m0_v_escape"]

    command = script_module._benchmark_command(
        args,
        layer_id=19,
        context=16384,
        selector_mode="layer_full_context",
        kv_mode="m0_v_escape",
    )
    assert "--layer-profile" not in command
    assert "--execution-full-context-layer" in command
    assert "19" in command
    assert "--execution-value-escape-layer" in command
    assert "--value-mode-override" in command
    assert "layer:19=M0" in command


def test_qwen35_value_escape_layer_scan_presets_apply_expected_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    script_spec = importlib.util.spec_from_file_location(
        "run_qwen35_value_escape_layer_scan",
        repo_root / "scripts" / "run_qwen35_value_escape_layer_scan.py",
    )
    assert script_spec is not None and script_spec.loader is not None
    script_module = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(script_module)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_value_escape_layer_scan.py",
            "--preset",
            "qwen35_4b_initial_scan",
            "--quality-check",
        ],
    )
    args = script_module.parse_args()
    assert args.model_id == "Qwen/Qwen3.5-4B"
    assert args.layer_profile == "none"
    assert args.layers == [3, 7, 11, 15, 19, 23]
    assert args.contexts == [16384]
    assert args.selector_modes == ["approx_shortlist"]
    assert args.kv_modes == ["exact_m0", "m0_v_escape"]

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_value_escape_layer_scan.py",
            "--preset",
            "qwen35_4b_confirm_32768",
            "--quality-check",
        ],
    )
    args = script_module.parse_args()
    assert args.layers == [7, 19]
    assert args.contexts == [32768]
    assert args.selector_modes == ["approx_shortlist", "layer_full_context"]
    assert args.kv_modes == ["exact_m0", "m0_v_escape"]

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_value_escape_layer_scan.py",
            "--preset",
            "qwen35_9b_initial_scan",
            "--quality-check",
        ],
    )
    args = script_module.parse_args()
    assert args.model_id == "Qwen/Qwen3.5-9B"
    assert args.weight_quantization == "bnb_8bit"
    assert args.layers == [3, 7, 11, 15, 19, 23]
    assert args.contexts == [8192]
    assert args.selector_modes == ["approx_shortlist"]
    assert args.kv_modes == ["exact_m0", "m0_v_escape"]


def test_qwen35_value_escape_reference_presets_build_expected_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    script_spec = importlib.util.spec_from_file_location(
        "run_qwen35_value_escape_reference",
        repo_root / "scripts" / "run_qwen35_value_escape_reference.py",
    )
    assert script_spec is not None and script_spec.loader is not None
    script_module = importlib.util.module_from_spec(script_spec)
    script_spec.loader.exec_module(script_module)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_value_escape_reference.py",
            "--preset",
            "qwen35_0p8b_best",
            "--contexts",
            "49152",
            "--quality-check",
        ],
    )
    args = script_module.parse_args()
    command = script_module._benchmark_command(args, context=49152)
    assert "Qwen/Qwen3.5-0.8B" in command
    assert "--layer-profile" in command
    assert "qwen35_0p8b_attention_subset_cuda_value_escape_best.yaml" in " ".join(command)
    assert "--execution-value-escape-layer" not in command
    assert "--execution-value-escape-prewarm" not in command

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_value_escape_reference.py",
            "--preset",
            "qwen35_4b_best",
            "--contexts",
            "65536",
            "--quality-check",
        ],
    )
    args = script_module.parse_args()
    command = script_module._benchmark_command(args, context=65536)
    assert "Qwen/Qwen3.5-4B" in command
    assert "--layer-profile" in command
    assert "qwen35_4b_attention_subset_cuda_value_escape_best.yaml" in " ".join(command)
    assert "--execution-value-escape-layer" not in command
    assert "--execution-value-escape-prewarm" not in command


def test_qwen35_cuda_layer_profile_loads_context_aware_shortlist_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    serving_spec = importlib.util.spec_from_file_location(
        "bench_qwen35_attention_subset_dotcache_serving",
        repo_root / "benchmarks" / "bench_qwen35_attention_subset_dotcache_serving.py",
    )
    assert serving_spec is not None and serving_spec.loader is not None
    serving_module = importlib.util.module_from_spec(serving_spec)
    serving_spec.loader.exec_module(serving_module)

    loss_spec = importlib.util.spec_from_file_location(
        "bench_qwen35_attention_subset_dotcache_loss",
        repo_root / "benchmarks" / "bench_qwen35_attention_subset_dotcache_loss.py",
    )
    assert loss_spec is not None and loss_spec.loader is not None
    loss_module = importlib.util.module_from_spec(loss_spec)
    loss_spec.loader.exec_module(loss_module)

    profile_path = (
        repo_root
        / "configs"
        / "layer_profiles"
        / "qwen35_0p8b_attention_subset_cuda_shortlist_baseline.yaml"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_attention_subset_dotcache_serving.py",
            "--layer-profile",
            str(profile_path),
        ],
    )
    serving_args = serving_module.parse_args()
    serving_module._resolve_args_from_layer_profile(serving_args)
    assert serving_args.execution_recent_window == 1024
    assert serving_args.execution_sink_window == 256
    assert serving_args.execution_relevance_top_k == 4
    assert serving_args.execution_relevance_mode == "envelope"
    assert serving_args.execution_relevance_top_k_context_layer == ["layer:23:min_ctx:32768=8"]
    assert serving_args.execution_exact_promote_top_k == 0
    assert serving_args.execution_exact_promote_margin_threshold == 0.0
    assert serving_args.execution_exact_promote_layer == []

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_attention_subset_dotcache_loss.py",
            "--layer-profile",
            str(profile_path),
        ],
    )
    loss_args = loss_module.parse_args()
    loss_module._resolve_args_from_layer_profile(loss_args)
    assert loss_args.execution_recent_window == 1024
    assert loss_args.execution_sink_window == 256
    assert loss_args.execution_relevance_top_k == 4
    assert loss_args.execution_relevance_mode == "envelope"
    assert loss_args.execution_relevance_top_k_context_layer == ["layer:23:min_ctx:32768=8"]
    assert loss_args.execution_exact_promote_top_k == 0
    assert loss_args.execution_exact_promote_margin_threshold == 0.0
    assert loss_args.execution_exact_promote_layer == []


def test_qwen35_cuda_layer_profile_loads_shortlist_quality_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    serving_spec = importlib.util.spec_from_file_location(
        "bench_qwen35_attention_subset_dotcache_serving",
        repo_root / "benchmarks" / "bench_qwen35_attention_subset_dotcache_serving.py",
    )
    assert serving_spec is not None and serving_spec.loader is not None
    serving_module = importlib.util.module_from_spec(serving_spec)
    serving_spec.loader.exec_module(serving_module)

    loss_spec = importlib.util.spec_from_file_location(
        "bench_qwen35_attention_subset_dotcache_loss",
        repo_root / "benchmarks" / "bench_qwen35_attention_subset_dotcache_loss.py",
    )
    assert loss_spec is not None and loss_spec.loader is not None
    loss_module = importlib.util.module_from_spec(loss_spec)
    loss_spec.loader.exec_module(loss_module)

    profile_path = (
        repo_root
        / "configs"
        / "layer_profiles"
        / "qwen35_0p8b_attention_subset_cuda_shortlist_quality.yaml"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_attention_subset_dotcache_serving.py",
            "--layer-profile",
            str(profile_path),
        ],
    )
    serving_args = serving_module.parse_args()
    serving_module._resolve_args_from_layer_profile(serving_args)
    assert serving_args.execution_relevance_top_k_context_layer == ["layer:23:min_ctx:32768=8"]
    assert serving_args.execution_exact_promote_top_k == 2
    assert serving_args.execution_exact_promote_margin_threshold == 0.0558
    assert serving_args.execution_exact_promote_layer == [23]

    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_attention_subset_dotcache_loss.py",
            "--layer-profile",
            str(profile_path),
        ],
    )
    loss_args = loss_module.parse_args()
    loss_module._resolve_args_from_layer_profile(loss_args)
    assert loss_args.execution_relevance_top_k_context_layer == ["layer:23:min_ctx:32768=8"]
    assert loss_args.execution_exact_promote_top_k == 2
    assert loss_args.execution_exact_promote_margin_threshold == 0.0558
    assert loss_args.execution_exact_promote_layer == [23]


def test_qwen35_cuda_layer_profile_loads_value_escape_reference_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    serving_spec = importlib.util.spec_from_file_location(
        "bench_qwen35_attention_subset_dotcache_serving",
        repo_root / "benchmarks" / "bench_qwen35_attention_subset_dotcache_serving.py",
    )
    assert serving_spec is not None and serving_spec.loader is not None
    serving_module = importlib.util.module_from_spec(serving_spec)
    serving_spec.loader.exec_module(serving_module)

    profile_0p8b = (
        repo_root
        / "configs"
        / "layer_profiles"
        / "qwen35_0p8b_attention_subset_cuda_value_escape_best.yaml"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_attention_subset_dotcache_serving.py",
            "--layer-profile",
            str(profile_0p8b),
        ],
    )
    serving_args = serving_module.parse_args()
    serving_module._resolve_args_from_layer_profile(serving_args)
    assert serving_args.execution_builtin_selector_cache is True
    assert serving_args.execution_builtin_selector_candidate_only is True
    assert serving_args.execution_relevance_top_k_context_layer == ["layer:23:min_ctx:8192=8"]
    assert serving_args.key_mode_override == ["layer:23=M0"]
    assert serving_args.value_mode_override == ["layer:23=M0"]
    assert serving_args.execution_value_escape_layer == [23]
    assert serving_args.execution_value_escape_mode == "M3"
    assert serving_args.execution_value_escape_prewarm is True
    assert serving_args.execution_value_escape_prewarm_min_context == 49152

    profile_4b = (
        repo_root
        / "configs"
        / "layer_profiles"
        / "qwen35_4b_attention_subset_cuda_value_escape_best.yaml"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "bench_qwen35_attention_subset_dotcache_serving.py",
            "--model-id",
            "Qwen/Qwen3.5-4B",
            "--layer-profile",
            str(profile_4b),
        ],
    )
    serving_args = serving_module.parse_args()
    serving_module._resolve_args_from_layer_profile(serving_args)
    assert serving_args.execution_builtin_selector_cache is True
    assert serving_args.execution_builtin_selector_candidate_only is True
    assert serving_args.execution_relevance_top_k_context_layer == ["layer:7:min_ctx:8192=8"]
    assert serving_args.key_mode_override == ["layer:7=M0"]
    assert serving_args.value_mode_override == ["layer:7=M0"]
    assert serving_args.execution_value_escape_layer == [7]
    assert serving_args.execution_value_escape_mode == "M3"
    assert serving_args.execution_value_escape_prewarm is False
    assert serving_args.execution_value_escape_prewarm_min_context == 0


def test_qwen35_cuda_profile_bakeoff_cli_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "run_qwen35_cuda_profile_bakeoff",
        repo_root / "scripts" / "run_qwen35_cuda_profile_bakeoff.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_qwen35_cuda_profile_bakeoff.py",
            "--profiles",
            "configs/layer_profiles/a.yaml",
            "configs/layer_profiles/b.yaml",
            "--modes",
            "quality",
            "scorer",
            "--contexts",
            "16384",
            "32768",
            "--timeout-seconds",
            "300",
            "--output-dir",
            "benchmarks/results/test_bakeoff",
        ],
    )
    args = module.parse_args()
    assert args.profiles == ["configs/layer_profiles/a.yaml", "configs/layer_profiles/b.yaml"]
    assert args.modes == ["quality", "scorer"]
    assert args.contexts == [16384, 32768]
    assert args.timeout_seconds == 300
    assert args.output_dir == "benchmarks/results/test_bakeoff"


def test_qwen35_scorer_report_summarizes_layer23(tmp_path: Path) -> None:
    import json
    import importlib.util
    from io import StringIO
    from pathlib import Path
    from contextlib import redirect_stdout

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "report_qwen35_scorer_diagnostic",
        repo_root / "scripts" / "report_qwen35_scorer_diagnostic.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sample = {
        "prompt_length": 32768,
        "scorer_diagnostic": True,
        "scorer_worst_layer_id": "11",
        "scorer_layer_records": [
            {
                "layer_id": 23,
                "groups": [
                    {
                        "exact_top_recall": 0.8,
                        "approx_boundary_margin_normalized": 0.02,
                        "score_rank_correlation": 0.95,
                        "score_value_correlation": 0.96,
                    },
                    {
                        "exact_top_recall": 0.7,
                        "approx_boundary_margin_normalized": 0.01,
                        "score_rank_correlation": 0.94,
                        "score_value_correlation": 0.97,
                    },
                ],
            },
            {
                "layer_id": 11,
                "groups": [
                    {
                        "exact_top_recall": 0.5,
                        "score_rank_correlation": 0.49,
                        "score_value_correlation": 0.46,
                    }
                ],
            },
        ],
    }
    input_path = tmp_path / "sample.jsonl"
    input_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    monkeypatch_args = [
        "report_qwen35_scorer_diagnostic.py",
        "--input",
        str(input_path),
    ]
    import sys

    old_argv = sys.argv
    sys.argv = monkeypatch_args
    try:
        buf = StringIO()
        with redirect_stdout(buf):
            module.main()
    finally:
        sys.argv = old_argv
    output = buf.getvalue()
    assert "sample.jsonl" in output
    assert "32768" in output
    assert "23" in output
    assert "0.750" in output
    assert "0.945" in output


def test_qwen35_statecache_serving_repeat_summary_aggregates_measurements() -> None:
    import benchmarks.bench_qwen35_deltanet_statecache_serving as serving_bench

    summary = serving_bench._summarize_in_process_repeat_records(
        [
            {
                "prefill_ms": 10.0,
                "deltanet_statecache_decode_ms_per_step": 20.0,
                "deltanet_statecache_prefill_cuda_peak_memory_reserved_bytes": 100,
                "deltanet_statecache_decode_cuda_peak_memory_reserved_bytes": 200,
                "deltanet_statecache_generated_ids": [1, 2],
            },
            {
                "prefill_ms": 14.0,
                "deltanet_statecache_decode_ms_per_step": 28.0,
                "deltanet_statecache_prefill_cuda_peak_memory_reserved_bytes": 140,
                "deltanet_statecache_decode_cuda_peak_memory_reserved_bytes": 260,
                "deltanet_statecache_generated_ids": [1, 2],
            },
        ],
        warmup_in_process_repeats=1,
    )

    assert summary["benchmark_measurement_mode"] == "in_process_repeated"
    assert summary["warmup_in_process_repeats"] == 1
    assert summary["in_process_repeats"] == 2
    assert summary["prefill_ms"] == 12.0
    assert summary["deltanet_statecache_decode_ms_per_step"] == 24.0
    assert summary["deltanet_statecache_prefill_cuda_peak_memory_reserved_bytes"] == 120
    assert summary["deltanet_statecache_decode_cuda_peak_memory_reserved_bytes"] == 230
    assert summary["in_process_repeat_generated_ids_consistent"] is True
    assert summary["in_process_repeat_decode_ms_per_step_values"] == [20.0, 28.0]


def test_qwen35_statecache_serving_paired_repeat_summary_aggregates_measurements() -> None:
    import benchmarks.bench_qwen35_deltanet_statecache_serving as serving_bench

    summary = serving_bench._summarize_paired_in_process_repeat_records(
        [
            {
                "prefill_ms": 10.0,
                "deltanet_statecache_decode_ms_per_step": 20.0,
                "deltanet_statecache_effective_recurrent_compression_ratio": 3.2,
                "deltanet_statecache_generated_ids": [1, 2],
            },
            {
                "prefill_ms": 14.0,
                "deltanet_statecache_decode_ms_per_step": 24.0,
                "deltanet_statecache_effective_recurrent_compression_ratio": 3.2,
                "deltanet_statecache_generated_ids": [1, 2],
            },
        ],
        [
            {
                "prefill_ms": 11.0,
                "deltanet_statecache_decode_ms_per_step": 18.0,
                "deltanet_statecache_effective_recurrent_compression_ratio": 2.57,
                "deltanet_statecache_generated_ids": [1, 2],
            },
            {
                "prefill_ms": 15.0,
                "deltanet_statecache_decode_ms_per_step": 22.0,
                "deltanet_statecache_effective_recurrent_compression_ratio": 2.57,
                "deltanet_statecache_generated_ids": [1, 2],
            },
        ],
        warmup_in_process_repeats=1,
        in_process_repeats=1,
        candidate_label="m3_outliers",
        order_schedule="AB",
    )

    assert summary["benchmark_measurement_mode"] == "in_process_paired_repeated"
    assert summary["warmup_in_process_repeats"] == 1
    assert summary["in_process_repeats"] == 1
    assert summary["paired_order_schedule"] == "AB"
    assert summary["paired_baseline_sample_count"] == 2
    assert summary["paired_candidate_sample_count"] == 2
    assert summary["paired_candidate_label"] == "m3_outliers"
    assert summary["paired_generated_ids_match_all"] is True
    assert summary["baseline_decode_ms_per_step"] == 22.0
    assert summary["candidate_decode_ms_per_step"] == 20.0
    assert summary["paired_decode_ms_per_step_delta"] == -2.0
    assert summary["baseline_recurrent_compression_ratio"] == 3.2
    assert summary["candidate_recurrent_compression_ratio"] == 2.57
