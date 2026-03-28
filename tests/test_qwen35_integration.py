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
    run_qwen35_attention_subset_dotcache_harness,
    run_qwen35_attention_subset_replay_harness,
    run_qwen35_deltanet_state_ablation_harness,
    run_qwen35_text_generation_harness,
    run_qwen35_text_loss_harness,
    summarize_qwen35_dotcache_fit,
)


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
