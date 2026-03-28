from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

from dotcache.integrations.qwen35 import (
    Qwen35TextHarness,
    Qwen35TextModelAdapter,
    inspect_qwen35_hybrid_state,
    load_qwen35_text_only_from_pretrained,
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
    assert len(result["hybrid_state_layers"]) == 4
    assert result["hybrid_state_layers"][0]["layer_type"] == "linear_attention"
    assert result["hybrid_state_layers"][3]["layer_type"] == "full_attention"
