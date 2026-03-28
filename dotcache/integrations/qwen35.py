from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .llama import (
    _default_model_device,
    _ensure_attention_mask,
    _normalize_input_ids,
    _require_transformers,
    _run_inference,
    _synchronize_device,
    _timed_call,
    resolve_hf_auth_kwargs,
    transformers_available,
)

if transformers_available():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
else:  # pragma: no cover - exercised in environments without transformers
    torch = None
    AutoTokenizer = None
    Qwen3_5ForConditionalGeneration = None


Qwen35Mode = Literal["dense"]


def _require_qwen35_model_class() -> None:
    _require_transformers()
    if Qwen3_5ForConditionalGeneration is None:
        raise RuntimeError("transformers installation does not expose Qwen3_5ForConditionalGeneration")


def _text_only_error() -> ValueError:
    return ValueError("Qwen3.5 v1 is text-only; image/video or multimodal inputs are not supported")


def _hybrid_layer_types(model_or_config: Any) -> tuple[str, ...]:
    config = getattr(model_or_config, "config", model_or_config)
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return ()
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is None:
        return ()
    return tuple(str(layer_type) for layer_type in layer_types)


def _hybrid_block_summary(model_or_config: Any) -> dict[str, Any]:
    layer_types = _hybrid_layer_types(model_or_config)
    summary = {
        "hybrid_family": "qwen3_5",
        "hybrid_layer_count": len(layer_types),
        "hybrid_layer_types": list(layer_types),
        "hybrid_linear_attention_layer_count": sum(1 for layer_type in layer_types if layer_type == "linear_attention"),
        "hybrid_full_attention_layer_count": sum(1 for layer_type in layer_types if layer_type == "full_attention"),
        "hybrid_other_layer_type_count": sum(
            1 for layer_type in layer_types if layer_type not in {"linear_attention", "full_attention"}
        ),
    }
    config = getattr(model_or_config, "config", model_or_config)
    summary["vision_config_present"] = bool(getattr(config, "vision_config", None) is not None)
    return summary


def _qwen35_text_model(model_or_config: Any) -> Any | None:
    model = getattr(model_or_config, "model", model_or_config)
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        return language_model
    root_model = getattr(model, "model", None)
    if root_model is not None:
        return getattr(root_model, "language_model", None)
    return None


def _hybrid_layer_records(model_or_config: Any) -> list[dict[str, Any]]:
    layer_types = _hybrid_layer_types(model_or_config)
    text_model = _qwen35_text_model(model_or_config)
    layers = getattr(text_model, "layers", None)
    records: list[dict[str, Any]] = []
    for layer_id, layer_type in enumerate(layer_types):
        record = {
            "layer_id": int(layer_id),
            "layer_type": layer_type,
            "state_family": "attention_kv" if layer_type == "full_attention" else "linear_recurrent",
            "dotcache_candidate": bool(layer_type == "full_attention"),
            "requires_hybrid_state": bool(layer_type != "full_attention"),
            "token_mixer_module": None,
            "layer_module": None,
        }
        if layers is not None and layer_id < len(layers):
            layer = layers[layer_id]
            record["layer_module"] = type(layer).__name__
            if hasattr(layer, "self_attn"):
                record["token_mixer_module"] = type(layer.self_attn).__name__
            elif hasattr(layer, "linear_attn"):
                record["token_mixer_module"] = type(layer.linear_attn).__name__
        records.append(record)
    return records


def _iter_cache_tensors(cache: Any, *, _seen: set[int] | None = None):
    if _seen is None:
        _seen = set()
    if cache is None:
        return
    cache_id = id(cache)
    if cache_id in _seen:
        return
    _seen.add(cache_id)

    if torch is not None and torch.is_tensor(cache):
        yield cache
        return
    if isinstance(cache, (list, tuple)):
        for item in cache:
            yield from _iter_cache_tensors(item, _seen=_seen)
        return

    for attr_name in (
        "key_cache",
        "value_cache",
        "kv_states",
        "conv_states",
        "ssm_states",
        "recurrent_states",
        "attention_mask_cache",
        "position_cache",
    ):
        if hasattr(cache, attr_name):
            yield from _iter_cache_tensors(getattr(cache, attr_name), _seen=_seen)

    to_legacy = getattr(cache, "to_legacy_cache", None)
    if callable(to_legacy):
        try:
            legacy = to_legacy()
        except Exception:  # pragma: no cover - defensive only
            legacy = None
        if legacy is not None and legacy is not cache:
            yield from _iter_cache_tensors(legacy, _seen=_seen)


def _hybrid_cache_nbytes(cache: Any) -> int:
    total = 0
    for tensor in _iter_cache_tensors(cache):
        total += int(tensor.nelement() * tensor.element_size())
    return total


def _cache_component_nbytes(cache: Any, attr_name: str, layer_id: int) -> int:
    values = getattr(cache, attr_name, None)
    if values is None:
        return 0
    if not isinstance(values, list | tuple):
        return 0
    if layer_id >= len(values):
        return 0
    value = values[layer_id]
    if value is None:
        return 0
    return _hybrid_cache_nbytes(value)


def summarize_qwen35_hybrid_state(cache: Any, model_or_config: Any) -> dict[str, Any]:
    layer_records = _hybrid_layer_records(model_or_config)
    attention_kv_bytes = 0
    linear_conv_bytes = 0
    linear_recurrent_bytes = 0
    for record in layer_records:
        layer_id = int(record["layer_id"])
        key_bytes = _cache_component_nbytes(cache, "key_cache", layer_id)
        value_bytes = _cache_component_nbytes(cache, "value_cache", layer_id)
        conv_bytes = _cache_component_nbytes(cache, "conv_states", layer_id)
        recurrent_bytes = _cache_component_nbytes(cache, "recurrent_states", layer_id)
        record["key_cache_bytes"] = int(key_bytes)
        record["value_cache_bytes"] = int(value_bytes)
        record["conv_state_bytes"] = int(conv_bytes)
        record["recurrent_state_bytes"] = int(recurrent_bytes)
        record["layer_state_bytes"] = int(key_bytes + value_bytes + conv_bytes + recurrent_bytes)
        attention_kv_bytes += key_bytes + value_bytes
        linear_conv_bytes += conv_bytes
        linear_recurrent_bytes += recurrent_bytes
    total_state_bytes = attention_kv_bytes + linear_conv_bytes + linear_recurrent_bytes
    return {
        "hybrid_state_total_bytes": int(total_state_bytes),
        "hybrid_attention_kv_bytes": int(attention_kv_bytes),
        "hybrid_linear_conv_state_bytes": int(linear_conv_bytes),
        "hybrid_linear_recurrent_state_bytes": int(linear_recurrent_bytes),
        "hybrid_state_layers": layer_records,
    }


def summarize_qwen35_dotcache_fit(model_or_config: Any) -> dict[str, Any]:
    layer_records = _hybrid_layer_records(model_or_config)
    attention_candidate_layers = [record["layer_id"] for record in layer_records if record["dotcache_candidate"]]
    hybrid_only_layers = [record["layer_id"] for record in layer_records if record["requires_hybrid_state"]]
    linear_layer_count = len(hybrid_only_layers)
    full_attention_layer_count = len(attention_candidate_layers)
    suggested_next_step = (
        "attention_subset_only_then_generalize_state"
        if linear_layer_count > 0 and full_attention_layer_count > 0
        else "dotcache_attention_path_only"
        if full_attention_layer_count > 0
        else "no_attention_subset_available"
    )
    return {
        "attention_candidate_layer_ids": attention_candidate_layers,
        "attention_candidate_layer_count": len(attention_candidate_layers),
        "hybrid_only_layer_ids": hybrid_only_layers,
        "hybrid_only_layer_count": len(hybrid_only_layers),
        "requires_hybrid_state_abstraction": bool(linear_layer_count > 0),
        "suggested_next_step": suggested_next_step,
    }


def _decode_text(tokenizer: Any | None, token_ids: list[int]) -> str | None:
    if tokenizer is None:
        return None
    return str(tokenizer.decode(token_ids, skip_special_tokens=True))


def load_qwen35_text_only_from_pretrained(
    model_id: str,
    *,
    device: str | None = None,
    torch_dtype: str = "float16",
):
    _require_qwen35_model_class()
    dtype = getattr(torch, torch_dtype)
    resolved_device = _default_model_device() if device is None else device
    auth_kwargs = resolve_hf_auth_kwargs()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        trust_remote_code=False,
        **auth_kwargs,
    )
    model.to(resolved_device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, **auth_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


@dataclass(slots=True)
class Qwen35TextModelAdapter:
    model: Any
    mode: Qwen35Mode = "dense"

    @property
    def device(self):
        return next(self.model.parameters()).device

    def set_mode(self, mode: str) -> None:
        if mode != "dense":
            raise ValueError(
                "Qwen3.5 v1 only supports dense mode; DotCache interception is not implemented for the hybrid text stack yet"
            )
        self.mode = "dense"

    def hybrid_block_summary(self) -> dict[str, Any]:
        return _hybrid_block_summary(self.model)

    def hybrid_layer_summary(self) -> list[dict[str, Any]]:
        return _hybrid_layer_records(self.model)

    def hybrid_fit_summary(self) -> dict[str, Any]:
        return summarize_qwen35_dotcache_fit(self.model)


@dataclass(slots=True)
class Qwen35TextHarness:
    model: Any
    tokenizer: Any | None
    adapter: Qwen35TextModelAdapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | None = None,
        torch_dtype: str = "float16",
    ) -> "Qwen35TextHarness":
        model, tokenizer = load_qwen35_text_only_from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        adapter = Qwen35TextModelAdapter(model=model)
        return cls(model=model, tokenizer=tokenizer, adapter=adapter)

    def tokenize_prompt(
        self,
        prompt: str,
        *,
        multimodal_inputs: Any | None = None,
    ) -> tuple[Any, Any]:
        if multimodal_inputs is not None:
            raise _text_only_error()
        if self.tokenizer is None:
            raise ValueError("tokenizer is unavailable for text prompt input")
        if not isinstance(prompt, str):
            raise ValueError("Qwen3.5 v1 expects a plain text prompt string")
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.adapter.device)
        attention_mask = encoded["attention_mask"].to(self.adapter.device)
        return input_ids, attention_mask

    def generate_greedy(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        max_new_tokens: int = 8,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_text_generation_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            tokenizer=self.tokenizer,
            multimodal_inputs=multimodal_inputs,
        )

    def evaluate_loss(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        prefix_length: int,
        eval_steps: int,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_text_loss_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_length=prefix_length,
            eval_steps=eval_steps,
            tokenizer=self.tokenizer,
            multimodal_inputs=multimodal_inputs,
        )

    def inspect_hybrid_state(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return inspect_qwen35_hybrid_state(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            multimodal_inputs=multimodal_inputs,
        )


def _normalize_text_inputs(
    adapter: Qwen35TextModelAdapter,
    *,
    prompt: str | None,
    input_ids,
    attention_mask,
    tokenizer,
    multimodal_inputs: Any | None,
) -> tuple[Any, Any]:
    if multimodal_inputs is not None:
        raise _text_only_error()
    if prompt is not None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when prompt text is provided")
        if not isinstance(prompt, str):
            raise ValueError("Qwen3.5 v1 expects a plain text prompt string")
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
    input_ids = _normalize_input_ids(input_ids, device=adapter.device)
    attention_mask = _ensure_attention_mask(input_ids, attention_mask, device=adapter.device)
    return input_ids, attention_mask


def _run_dense_prefill(model, *, input_ids, attention_mask):
    return _run_inference(lambda: model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True))


def _run_dense_decode_step(
    model,
    *,
    decode_input_ids,
    attention_mask,
    past_key_values,
    cache_position,
):
    return _run_inference(
        lambda: model(
            input_ids=decode_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            cache_position=cache_position,
        )
    )


def run_qwen35_text_generation_harness(
    model,
    adapter: Qwen35TextModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    max_new_tokens: int = 8,
    tokenizer=None,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_mode("dense")
    input_ids, attention_mask = _normalize_text_inputs(
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        multimodal_inputs=multimodal_inputs,
    )

    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=input_ids.device,
    )
    prefill_cache_bytes = _hybrid_cache_nbytes(prefill_outputs.past_key_values)
    generated_ids: list[int] = []
    dense_decode_ms_total = 0.0
    final_past_key_values = prefill_outputs.past_key_values

    if max_new_tokens > 0:
        current_input_ids = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(int(current_input_ids.item()))
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
        past_key_values = prefill_outputs.past_key_values

        for _ in range(max(max_new_tokens - 1, 0)):
            outputs, step_ms = _timed_call(
                lambda: _run_dense_decode_step(
                    model,
                    decode_input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                ),
                device=input_ids.device,
            )
            dense_decode_ms_total += step_ms
            past_key_values = outputs.past_key_values
            final_past_key_values = outputs.past_key_values
            current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(int(current_input_ids.item()))
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                dim=1,
            )
            cache_position = cache_position + 1

    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": max(max_new_tokens - 1, 0),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_decode_ms_total / max(max_new_tokens - 1, 1)) if max_new_tokens > 1 else 0.0,
        "dense_generated_ids": list(generated_ids),
        "dense_prefill_cache_bytes": int(prefill_cache_bytes),
        "dense_final_cache_bytes": int(_hybrid_cache_nbytes(final_past_key_values)),
        "cache_metric_kind": "hybrid_cache_bytes",
        "text_only": True,
        "dotcache_ready": False,
        "runtime_mode": "dense",
        "uses_native_qwen35_class": True,
    }
    result.update(adapter.hybrid_block_summary())
    decoded_text = _decode_text(tokenizer, generated_ids)
    if decoded_text is not None:
        result["dense_text"] = decoded_text
    return result


def run_qwen35_text_loss_harness(
    model,
    adapter: Qwen35TextModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    prefix_length: int,
    eval_steps: int,
    tokenizer=None,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_mode("dense")
    input_ids, attention_mask = _normalize_text_inputs(
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        multimodal_inputs=multimodal_inputs,
    )
    if prefix_length <= 0 or prefix_length >= int(input_ids.shape[1]):
        raise ValueError("prefix_length must be in [1, sequence_length)")
    available_eval_steps = int(input_ids.shape[1]) - prefix_length
    if eval_steps <= 0 or eval_steps > available_eval_steps:
        raise ValueError("eval_steps must be positive and fit inside the provided sequence after prefix_length")

    prefix_input_ids = input_ids[:, :prefix_length]
    prefix_attention_mask = attention_mask[:, :prefix_length]
    continuation_ids = input_ids[:, prefix_length : prefix_length + eval_steps]

    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=prefix_input_ids, attention_mask=prefix_attention_mask),
        device=input_ids.device,
    )
    logits_list = [prefill_outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu()]
    past_key_values = prefill_outputs.past_key_values
    current_attention_mask = torch.cat(
        [prefix_attention_mask, torch.ones((1, 1), dtype=prefix_attention_mask.dtype, device=prefix_attention_mask.device)],
        dim=1,
    )
    cache_position = torch.tensor([prefix_input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
    dense_decode_ms_total = 0.0

    for step_index in range(max(eval_steps - 1, 0)):
        decode_input_ids = continuation_ids[:, step_index : step_index + 1]
        outputs, step_ms = _timed_call(
            lambda: _run_dense_decode_step(
                model,
                decode_input_ids=decode_input_ids,
                attention_mask=current_attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            ),
            device=input_ids.device,
        )
        dense_decode_ms_total += step_ms
        past_key_values = outputs.past_key_values
        logits_list.append(outputs.logits[:, -1, :].detach().to(dtype=torch.float32).cpu())
        current_attention_mask = torch.cat(
            [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
            dim=1,
        )
        cache_position = cache_position + 1

    logits = torch.cat(logits_list, dim=0).numpy()
    target_tokens = continuation_ids[0, : logits.shape[0]].detach().cpu().numpy()
    max_logits = logits.max(axis=-1, keepdims=True)
    stabilized = logits - max_logits
    log_probs = stabilized - torch.from_numpy(stabilized).exp().sum(dim=-1, keepdim=True).log().numpy()
    token_losses = -log_probs[range(target_tokens.shape[0]), target_tokens]
    mean_loss = float(token_losses.mean())
    perplexity = float(math.exp(min(mean_loss, 50.0)))
    predictions = logits.argmax(axis=-1)
    result = {
        "sequence_length": int(input_ids.shape[1]),
        "prefix_length": int(prefix_length),
        "eval_steps": int(eval_steps),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_decode_ms_total / max(eval_steps - 1, 1)) if eval_steps > 1 else 0.0,
        "dense_teacher_forced_loss": mean_loss,
        "dense_teacher_forced_perplexity": perplexity,
        "dense_teacher_forced_target_match_rate": float((predictions == target_tokens).mean()),
        "dense_prefill_cache_bytes": int(_hybrid_cache_nbytes(prefill_outputs.past_key_values)),
        "dense_final_cache_bytes": int(_hybrid_cache_nbytes(past_key_values)),
        "cache_metric_kind": "hybrid_cache_bytes",
        "text_only": True,
        "dotcache_ready": False,
        "runtime_mode": "dense",
        "uses_native_qwen35_class": True,
    }
    result.update(adapter.hybrid_block_summary())
    return result


def inspect_qwen35_hybrid_state(
    model,
    adapter: Qwen35TextModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    multimodal_inputs: Any | None = None,
) -> dict[str, Any]:
    _require_qwen35_model_class()
    adapter.set_mode("dense")
    input_ids, attention_mask = _normalize_text_inputs(
        adapter,
        prompt=prompt,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        multimodal_inputs=multimodal_inputs,
    )
    prefill_outputs, prefill_ms = _timed_call(
        lambda: _run_dense_prefill(model, input_ids=input_ids, attention_mask=attention_mask),
        device=input_ids.device,
    )
    cache = prefill_outputs.past_key_values
    result = {
        "prompt_length": int(input_ids.shape[1]),
        "prefill_ms": float(prefill_ms),
        "text_only": True,
        "dotcache_ready": False,
        "runtime_mode": "dense",
        "uses_native_qwen35_class": True,
    }
    result.update(adapter.hybrid_block_summary())
    result.update(summarize_qwen35_hybrid_state(cache, model))
    result.update(adapter.hybrid_fit_summary())
    return result


__all__ = [
    "Qwen35TextHarness",
    "Qwen35TextModelAdapter",
    "inspect_qwen35_hybrid_state",
    "load_qwen35_text_only_from_pretrained",
    "run_qwen35_text_generation_harness",
    "run_qwen35_text_loss_harness",
    "summarize_qwen35_dotcache_fit",
    "summarize_qwen35_hybrid_state",
]
