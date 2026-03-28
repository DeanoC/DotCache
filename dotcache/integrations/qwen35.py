from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .llama import (
    _default_model_device,
    _ensure_attention_mask,
    _normalize_input_ids,
    _require_transformers,
    _run_inference,
    LlamaReplayRecord,
    _timed_call,
    resolve_hf_auth_kwargs,
    transformers_available,
)

if transformers_available():
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    import torch.nn as nn
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_mod
else:  # pragma: no cover - exercised in environments without transformers
    torch = None
    AutoTokenizer = None
    Qwen3_5ForConditionalGeneration = None
    nn = object  # type: ignore[assignment]
    qwen35_mod = None


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


class DotCacheQwen35AttentionSubset(nn.Module):
    def __init__(self, base_attention: nn.Module, adapter: "Qwen35AttentionSubsetModelAdapter") -> None:
        super().__init__()
        self.base_attention = base_attention
        self.adapter = adapter
        self.layer_idx = int(base_attention.layer_idx)
        self.config = base_attention.config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.adapter.capture_enabled:
            return self.base_attention(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        return self._forward_dense_with_capture(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for the Qwen3.5 attention subset capture path")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.base_attention.head_dim)
        query_states, gate = torch.chunk(
            self.base_attention.q_proj(hidden_states).view(*input_shape, -1, self.base_attention.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.base_attention.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.base_attention.k_norm(self.base_attention.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.base_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = qwen35_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states, gate

    def _forward_dense_with_capture(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        query_states, key_states, value_states, gate = self._project_qkv(hidden_states, position_embeddings)
        fresh_key_states = key_states
        fresh_value_states = value_states

        if past_key_values is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = qwen35_mod.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.base_attention.config._attn_implementation,
            qwen35_mod.eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self.base_attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.base_attention.attention_dropout,
            scaling=self.base_attention.scaling,
            **kwargs,
        )
        gated_context = attn_output.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
        projected_output = self.base_attention.o_proj(gated_context)

        if self.adapter.capture_enabled and tuple(hidden_states.shape[:2]) == (1, 1):
            token_index = self.adapter.current_token_index(cache_position)
            self.adapter.record_replay(
                LlamaReplayRecord(
                    step_index=self.adapter.capture_step_index,
                    layer_id=self.layer_idx,
                    token_index=token_index,
                    query_states=query_states[0, :, 0, :].detach().to(dtype=torch.float32).cpu().numpy(),
                    key_states=fresh_key_states[0, :, 0, :].detach().to(dtype=torch.float32).cpu().numpy(),
                    value_states=fresh_value_states[0, :, 0, :].detach().to(dtype=torch.float32).cpu().numpy(),
                    context_states=gated_context[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                    output_states=projected_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                )
            )
        return projected_output, attn_weights


@dataclass(slots=True)
class Qwen35AttentionSubsetModelAdapter(Qwen35TextModelAdapter):
    capture_enabled: bool = False
    capture_step_index: int = -1
    _pending_records: list[LlamaReplayRecord] = field(default_factory=list, init=False, repr=False)
    _wrappers: list[DotCacheQwen35AttentionSubset] = field(default_factory=list, init=False, repr=False)
    _current_token_index_override: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._install_wrappers()

    def _install_wrappers(self) -> None:
        text_model = _qwen35_text_model(self.model)
        layers = getattr(text_model, "layers", None)
        if layers is None:
            return
        layer_types = _hybrid_layer_types(self.model)
        for layer_id, layer in enumerate(layers[: len(layer_types)]):
            if layer_types[layer_id] != "full_attention" or not hasattr(layer, "self_attn"):
                continue
            wrapper = DotCacheQwen35AttentionSubset(layer.self_attn, self)
            layer.self_attn = wrapper
            self._wrappers.append(wrapper)

    def begin_capture_step(self, step_index: int) -> None:
        self.capture_step_index = int(step_index)
        self.capture_enabled = True
        self._pending_records = []

    def end_capture_step(self) -> list[LlamaReplayRecord]:
        records = list(self._pending_records)
        self.capture_step_index = -1
        self.capture_enabled = False
        self._pending_records = []
        return records

    def record_replay(self, record: LlamaReplayRecord) -> None:
        if self.capture_step_index < 0:
            return
        self._pending_records.append(record)

    def current_token_index(self, cache_position) -> int:
        if self._current_token_index_override is not None:
            return self._current_token_index_override
        if cache_position is None:
            raise ValueError("cache_position is required for Qwen3.5 attention-subset capture")
        token_positions = cache_position.reshape(-1)
        if token_positions.numel() != 1:
            raise ValueError("Qwen3.5 attention-subset capture requires a single cache_position per decode step")
        return int(token_positions.item())

    def set_current_token_index(self, token_index: int | None) -> None:
        self._current_token_index_override = None if token_index is None else int(token_index)

    def attention_subset_layer_ids(self) -> list[int]:
        return self.hybrid_fit_summary()["attention_candidate_layer_ids"]


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


@dataclass(slots=True)
class Qwen35AttentionSubsetHarness:
    model: Any
    tokenizer: Any | None
    adapter: Qwen35AttentionSubsetModelAdapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: str | None = None,
        torch_dtype: str = "float16",
    ) -> "Qwen35AttentionSubsetHarness":
        model, tokenizer = load_qwen35_text_only_from_pretrained(
            model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        adapter = Qwen35AttentionSubsetModelAdapter(model=model)
        return cls(model=model, tokenizer=tokenizer, adapter=adapter)

    def tokenize_prompt(
        self,
        prompt: str,
        *,
        multimodal_inputs: Any | None = None,
    ) -> tuple[Any, Any]:
        helper = Qwen35TextHarness(model=self.model, tokenizer=self.tokenizer, adapter=self.adapter)
        return helper.tokenize_prompt(prompt, multimodal_inputs=multimodal_inputs)

    def run_attention_subset_replay(
        self,
        *,
        prompt: str | None = None,
        input_ids=None,
        attention_mask=None,
        decode_steps: int = 4,
        multimodal_inputs: Any | None = None,
    ) -> dict[str, Any]:
        return run_qwen35_attention_subset_replay_harness(
            self.model,
            self.adapter,
            prompt=prompt,
            input_ids=input_ids,
            attention_mask=attention_mask,
            tokenizer=self.tokenizer,
            decode_steps=decode_steps,
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


def run_qwen35_attention_subset_replay_harness(
    model,
    adapter: Qwen35AttentionSubsetModelAdapter,
    *,
    prompt: str | None = None,
    input_ids=None,
    attention_mask=None,
    tokenizer=None,
    decode_steps: int = 4,
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
    if decode_steps <= 0:
        per_step_records: list[list[LlamaReplayRecord]] = []
        dense_decode_ms_total = 0.0
    else:
        current_input_ids = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )
        cache_position = torch.tensor([input_ids.shape[1]], dtype=torch.long, device=input_ids.device)
        past_key_values = prefill_outputs.past_key_values
        per_step_records = []
        dense_decode_ms_total = 0.0
        for step_index in range(decode_steps):
            adapter.begin_capture_step(step_index)
            adapter.set_current_token_index(int(input_ids.shape[1] + step_index))
            try:
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
            finally:
                adapter.set_current_token_index(None)
            dense_decode_ms_total += step_ms
            per_step_records.append(adapter.end_capture_step())
            past_key_values = outputs.past_key_values
            current_input_ids = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            current_attention_mask = torch.cat(
                [current_attention_mask, torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)],
                dim=1,
            )
            cache_position = cache_position + 1

    records = [record for step_records in per_step_records for record in step_records]
    per_layer_counts: dict[int, int] = {}
    per_layer_shapes: dict[int, dict[str, list[int]]] = {}
    for record in records:
        per_layer_counts[record.layer_id] = per_layer_counts.get(record.layer_id, 0) + 1
        per_layer_shapes.setdefault(
            record.layer_id,
            {
                "query_states": list(record.query_states.shape),
                "key_states": list(record.key_states.shape),
                "value_states": list(record.value_states.shape),
                "context_states": list(record.context_states.shape),
            },
        )

    result = {
        "prompt_length": int(input_ids.shape[1]),
        "decode_steps": int(decode_steps),
        "prefill_ms": float(prefill_ms),
        "dense_decode_ms_per_step": float(dense_decode_ms_total / max(decode_steps, 1)) if decode_steps > 0 else 0.0,
        "attention_subset_layer_ids": adapter.attention_subset_layer_ids(),
        "attention_subset_capture_layer_count": len(adapter.attention_subset_layer_ids()),
        "attention_subset_capture_record_count": len(records),
        "attention_subset_capture_counts_by_layer": {str(layer_id): count for layer_id, count in sorted(per_layer_counts.items())},
        "attention_subset_capture_shapes_by_layer": {str(layer_id): shapes for layer_id, shapes in sorted(per_layer_shapes.items())},
        "text_only": True,
        "dotcache_ready": False,
        "runtime_mode": "dense_attention_subset_capture",
        "uses_native_qwen35_class": True,
    }
    result.update(adapter.hybrid_block_summary())
    result.update(adapter.hybrid_fit_summary())
    return result


__all__ = [
    "Qwen35AttentionSubsetHarness",
    "Qwen35AttentionSubsetModelAdapter",
    "Qwen35TextHarness",
    "Qwen35TextModelAdapter",
    "inspect_qwen35_hybrid_state",
    "load_qwen35_text_only_from_pretrained",
    "run_qwen35_attention_subset_replay_harness",
    "run_qwen35_text_generation_harness",
    "run_qwen35_text_loss_harness",
    "summarize_qwen35_dotcache_fit",
    "summarize_qwen35_hybrid_state",
]
