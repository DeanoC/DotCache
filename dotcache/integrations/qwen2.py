from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import DotCacheConfig
from .llama import (
    LlamaDotCacheHarness,
    LlamaDotCacheModelAdapter,
    LlamaReplayRecord,
    _default_model_device,
    _require_transformers,
    _timed_call,
    _torch_backend_matches_device,
    run_llama_generation_harness,
    run_llama_loss_harness,
    run_llama_replay_harness,
    transformers_available,
)

if transformers_available():
    import torch
    import torch.nn as nn
    import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
    from transformers import AutoModelForCausalLM, AutoTokenizer
else:  # pragma: no cover - exercised in environments without transformers
    torch = None
    nn = object  # type: ignore[assignment]
    qwen2_mod = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class DotCacheQwen2Attention(nn.Module):
    def __init__(self, base_attention: nn.Module, adapter: "Qwen2DotCacheModelAdapter") -> None:
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
        if self.adapter.mode == "dense" and not self.adapter.capture_enabled:
            return self.base_attention(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        if self.adapter.mode == "dense":
            return self._forward_dense_with_capture(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        return self._forward_dotcache(
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            raise ValueError("position_embeddings are required for the Qwen2 attention path")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.base_attention.head_dim)
        query_states = self.base_attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.base_attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.base_attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = qwen2_mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)
        return query_states, key_states, value_states

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
        query_states, key_states, value_states = self._project_qkv(hidden_states, position_embeddings)
        fresh_key_states = key_states
        fresh_value_states = value_states

        if past_key_values is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = qwen2_mod.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.base_attention.config._attn_implementation,
            qwen2_mod.eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self.base_attention,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.base_attention.attention_dropout,
            scaling=self.base_attention.scaling,
            sliding_window=self.base_attention.sliding_window,
            **kwargs,
        )
        reshaped_output = attn_output.reshape(*input_shape, -1).contiguous()
        projected_output = self.base_attention.o_proj(reshaped_output)

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
                    context_states=attn_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                    output_states=projected_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                )
            )
        return projected_output, attn_weights

    def _forward_dotcache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        del attention_mask, kwargs
        if past_key_values is not None:
            raise ValueError("DotCache decode mode manages its own KV cache and requires past_key_values=None")
        if tuple(hidden_states.shape[:2]) != (1, 1):
            raise ValueError("DotCache decode mode only supports batch=1 and query_len=1")
        token_index = self.adapter.current_token_index(cache_position)
        (query_states, key_states, value_states), qkv_ms = _timed_call(
            lambda: self._project_qkv(hidden_states, position_embeddings),
            device=hidden_states.device,
        )
        query_step = query_states[0, :, 0, :].detach().to(dtype=torch.float32)
        key_step = key_states[0].detach().to(dtype=torch.float32)
        value_step = value_states[0].detach().to(dtype=torch.float32)
        self.adapter.record_layer_runtime(self.layer_idx, qkv_projection_ms=qkv_ms)

        _, append_ms = _timed_call(
            lambda: self.adapter.model_kv_cache.append_step_torch(
                self.layer_idx,
                key_step,
                value_step,
                token_index,
                trace=self.adapter.active_trace,
            )
            if _torch_backend_matches_device(self.adapter.backend, hidden_states.device.type)
            else self.adapter.model_kv_cache.append_step(
                self.layer_idx,
                key_step.cpu().numpy(),
                value_step.cpu().numpy(),
                token_index,
                trace=self.adapter.active_trace,
            ),
            device=hidden_states.device,
        )
        self.adapter.append_runtime_ms_total += append_ms

        context_states, decode_ms = _timed_call(
            lambda: self.adapter.model_kv_cache.decode_layer_torch(
                self.layer_idx,
                query_step,
                self.adapter.q_head_to_kv_head,
                query_scale=float(self.base_attention.scaling),
                trace=self.adapter.active_trace,
            )
            if _torch_backend_matches_device(self.adapter.backend, hidden_states.device.type)
            else self.adapter.model_kv_cache.decode_layer(
                self.layer_idx,
                query_step.detach().cpu().numpy(),
                self.adapter.q_head_to_kv_head,
                query_scale=float(self.base_attention.scaling),
                trace=self.adapter.active_trace,
            ),
            device=hidden_states.device,
        )
        self.adapter.decode_runtime_ms_total += decode_ms

        def _project_output():
            local_context_states = context_states
            if not torch.is_tensor(local_context_states):
                local_context_states = torch.as_tensor(local_context_states, dtype=torch.float32, device=hidden_states.device)
            context_tensor = local_context_states.to(dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(0)
            return self.base_attention.o_proj(context_tensor.reshape(1, 1, -1).contiguous())

        projected_output, output_projection_ms = _timed_call(_project_output, device=hidden_states.device)
        self.adapter.record_layer_runtime(
            self.layer_idx,
            append_ms=append_ms,
            decode_ms=decode_ms,
            output_projection_ms=output_projection_ms,
        )

        if self.adapter.capture_enabled:
            self.adapter.record_replay(
                LlamaReplayRecord(
                    step_index=self.adapter.capture_step_index,
                    layer_id=self.layer_idx,
                    token_index=token_index,
                    query_states=query_step.detach().cpu().numpy(),
                    key_states=key_step[:, 0, :].detach().cpu().numpy(),
                    value_states=value_step[:, 0, :].detach().cpu().numpy(),
                    context_states=context_states.detach().cpu().numpy().astype(np.float32, copy=False),
                    output_states=projected_output[0, 0].detach().to(dtype=torch.float32).cpu().numpy(),
                )
            )
        return projected_output, None


class Qwen2DotCacheModelAdapter(LlamaDotCacheModelAdapter):
    def _install_wrappers(self) -> None:
        for layer in self.model.model.layers[: self.model.config.num_hidden_layers]:
            wrapper = DotCacheQwen2Attention(layer.self_attn, self)
            layer.self_attn = wrapper
            self._wrappers.append(wrapper)


@dataclass(slots=True)
class Qwen2DotCacheHarness(LlamaDotCacheHarness):
    adapter: Qwen2DotCacheModelAdapter

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        dotcache_config: DotCacheConfig,
        *,
        backend: str = "auto",
        device: str | None = None,
        torch_dtype: str = "float16",
    ) -> "Qwen2DotCacheHarness":
        _require_transformers()
        dtype = getattr(torch, torch_dtype)
        resolved_device = _default_model_device() if device is None else device
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model.to(resolved_device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        adapter = Qwen2DotCacheModelAdapter(model, dotcache_config, backend=backend)
        return cls(model=model, tokenizer=tokenizer, adapter=adapter)


run_qwen2_replay_harness = run_llama_replay_harness
run_qwen2_generation_harness = run_llama_generation_harness
run_qwen2_loss_harness = run_llama_loss_harness

