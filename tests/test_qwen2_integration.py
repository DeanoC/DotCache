import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from dotcache.backends import mps_available
from dotcache.config import DotCacheConfig
from dotcache.integrations.qwen2 import (
    Qwen2DotCacheModelAdapter,
    run_qwen2_generation_harness,
    run_qwen2_loss_harness,
    run_qwen2_replay_harness,
)

Qwen2Config = transformers.Qwen2Config
Qwen2ForCausalLM = transformers.Qwen2ForCausalLM

requires_mps = pytest.mark.skipif(not mps_available(), reason="torch_mps is unavailable")


def _tiny_qwen2_model(*, device: str = "cpu"):
    config = Qwen2Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=128,
        max_position_embeddings=64,
    )
    model = Qwen2ForCausalLM(config)
    model.to(device)
    model.eval()
    dotcache_config = DotCacheConfig(head_dim=32, group_size=32, bits_k=4, bits_v=4, tokens_per_page=4)
    adapter = Qwen2DotCacheModelAdapter(model, dotcache_config, backend="torch_mps" if device == "mps" else "cpu_ref")
    return model, adapter


def test_qwen2_replay_harness_runs_on_tiny_random_model() -> None:
    model, adapter = _tiny_qwen2_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

    result = run_qwen2_replay_harness(model, adapter, input_ids=input_ids, decode_steps=3)

    assert result["decode_steps"] == 3
    assert np.isfinite(result["replay_context_max_abs_error"])
    assert np.isfinite(result["teacher_forced_logit_max_abs_error"])


def test_qwen2_generation_harness_runs_on_tiny_random_model() -> None:
    model, adapter = _tiny_qwen2_model()
    input_ids = torch.tensor([[7, 8, 9, 10, 11, 12]], dtype=torch.long)

    result = run_qwen2_generation_harness(model, adapter, input_ids=input_ids, max_new_tokens=4)

    assert len(result["dense_generated_ids"]) == 4
    assert len(result["dotcache_generated_ids"]) == 4
    assert np.isfinite(result["decode_ms_per_step"])
    assert np.isfinite(result["teacher_forced_logit_max_abs_error"])


def test_qwen2_loss_harness_runs_on_tiny_random_model() -> None:
    model, adapter = _tiny_qwen2_model()
    input_ids = torch.tensor([[7, 8, 9, 10, 11, 12, 13, 14]], dtype=torch.long)

    result = run_qwen2_loss_harness(model, adapter, input_ids=input_ids, prefix_length=4, eval_steps=4)

    assert result["sequence_length"] == 8
    assert result["prefix_length"] == 4
    assert result["eval_steps"] == 4
    assert np.isfinite(result["dense_teacher_forced_loss"])
    assert np.isfinite(result["dotcache_teacher_forced_loss"])
    assert np.isfinite(result["teacher_forced_loss_delta"])
    assert 0.0 <= result["teacher_forced_token_agreement_rate"] <= 1.0


@requires_mps
def test_qwen2_generation_harness_runs_on_mps_tiny_random_model() -> None:
    model, adapter = _tiny_qwen2_model(device="mps")
    input_ids = torch.tensor([[2, 4, 6, 8]], dtype=torch.long, device="mps")

    result = run_qwen2_generation_harness(model, adapter, input_ids=input_ids, max_new_tokens=3)

    assert len(result["dotcache_generated_ids"]) == 3
    assert result["resident_bytes"] >= 0
