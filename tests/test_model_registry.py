from dotcache.model_registry import get_model_spec, list_model_specs


def test_model_registry_contains_expected_keys() -> None:
    specs = {spec.key: spec for spec in list_model_specs()}
    assert "tinyllama_hf" in specs
    assert "smollm2_360m_hf" in specs
    assert "llama32_3b_hf" in specs
    assert "qwen25_3b_hf" in specs
    assert "llama32_3b_gguf" in specs


def test_model_registry_records_dotcache_ready_models_as_harness_backed() -> None:
    spec = get_model_spec("llama32_3b_hf")
    assert spec.dotcache_ready is True
    assert spec.benchmark_harness == "llama_compare"
    assert spec.runtime == "dotcache_hf"
    assert spec.tokenizer_model_id == spec.model_id


def test_model_registry_marks_qwen35_as_reference_only() -> None:
    spec = get_model_spec("qwen35_4b_hf")
    assert spec.dotcache_ready is False
    assert spec.local_tier == "reference_only"
    assert spec.family == "qwen3_5_hybrid"


def test_model_registry_marks_gguf_lanes_as_external_runnable() -> None:
    spec = get_model_spec("llama32_3b_gguf")
    assert spec.runtime == "llama_cpp"
    assert spec.benchmark_harness == "gguf_external"
    assert spec.tokenizer_model_id == "meta-llama/Llama-3.2-3B-Instruct"
