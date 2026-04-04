from dotcache.model_registry import get_model_spec, list_model_specs


def test_model_registry_contains_expected_keys() -> None:
    specs = {spec.key: spec for spec in list_model_specs()}
    assert "tinyllama_hf" in specs
    assert "smollm2_360m_hf" in specs
    assert "smollm2_1p7b_hf" in specs
    assert "llama32_3b_hf" in specs
    assert "qwen25_1p5b_hf" in specs
    assert "qwen25_3b_hf" in specs
    assert "qwen25_7b_hf" in specs
    assert "qwen35_0p8b_hf" in specs
    assert "llama32_3b_gguf" in specs
    assert "qwen25_7b_gguf" in specs


def test_model_registry_records_dotcache_ready_models_as_harness_backed() -> None:
    spec = get_model_spec("llama32_3b_hf")
    assert spec.dotcache_ready is True
    assert spec.benchmark_harness == "llama_compare"
    assert spec.runtime == "dotcache_hf"
    assert spec.tokenizer_model_id == spec.model_id

    qwen_spec = get_model_spec("qwen25_3b_hf")
    assert qwen_spec.dotcache_ready is True
    assert qwen_spec.benchmark_harness == "qwen2_compare"
    assert qwen_spec.runtime == "dotcache_hf"

    qwen7b_spec = get_model_spec("qwen25_7b_hf")
    assert qwen7b_spec.dotcache_ready is True
    assert qwen7b_spec.benchmark_harness == "qwen2_compare"
    assert qwen7b_spec.runtime == "dotcache_hf"

    qwen15b_spec = get_model_spec("qwen25_1p5b_hf")
    assert qwen15b_spec.dotcache_ready is True
    assert qwen15b_spec.benchmark_harness == "qwen2_compare"
    assert qwen15b_spec.runtime == "dotcache_hf"

    smollm17b_spec = get_model_spec("smollm2_1p7b_hf")
    assert smollm17b_spec.dotcache_ready is True
    assert smollm17b_spec.benchmark_harness == "llama_compare"
    assert smollm17b_spec.runtime == "dotcache_hf"


def test_model_registry_marks_qwen35_4b_as_local_stretch_lane() -> None:
    spec = get_model_spec("qwen35_4b_hf")
    assert spec.dotcache_ready is False
    assert spec.local_tier == "stretch_here"
    assert spec.family == "qwen3_5_hybrid"
    assert spec.runtime == "transformers"
    assert spec.benchmark_harness == "qwen35_text"


def test_model_registry_marks_qwen35_0p8b_as_runnable_dense_text_lane() -> None:
    spec = get_model_spec("qwen35_0p8b_hf")
    assert spec.dotcache_ready is False
    assert spec.local_tier == "stretch_here"
    assert spec.family == "qwen3_5_hybrid"
    assert spec.runtime == "transformers"
    assert spec.benchmark_harness == "qwen35_text"


def test_model_registry_marks_gguf_lanes_as_external_runnable() -> None:
    spec = get_model_spec("llama32_3b_gguf")
    assert spec.runtime == "llama_cpp"
    assert spec.benchmark_harness == "gguf_external"
    assert spec.tokenizer_model_id == "meta-llama/Llama-3.2-3B-Instruct"

    qwen_spec = get_model_spec("qwen25_7b_gguf")
    assert qwen_spec.runtime == "llama_cpp"
    assert qwen_spec.benchmark_harness == "gguf_external"
    assert qwen_spec.tokenizer_model_id == "Qwen/Qwen2.5-7B-Instruct"
