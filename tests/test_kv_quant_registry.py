from dotcache.kv_quant_registry import get_kv_quant_baseline, list_kv_quant_baselines


def test_kv_quant_registry_contains_expected_rows() -> None:
    rows = {row.key: row for row in list_kv_quant_baselines()}
    assert "dense_kv" in rows
    assert "dotcache_policy_adaptive" in rows
    assert "turboquant" in rows
    assert "kvquant" in rows
    assert "kivi" in rows
    assert "qjl" in rows


def test_kv_quant_registry_marks_local_and_reference_rows_distinctly() -> None:
    adaptive = get_kv_quant_baseline("dotcache_policy_adaptive")
    assert adaptive.kind == "local_mechanism"
    assert adaptive.local_status == "implemented"

    turboquant = get_kv_quant_baseline("turboquant")
    assert turboquant.kind == "external_runner"
    assert turboquant.local_status == "external"

    kvquant = get_kv_quant_baseline("kvquant")
    assert kvquant.kind == "paper_reference_only"
    assert kvquant.local_status == "paper_only"
