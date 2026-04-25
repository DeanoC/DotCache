import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_experiment_v2_sweep", REPO / "benchmarks" / "run_experiment_v2_sweep.py",
)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def test_v2_default_cells_are_tier1_plus_pg19_128k():
    cells = sweep.build_cells(include_tier2=True, include_tier3=False, include_tier4=False)
    assert len(cells) == 10
    assert {c["ctx"] for c in cells if c["tier"] == 1} == {8192, 32768, 65536}
    assert cells[-1]["bench"] == "pg19"
    assert cells[-1]["ctx"] == 131072
    assert cells[-1]["chunks"] == 5


def test_v2_pg19_uses_20_chunks_except_128k(tmp_path: Path):
    cells = sweep.build_cells()
    pg19_8k = next(c for c in cells if c["bench"] == "pg19" and c["ctx"] == 8192)
    pg19_128k = next(c for c in cells if c["bench"] == "pg19" and c["ctx"] == 131072)
    args_8k = sweep._cli_for_cell(pg19_8k, tmp_path / "8k.json", smoke=False)
    args_128k = sweep._cli_for_cell(pg19_128k, tmp_path / "128k.json", smoke=False)
    assert args_8k[args_8k.index("--num-chunks") + 1] == "20"
    assert args_128k[args_128k.index("--num-chunks") + 1] == "5"


def test_v2_niah_uses_100_trials_via_10_needles(tmp_path: Path):
    cell = next(c for c in sweep.build_cells() if c["bench"] == "niah" and c["ctx"] == 32768)
    args = sweep._cli_for_cell(cell, tmp_path / "niah.json", smoke=False)
    assert args[args.index("--needles") + 1] == "10"


def test_v2_common_cert_flags_match_experiment_spec(tmp_path: Path):
    cell = next(c for c in sweep.build_cells() if c["bench"] == "ruler" and c["ctx"] == 65536)
    args = sweep._cli_for_cell(cell, tmp_path / "ruler.json", smoke=False)
    expected_pairs = {
        "--model": "NousResearch/Meta-Llama-3.1-8B",
        "--v-tolerance": "0.05",
        "--group-size": "16",
        "--tau-cov": "0.995",
        "--k-max": "128",
        "--fp16-value-cache-blocks": "64",
    }
    for flag, value in expected_pairs.items():
        assert args[args.index(flag) + 1] == value
    for flag in ("--use-int4-values", "--ranking-fallback", "--score-consistency-check"):
        assert flag in args


def test_v2_pg19_wrapped_summary_uses_paired_delta_stats():
    native = {
        "dense": {"perplexity": 10.0},
        "certified": {
            "perplexity": 10.1,
            "paired_delta_stats": {"n_chunks": 20, "mean_delta_ppl": 0.1},
            "telemetry": {"e_key_step_max": 0.04, "e_val_step_max": 0.01},
        },
        "delta": 0.1,
        "ratio": 1.01,
    }
    summary = sweep._results_summary("pg19", native)
    assert summary["paired_delta_stats"]["n_chunks"] == 20
    assert summary["ratio"] == 1.01
    telemetry = sweep._native_telemetry("pg19", native)
    assert telemetry["e_key_max"] == 0.04
    assert telemetry["e_val_max"] == 0.01
