import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_arxiv_v1_sweep", REPO / "benchmarks" / "run_arxiv_v1_sweep.py",
)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def test_paper_sweep_targets_current_branch_and_output_dir():
    assert sweep.BRANCH == "port-to-paper-20260424"
    assert sweep.OUT_DIR.name == "paper_v1_20260424"
    assert "arxiv_v1_20260420" not in str(sweep.OUT_DIR)


def test_paper_sweep_cells_are_paired_summary_matrix():
    cells = sweep.build_cells()
    assert len(cells) == 12
    assert cells[0] == {"idx": "01", "bench": "pg19", "ctx": 4096}
    assert cells[-1] == {"idx": "12", "bench": "ruler", "ctx": 32768}


def test_paper_sweep_pg19_sample_counts_match_root_table(tmp_path: Path):
    args_4k = sweep._cli_for_pg19(4096, tmp_path / "pg19_4k.json", smoke=False)
    args_32k = sweep._cli_for_pg19(32768, tmp_path / "pg19_32k.json", smoke=False)
    assert args_4k[args_4k.index("--num-chunks") + 1] == "5"
    assert args_32k[args_32k.index("--num-chunks") + 1] == "20"
    assert args_4k[args_4k.index("--telemetry-mode") + 1] == "summary"


def test_paper_sweep_niah_trial_counts_match_root_table(tmp_path: Path):
    args_4k = sweep._cli_for_niah(4096, tmp_path / "niah_4k.json", smoke=False)
    args_8k = sweep._cli_for_niah(8192, tmp_path / "niah_8k.json", smoke=False)
    assert args_4k[args_4k.index("--needles") + 1] == "3"
    assert args_8k[args_8k.index("--needles") + 1] == "10"


def test_paper_sweep_certified_flags_match_section7(tmp_path: Path):
    args = sweep._cli_for_ruler(4096, tmp_path / "ruler.json", smoke=False)
    expected_pairs = {
        "--v-tolerance": "0.05",
        "--group-size": "16",
        "--tau-cov": "0.995",
        "--k-min": "2",
        "--k-max": "128",
        "--ranking-r": "1",
        "--ranking-fallback-mode": "full",
        "--eps-guard": "0.01",
        "--exploration-rate": "0.02",
        "--rung1-threshold": "0.02",
        "--rung1-multiplier": "2.0",
    }
    for flag, value in expected_pairs.items():
        assert args[args.index(flag) + 1] == value
    for flag in ("--use-int4-values", "--ranking-fallback"):
        assert flag in args
    assert "--score-consistency-check" not in args


def test_paper_sweep_wraps_niah_inferential_stats():
    native = {
        "dense_accuracy": 0.93,
        "certified_accuracy": 0.91,
        "critical_failures": 4,
        "paired_stats": {
            "n": 100,
            "delta_pp": -2.0,
            "bootstrap_ci_pp_lo": -7.0,
            "bootstrap_ci_pp_hi": 3.0,
            "mcnemar_p": 0.38,
        },
    }
    quality = sweep._quality("niah", native)
    assert quality["delta"] == pytest.approx(-0.02)
    assert quality["n"] == 100
    assert quality["delta_pp"] == pytest.approx(-2.0)
    assert quality["bootstrap_ci_pp"] == [-7.0, 3.0]
    assert quality["mcnemar_p"] == pytest.approx(0.38)
