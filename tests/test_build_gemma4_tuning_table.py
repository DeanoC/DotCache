import importlib.util
import json
import sys
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_build_gemma4_tuning_table_writes_expected_schema(tmp_path: Path) -> None:
    module = _load_module("build_gemma4_tuning_table", "scripts/build_gemma4_tuning_table.py")

    output_path = tmp_path / "gemma4_text_tuning_table.json"
    written_path = module.write_tuning_table(output_path)

    assert written_path == output_path
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["model_family"] == "gemma4_text"
    assert len(payload["rules"]) == 7
    assert payload["rules"][0]["preset"]["profile"] == "balanced_layer0_8"
    assert payload["rules"][-1]["preset"]["exact_value_layers"] == [0, 4, 8, 9, 14]


def test_run_gemma4_profile_sweep_refreshes_tuning_table(monkeypatch) -> None:
    module = _load_module("run_gemma4_profile_sweep", "scripts/run_gemma4_profile_sweep.py")
    called: dict[str, object] = {}

    def _fake_write_tuning_table():
        path = Path("/tmp/fake-gemma4-text-tuning-table.json")
        called["path"] = path
        return path

    monkeypatch.setattr(module, "write_tuning_table", _fake_write_tuning_table)

    refreshed = module._refresh_tuning_table()

    assert refreshed == Path("/tmp/fake-gemma4-text-tuning-table.json")
    assert called["path"] == refreshed
