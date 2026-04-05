import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
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


def test_run_gemma4_apple_smoke_runner_writes_success_summary(monkeypatch, tmp_path, capsys) -> None:
    module = _load_module("run_gemma4_apple_smoke_success", "scripts/run_gemma4_apple_smoke.py")

    output_dir = tmp_path / "smoke"
    probe_record = {"status": "ok", "mode": "dotcache", "dense_generated_ids": [1], "dotcache_generated_ids": [1]}

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            model_id="google/gemma-4-E2B",
            prompt="hello",
            max_new_tokens=1,
            timeout_seconds=30,
            output_dir=str(output_dir),
        ),
    )

    class _FakeProcess:
        pid = 1234
        returncode = 0

        def communicate(self, timeout=None):
            (output_dir / "dotcache_mps_balanced.json").write_text(json.dumps(probe_record), encoding="utf-8")
            return ('{"status":"ok"}\n', "")

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )

    assert module.main() == 0
    output = json.loads(capsys.readouterr().out)
    summary_path = output_dir / "smoke_runner.json"

    assert output["status"] == "completed"
    assert output["probe_record"]["status"] == "ok"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text())["probe_record"] == probe_record


def test_run_gemma4_apple_smoke_runner_reports_timeout(monkeypatch, tmp_path, capsys) -> None:
    module = _load_module("run_gemma4_apple_smoke_timeout", "scripts/run_gemma4_apple_smoke.py")

    output_dir = tmp_path / "smoke"
    kill_calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            model_id="google/gemma-4-E2B",
            prompt="hello",
            max_new_tokens=1,
            timeout_seconds=5,
            output_dir=str(output_dir),
        ),
    )

    class _FakeProcess:
        pid = 4321
        returncode = -9

        def __init__(self) -> None:
            self._timed_out = False

        def communicate(self, timeout=None):
            if not self._timed_out:
                self._timed_out = True
                raise subprocess.TimeoutExpired(cmd=["fake"], timeout=timeout)
            return ("", "")

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: kill_calls.append((pid, sig)))

    assert module.main() == 1
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "timeout"
    assert output["error_type"] == "TimeoutExpired"
    assert kill_calls == [(4321, module.signal.SIGKILL)]


def test_run_gemma4_apple_smoke_runner_rejects_stale_success_when_probe_process_fails(
    monkeypatch, tmp_path, capsys
) -> None:
    module = _load_module("run_gemma4_apple_smoke_stale", "scripts/run_gemma4_apple_smoke.py")

    output_dir = tmp_path / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_probe_path = output_dir / "dotcache_mps_balanced.json"
    stale_probe_path.write_text(json.dumps({"status": "ok", "mode": "dotcache"}), encoding="utf-8")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            model_id="google/gemma-4-E2B",
            prompt="hello",
            max_new_tokens=1,
            timeout_seconds=30,
            output_dir=str(output_dir),
        ),
    )

    class _FakeProcess:
        pid = 999
        returncode = 2

        def communicate(self, timeout=None):
            return ("", "import failed")

    monkeypatch.setattr(
        module.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(),
    )

    assert module.main() == 1
    output = json.loads(capsys.readouterr().out)

    assert output["status"] == "error"
    assert output["error_type"] == "ProbeProcessError"
    assert output["probe_record"] is None
