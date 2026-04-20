from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyro")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _make_spec(tmp_path: Path, name: str) -> dict[str, Path | str]:
    data = tmp_path / f"{name}.dat"
    out = tmp_path / f"{name}.pt"
    dnn = tmp_path / f"{name}_dnn.pkl"
    script = tmp_path / f"{name}_train.py"
    for path in (data, dnn, script):
        path.write_text("placeholder", encoding="utf-8")
    return {
        "name": name,
        "script": script,
        "data": data,
        "out": out,
        "dnn": dnn,
    }


def _parse_arg(command: list[str], flag: str) -> str:
    idx = command.index(flag)
    return command[idx + 1]


def test_train_bnn_matrix_resume_skips_completed(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "karolina" / "train_bnn_surrogates.py",
        "train_bnn_surrogates_resume_test",
    )
    spec_a = _make_spec(tmp_path, "spec_a")
    spec_b = _make_spec(tmp_path, "spec_b")
    monkeypatch.setattr(module, "SPECS", [spec_a, spec_b])

    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state.json"

    # Pre-mark first spec as completed.
    Path(spec_a["out"]).write_text("artifact", encoding="utf-8")
    (output_root / "spec_a.json").write_text(
        json.dumps({"training": {"parity_passed": True}}),
        encoding="utf-8",
    )

    called: list[list[str]] = []

    def fake_run(command, cwd, check, timeout):
        called.append(command)
        out_path = Path(_parse_arg(command, "--out"))
        report_path = Path(_parse_arg(command, "--report-path"))
        out_path.write_text("artifact", encoding="utf-8")
        report_path.write_text(
            json.dumps({"training": {"parity_passed": True}}),
            encoding="utf-8",
        )
        class _Done:
            returncode = 0
        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--python-bin",
            sys.executable,
            "--output-root",
            str(output_root),
            "--state-path",
            str(state_path),
        ]
    )
    assert rc == 0
    assert len(called) == 1
    assert str(spec_b["script"]) in called[0]

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "passed"
    runs = {item["name"]: item for item in state["runs"]}
    assert runs["spec_a"]["status"] == "passed"
    assert runs["spec_a"]["skipped"] is True
    assert runs["spec_b"]["status"] == "passed"
    assert runs["spec_b"]["skipped"] is False


def test_train_bnn_matrix_start_from_filters_sequence(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "karolina" / "train_bnn_surrogates.py",
        "train_bnn_surrogates_start_from_test",
    )
    spec_a = _make_spec(tmp_path, "spec_a")
    spec_b = _make_spec(tmp_path, "spec_b")
    spec_c = _make_spec(tmp_path, "spec_c")
    monkeypatch.setattr(module, "SPECS", [spec_a, spec_b, spec_c])

    output_root = tmp_path / "out"
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "state.json"

    called_names: list[str] = []

    def fake_run(command, cwd, check, timeout):
        report_path = Path(_parse_arg(command, "--report-path"))
        called_names.append(report_path.stem)
        Path(_parse_arg(command, "--out")).write_text("artifact", encoding="utf-8")
        report_path.write_text(
            json.dumps({"training": {"parity_passed": True}}),
            encoding="utf-8",
        )
        class _Done:
            returncode = 0
        return _Done()

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    rc = module.main(
        [
            "--python-bin",
            sys.executable,
            "--output-root",
            str(output_root),
            "--state-path",
            str(state_path),
            "--start-from",
            "spec_b",
            "--no-resume",
        ]
    )
    assert rc == 0
    assert called_names == ["spec_b", "spec_c"]
