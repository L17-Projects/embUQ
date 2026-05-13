import importlib.util
import subprocess
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path

import pytest


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_merge_ready_merge_gate_parser_defaults():
    module = _load_module(
        Path(__file__).resolve().parents[1] / "scripts" / "qa" / "ci" / "run_merge_ready_coverage.py",
        "run_merge_ready_coverage_parser",
    )

    parsed = module._build_arg_parser().parse_args([])
    assert parsed.base_ref == "origin/main"
    assert parsed.base_json is None
    assert parsed.head_json is None
    assert parsed.test_command == "pytest"


def test_merge_ready_merge_gate_parses_blank_test_command():
    module = _load_module(
        Path(__file__).resolve().parents[1] / "scripts" / "qa" / "ci" / "run_merge_ready_coverage.py",
        "run_merge_ready_coverage_parse_blank",
    )

    assert module._parse_test_command("   ") == ["pytest"]


def test_merge_ready_logged_command_writes_logs_and_raises(monkeypatch, tmp_path, capsys):
    module = _load_module(
        Path(__file__).resolve().parents[1] / "scripts" / "qa" / "ci" / "run_merge_ready_coverage.py",
        "run_merge_ready_coverage_logged_command",
    )
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="warn\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    stdout_log = tmp_path / "logs" / "stdout.log"
    stderr_log = tmp_path / "logs" / "stderr.log"

    module._run_logged_command(
        ["python", "-m", "pytest"],
        cwd=tmp_path,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    )

    captured = capsys.readouterr()
    assert calls == [["python", "-m", "pytest"]]
    assert stdout_log.read_text(encoding="utf-8") == "ok\n"
    assert stderr_log.read_text(encoding="utf-8") == "warn\n"
    assert "ok" in captured.out
    assert "warn" in captured.err

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=3, stdout="bad", stderr="err"),
    )
    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        module._run_logged_command(
            ["pytest"],
            cwd=tmp_path,
            stdout_log=tmp_path / "fail.stdout.log",
            stderr_log=tmp_path / "fail.stderr.log",
        )
    assert excinfo.value.returncode == 3


def test_merge_ready_base_worktree_adds_and_removes_worktree(monkeypatch, tmp_path):
    module = _load_module(
        Path(__file__).resolve().parents[1] / "scripts" / "qa" / "ci" / "run_merge_ready_coverage.py",
        "run_merge_ready_coverage_base_worktree",
    )
    git_calls: list[tuple[str, ...]] = []
    removed: list[Path] = []

    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", lambda prefix: _FakeTemporaryDirectory(tmp_path))
    monkeypatch.setattr(module, "_run_git", lambda *args: git_calls.append(args))
    monkeypatch.setattr(module.shutil, "rmtree", lambda path, ignore_errors=False: removed.append(Path(path)))

    with module._base_worktree("origin/main") as base_repo:
        assert base_repo == tmp_path / "repo"

    assert git_calls == [
        ("fetch", "origin"),
        ("worktree", "add", "--detach", str(tmp_path / "repo"), "origin/main"),
        ("worktree", "remove", "--force", str(tmp_path / "repo")),
    ]
    assert removed == [tmp_path / "repo"]


class _FakeTemporaryDirectory:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_merge_ready_collect_coverage_json_runs_coverage_commands(monkeypatch, tmp_path):
    module = _load_module(
        Path(__file__).resolve().parents[1] / "scripts" / "qa" / "ci" / "run_merge_ready_coverage.py",
        "run_merge_ready_coverage_collect",
    )
    calls: list[list[str]] = []

    def fake_logged(command, **_kwargs) -> None:
        calls.append(command)

    monkeypatch.setattr(module, "_run_logged_command", fake_logged)
    module._collect_coverage_json(
        python_bin="/venv/bin/python",
        repo_root=tmp_path,
        test_command="pytest -q tests/unit",
        output_root=tmp_path / "out",
        coverage_json_path=tmp_path / "coverage.json",
    )

    assert calls == [
        ["/venv/bin/python", "-m", "coverage", "run", "-m", "pytest", "-q", "tests/unit"],
        ["/venv/bin/python", "-m", "coverage", "json", "-o", str(tmp_path / "coverage.json")],
    ]


def test_merge_ready_delta_gate_loader_failure(monkeypatch, tmp_path):
    module = _load_module(
        Path(__file__).resolve().parents[1] / "scripts" / "qa" / "ci" / "run_merge_ready_coverage.py",
        "run_merge_ready_coverage_delta_loader_failure",
    )

    monkeypatch.setattr(module.importlib.util, "spec_from_file_location", lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="Could not load coverage checker"):
        module._run_delta_gate(
            base_json=tmp_path / "base.json",
            head_json=tmp_path / "head.json",
            markdown_path=tmp_path / "out.md",
        )


def test_merge_ready_merge_gate_uses_provided_coverage_files(monkeypatch, tmp_path):
    module = _load_module(
        Path(__file__).resolve().parents[1] / "scripts" / "qa" / "ci" / "run_merge_ready_coverage.py",
        "run_merge_ready_coverage_run",
    )

    base_json = tmp_path / "base.json"
    head_json = tmp_path / "head.json"
    base_json.write_text('{"totals":{"num_statements":10,"covered_lines":5}}', encoding="utf-8")
    head_json.write_text('{"totals":{"num_statements":10,"covered_lines":6}}', encoding="utf-8")

    calls: list[str] = []

    monkeypatch.setattr(module, "_collect_coverage_json", lambda *args, **kwargs: calls.append("collect"))
    result = module.main(
        [
            "--base-json",
            str(base_json),
            "--head-json",
            str(head_json),
        ]
    )

    assert result == 0
    assert calls == []


def test_merge_ready_merge_gate_runs_base_worktree_when_base_json_is_missing(monkeypatch, tmp_path):
    module = _load_module(
        Path(__file__).resolve().parents[1] / "scripts" / "qa" / "ci" / "run_merge_ready_coverage.py",
        "run_merge_ready_coverage_run_fallback",
    )

    output_root = tmp_path / "out"
    base_repo = tmp_path / "base-repo"
    base_repo.mkdir()
    head_json = tmp_path / "head.json"
    head_json.write_text("{}", encoding="utf-8")

    collect_calls: list[tuple[Path, Path]] = []
    check_calls: list[tuple[Path, Path]] = []

    @contextmanager
    def fake_base_worktree(base_ref: str):
        assert base_ref == "origin/main"
        yield base_repo

    def fake_collect(*, repo_root: Path, coverage_json_path: Path, **_kwargs) -> None:
        collect_calls.append((repo_root, coverage_json_path))
        coverage_json_path.write_text("{}", encoding="utf-8")

    def fake_run_delta_gate(*, base_json: Path, head_json: Path, markdown_path: Path) -> int:
        check_calls.append((base_json, head_json))
        assert str(base_json) != str(head_json)
        return 0

    monkeypatch.setattr(module, "_base_worktree", fake_base_worktree)
    monkeypatch.setattr(module, "_collect_coverage_json", fake_collect)
    monkeypatch.setattr(module, "_run_delta_gate", fake_run_delta_gate)

    result = module.main(
        [
            "--head-json",
            str(head_json),
            "--output-root",
            str(output_root),
        ]
    )

    assert result == 0
    assert len(collect_calls) == 1
    assert collect_calls[0][0] == base_repo
    assert len(check_calls) == 1
    assert check_calls[0][0] == output_root / "coverage-base.json"
    assert check_calls[0][1] == head_json


def test_merge_ready_merge_gate_rejects_missing_coverage_files(tmp_path):
    module = _load_module(
        Path(__file__).resolve().parents[1] / "scripts" / "qa" / "ci" / "run_merge_ready_coverage.py",
        "run_merge_ready_coverage_missing_files",
    )

    with pytest.raises(FileNotFoundError, match="Base coverage file not found"):
        module.main(["--base-json", str(tmp_path / "missing-base.json")])

    base_json = tmp_path / "base.json"
    base_json.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Head coverage file not found"):
        module.main(
            [
                "--base-json",
                str(base_json),
                "--head-json",
                str(tmp_path / "missing-head.json"),
            ]
        )
