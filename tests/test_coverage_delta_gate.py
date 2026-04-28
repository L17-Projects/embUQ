import importlib.util
import json
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_coverage_json(path: Path, covered: int, statements: int):
    totals = {"covered_lines": covered, "num_statements": statements}
    path.write_text(
        json.dumps({"totals": totals}),
        encoding="utf-8",
    )


def test_coverage_delta_gate_passes_when_head_increases(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "qa" / "ci" / "check_coverage_increase.py",
        "check_coverage_increase_test",
    )

    base_json = tmp_path / "base.json"
    head_json = tmp_path / "head.json"
    summary = tmp_path / "summary.md"

    _write_coverage_json(base_json, covered=50, statements=100)
    _write_coverage_json(head_json, covered=51, statements=100)

    rc = module.main(
        [
            "--base-json",
            str(base_json),
            "--head-json",
            str(head_json),
            "--markdown-path",
            str(summary),
        ]
    )

    assert rc == 0
    assert "Result: PASS" in summary.read_text(encoding="utf-8")


def test_coverage_delta_gate_fails_when_head_not_higher(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "qa" / "ci" / "check_coverage_increase.py",
        "check_coverage_increase_test_equal",
    )

    base_json = tmp_path / "base.json"
    head_json = tmp_path / "head.json"

    _write_coverage_json(base_json, covered=50, statements=100)
    _write_coverage_json(head_json, covered=50, statements=100)

    rc = module.main(["--base-json", str(base_json), "--head-json", str(head_json)])

    assert rc == 1


def test_coverage_delta_gate_fallback_to_percent_covered(tmp_path):
    """When num_statements == 0, _extract_percent falls back to percent_covered."""
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "qa" / "ci" / "check_coverage_increase.py",
        "check_coverage_increase_test_fallback",
    )

    # Write payloads with num_statements=0 so the fallback branch is taken
    def _write_fallback_json(path: Path, percent: float):
        path.write_text(
            json.dumps({"totals": {"num_statements": 0, "percent_covered": percent}}),
            encoding="utf-8",
        )

    base_json = tmp_path / "base.json"
    head_json = tmp_path / "head.json"

    _write_fallback_json(base_json, 80.0)
    _write_fallback_json(head_json, 81.0)

    rc = module.main(["--base-json", str(base_json), "--head-json", str(head_json)])
    assert rc == 0


def test_coverage_delta_gate_ignores_optional_bnn_files(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "qa" / "ci" / "check_coverage_increase.py",
        "check_coverage_increase_test_excludes",
    )

    def _write_files_json(path: Path, *, generic_miss: int) -> None:
        path.write_text(
            json.dumps(
                {
                    "files": {
                        "src/meso_uq/core.py": {
                            "summary": {"num_statements": 10, "missing_lines": generic_miss}
                        },
                        "src/meso_uq/surrogate/bnn.py": {
                            "summary": {"num_statements": 100, "missing_lines": 100}
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

    base_json = tmp_path / "base.json"
    head_json = tmp_path / "head.json"

    _write_files_json(base_json, generic_miss=5)
    _write_files_json(head_json, generic_miss=4)

    rc = module.main(["--base-json", str(base_json), "--head-json", str(head_json)])
    assert rc == 0


def test_coverage_delta_gate_falls_back_to_totals_when_file_entries_are_invalid(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(
        repo_root / "scripts" / "qa" / "ci" / "check_coverage_increase.py",
        "check_coverage_increase_test_invalid_files",
    )

    def _write_files_json(path: Path, *, covered: int, statements: int) -> None:
        path.write_text(
            json.dumps(
                {
                    "files": {
                        "src/meso_uq/core.py": "invalid-entry",
                        "src/meso_uq/metrics.py": {"summary": "invalid-summary"},
                        "src/meso_uq/surrogate/bnn_training.py": {
                            "summary": {"num_statements": 99, "missing_lines": 99}
                        },
                    },
                    "totals": {"covered_lines": covered, "num_statements": statements},
                }
            ),
            encoding="utf-8",
        )

    base_json = tmp_path / "base.json"
    head_json = tmp_path / "head.json"

    _write_files_json(base_json, covered=80, statements=100)
    _write_files_json(head_json, covered=81, statements=100)

    rc = module.main(["--base-json", str(base_json), "--head-json", str(head_json)])
    assert rc == 0
