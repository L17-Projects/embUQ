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


def test_coverage_summary_script_writes_markdown(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    module = _load_module(repo_root / "scripts" / "ci" / "summarize_coverage.py", "summarize_coverage_test")

    coverage_json = tmp_path / "coverage.json"
    coverage_json.write_text(
        json.dumps(
            {
                "files": {
                    "src/meso_uq/experiments.py": {
                        "summary": {
                            "num_statements": 20,
                            "covered_lines": 15,
                            "missing_lines": 5,
                            "percent_covered": 75.0,
                        }
                    },
                    "indentation/evalkit/tools.py": {
                        "summary": {
                            "num_statements": 10,
                            "covered_lines": 5,
                            "missing_lines": 5,
                            "percent_covered": 50.0,
                        }
                    },
                },
                "totals": {
                    "num_statements": 30,
                    "covered_lines": 20,
                    "percent_covered": 66.7,
                },
            }
        ),
        encoding="utf-8",
    )

    markdown_path = tmp_path / "coverage-summary.md"
    rc = module.main(["--json-path", str(coverage_json), "--markdown-path", str(markdown_path), "--max-files", "1"])

    assert rc == 0
    summary = markdown_path.read_text(encoding="utf-8")
    assert "Total coverage: 66.7%" in summary
    assert "`indentation/evalkit/tools.py`" in summary
    assert "`src/meso_uq/experiments.py`" not in summary
