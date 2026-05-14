from configparser import ConfigParser
from pathlib import Path

import yaml


def _load_coveragerc(repo_root: Path) -> ConfigParser:
    parser = ConfigParser()
    parser.read(repo_root / ".coveragerc", encoding="utf-8")
    return parser


def _coverage_omit_entries(parser: ConfigParser) -> list[str]:
    return [
        line.strip()
        for line in parser.get("run", "omit", fallback="").splitlines()
        if line.strip()
    ]


def _coverage_source_entries(parser: ConfigParser) -> list[str]:
    return [
        line.strip()
        for line in parser.get("run", "source", fallback="").splitlines()
        if line.strip()
    ]


def test_coveragerc_includes_canonical_and_compatibility_source_roots():
    repo_root = Path(__file__).resolve().parents[1]
    parser = _load_coveragerc(repo_root)
    source_entries = _coverage_source_entries(parser)

    assert source_entries == [
        "src/meso_uq",
        "compression",
        "indentation",
        "inference",
        "propagation",
        "reduced",
        "scripts",
    ]


def test_coveragerc_excludes_generated_vendor_checkpoint_and_ops_paths():
    repo_root = Path(__file__).resolve().parents[1]
    parser = _load_coveragerc(repo_root)
    omit_entries = _coverage_omit_entries(parser)

    expected_omits = {
        "*/__pycache__/*",
        "*/.ipynb_checkpoints/*",
        "*.ipynb",
        "tests/*",
        "docs/*",
        "extern/*",
        "papers/*",
        "runtime/*",
        "_ci/*",
        "_out/*",
        "_runs/*",
        "logs/*",
        "*/generated/*",
        "*/artifacts/*",
        "*/archive/*",
        "*/archives/*",
    }

    assert expected_omits.issubset(set(omit_entries))
    assert not any(entry.startswith("src/meso_uq") for entry in omit_entries)


def test_codecov_policy_stays_relaxed_and_informational():
    repo_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repo_root / "codecov.yml").read_text(encoding="utf-8"))

    assert config["codecov"]["require_ci_to_pass"] is False
    assert config["coverage"]["status"]["project"]["default"] == {
        "target": "auto",
        "threshold": "0%",
    }
    assert config["coverage"]["status"]["patch"]["default"] == {
        "target": "auto",
        "threshold": "0%",
        "informational": True,
    }


def test_coverage_policy_docs_exist_and_describe_tightening_path():
    repo_root = Path(__file__).resolve().parents[1]
    docs_path = repo_root / "docs" / "COVERAGE_POLICY.md"

    contents = docs_path.read_text(encoding="utf-8")

    assert docs_path.exists()
    assert "Coverage Policy" in contents
    assert "Unit tests" in contents
    assert "Integration tests" in contents
    assert "Operational tests" in contents
    assert "Regression tests" in contents
    assert "Future tightening path" in contents
    assert "GPU or HPC" in contents
