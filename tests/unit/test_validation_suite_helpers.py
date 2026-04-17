"""Tests for pure helper functions in scripts/vega/run_validation_suite.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    key = "mesouq_test_run_validation_suite"
    if key in sys.modules:
        return sys.modules[key]
    module_path = repo_root / "scripts" / "vega" / "run_validation_suite.py"
    spec = importlib.util.spec_from_file_location(key, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _set_population_settings
# ---------------------------------------------------------------------------


def test_set_population_settings_none_returns_unchanged() -> None:
    mod = _load_module()
    config = {"pop_size": 1000, "hbi_pop_size": 2000, "phase3b_pop_size": 3000, "other": "x"}
    result = mod._set_population_settings(config, population_size=None)
    assert result == config


def test_set_population_settings_overrides_all_size_keys() -> None:
    mod = _load_module()
    config = {"pop_size": 1000, "hbi_pop_size": 2000, "phase3b_pop_size": 3000}
    result = mod._set_population_settings(config, population_size=50)
    assert result["pop_size"] == 50
    assert result["hbi_pop_size"] == 50
    assert result["phase3b_pop_size"] == 50


def test_set_population_settings_only_overrides_present_keys() -> None:
    mod = _load_module()
    config = {"pop_size": 1000, "other": "kept"}
    result = mod._set_population_settings(config, population_size=10)
    assert result["pop_size"] == 10
    assert result["other"] == "kept"
    assert "hbi_pop_size" not in result


def test_set_population_settings_does_not_mutate_original() -> None:
    mod = _load_module()
    config = {"pop_size": 1000}
    _ = mod._set_population_settings(config, population_size=10)
    assert config["pop_size"] == 1000


def test_set_population_settings_casts_to_int() -> None:
    mod = _load_module()
    config = {"pop_size": 1000}
    result = mod._set_population_settings(config, population_size=10.7)
    assert isinstance(result["pop_size"], int)
    assert result["pop_size"] == 10


# ---------------------------------------------------------------------------
# _build_env
# ---------------------------------------------------------------------------


def test_build_env_contains_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = mod._build_env(korali_pythonpath=None)
    assert "PYTHONPATH" in env
    # Project root should be in PYTHONPATH
    assert str(mod.PROJECT_ROOT) in env["PYTHONPATH"]


def test_build_env_prepends_korali_pythonpath(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = mod._build_env(korali_pythonpath="/opt/korali/site-packages")
    assert env["PYTHONPATH"].startswith("/opt/korali/site-packages")


def test_build_env_without_korali_pythonpath_no_extra_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _load_module()
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = mod._build_env(korali_pythonpath=None)
    assert "/opt/korali" not in env["PYTHONPATH"]


def test_build_env_preserves_existing_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setenv("MY_CUSTOM_VAR", "hello")
    env = mod._build_env(korali_pythonpath=None)
    assert env.get("MY_CUSTOM_VAR") == "hello"


# ---------------------------------------------------------------------------
# _resolve_validation_selection
# ---------------------------------------------------------------------------


def test_resolve_validation_selection_valid_returns_key() -> None:
    mod = _load_module()
    key = mod._resolve_validation_selection("compression:full-model:validation")
    assert key == "compression:full-model:validation"


def test_resolve_validation_selection_all_valid_workflows() -> None:
    mod = _load_module()
    valid = [
        "compression:reduced-model:validation",
        "compression:full-model:validation",
        "indentation:reduced-model:validation",
        "indentation:full-model:validation",
    ]
    for workflow in valid:
        key = mod._resolve_validation_selection(workflow)
        assert key == workflow


def test_resolve_validation_selection_non_validation_profile_raises() -> None:
    mod = _load_module()
    with pytest.raises(ValueError, match="validation"):
        mod._resolve_validation_selection("compression:full-model:production")


def test_resolve_validation_selection_unsupported_selection_raises() -> None:
    mod = _load_module()
    # "bogus" experiment is not in WORKFLOW_CONFIGS
    with pytest.raises((ValueError, Exception)):
        mod._resolve_validation_selection("bogus:full-model:validation")


# ---------------------------------------------------------------------------
# _phase2_command
# ---------------------------------------------------------------------------


def test_phase2_command_multi_rank_uses_mpirun(tmp_path: Path) -> None:
    mod = _load_module()
    script = tmp_path / "run_phase_2.py"
    script.touch()
    config = tmp_path / "config.yaml"
    config.touch()
    results = tmp_path / "results"

    cmd = mod._phase2_command("python", 4, script, config, results)
    assert cmd[0] == "mpirun"
    assert "-np" in cmd
    assert "4" in cmd


def test_phase2_command_single_rank_uses_direct_python(tmp_path: Path) -> None:
    mod = _load_module()
    script = tmp_path / "run_phase_2.py"
    config = tmp_path / "config.yaml"
    results = tmp_path / "results"

    cmd = mod._phase2_command("python", 1, script, config, results)
    assert cmd[0] == "python"
    assert "mpirun" not in cmd


def test_phase2_command_includes_config_and_output(tmp_path: Path) -> None:
    mod = _load_module()
    script = tmp_path / "run_phase_2.py"
    config = tmp_path / "config.yaml"
    results = tmp_path / "results"

    cmd = mod._phase2_command("python", 1, script, config, results)
    assert "--config" in cmd
    assert str(config) in cmd
    assert "--output-dir" in cmd
    assert str(results) in cmd


# ---------------------------------------------------------------------------
# _write_derived_config
# ---------------------------------------------------------------------------


def test_write_derived_config_creates_output_yaml(tmp_path: Path) -> None:
    mod = _load_module()
    base = tmp_path / "base.yaml"
    base.write_bytes(b"pop_size: 1000\nhbi_pop_size: 2000\n")
    output = tmp_path / "out" / "config.yaml"

    result = mod._write_derived_config(base, output, population_size=50)
    assert output.exists()
    assert result["pop_size"] == 50
    assert result["hbi_pop_size"] == 50


def test_write_derived_config_no_population_keeps_original(tmp_path: Path) -> None:
    mod = _load_module()
    base = tmp_path / "base.yaml"
    base.write_bytes(b"pop_size: 1000\n")
    output = tmp_path / "config.yaml"

    result = mod._write_derived_config(base, output, population_size=None)
    assert result["pop_size"] == 1000
