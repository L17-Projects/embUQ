"""Unit tests for src/meso_uq/config/loader.py."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from meso_uq.config.loader import (
    load_inference_config_with_overrides,
    load_yaml,
    merge_configs,
    resolve_inference_config_path,
    validate_config_file,
)

# ---------------------------------------------------------------------------
# resolve_inference_config_path
# ---------------------------------------------------------------------------


def test_resolve_inference_config_path_returns_existing_file(tmp_path: Path) -> None:
    config_dir = tmp_path / "inference" / "configs" / "production"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "inference_config_compression.yaml"
    config_file.write_text("emb_diameters: [2.1]\n", encoding="utf-8")

    result = resolve_inference_config_path(tmp_path, experiment="compression", mode="production")
    assert result == config_file


def test_resolve_inference_config_path_indentation_production(tmp_path: Path) -> None:
    config_dir = tmp_path / "inference" / "configs" / "production"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "inference_config_indentation.yaml"
    config_file.write_text("emb_diameters: [2.1]\n", encoding="utf-8")

    result = resolve_inference_config_path(tmp_path, experiment="indentation", mode="production")
    assert result == config_file


def test_resolve_inference_config_path_test_mode_compression(tmp_path: Path) -> None:
    config_dir = tmp_path / "inference" / "configs" / "test"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "inference_config.yaml"
    config_file.write_text("emb_diameters: [2.1]\n", encoding="utf-8")

    result = resolve_inference_config_path(tmp_path, experiment="compression", mode="test")
    assert result == config_file


def test_resolve_inference_config_path_env_override_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "override.yaml"
    override.write_text("emb_diameters: [2.1]\n", encoding="utf-8")

    monkeypatch.setenv("HUQ_INFERENCE_CONFIG", str(override))
    result = resolve_inference_config_path(tmp_path)
    assert result == override


def test_resolve_inference_config_path_config_path_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "myconfig.yaml"
    override.write_text("emb_diameters: [2.9]\n", encoding="utf-8")

    monkeypatch.delenv("HUQ_INFERENCE_CONFIG", raising=False)
    monkeypatch.setenv("CONFIG_PATH", str(override))
    result = resolve_inference_config_path(tmp_path)
    assert result == override


def test_resolve_inference_config_path_env_override_relative_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "rel_config.yaml"
    override.write_text("emb_diameters: [3.0]\n", encoding="utf-8")

    monkeypatch.setenv("HUQ_INFERENCE_CONFIG", "rel_config.yaml")
    os.chdir(tmp_path)
    # Relative path that exists (cwd)
    result = resolve_inference_config_path(tmp_path)
    assert result == override


def test_resolve_inference_config_path_invalid_mode_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_inference_config_path(tmp_path, experiment="compression", mode="bogus")


def test_resolve_inference_config_path_invalid_experiment_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_inference_config_path(tmp_path, experiment="unknown_exp", mode="production")


def test_resolve_inference_config_path_missing_file_raises(tmp_path: Path) -> None:
    config_dir = tmp_path / "inference" / "configs" / "production"
    config_dir.mkdir(parents=True)
    # No config files present
    with pytest.raises(FileNotFoundError):
        resolve_inference_config_path(tmp_path, experiment="compression", mode="production")


# ---------------------------------------------------------------------------
# load_yaml
# ---------------------------------------------------------------------------


def test_load_yaml_returns_dict_for_valid_file(tmp_path: Path) -> None:
    f = tmp_path / "config.yaml"
    f.write_text("key: value\nnumber: 42\n", encoding="utf-8")
    result = load_yaml(f)
    assert result == {"key": "value", "number": 42}


def test_load_yaml_empty_file_returns_empty_dict(tmp_path: Path) -> None:
    f = tmp_path / "empty.yaml"
    f.write_text("", encoding="utf-8")
    result = load_yaml(f)
    assert result == {}


def test_load_yaml_null_file_returns_empty_dict(tmp_path: Path) -> None:
    f = tmp_path / "null.yaml"
    f.write_text("null\n", encoding="utf-8")
    result = load_yaml(f)
    assert result == {}


def test_load_yaml_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_yaml(tmp_path / "nonexistent.yaml")


def test_load_yaml_accepts_string_path(tmp_path: Path) -> None:
    f = tmp_path / "s.yaml"
    f.write_text("x: 1\n", encoding="utf-8")
    result = load_yaml(str(f))
    assert result["x"] == 1


# ---------------------------------------------------------------------------
# merge_configs
# ---------------------------------------------------------------------------


def test_merge_configs_flat_override() -> None:
    base = {"a": 1, "b": 2}
    override = {"b": 99, "c": 3}
    result = merge_configs(base, override)
    assert result == {"a": 1, "b": 99, "c": 3}


def test_merge_configs_deep_merge() -> None:
    base = {"nested": {"x": 1, "y": 2}, "top": "base"}
    override = {"nested": {"y": 99, "z": 3}}
    result = merge_configs(base, override)
    assert result["nested"] == {"x": 1, "y": 99, "z": 3}
    assert result["top"] == "base"


def test_merge_configs_does_not_mutate_base() -> None:
    base = {"a": 1, "nested": {"b": 2}}
    override = {"nested": {"c": 3}}
    merge_configs(base, override)
    assert base["nested"] == {"b": 2}


def test_merge_configs_non_dict_override_replaces() -> None:
    base = {"a": {"x": 1}}
    override = {"a": "scalar"}
    result = merge_configs(base, override)
    assert result["a"] == "scalar"


def test_merge_configs_empty_override_returns_copy_of_base() -> None:
    base = {"a": 1}
    result = merge_configs(base, {})
    assert result == base
    assert result is not base


# ---------------------------------------------------------------------------
# load_inference_config_with_overrides
# ---------------------------------------------------------------------------


def _minimal_inference_yaml() -> str:
    return (
        "emb_diameters: [2.1]\n"
        "pop_size: 1000\n"
        "max_gen: -1\n"
        "target_cov: 0.8\n"
        "covariance_scaling: 0.04\n"
        "prior_Yt: [1.0e7, 5.0e7]\n"
        "prior_kb: [100.0, 1000.0]\n"
        "prior_b1: [0.0, 3.0]\n"
        "prior_b2: [0.0, 10.0]\n"
        "prior_a3: [-2.5, 3.0]\n"
        "prior_a4: [0.0, 4.0]\n"
        "prior_d0: [0.0, 0.5]\n"
        "prior_sigma: [0.0, 1.0]\n"
    )


def test_load_inference_config_with_overrides_base_only(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_inference_yaml(), encoding="utf-8")

    config = load_inference_config_with_overrides(base)
    assert config.pop_size == 1000


def test_load_inference_config_with_overrides_applies_override_file(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_inference_yaml(), encoding="utf-8")

    override = tmp_path / "override.yaml"
    override.write_text("pop_size: 500\n", encoding="utf-8")

    config = load_inference_config_with_overrides(base, override_path=override)
    assert config.pop_size == 500


def test_load_inference_config_with_overrides_applies_cli_overrides(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_inference_yaml(), encoding="utf-8")

    config = load_inference_config_with_overrides(base, cli_overrides={"pop_size": 200})
    assert config.pop_size == 200


def test_load_inference_config_with_overrides_cli_wins_over_file(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(_minimal_inference_yaml(), encoding="utf-8")

    override = tmp_path / "override.yaml"
    override.write_text("pop_size: 500\n", encoding="utf-8")

    config = load_inference_config_with_overrides(
        base, override_path=override, cli_overrides={"pop_size": 100}
    )
    assert config.pop_size == 100


# ---------------------------------------------------------------------------
# validate_config_file
# ---------------------------------------------------------------------------


def test_validate_config_file_valid_inference(tmp_path: Path) -> None:
    f = tmp_path / "config.yaml"
    f.write_text(_minimal_inference_yaml(), encoding="utf-8")

    ok, msg = validate_config_file(f, config_type="inference")
    assert ok is True
    assert msg == ""


def test_validate_config_file_invalid_inference_schema(tmp_path: Path) -> None:
    f = tmp_path / "bad.yaml"
    f.write_text("emb_diameters: [2.1]\npop_size: -99\n", encoding="utf-8")

    ok, msg = validate_config_file(f, config_type="inference")
    assert ok is False
    assert msg


def test_validate_config_file_missing_file(tmp_path: Path) -> None:
    ok, msg = validate_config_file(tmp_path / "missing.yaml", config_type="inference")
    assert ok is False
    assert "not found" in msg.lower() or "File" in msg


def test_validate_config_file_unknown_type(tmp_path: Path) -> None:
    f = tmp_path / "config.yaml"
    f.write_text("{}\n", encoding="utf-8")

    ok, msg = validate_config_file(f, config_type="badtype")
    assert ok is False
    assert "Unknown" in msg


def test_validate_config_file_sampling_type(tmp_path: Path) -> None:
    f = tmp_path / "sampling.yaml"
    f.write_text(
        "n_samples: 100\nseed: 42\ndiameter_um: 2.1\noutput_dir: out\n" "parameter_bounds: {}\n",
        encoding="utf-8",
    )

    ok, _ = validate_config_file(f, config_type="sampling")
    assert ok is True


def test_validate_config_file_propagation_type(tmp_path: Path) -> None:
    f = tmp_path / "prop.yaml"
    f.write_text(
        "method: mc\ndiameter_um: 2.1\nposterior_samples_file: samples.csv\noutput_dir: out\n",
        encoding="utf-8",
    )

    ok, _ = validate_config_file(f, config_type="propagation")
    assert ok is True
