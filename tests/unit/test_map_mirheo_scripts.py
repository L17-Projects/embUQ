"""Tests for PR 4/6: MAP Mirheo evaluation scripts and validate_training_baseline."""
from __future__ import annotations

import py_compile
from pathlib import Path
import types
import importlib
import sys

import pytest

from meso_uq.mirheo.baseline import (
    EXPECTED_TRAINING_FSCALE,
    EXPECTED_TRAINING_NUMSTEPS,
    EXPECTED_TRAINING_NUMSTEPS_EQ,
    EXPECTED_TRAINING_SHELL_TH,
    validate_training_baseline,
)

REPO = Path(__file__).resolve().parents[2]
_IND_SCRIPT = REPO / "propagation" / "scripts" / "evaluate_map_mirheo_optimized_indentation.py"
_CMP_SCRIPT = REPO / "propagation" / "scripts" / "evaluate_map_mirheo_optimized.py"


# ---------------------------------------------------------------------------
# Syntax checks (no Mirheo needed)
# ---------------------------------------------------------------------------

def test_evaluate_indentation_script_parseable():
    py_compile.compile(str(_IND_SCRIPT), doraise=True)


def test_evaluate_compression_script_parseable():
    py_compile.compile(str(_CMP_SCRIPT), doraise=True)


def test_convert_map_manifest_script_parseable():
    script = REPO / "scripts" / "vega" / "convert_map_manifest.py"
    py_compile.compile(str(script), doraise=True)


# ---------------------------------------------------------------------------
# Baseline constants sanity check
# ---------------------------------------------------------------------------

def test_baseline_constants_values():
    assert EXPECTED_TRAINING_FSCALE == pytest.approx(0.0074)
    assert EXPECTED_TRAINING_SHELL_TH == pytest.approx(5.0e-9)
    assert EXPECTED_TRAINING_NUMSTEPS == 5000
    assert EXPECTED_TRAINING_NUMSTEPS_EQ == 10000


# ---------------------------------------------------------------------------
# validate_training_baseline — pure Python, no Mirheo/korali needed
# ---------------------------------------------------------------------------

def _valid_params():
    return {
        "fscale": 0.0074,
        "shell_th": 5.0e-9,
        "numsteps": 5000,
        "numsteps_eq": 10000,
    }


def test_validate_training_baseline_passes():
    validate_training_baseline(_valid_params(), "test")  # must not raise


def test_validate_training_baseline_wrong_fscale():
    params = {**_valid_params(), "fscale": 0.009}
    with pytest.raises(ValueError, match="fscale"):
        validate_training_baseline(params, "test")


def test_validate_training_baseline_wrong_shell_th():
    params = {**_valid_params(), "shell_th": 1.0e-9}
    with pytest.raises(ValueError, match="shell_th"):
        validate_training_baseline(params, "test")


def test_validate_training_baseline_wrong_numsteps():
    params = {**_valid_params(), "numsteps": 999}
    with pytest.raises(ValueError, match="numsteps"):
        validate_training_baseline(params, "test")


def test_validate_training_baseline_wrong_numsteps_eq():
    params = {**_valid_params(), "numsteps_eq": 1}
    with pytest.raises(ValueError, match="numsteps_eq"):
        validate_training_baseline(params, "test")


def test_compression_map_init_uses_fresh_src_template(tmp_path, monkeypatch):
    import yaml

    fake_mpi = types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(Barrier=lambda: None))
    monkeypatch.setitem(sys.modules, "mpi4py", types.SimpleNamespace(MPI=fake_mpi))
    fake_posterior = types.SimpleNamespace(compute_compression=lambda *args, **kwargs: None)
    fake_tools = types.SimpleNamespace(
        datedPrint=lambda *args, **kwargs: None,
        getReferencePoints=lambda diameter_um: [0.0, 1.0],
    )
    monkeypatch.setitem(sys.modules, "compression.evalkit.posterior_compression", fake_posterior)
    monkeypatch.setitem(sys.modules, "compression.evalkit.tools", fake_tools)
    mod = importlib.import_module("propagation.scripts.evaluate_map_mirheo_optimized")

    project_root = tmp_path / "repo"
    src_dir = project_root / "compression" / "src"
    (src_dir / "microbubble").mkdir(parents=True)
    (src_dir / "microbubble" / "sphere_icosphere.py").write_text("# stub\n")

    def fake_generate_sim(**kwargs):
        simu_path = Path(kwargs["simu_path"])
        (simu_path / "parameter").mkdir(parents=True, exist_ok=True)

    def fake_write_parameters(source_path, simu_path, simnum):
        param_dir = Path(simu_path) / "parameter"
        param_dir.mkdir(parents=True, exist_ok=True)
        (param_dir / "parameters-default00001.yaml").write_text(
            yaml.dump(
                {
                    "ul": 2.5e-7,
                    "radp": 1.0,
                    "Lx": 1.0,
                    "Ly": 1.0,
                    "Lz": 1.0,
                    "dt": 1.0e-4,
                    "dt_eq": 1.0e-4,
                    "numsteps": 20000,
                    "numsteps_eq": 40000,
                }
            )
        )

    fake_generate = types.SimpleNamespace(generate_sim=fake_generate_sim)
    fake_parameters = types.SimpleNamespace(write_parameters=fake_write_parameters)

    monkeypatch.setattr(mod, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        mod,
        "import_module",
        lambda name: fake_generate if name == "generate" else fake_parameters,
    )

    out = mod.setup_map_specific_init_dir(
        diameter_um=3.0,
        retry_attempt=0,
        dt_scale_factor=0.5,
        rank=0,
        scratch_root=str(tmp_path / "scratch"),
    )

    assert out.endswith("scratch")
    param_file = tmp_path / "scratch" / "parameter" / "parameters-default00001.yaml"
    params = yaml.safe_load(param_file.read_text())
    assert params["radp"] == pytest.approx(6.0)
    assert params["Lx"] == pytest.approx(17.0)
    assert params["Ly"] == pytest.approx(17.0)
    assert params["Lz"] == pytest.approx(22.0)
