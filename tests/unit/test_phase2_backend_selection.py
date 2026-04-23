from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _load_phase2_module():
    repo_root = Path(__file__).resolve().parents[2]
    key = "mesouq_test_inference_phase2_backend"
    sys.modules.pop(key, None)
    module_path = repo_root / "inference" / "scripts" / "run_phase_2.py"
    # The helper tests below only exercise pure helper functions. Stub heavy
    # runtime modules so this file can be imported without a full Korali stack.
    if "korali" not in sys.modules:
        sys.modules["korali"] = types.ModuleType("korali")
    if "mpi4py" not in sys.modules:
        mpi4py_mod = types.ModuleType("mpi4py")
        mpi4py_mod.MPI = types.SimpleNamespace(
            COMM_WORLD=types.SimpleNamespace(Get_rank=lambda: 0, Get_size=lambda: 1)
        )
        sys.modules["mpi4py"] = mpi4py_mod
    spec = importlib.util.spec_from_file_location(key, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_phase2_backend_profile_defaults() -> None:
    mod = _load_phase2_module()
    assert mod._resolve_phase2_backend({}, None, profile_hint="production") == "native-cuda"
    assert mod._resolve_phase2_backend({}, None, profile_hint="validation") == "cpu-mpi"


def test_resolve_phase2_backend_explicit_override_wins() -> None:
    mod = _load_phase2_module()
    cfg = {"phase2_backend": "cpu-mpi"}
    assert mod._resolve_phase2_backend(cfg, "native-cuda", profile_hint="validation") == "native-cuda"


def test_resolve_phase2_backend_config_value_used_when_explicit_missing() -> None:
    mod = _load_phase2_module()
    cfg = {"phase2_backend": "native-cuda"}
    assert mod._resolve_phase2_backend(cfg, None, profile_hint="validation") == "native-cuda"


def test_resolve_phase2_backend_rejects_unknown_backend() -> None:
    mod = _load_phase2_module()
    with pytest.raises(ValueError, match="Unsupported phase2_backend"):
        mod._resolve_phase2_backend({}, "gpu-batch", profile_hint="production")


def test_detect_native_cuda_batch_support_reports_disabled(tmp_path: Path) -> None:
    mod = _load_phase2_module()
    build_opts = (
        tmp_path / "_vega" / "korali" / "build" / "meson-info" / "intro-buildoptions.json"
    )
    build_opts.parent.mkdir(parents=True, exist_ok=True)
    build_opts.write_text(
        json.dumps([{"name": "native_cuda_batch", "value": False}]),
        encoding="utf-8",
    )

    supported, source = mod._detect_native_cuda_batch_support(tmp_path)
    assert supported is False
    assert "native_cuda_batch" in source


def test_detect_native_cuda_batch_support_reports_missing_metadata(tmp_path: Path) -> None:
    mod = _load_phase2_module()
    supported, source = mod._detect_native_cuda_batch_support(tmp_path)
    assert supported is None
    assert "build options not found" in source
