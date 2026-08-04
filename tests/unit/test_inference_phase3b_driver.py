"""Tests for pure helpers in inference/scripts/run_phase_3b.py.

Heavy dependencies (korali, mpi4py, evalkit) are stubbed so the module can be
imported and its pure functions exercised without a real solver.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Module loading helpers (mirrors test_phase3b_config_bootstrap.py pattern)
# ---------------------------------------------------------------------------


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    for extra in [
        repo_root,
        repo_root / "emb" / "compression",
        repo_root / "emb" / "compression" / "evalkit",
        repo_root / "emb" / "indentation",
        repo_root / "emb" / "indentation" / "evalkit",
    ]:
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    # Stub heavy dependencies before loading the module
    sys.modules.setdefault("korali", types.ModuleType("korali"))

    mpi4py_mod = types.ModuleType("mpi4py")

    class _FakeComm:
        def Get_rank(self) -> int:
            return 0

        def Get_size(self) -> int:
            return 1

        def Barrier(self) -> None:
            pass

    mpi4py_mod.MPI = types.SimpleNamespace(COMM_WORLD=_FakeComm())
    sys.modules.setdefault("mpi4py", mpi4py_mod)

    comp_mod = types.ModuleType("emb.compression.evalkit.posterior_compression")
    comp_mod.compute_compression_surrogate = lambda *a, **kw: None
    comp_mod.compute_compression_surrogate_batch = lambda *a, **kw: None
    comp_mod.preload_compression_surrogate = lambda *a, **kw: None
    sys.modules.setdefault("emb.compression.evalkit.posterior_compression", comp_mod)

    ind_mod = types.ModuleType("emb.indentation.evalkit.posterior_indentation")
    ind_mod.compute_indentation_surrogate = lambda *a, **kw: None
    ind_mod.compute_indentation_surrogate_batch = lambda *a, **kw: None
    ind_mod.preload_indentation_surrogate = lambda *a, **kw: None
    sys.modules.setdefault("emb.indentation.evalkit.posterior_indentation", ind_mod)

    module_path = repo_root / "inference" / "scripts" / "run_phase_3b.py"
    key = "mesouq_test_phase3b_driver"
    spec = importlib.util.spec_from_file_location(key, module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _resolve_config_path
# ---------------------------------------------------------------------------


def test_resolve_config_path_absolute_is_returned_as_is(tmp_path: Path) -> None:
    mod = _load_module()
    config_file = tmp_path / "myconfig.yaml"
    config_file.write_text("", encoding="utf-8")

    result = mod._resolve_config_path(str(config_file))
    assert result == config_file.resolve()


def test_resolve_config_path_none_calls_resolve_inference_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_module()
    dummy = tmp_path / "dummy.yaml"
    dummy.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        mod,
        "resolve_inference_config_path",
        lambda *a, **kw: dummy,
    )
    result = mod._resolve_config_path(None)
    assert result == dummy.resolve()


def test_resolve_config_path_relative_joined_to_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_module()
    # Patch project_root so we can predict the result
    monkeypatch.setattr(mod, "project_root", str(tmp_path))
    (tmp_path / "sub.yaml").write_text("", encoding="utf-8")

    result = mod._resolve_config_path("sub.yaml")
    assert result == (tmp_path / "sub.yaml").resolve()


# ---------------------------------------------------------------------------
# _resolve_output_root
# ---------------------------------------------------------------------------


def test_resolve_output_root_absolute_returns_resolved(tmp_path: Path) -> None:
    mod = _load_module()
    result = mod._resolve_output_root(str(tmp_path))
    assert result == tmp_path.resolve()


def test_resolve_output_root_relative_joined_to_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "project_root", str(tmp_path))

    result = mod._resolve_output_root("myoutput")
    assert result == (tmp_path / "myoutput").resolve()


def test_resolve_output_root_expands_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "project_root", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    result = mod._resolve_output_root("~/outdir")
    assert result == (tmp_path / "outdir").resolve()


# ---------------------------------------------------------------------------
# _resolve_surrogate_backend
# ---------------------------------------------------------------------------


def test_resolve_surrogate_backend_accepts_none_and_rejects_invalid_values() -> None:
    mod = _load_module()
    assert mod._resolve_surrogate_backend({"surrogate": None}) == "dnn"

    with pytest.raises(ValueError, match="Expected 'surrogate' config section"):
        mod._resolve_surrogate_backend({"surrogate": []})

    with pytest.raises(ValueError, match="Unsupported surrogate backend"):
        mod._resolve_surrogate_backend({"surrogate": {"backend": "foo"}})


# ---------------------------------------------------------------------------
# _extract_reference_data
# ---------------------------------------------------------------------------


def test_extract_reference_data_list_subscript() -> None:
    mod = _load_module()
    sub = {"Problem": {"Reference Data": [1.0, 2.0, 3.0]}}
    result = mod._extract_reference_data(sub)
    assert result == [1.0, 2.0, 3.0]


def test_extract_reference_data_missing_key_returns_none() -> None:
    mod = _load_module()
    sub = {"Problem": {}}
    result = mod._extract_reference_data(sub)
    assert result is None


def test_extract_reference_data_none_value_returns_none() -> None:
    mod = _load_module()
    sub = {"Problem": {"Reference Data": None}}
    result = mod._extract_reference_data(sub)
    assert result is None


def test_extract_reference_data_non_subscriptable_key_returns_none() -> None:
    mod = _load_module()

    class _Bad:
        def __iter__(self):
            raise TypeError("not iterable")

        def __len__(self):
            raise TypeError("no len")

        def __getitem__(self, i):
            raise TypeError("no index")

    sub = {"Problem": {"Reference Data": _Bad()}}
    result = mod._extract_reference_data(sub)
    assert result is None


def test_extract_reference_data_object_with_len() -> None:
    mod = _load_module()

    class _LenObj:
        def __len__(self):
            return 2

        def __getitem__(self, i):
            return float(i)

    sub = {"Problem": {"Reference Data": _LenObj()}}
    result = mod._extract_reference_data(sub)
    assert result == [0.0, 1.0]


def test_extract_reference_data_len_object_with_failing_index_returns_none() -> None:
    mod = _load_module()

    class _LenButBadIndex:
        def __len__(self):
            return 1

        def __getitem__(self, i):
            raise TypeError("bad index")

    sub = {"Problem": {"Reference Data": _LenButBadIndex()}}
    result = mod._extract_reference_data(sub)
    assert result is None


# ---------------------------------------------------------------------------
# _align_sub_reference
# ---------------------------------------------------------------------------


def test_align_sub_reference_matching_lengths_unchanged() -> None:
    mod = _load_module()
    sub = {"Problem": {"Reference Data": [1.0, 2.0, 3.0]}}
    ref_points = [0.1, 0.2, 0.3]

    result = mod._align_sub_reference(sub, ref_points, "test_exp", rank=0)
    assert result == ref_points


def test_align_sub_reference_mismatch_trims_to_shorter() -> None:
    mod = _load_module()
    sub = {"Problem": {"Reference Data": [1.0, 2.0]}}
    ref_points = [0.1, 0.2, 0.3]

    result = mod._align_sub_reference(sub, ref_points, "test_exp", rank=0)
    assert result == [0.1, 0.2]
    assert sub["Problem"]["Reference Data"] == [1.0, 2.0]


def test_align_sub_reference_ref_data_longer_trims_to_ref_points() -> None:
    mod = _load_module()
    sub = {"Problem": {"Reference Data": [1.0, 2.0, 3.0, 4.0]}}
    ref_points = [0.1, 0.2]

    result = mod._align_sub_reference(sub, ref_points, "test_exp", rank=0)
    assert result == [0.1, 0.2]
    assert sub["Problem"]["Reference Data"] == [1.0, 2.0]


def test_align_sub_reference_none_ref_data_returns_ref_points() -> None:
    mod = _load_module()
    sub = {"Problem": {}}  # no Reference Data key
    ref_points = [0.1, 0.2, 0.3]

    result = mod._align_sub_reference(sub, ref_points, "test_exp", rank=0)
    assert result is ref_points


def test_align_sub_reference_non_zero_rank_no_print_side_effect() -> None:
    mod = _load_module()
    sub = {"Problem": {"Reference Data": [1.0]}}
    ref_points = [0.1, 0.2, 0.3]

    # Should not raise even for rank != 0
    result = mod._align_sub_reference(sub, ref_points, "test_exp", rank=1)
    assert len(result) == 1


def test_select_phase3b_targets_filters_by_dataset_name() -> None:
    mod = _load_module()

    class _Spec:
        name = "compression"
        diameters = [2.1, 2.9]

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"compression_{diameter_um}um"

    selected = mod._select_phase3b_targets([_Spec()], dataset_name="compression_2.9um")
    assert len(selected) == 1
    assert selected[0][1] == 2.9


def test_select_phase3b_targets_filters_by_diameter() -> None:
    mod = _load_module()

    class _Spec:
        name = "compression"
        diameters = [2.1, 2.9]

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"compression_{diameter_um}um"

    selected = mod._select_phase3b_targets([_Spec()], diameter=2.1)
    assert len(selected) == 1
    assert selected[0][1] == 2.1


def test_select_phase3b_targets_filters_by_experiment_and_diameters() -> None:
    mod = _load_module()

    class _IndentationSpec:
        name = "indentation"
        diameters = [3.2, 3.4, 5.8]

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"indentation_{diameter_um}um"

    class _ResonanceSpec:
        name = "resonance"
        diameters = [2.6, 3.2, 4.0]

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"resonance_{diameter_um}um"

    selected = mod._select_phase3b_targets(
        [_IndentationSpec(), _ResonanceSpec()],
        experiment_names=["indentation"],
        diameters=[3.2, 5.8],
    )

    assert [(exp.name, diameter) for exp, diameter in selected] == [
        ("indentation", 3.2),
        ("indentation", 5.8),
    ]


def test_select_phase3b_targets_rejects_conflicting_filters() -> None:
    mod = _load_module()

    class _Spec:
        name = "compression"
        diameters = [2.1]

        @staticmethod
        def dataset_name(diameter_um: float) -> str:
            return f"compression_{diameter_um}um"

    with pytest.raises(ValueError, match="Use either dataset_name or diameter, not both."):
        mod._select_phase3b_targets([_Spec()], dataset_name="compression_2.1um", diameter=2.1)

    with pytest.raises(ValueError, match="dataset_name or config target filters"):
        mod._select_phase3b_targets(
            [_Spec()],
            dataset_name="compression_2.1um",
            experiment_names=["compression"],
        )
