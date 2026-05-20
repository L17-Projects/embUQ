from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from emb.compression.evalkit import posterior_compression
from emb.indentation.evalkit import posterior_indentation


class _CompressionDnnStub:
    def __init__(self) -> None:
        self.single_call = None
        self.batch_call = None

    def evaluate_compression(self, *, x, disp):
        self.single_call = {"x": x, "disp": disp}
        return [2.0, 4.0]

    def evaluate_compression_batch(self, theta, *, disp, d0, chunk_size, **kwargs):
        self.batch_call = {
            "theta": theta,
            "disp": disp,
            "d0": d0,
            "chunk_size": chunk_size,
            "kwargs": kwargs,
        }
        return np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


class _CompressionBnnStub:
    def __init__(self) -> None:
        self.single_call = None
        self.batch_call = None

    def evaluate_compression(self, *, x, disp, predictive_mc_samples, predictive_mc_chunk_size):
        self.single_call = {
            "x": x,
            "disp": disp,
            "predictive_mc_samples": predictive_mc_samples,
            "predictive_mc_chunk_size": predictive_mc_chunk_size,
        }
        return [2.0, 4.0], [0.3, 0.4]

    def evaluate_compression_batch(
        self,
        theta,
        *,
        disp,
        d0,
        chunk_size,
        predictive_mc_samples,
        predictive_mc_chunk_size,
    ):
        self.batch_call = {
            "theta": theta,
            "disp": disp,
            "d0": d0,
            "chunk_size": chunk_size,
            "predictive_mc_samples": predictive_mc_samples,
            "predictive_mc_chunk_size": predictive_mc_chunk_size,
        }
        return (
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        )


class _IndentationDnnStub:
    def __init__(self) -> None:
        self.single_call = None
        self.batch_call = None

    def evaluate_indentation(self, *, x, forces):
        self.single_call = {"x": x, "forces": forces}
        return [1.0, 3.0]

    def evaluate_indentation_batch(self, theta, *, forces, d0, chunk_size, **kwargs):
        self.batch_call = {
            "theta": theta,
            "forces": forces,
            "d0": d0,
            "chunk_size": chunk_size,
            "kwargs": kwargs,
        }
        return np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


class _IndentationBnnStub:
    def __init__(self) -> None:
        self.single_call = None
        self.batch_call = None

    def evaluate_indentation(self, *, x, forces, predictive_mc_samples, predictive_mc_chunk_size):
        self.single_call = {
            "x": x,
            "forces": forces,
            "predictive_mc_samples": predictive_mc_samples,
            "predictive_mc_chunk_size": predictive_mc_chunk_size,
        }
        return [1.0, 3.0], [0.2, 0.5]

    def evaluate_indentation_batch(
        self,
        theta,
        *,
        forces,
        d0,
        chunk_size,
        predictive_mc_samples,
        predictive_mc_chunk_size,
    ):
        self.batch_call = {
            "theta": theta,
            "forces": forces,
            "d0": d0,
            "chunk_size": chunk_size,
            "predictive_mc_samples": predictive_mc_samples,
            "predictive_mc_chunk_size": predictive_mc_chunk_size,
        }
        return (
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        )


@pytest.fixture(autouse=True)
def _clear_evalkit_caches() -> None:
    posterior_compression._CONFIG_CACHE.clear()
    posterior_compression._SURROGATE_CACHE.clear()
    posterior_compression._resolve_project_root.cache_clear()
    posterior_indentation._CONFIG_CACHE.clear()
    posterior_indentation._SURROGATE_CACHE.clear()
    posterior_indentation._resolve_project_root.cache_clear()
    posterior_indentation._DUMP_FLAG = None


def test_compression_worker_comm_prefers_korali_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        posterior_compression,
        "korali",
        types.SimpleNamespace(getWorkerMPIComm=lambda: sentinel),
    )
    assert posterior_compression._get_worker_comm() is sentinel

    def _boom():
        raise RuntimeError("boom")

    fallback = object()
    monkeypatch.setattr(
        posterior_compression,
        "korali",
        types.SimpleNamespace(getWorkerMPIComm=_boom),
    )
    monkeypatch.setattr(posterior_compression, "MPI", types.SimpleNamespace(COMM_WORLD=fallback))
    assert posterior_compression._get_worker_comm() is fallback


def test_indentation_worker_comm_prefers_korali_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        posterior_indentation,
        "korali",
        types.SimpleNamespace(getWorkerMPIComm=lambda: sentinel),
    )
    assert posterior_indentation._get_worker_comm() is sentinel

    def _boom():
        raise RuntimeError("boom")

    fallback = object()
    monkeypatch.setattr(
        posterior_indentation,
        "korali",
        types.SimpleNamespace(getWorkerMPIComm=_boom),
    )
    monkeypatch.setattr(posterior_indentation, "MPI", types.SimpleNamespace(COMM_WORLD=fallback))
    assert posterior_indentation._get_worker_comm() is fallback


def test_compression_resolve_project_root_from_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "emb" / "compression" / "src").mkdir(parents=True)
    workdir = project / "nested" / "child"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)
    assert posterior_compression._resolve_project_root() == str(project)


def test_compression_resolve_project_root_from_relocated_evalkit_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    (project / "emb" / "compression" / "src").mkdir(parents=True)
    evalkit_dir = project / "emb" / "compression" / "evalkit"
    evalkit_dir.mkdir(parents=True)
    monkeypatch.chdir(evalkit_dir)
    assert posterior_compression._resolve_project_root() == str(project)


def test_indentation_resolve_project_root_from_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "emb" / "indentation" / "src").mkdir(parents=True)
    workdir = project / "nested" / "child"
    workdir.mkdir(parents=True)
    monkeypatch.chdir(workdir)
    assert posterior_indentation._resolve_project_root() == str(project)


def test_indentation_resolve_project_root_from_relocated_evalkit_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    (project / "emb" / "indentation" / "src").mkdir(parents=True)
    evalkit_dir = project / "emb" / "indentation" / "evalkit"
    evalkit_dir.mkdir(parents=True)
    monkeypatch.chdir(evalkit_dir)
    assert posterior_indentation._resolve_project_root() == str(project)


def test_compression_resolve_project_root_raises_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(posterior_compression.os.path, "exists", lambda _path: False)
    with pytest.raises(RuntimeError, match="emb/compression/src"):
        posterior_compression._resolve_project_root()


def test_indentation_resolve_project_root_raises_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(posterior_indentation.os.path, "exists", lambda _path: False)
    with pytest.raises(RuntimeError, match="emb/indentation/src"):
        posterior_indentation._resolve_project_root()


@pytest.mark.parametrize(
    ("module", "relative_path"),
    [
        (posterior_compression, "inference/configs/production/inference_config_compression.yaml"),
        (posterior_indentation, "inference/configs/production/inference_config_indentation.yaml"),
    ],
)
def test_resolve_config_path_prefers_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module,
    relative_path: str,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    override = project_root / "custom.yaml"
    override.write_text("key: value\n", encoding="utf-8")
    monkeypatch.setenv("HUQ_INFERENCE_CONFIG", "custom.yaml")
    assert module._resolve_config_path(str(project_root)) == override


@pytest.mark.parametrize(
    ("module", "relative_path"),
    [
        (posterior_compression, "inference/configs/production/inference_config_compression.yaml"),
        (posterior_indentation, "inference/configs/production/inference_config_indentation.yaml"),
    ],
)
def test_resolve_config_path_falls_back_to_standard_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module,
    relative_path: str,
) -> None:
    project_root = tmp_path / "project"
    config_path = project_root / relative_path
    config_path.parent.mkdir(parents=True)
    config_path.write_text("value: 1\n", encoding="utf-8")
    monkeypatch.delenv("HUQ_INFERENCE_CONFIG", raising=False)
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    assert module._resolve_config_path(str(project_root)) == config_path


@pytest.mark.parametrize(
    ("module", "missing_fragment"),
    [
        (posterior_compression, "inference_config_compression"),
        (posterior_indentation, "inference_config_indentation"),
    ],
)
def test_resolve_config_path_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    module,
    missing_fragment: str,
) -> None:
    monkeypatch.delenv("HUQ_INFERENCE_CONFIG", raising=False)
    monkeypatch.delenv("CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=missing_fragment):
        module._resolve_config_path(str(tmp_path))


def test_compression_load_config_caches_by_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "emb.compression.yaml"
    config_path.write_text("value: 1\n", encoding="utf-8")
    monkeypatch.setattr(posterior_compression, "_resolve_config_path", lambda _root: config_path)
    first = posterior_compression._load_config(str(tmp_path))
    config_path.write_text("value: 2\n", encoding="utf-8")
    second = posterior_compression._load_config(str(tmp_path))
    assert first is second
    assert second["value"] == 1


def test_indentation_load_config_caches_and_dump_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "emb.indentation.yaml"
    config_path.write_text("dump: true\n", encoding="utf-8")
    monkeypatch.setattr(posterior_indentation, "_resolve_config_path", lambda _root: config_path)
    first = posterior_indentation._load_config(str(tmp_path))
    config_path.write_text("dump: false\n", encoding="utf-8")
    second = posterior_indentation._load_config(str(tmp_path))
    assert first is second
    assert posterior_indentation._get_dump_flag() is True


def test_preload_surrogates_use_cached_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    compression_calls = []
    indentation_calls = []
    monkeypatch.setattr(posterior_compression, "_resolve_project_root", lambda: "/repo")
    monkeypatch.setattr(
        posterior_compression,
        "_get_surrogate",
        lambda *args, **kwargs: compression_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(posterior_indentation, "_resolve_project_root", lambda: "/repo")
    monkeypatch.setattr(
        posterior_indentation,
        "_get_surrogate",
        lambda *args, **kwargs: indentation_calls.append((args, kwargs)),
    )

    posterior_compression.preload_compression_surrogate(2.1, device="gpu", backend="bnn")
    posterior_indentation.preload_indentation_surrogate(3.2, device="gpu", backend="bnn")

    assert compression_calls == [(("/repo", 2.1), {"device": "gpu", "backend": "bnn"})]
    assert indentation_calls == [(("/repo", 3.2), {"device": "gpu", "backend": "bnn"})]


def test_surrogate_getters_cache_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    compression_builds = []
    indentation_builds = []
    monkeypatch.setattr(
        posterior_compression,
        "_build_surrogate",
        lambda *args, **kwargs: compression_builds.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(
        posterior_indentation,
        "_build_surrogate",
        lambda *args, **kwargs: indentation_builds.append((args, kwargs)) or object(),
    )

    first_compression = posterior_compression._get_surrogate("/repo", 2.9, device="cpu", backend="dnn")
    second_compression = posterior_compression._get_surrogate("/repo", 2.9, device="cpu", backend="dnn")
    first_indentation = posterior_indentation._get_surrogate("/repo", 3.2, device="cpu", backend="dnn")
    second_indentation = posterior_indentation._get_surrogate("/repo", 3.2, device="cpu", backend="dnn")

    assert first_compression is second_compression
    assert first_indentation is second_indentation
    assert len(compression_builds) == 1
    assert len(indentation_builds) == 1


def test_compute_compression_surrogate_dnn_handles_debug_and_d0(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    stub = _CompressionDnnStub()
    monkeypatch.setattr(posterior_compression, "_resolve_project_root", lambda: "/repo")
    monkeypatch.setattr(
        posterior_compression,
        "_load_config",
        lambda _root: {"debug": 1, "surrogate": {"backend": "dnn"}},
    )
    monkeypatch.setattr(
        posterior_compression,
        "expand_parameter_vector",
        lambda params, fixed_params=None: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25, 0.5]),
    )
    monkeypatch.setattr(posterior_compression, "get_fixed_parameters", lambda _config: {})
    monkeypatch.setattr(posterior_compression, "_get_surrogate", lambda *args, **kwargs: stub)

    sample = {"Parameters": [0.0] * 8}
    posterior_compression.compute_compression_surrogate(sample, displ=[0.0, 1.0], diameter_um=2.1)

    assert stub.single_call == {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "disp": [0.0, 0.75],
    }
    assert sample["Reference Evaluations"] == [2.0, 4.0]
    assert sample["Standard Deviation"] == [1.0, 2.0]
    assert "compute_compression_surrogate" in capsys.readouterr().out


def test_compute_indentation_surrogate_dnn_applies_positive_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _IndentationDnnStub()
    monkeypatch.setattr(posterior_indentation, "_resolve_project_root", lambda: "/repo")
    monkeypatch.setattr(
        posterior_indentation,
        "_load_config",
        lambda _root: {"dump": False, "surrogate": {"backend": "dnn"}},
    )
    monkeypatch.setattr(posterior_indentation, "_get_dump_flag", lambda: False)
    monkeypatch.setattr(
        posterior_indentation,
        "expand_parameter_vector",
        lambda params, fixed_params=None: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -2.0, 0.1]),
    )
    monkeypatch.setattr(posterior_indentation, "get_fixed_parameters", lambda _config: {})
    monkeypatch.setattr(posterior_indentation, "_get_surrogate", lambda *args, **kwargs: stub)

    sample = {"Parameters": [0.0] * 8}
    posterior_indentation.compute_indentation_surrogate(sample, forces=[-1.0, 2.0], diameter_um=3.2)

    assert stub.single_call == {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "forces": [-1.0, 2.0],
    }
    assert sample["Reference Evaluations"] == [0.0, 1.0]
    assert sample["Standard Deviation"] == pytest.approx([0.0, 0.1])


def test_compute_compression_surrogate_bnn_preserves_legacy_quadrature(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _CompressionBnnStub()
    monkeypatch.setattr(posterior_compression, "_resolve_project_root", lambda: "/repo")
    monkeypatch.setattr(posterior_compression, "_load_config", lambda _root: {"surrogate": {"backend": "bnn"}})
    monkeypatch.setattr(posterior_compression, "_resolve_surrogate_runtime", lambda _config: ("bnn", 7, 3))
    monkeypatch.setattr(
        posterior_compression,
        "expand_parameter_vector",
        lambda params, fixed_params=None: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25, 0.5]),
    )
    monkeypatch.setattr(posterior_compression, "get_fixed_parameters", lambda _config: {})
    monkeypatch.setattr(posterior_compression, "_get_surrogate", lambda *args, **kwargs: stub)

    sample = {"Parameters": [0.0] * 8}
    posterior_compression.compute_compression_surrogate(sample, displ=[0.0, 1.0], diameter_um=2.1)

    assert stub.single_call == {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "disp": [0.0, 0.75],
        "predictive_mc_samples": 7,
        "predictive_mc_chunk_size": 3,
    }
    assert sample["Reference Evaluations"] == [2.0, 4.0]
    assert sample["Standard Deviation"] == pytest.approx(
        [np.sqrt(0.3**2 + 1.0**2), np.sqrt(0.4**2 + 2.0**2)]
    )


def test_compute_indentation_surrogate_bnn_preserves_legacy_shifted_quadrature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _IndentationBnnStub()
    monkeypatch.setattr(posterior_indentation, "_resolve_project_root", lambda: "/repo")
    monkeypatch.setattr(posterior_indentation, "_load_config", lambda _root: {"dump": False, "surrogate": {"backend": "bnn"}})
    monkeypatch.setattr(posterior_indentation, "_get_dump_flag", lambda: False)
    monkeypatch.setattr(posterior_indentation, "_resolve_surrogate_runtime", lambda _config: ("bnn", 11, 5))
    monkeypatch.setattr(
        posterior_indentation,
        "expand_parameter_vector",
        lambda params, fixed_params=None: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -2.0, 0.1]),
    )
    monkeypatch.setattr(posterior_indentation, "get_fixed_parameters", lambda _config: {})
    monkeypatch.setattr(posterior_indentation, "_get_surrogate", lambda *args, **kwargs: stub)

    sample = {"Parameters": [0.0] * 8}
    posterior_indentation.compute_indentation_surrogate(sample, forces=[-1.0, 2.0], diameter_um=3.2)

    assert stub.single_call == {
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "forces": [-1.0, 2.0],
        "predictive_mc_samples": 11,
        "predictive_mc_chunk_size": 5,
    }
    assert sample["Reference Evaluations"] == [0.0, 1.0]
    assert sample["Standard Deviation"] == pytest.approx([0.2, np.sqrt(0.5**2 + 0.1**2)])


def test_compute_compression_surrogate_batch_supports_all_parameter_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(posterior_compression, "_resolve_project_root", lambda: "/repo")
    monkeypatch.setattr(
        posterior_compression,
        "_load_config",
        lambda _root: {"surrogate": {"backend": "dnn"}},
    )
    monkeypatch.setattr(posterior_compression, "get_fixed_parameters", lambda _config: {})

    cases = []

    def _expanded(batch_params, fixed_params=None):
        cases.append(("expanded", batch_params.copy()))
        return np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7, 0.8],
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.3, 0.2],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(posterior_compression, "expand_reduced_parameters", _expanded)

    for batch_params, expected_d0, expected_sigma in [
        (
            np.asarray(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7, 0.8],
                    [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.3, 0.2],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.7, 0.3], dtype=np.float32),
            np.asarray([0.8, 0.2], dtype=np.float32),
        ),
        (
            np.asarray(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.8],
                    [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.2],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0.8, 0.2], dtype=np.float32),
        ),
        (
            np.asarray(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [9.0, 8.0, 7.0, 6.0],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.7, 0.3], dtype=np.float32),
            np.asarray([0.8, 0.2], dtype=np.float32),
        ),
    ]:
        stub = _CompressionDnnStub()
        monkeypatch.setattr(posterior_compression, "_get_surrogate", lambda *args, **kwargs: stub)
        sample = {"Batch Parameters": batch_params}
        posterior_compression.compute_compression_surrogate_batch(
            sample,
            displ=[0.0, 1.0],
            diameter_um=2.1,
            particle_batch_size=16,
        )
        assert np.allclose(stub.batch_call["d0"], expected_d0)
        assert np.allclose(sample["Batch Standard Deviation"], (expected_sigma[:, None] * np.asarray([[1.0, 2.0], [3.0, 4.0]])).tolist())


def test_compute_indentation_surrogate_batch_supports_all_parameter_layouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(posterior_indentation, "_resolve_project_root", lambda: "/repo")
    monkeypatch.setattr(
        posterior_indentation,
        "_load_config",
        lambda _root: {"surrogate": {"backend": "dnn"}},
    )
    monkeypatch.setattr(posterior_indentation, "get_fixed_parameters", lambda _config: {})

    def _expanded(batch_params, fixed_params=None):
        return np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7, 0.8],
                [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.3, 0.2],
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(posterior_indentation, "expand_reduced_parameters", _expanded)

    for batch_params, expected_d0, expected_sigma in [
        (
            np.asarray(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.7, 0.8],
                    [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.3, 0.2],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.7, 0.3], dtype=np.float32),
            np.asarray([0.8, 0.2], dtype=np.float32),
        ),
        (
            np.asarray(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.8],
                    [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 0.2],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.0, 0.0], dtype=np.float32),
            np.asarray([0.8, 0.2], dtype=np.float32),
        ),
        (
            np.asarray(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [9.0, 8.0, 7.0, 6.0],
                ],
                dtype=np.float32,
            ),
            np.asarray([0.7, 0.3], dtype=np.float32),
            np.asarray([0.8, 0.2], dtype=np.float32),
        ),
    ]:
        stub = _IndentationDnnStub()
        monkeypatch.setattr(posterior_indentation, "_get_surrogate", lambda *args, **kwargs: stub)
        sample = {"Batch Parameters": batch_params}
        posterior_indentation.compute_indentation_surrogate_batch(
            sample,
            forces=[-1.0, 2.0],
            diameter_um=3.2,
            particle_batch_size=16,
        )
        assert np.allclose(stub.batch_call["d0"], expected_d0)
        assert np.allclose(
            sample["Batch Standard Deviation"],
            (expected_sigma[:, None] * np.asarray([[1.0, 2.0], [3.0, 4.0]])).tolist(),
        )


def test_compression_surrogate_batch_rejects_invalid_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(posterior_compression, "_resolve_project_root", lambda: "/repo")
    monkeypatch.setattr(
        posterior_compression,
        "_load_config",
        lambda _root: {"surrogate": {"backend": "dnn"}},
    )
    monkeypatch.setattr(posterior_compression, "get_fixed_parameters", lambda _config: {})
    with pytest.raises(ValueError, match="Expected 2D batch params"):
        posterior_compression.compute_compression_surrogate_batch(
            {"Batch Parameters": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)},
            displ=[0.0],
            diameter_um=2.1,
        )
    with pytest.raises(ValueError, match="Expected 4, 7, or 8 params"):
        posterior_compression.compute_compression_surrogate_batch(
            {"Batch Parameters": np.ones((1, 5), dtype=np.float32)},
            displ=[0.0],
            diameter_um=2.1,
        )


def test_indentation_surrogate_batch_rejects_invalid_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(posterior_indentation, "_resolve_project_root", lambda: "/repo")
    monkeypatch.setattr(
        posterior_indentation,
        "_load_config",
        lambda _root: {"surrogate": {"backend": "dnn"}},
    )
    monkeypatch.setattr(posterior_indentation, "get_fixed_parameters", lambda _config: {})
    with pytest.raises(ValueError, match="Expected 2D batch params"):
        posterior_indentation.compute_indentation_surrogate_batch(
            {"Batch Parameters": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)},
            forces=[0.0],
            diameter_um=3.2,
        )
    with pytest.raises(ValueError, match="Expected 4, 7, or 8 params"):
        posterior_indentation.compute_indentation_surrogate_batch(
            {"Batch Parameters": np.ones((1, 5), dtype=np.float32)},
            forces=[0.0],
            diameter_um=3.2,
        )


def test_compression_prepare_simulation_parameters_updates_templates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_path = tmp_path / "source"
    init_path = tmp_path / "init"
    simu_path = tmp_path / "simu"
    (source_path / "mesh").mkdir(parents=True)
    (init_path / "parameter").mkdir(parents=True)

    commands = []
    adjust_calls = []
    write_calls = []
    monkeypatch.setattr(posterior_compression.os, "system", lambda cmd: commands.append(cmd) or 0)
    monkeypatch.setattr(
        posterior_compression,
        "adjust_simu_params",
        lambda sample_param, filename_1_simu, filename_2_simu: adjust_calls.append(
            (sample_param, filename_1_simu, filename_2_simu)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "parameters",
        types.SimpleNamespace(
            write_parameters=lambda **kwargs: write_calls.append(kwargs),
        ),
    )

    posterior_compression.prepare_simulation_parameters(
        str(source_path) + "/",
        str(init_path) + "/",
        str(simu_path) + "/",
        "00001",
        0.25,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        2.1,
    )

    assert any("mesh/cantilever.off" in command for command in commands)
    assert len(adjust_calls) == 2
    assert write_calls == [
        {
            "source_path": str(init_path) + "/",
            "simu_path": str(simu_path) + "/",
            "simnum": "00001",
        }
    ]


def test_indentation_adjust_and_prepare_simulation_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    param_a = tmp_path / "a.yaml"
    param_b = tmp_path / "b.yaml"
    for path in [param_a, param_b]:
        path.write_text(yaml.safe_dump({"disp": 0.0, "Yt": 1.0}), encoding="utf-8")

    posterior_indentation.adjust_simu_params({"disp": 3.5, "Yt": 4.5}, str(param_a), str(param_b))
    for path in [param_a, param_b]:
        params = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert params["disp"] == pytest.approx(3.5)
        assert params["Yt"] == pytest.approx(4.5)

    source_path = tmp_path / "source"
    init_path = tmp_path / "init"
    simu_path = tmp_path / "simu"
    (init_path / "parameter").mkdir(parents=True)
    commands = []
    adjust_calls = []
    write_calls = []
    monkeypatch.setattr(posterior_indentation.os, "system", lambda cmd: commands.append(cmd) or 0)
    monkeypatch.setattr(
        posterior_indentation,
        "adjust_simu_params",
        lambda sample_param, filename_1_simu, filename_2_simu: adjust_calls.append(
            (sample_param, filename_1_simu, filename_2_simu)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "emb.indentation.src.parameters",
        types.SimpleNamespace(
            write_parameters=lambda **kwargs: write_calls.append(kwargs),
        ),
    )

    posterior_indentation.prepare_simulation_parameters(
        str(source_path),
        str(init_path) + "/",
        str(simu_path) + "/",
        "00001",
        0.25,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        3.2,
    )

    assert any("mkdir -p" in command for command in commands)
    assert len(adjust_calls) == 2
    assert write_calls == [
        {
            "source_path": str(init_path) + "/",
            "simu_path": str(simu_path) + "/",
            "simnum": "00001",
        }
    ]
