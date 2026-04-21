from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("pyro")

from compression.surrogate import evaluate_bnn as compression_evaluate_bnn
from indentation.surrogate import evaluate_bnn as indentation_evaluate_bnn
from meso_uq.surrogate import bnn as bnn_runtime


def test_resolve_torch_device_falls_back_to_cpu_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bnn_runtime.torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        resolved = bnn_runtime._resolve_torch_device("cuda")
    assert resolved.type == "cpu"
    assert any("Falling back to CPU" in str(w.message) for w in captured)


def test_compression_bnn_surrogate_requires_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="compression BNN artifact"):
        compression_evaluate_bnn.Surrogate(str(tmp_path))


def test_indentation_bnn_surrogate_requires_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="indentation BNN artifact"):
        indentation_evaluate_bnn.Surrogate(str(tmp_path))


def test_compression_bnn_surrogate_accepts_single_discovered_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    discovered = tmp_path / "my_custom_bnn.pt"
    discovered.write_text("placeholder", encoding="utf-8")
    captured: dict[str, str] = {}

    class _DummyPredictor:
        def __init__(self, artifact_path: str, *, device: str = "cpu") -> None:
            captured["artifact"] = artifact_path
            captured["device"] = device

    monkeypatch.setattr(compression_evaluate_bnn, "VariationalBNNPredictor", _DummyPredictor)
    compression_evaluate_bnn.Surrogate(str(tmp_path), device="cpu")
    assert captured["artifact"] == str(discovered)


def test_indentation_bnn_surrogate_accepts_single_discovered_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    discovered = tmp_path / "indentation_BNN_variant.pth"
    discovered.write_text("placeholder", encoding="utf-8")
    captured: dict[str, str] = {}

    class _DummyPredictor:
        def __init__(self, artifact_path: str, *, device: str = "cpu") -> None:
            captured["artifact"] = artifact_path
            captured["device"] = device

    monkeypatch.setattr(indentation_evaluate_bnn, "VariationalBNNPredictor", _DummyPredictor)
    indentation_evaluate_bnn.Surrogate(str(tmp_path), device="cpu")
    assert captured["artifact"] == str(discovered)


def test_variational_predictors_restore_own_pyro_param_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pyro = pytest.importorskip("pyro")

    def _write_artifact(path: Path, *, width: int) -> None:
        pyro = bnn_runtime._require_pyro()
        pyro.clear_param_store()
        device = torch.device("cpu")
        _, _, _, guide = bnn_runtime.build_variational_components(
            input_dim=2,
            width=width,
            depth=1,
            prior_scale=1.0,
            obs_noise_prior_scale=1.0,
            device=device,
        )
        with torch.inference_mode():
            guide(torch.zeros((1, 2), dtype=torch.float32, device=device))
        store = pyro.get_param_store()
        payload = bnn_runtime.make_artifact_payload(
            xshift=[0.0, 0.0],
            xscale=[1.0, 1.0],
            yshift=[0.0],
            yscale=[1.0],
            input_dim=2,
            width=width,
            depth=1,
            prior_scale=1.0,
            obs_noise_prior_scale=1.0,
            pyro_param_values={
                name: value.detach().clone().cpu() for name, value in store._params.items()
            },
        )
        torch.save(payload, path)

    artifact_a = tmp_path / "bnn_a.pt"
    artifact_b = tmp_path / "bnn_b.pt"
    _write_artifact(artifact_a, width=4)
    _write_artifact(artifact_b, width=5)

    class _FakePredictive:
        def __init__(self, model: object, guide: object, num_samples: int, return_sites: tuple[str, ...]):
            self.num_samples = int(num_samples)
            self.return_sites = tuple(return_sites)

        def __call__(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
            loc = pyro.get_param_store()._params["AutoDiagonalNormal.loc"]
            out = torch.full(
                (self.num_samples, int(inputs.shape[0])),
                float(loc.numel()),
                dtype=torch.float32,
                device=inputs.device,
            )
            return {self.return_sites[0]: out}

    monkeypatch.setattr(pyro.infer, "Predictive", _FakePredictive)

    predictor_a = bnn_runtime.VariationalBNNPredictor(str(artifact_a), device="cpu")
    predictor_b = bnn_runtime.VariationalBNNPredictor(str(artifact_b), device="cpu")

    inputs = np.asarray([[0.2, -0.1], [0.5, 0.3]], dtype=np.float32)
    mean_a, _ = predictor_a.predict_mean_std(
        inputs, predictive_mc_samples=2, predictive_mc_chunk_size=1
    )
    mean_b, _ = predictor_b.predict_mean_std(
        inputs, predictive_mc_samples=2, predictive_mc_chunk_size=1
    )
    assert not np.allclose(mean_a, mean_b)
