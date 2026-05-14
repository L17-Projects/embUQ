from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("pyro")

from emb.compression.surrogate import evaluate_bnn as compression_evaluate_bnn
from emb.indentation.surrogate import evaluate_bnn as indentation_evaluate_bnn
from meso_uq.surrogate import bnn as bnn_runtime


def test_resolve_torch_device_raises_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bnn_runtime.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        bnn_runtime._resolve_torch_device("cuda")


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


def test_format_v1_reload_is_independent_of_base_model_init_seed(tmp_path: Path) -> None:
    pyro = pytest.importorskip("pyro")
    torch.manual_seed(12345)
    device = torch.device("cpu")
    _, base_model, _, guide = bnn_runtime.build_variational_components(
        input_dim=2,
        width=4,
        depth=1,
        prior_scale=1.0,
        obs_noise_prior_scale=1.0,
        device=device,
    )
    pyro.clear_param_store()
    with torch.inference_mode():
        guide(
            torch.zeros((1, 2), dtype=torch.float32, device=device),
            torch.zeros((1, 1), dtype=torch.float32, device=device),
        )
    store = pyro.get_param_store()
    payload = bnn_runtime.make_artifact_payload(
        xshift=[0.0, 0.0],
        xscale=[1.0, 1.0],
        yshift=[0.0],
        yscale=[1.0],
        input_dim=2,
        width=4,
        depth=1,
        prior_scale=1.0,
        obs_noise_prior_scale=1.0,
        pyro_param_values={
            name: value.detach().clone().cpu() for name, value in store._params.items()
        },
        base_model_state_dict={
            name: value.detach().clone().cpu()
            for name, value in base_model.state_dict().items()
        },
        guide_state_dict={
            name: value.detach().clone().cpu()
            for name, value in guide.state_dict().items()
        },
    )
    artifact = tmp_path / "seed_stable_bnn.pt"
    torch.save(payload, artifact)

    inputs = np.asarray([[0.2, -0.1], [0.5, 0.3]], dtype=np.float32)

    torch.manual_seed(1)
    predictor_a = bnn_runtime.VariationalBNNPredictor(str(artifact), device="cpu")
    torch.manual_seed(20260422)
    mean_a, std_a = predictor_a.predict_mean_std(
        inputs, predictive_mc_samples=16, predictive_mc_chunk_size=16
    )

    torch.manual_seed(999)
    predictor_b = bnn_runtime.VariationalBNNPredictor(str(artifact), device="cpu")
    torch.manual_seed(20260422)
    mean_b, std_b = predictor_b.predict_mean_std(
        inputs, predictive_mc_samples=16, predictive_mc_chunk_size=16
    )

    np.testing.assert_allclose(mean_a, mean_b, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(std_a, std_b, rtol=0.0, atol=1e-6)
