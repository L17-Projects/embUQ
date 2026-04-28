from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from meso_uq.surrogate import bnn as bnn_runtime


class _FakeDistribution:
    def to_event(self, dim: int) -> "_FakeDistribution":
        _ = dim
        return self


class _FakeScale:
    def __init__(self, records: list[float], scale: float) -> None:
        self._records = records
        self._scale = float(scale)

    def __enter__(self) -> None:
        self._records.append(self._scale)
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type, exc, tb
        return False


class _FakePlate:
    def __init__(self, sizes: list[int], size: int) -> None:
        self._sizes = sizes
        self._size = int(size)

    def __enter__(self) -> None:
        self._sizes.append(self._size)
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type, exc, tb
        return False


def _fake_pyro(records: dict[str, list[object]]) -> SimpleNamespace:
    def _sample(name: str, dist, obs=None):  # noqa: ANN001
        _ = dist
        if name == "obs_noise":
            return torch.tensor(1.0)
        records["samples"].append((name, obs))
        return obs

    pyro = SimpleNamespace()
    pyro.distributions = SimpleNamespace(
        HalfNormal=lambda scale: _FakeDistribution(),
        Normal=lambda mean, scale: _FakeDistribution(),
    )
    pyro.sample = _sample
    pyro.random_module = lambda name, base_model, priors: (lambda: base_model)
    pyro.plate = lambda name, size: _FakePlate(records["plate_sizes"], size)
    pyro.poutine = SimpleNamespace(scale=lambda scale: _FakeScale(records["scales"], scale))
    pyro.infer = SimpleNamespace(
        autoguide=SimpleNamespace(AutoDiagonalNormal=lambda model: object()),
    )
    return pyro


def test_build_variational_components_scales_minibatch_likelihood(monkeypatch: pytest.MonkeyPatch) -> None:
    records: dict[str, list[object]] = {
        "samples": [],
        "scales": [],
        "plate_sizes": [],
    }
    monkeypatch.setattr(bnn_runtime, "_require_pyro", lambda: _fake_pyro(records))

    _, _, model, _ = bnn_runtime.build_variational_components(
        input_dim=2,
        width=2,
        depth=1,
        prior_scale=1.0,
        obs_noise_prior_scale=1.0,
        training_dataset_size=10,
        device=torch.device("cpu"),
    )

    x = torch.ones((4, 2), dtype=torch.float32)
    y = torch.zeros((4, 1), dtype=torch.float32)
    mean = model(x, y)

    assert mean.shape == (4, 1)
    assert records["plate_sizes"] == [4]
    assert records["scales"] == [pytest.approx(2.5)]
    assert len(records["samples"]) == 1
    assert records["samples"][0][0] == "obs"
    assert torch.equal(records["samples"][0][1], torch.zeros(4, dtype=torch.float32))


def test_build_variational_components_does_not_scale_predictive_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    records: dict[str, list[object]] = {
        "samples": [],
        "scales": [],
        "plate_sizes": [],
    }
    monkeypatch.setattr(bnn_runtime, "_require_pyro", lambda: _fake_pyro(records))

    _, _, model, _ = bnn_runtime.build_variational_components(
        input_dim=2,
        width=2,
        depth=1,
        prior_scale=1.0,
        obs_noise_prior_scale=1.0,
        training_dataset_size=10,
        device=torch.device("cpu"),
    )

    x = torch.ones((3, 2), dtype=torch.float32)
    mean = model(x, None)

    assert mean.shape == (3, 1)
    assert records["plate_sizes"] == [3]
    assert records["scales"] == []
    assert records["samples"] == [("obs", None)]


def test_build_variational_components_rejects_invalid_training_dataset_size() -> None:
    with pytest.raises(ValueError, match="training_dataset_size must be >= 1"):
        bnn_runtime.build_variational_components(
            input_dim=1,
            width=1,
            depth=1,
            prior_scale=1.0,
            obs_noise_prior_scale=1.0,
            training_dataset_size=0,
            device=torch.device("cpu"),
        )
