from __future__ import annotations

import numpy as np
import pytest

from compression.evalkit import posterior_compression as posterior_compression
from indentation.evalkit import posterior_indentation as posterior_indentation


class _CompressionBnnStub:
    def evaluate_compression(self, **kwargs):
        return [2.0, 4.0], [0.3, 0.4]


class _IndentationBnnStub:
    def evaluate_indentation(self, **kwargs):
        return [1.0, 3.0], [0.2, 0.5]


def test_compression_resolve_surrogate_runtime_defaults_to_dnn() -> None:
    backend, mc_samples, mc_chunk = posterior_compression._resolve_surrogate_runtime({})
    assert (backend, mc_samples, mc_chunk) == ("dnn", 32, 8)


def test_compression_resolve_surrogate_runtime_rejects_invalid_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported surrogate backend"):
        posterior_compression._resolve_surrogate_runtime({"surrogate": {"backend": "foo"}})


def test_indentation_resolve_surrogate_runtime_defaults_to_dnn() -> None:
    backend, mc_samples, mc_chunk = posterior_indentation._resolve_surrogate_runtime({})
    assert (backend, mc_samples, mc_chunk) == ("dnn", 32, 8)


def test_indentation_resolve_surrogate_runtime_rejects_invalid_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported surrogate backend"):
        posterior_indentation._resolve_surrogate_runtime({"surrogate": {"backend": "foo"}})


def test_compression_bnn_combines_epistemic_and_multiplicative_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior_compression._CONFIG_CACHE.clear()
    posterior_compression._SURROGATE_CACHE.clear()
    monkeypatch.setattr(posterior_compression, "_resolve_project_root", lambda: "/tmp/project")
    monkeypatch.setattr(
        posterior_compression,
        "_load_config",
        lambda _: {
            "surrogate": {
                "backend": "bnn",
                "predictive_mc_samples": 16,
                "predictive_mc_chunk_size": 4,
            }
        },
    )
    monkeypatch.setattr(
        posterior_compression,
        "expand_parameter_vector",
        lambda params, fixed_params=None: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.1, 0.5]),
    )
    monkeypatch.setattr(posterior_compression, "get_fixed_parameters", lambda _: {})
    monkeypatch.setattr(posterior_compression, "_get_surrogate", lambda *args, **kwargs: _CompressionBnnStub())

    sample = {"Parameters": [0.0] * 8}
    posterior_compression.compute_compression_surrogate(sample, displ=[0.0, 1.0], diameter_um=2.1)

    assert sample["Reference Evaluations"] == pytest.approx([2.0, 4.0])
    expected = np.sqrt(np.array([0.3**2 + 1.0**2, 0.4**2 + 2.0**2]))
    assert np.asarray(sample["Standard Deviation"]) == pytest.approx(expected)


def test_indentation_bnn_combines_epistemic_and_multiplicative_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posterior_indentation._CONFIG_CACHE.clear()
    posterior_indentation._SURROGATE_CACHE.clear()
    monkeypatch.setattr(posterior_indentation, "_resolve_project_root", lambda: "/tmp/project")
    monkeypatch.setattr(
        posterior_indentation,
        "_load_config",
        lambda _: {
            "dump": False,
            "surrogate": {
                "backend": "bnn",
                "predictive_mc_samples": 16,
                "predictive_mc_chunk_size": 4,
            },
        },
    )
    monkeypatch.setattr(
        posterior_indentation,
        "expand_parameter_vector",
        lambda params, fixed_params=None: np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.25, 0.1]),
    )
    monkeypatch.setattr(posterior_indentation, "get_fixed_parameters", lambda _: {})
    monkeypatch.setattr(posterior_indentation, "_get_surrogate", lambda *args, **kwargs: _IndentationBnnStub())

    sample = {"Parameters": [0.0] * 8}
    posterior_indentation.compute_indentation_surrogate(
        sample, forces=[-1.0, 2.0], diameter_um=3.2
    )

    # d0 shifts the BNN mean exactly as in DNN path.
    assert sample["Reference Evaluations"] == pytest.approx([1.25, 3.25])
    expected = np.sqrt(np.array([0.2**2 + 0.125**2, 0.5**2 + 0.325**2]))
    assert np.asarray(sample["Standard Deviation"]) == pytest.approx(expected)
