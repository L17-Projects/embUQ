from __future__ import annotations

import numpy as np
import pytest

from emb.compression.evalkit import posterior_compression


class _DummyLegacySurrogate:
    def evaluate_compression(self, x, disp):
        yt, kb, b1, b2, a3, a4 = x
        offset = float(yt) + float(kb) + float(b1) + float(b2) + float(a3) + float(a4)
        return [offset + float(value) for value in disp]

    def evaluate_compression_batch(self, x, disp, d0=None, chunk_size=2048):
        params = np.asarray(x, dtype=float)
        d0_arr = np.zeros(params.shape[0], dtype=float) if d0 is None else np.asarray(d0, dtype=float)
        rows = []
        for theta, offset in zip(params, d0_arr):
            rows.append([float(np.sum(theta)) + max(0.0, float(value) - float(offset)) for value in disp])
        return np.asarray(rows, dtype=float)


def test_compute_compression_surrogate_direct_applies_d0_and_relative_sigma(monkeypatch) -> None:
    monkeypatch.setattr(posterior_compression, "_resolve_project_root", lambda: "/tmp/project")
    monkeypatch.setattr(posterior_compression, "_load_config", lambda _project_root: {})
    monkeypatch.setattr(
        posterior_compression,
        "direct_parameters_to_legacy_vector",
        lambda params, **_kwargs: np.asarray([100.0, 20.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.1]),
    )
    monkeypatch.setattr(
        posterior_compression,
        "_get_surrogate",
        lambda *args, **kwargs: _DummyLegacySurrogate(),
    )
    sample = {"Parameters": [10.0, 20.0, 0.5, 0.1]}

    posterior_compression.compute_compression_surrogate_direct(sample, [0.25, 0.75], 4.10)

    assert sample["Reference Evaluations"] == pytest.approx([130.0, 130.25])
    assert sample["Standard Deviation"] == pytest.approx([13.0, 13.025])


def test_compute_compression_surrogate_batch_direct_uses_ka_kb_d0_sigma(monkeypatch) -> None:
    monkeypatch.setattr(posterior_compression, "_resolve_project_root", lambda: "/tmp/project")
    monkeypatch.setattr(posterior_compression, "_load_config", lambda _project_root: {})
    monkeypatch.setattr(
        posterior_compression,
        "direct_parameters_to_legacy_batch",
        lambda params, **_kwargs: np.asarray(
            [
                [100.0, 20.0, 1.0, 2.0, 3.0, 4.0, 0.5, 0.1],
                [10.0, 2.0, 0.5, 0.25, 0.125, 0.0, 0.0, 0.2],
            ],
            dtype=float,
        ),
    )
    monkeypatch.setattr(
        posterior_compression,
        "_get_surrogate",
        lambda *args, **kwargs: _DummyLegacySurrogate(),
    )
    sample = {
        "Batch Parameters": np.asarray(
            [
                [10.0, 20.0, 0.5, 0.1],
                [1.0, 2.0, 0.0, 0.2],
            ],
            dtype=float,
        )
    }

    posterior_compression.compute_compression_surrogate_batch_direct(sample, [0.25, 0.75], 4.10)

    np.testing.assert_allclose(
        sample["Batch Reference Evaluations"],
        [
            [130.0, 130.25],
            [13.125, 13.625],
        ],
    )
    np.testing.assert_allclose(
        sample["Batch Standard Deviation"],
        [
            [13.0, 13.025],
            [2.625, 2.725],
        ],
    )
