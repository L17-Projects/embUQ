from __future__ import annotations

import numpy as np

from compression.evalkit import posterior_compression
from indentation.evalkit import posterior_indentation


class _FakeCompressionSurrogate:
    def __init__(self) -> None:
        self.theta = None
        self.disp = None
        self.d0 = None
        self.chunk_size = None

    def evaluate_compression_batch(self, theta, disp, d0, chunk_size):
        self.theta = np.asarray(theta, dtype=np.float32)
        self.disp = list(disp)
        self.d0 = np.asarray(d0, dtype=np.float32)
        self.chunk_size = int(chunk_size)
        out = np.column_stack([self.theta[:, 0] + self.d0, self.theta[:, 1] + self.d0])
        return out.astype(np.float32)


class _FakeIndentationSurrogate:
    def __init__(self) -> None:
        self.theta = None
        self.forces = None
        self.d0 = None
        self.chunk_size = None

    def evaluate_indentation_batch(self, theta, forces, d0, chunk_size):
        self.theta = np.asarray(theta, dtype=np.float32)
        self.forces = list(forces)
        self.d0 = np.asarray(d0, dtype=np.float32)
        self.chunk_size = int(chunk_size)
        out = np.column_stack([self.theta[:, 0] - self.d0, self.theta[:, 1] - self.d0])
        return out.astype(np.float32)


def test_compression_batch_accepts_reduced_four_parameter_vectors(monkeypatch):
    fake_surrogate = _FakeCompressionSurrogate()
    monkeypatch.setattr(posterior_compression, "_resolve_project_root", lambda: "/tmp/project")
    monkeypatch.setattr(
        posterior_compression,
        "_load_config",
        lambda _project_root: {"fixed_params": {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0}},
    )
    monkeypatch.setattr(
        posterior_compression, "_get_surrogate", lambda *_args, **_kwargs: fake_surrogate
    )

    sample = {
        "Batch Parameters": [
            [10.0, 20.0, 0.3, 0.1],
            [11.0, 21.0, 0.4, 0.2],
        ]
    }

    posterior_compression.compute_compression_surrogate_batch(
        sample,
        displ=[0.0, 1.0],
        diameter_um=2.1,
        device="gpu",
        particle_batch_size=17,
    )

    expected_theta = np.array(
        [[10.0, 20.0, 0.0, 0.0, 0.0, 0.0], [11.0, 21.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    expected_d0 = np.array([0.3, 0.4], dtype=np.float32)

    assert fake_surrogate.disp == [0.0, 1.0]
    assert fake_surrogate.chunk_size == 17
    assert np.allclose(fake_surrogate.theta, expected_theta)
    assert np.allclose(fake_surrogate.d0, expected_d0)

    expected_eval = np.column_stack(
        [expected_theta[:, 0] + expected_d0, expected_theta[:, 1] + expected_d0]
    )
    expected_sigma = np.array([0.1, 0.2], dtype=np.float32)
    expected_std = expected_sigma[:, None] * expected_eval

    assert np.allclose(
        np.asarray(sample["Batch Reference Evaluations"], dtype=np.float32), expected_eval
    )
    assert np.allclose(
        np.asarray(sample["Batch Standard Deviation"], dtype=np.float32), expected_std
    )


def test_indentation_batch_accepts_reduced_four_parameter_vectors(monkeypatch):
    fake_surrogate = _FakeIndentationSurrogate()
    monkeypatch.setattr(posterior_indentation, "_resolve_project_root", lambda: "/tmp/project")
    monkeypatch.setattr(
        posterior_indentation,
        "_load_config",
        lambda _project_root: {"fixed_params": {"b1": 0.0, "b2": 0.0, "a3": 0.0, "a4": 0.0}},
    )
    monkeypatch.setattr(
        posterior_indentation, "_get_surrogate", lambda *_args, **_kwargs: fake_surrogate
    )

    sample = {
        "Batch Parameters": [
            [12.0, 22.0, 0.5, 0.3],
            [13.0, 23.0, 0.6, 0.4],
        ]
    }

    posterior_indentation.compute_indentation_surrogate_batch(
        sample,
        forces=[0.5, 1.5],
        diameter_um=3.2,
        device="gpu",
        particle_batch_size=19,
    )

    expected_theta = np.array(
        [[12.0, 22.0, 0.0, 0.0, 0.0, 0.0], [13.0, 23.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    expected_d0 = np.array([0.5, 0.6], dtype=np.float32)

    assert fake_surrogate.forces == [0.5, 1.5]
    assert fake_surrogate.chunk_size == 19
    assert np.allclose(fake_surrogate.theta, expected_theta)
    assert np.allclose(fake_surrogate.d0, expected_d0)

    expected_eval = np.column_stack(
        [expected_theta[:, 0] - expected_d0, expected_theta[:, 1] - expected_d0]
    )
    expected_sigma = np.array([0.3, 0.4], dtype=np.float32)
    expected_std = expected_sigma[:, None] * expected_eval

    assert np.allclose(
        np.asarray(sample["Batch Reference Evaluations"], dtype=np.float32), expected_eval
    )
    assert np.allclose(
        np.asarray(sample["Batch Standard Deviation"], dtype=np.float32), expected_std
    )
