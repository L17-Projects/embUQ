from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from compression.surrogate import evaluate as compression_evaluate
from compression.surrogate import evaluate_bnn as compression_evaluate_bnn
from indentation.surrogate import evaluate as indentation_evaluate
from indentation.surrogate import evaluate_bnn as indentation_evaluate_bnn


class _CompressionModel:
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, -1:].clone()


class _IndentationModel:
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, -1:].clone()


class _ModelWithTorchHooks:
    def __init__(self) -> None:
        self.to_calls: list[str] = []
        self.eval_calls = 0
        self.buffers: dict[str, tuple[torch.Tensor, bool]] = {}

    def to(self, device: torch.device):
        self.to_calls.append(str(device))
        return self

    def eval(self):
        self.eval_calls += 1
        return self

    def register_buffer(self, name: str, tensor: torch.Tensor, persistent: bool = True) -> None:
        self.buffers[name] = (tensor, persistent)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, -1:].clone()


def test_compression_resolve_torch_device_maps_gpu_alias_to_cuda() -> None:
    assert compression_evaluate._resolve_torch_device("gpu").type == "cuda"


def test_indentation_resolve_torch_device_maps_gpu_alias_to_cuda() -> None:
    assert indentation_evaluate._resolve_torch_device("gpu").type == "cuda"


def test_compression_surrogate_evaluates_and_clips_negative_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = {}

    def fake_load_model_states(path: str):
        captured["path"] = path
        return (
            _CompressionModel(),
            np.zeros(7),
            np.ones(7),
            np.array([0.0]),
            np.array([1.0]),
        )

    monkeypatch.setattr(compression_evaluate, "load_model_states", fake_load_model_states)

    surrogate = compression_evaluate.Surrogate(str(tmp_path))
    values = surrogate.evaluate_compression(
        [10.0, 20.0, 1.0, 2.0, 3.0, 4.0],
        [-1.5, 2.5],
    )

    assert captured["path"] == str(tmp_path / "microbubble_force_BEST.pkl")
    assert values == pytest.approx([0.0, 2.5])
    assert surrogate.evaluate_compression([10.0, 20.0, 1.0, 2.0, 3.0, 4.0], []) == []


def test_indentation_surrogate_uses_fallback_weights_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fallback = tmp_path / "microbubble_disp_BEST.pkl"
    fallback.write_text("weights", encoding="utf-8")
    captured = {}

    def fake_load_model_states(path: str):
        captured["path"] = path
        return (
            _IndentationModel(),
            np.zeros(7),
            np.ones(7),
            np.array([0.0]),
            np.array([1.0]),
        )

    monkeypatch.setattr(indentation_evaluate, "load_model_states", fake_load_model_states)

    surrogate = indentation_evaluate.Surrogate(str(tmp_path))
    values = surrogate.evaluate_indentation(
        [10.0, 20.0, 1.0, 2.0, 3.0, 4.0],
        [-1.0, 1.5],
    )

    assert captured["path"] == str(fallback)
    assert values == pytest.approx([0.0, 1.5])
    assert surrogate.evaluate_indentation([10.0, 20.0, 1.0, 2.0, 3.0, 4.0], []) == []


def test_indentation_surrogate_requires_weights_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not find indentation surrogate weights"):
        indentation_evaluate.Surrogate(str(tmp_path))


def test_compression_surrogate_initialization_moves_and_registers_tensors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _ModelWithTorchHooks()

    def fake_load_model_states(path: str):
        return (
            model,
            np.zeros(7),
            np.ones(7),
            np.array([0.0]),
            np.array([1.0]),
        )

    monkeypatch.setattr(compression_evaluate, "load_model_states", fake_load_model_states)

    compression_evaluate.Surrogate(str(tmp_path), device="cpu")

    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert set(model.buffers) == {
        "mesouq_cp_xscale",
        "mesouq_cp_xshift",
        "mesouq_cp_yscale",
        "mesouq_cp_yshift",
    }
    assert all(not persistent for _, persistent in model.buffers.values())
    assert all(tensor.device.type == "cpu" for tensor, _ in model.buffers.values())


def test_indentation_surrogate_initialization_moves_and_registers_tensors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fallback = tmp_path / "microbubble_disp_BEST.pkl"
    fallback.write_text("weights", encoding="utf-8")
    model = _ModelWithTorchHooks()

    def fake_load_model_states(path: str):
        return (
            model,
            np.zeros(7),
            np.ones(7),
            np.array([0.0]),
            np.array([1.0]),
        )

    monkeypatch.setattr(indentation_evaluate, "load_model_states", fake_load_model_states)

    indentation_evaluate.Surrogate(str(tmp_path), device="cpu")

    assert model.to_calls == ["cpu"]
    assert model.eval_calls == 1
    assert set(model.buffers) == {
        "mesouq_xshift",
        "mesouq_xscale",
        "mesouq_yshift",
        "mesouq_yscale",
    }
    assert all(not persistent for _, persistent in model.buffers.values())
    assert all(tensor.device.type == "cpu" for tensor, _ in model.buffers.values())


def test_compression_surrogate_batch_evaluates_chunked_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _ModelWithTorchHooks()

    def fake_load_model_states(path: str):
        return (
            model,
            np.zeros(7),
            np.ones(7),
            np.array([0.0]),
            np.array([1.0]),
        )

    monkeypatch.setattr(compression_evaluate, "load_model_states", fake_load_model_states)
    surrogate = compression_evaluate.Surrogate(str(tmp_path), device="cpu")
    if not hasattr(surrogate, "evaluate_compression_batch"):
        pytest.skip("Batch compression evaluator not available on this branch.")

    theta = np.array(
        [
            [10.0, 20.0, 1.0, 2.0, 3.0, 4.0],
            [11.0, 21.0, 1.1, 2.1, 3.1, 4.1],
        ],
        dtype=np.float32,
    )
    d0 = np.array([0.0, 0.5], dtype=np.float32)
    out = surrogate.evaluate_compression_batch(theta, disp=[0.0, 1.0], d0=d0, chunk_size=1)

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, np.array([[0.0, 1.0], [0.0, 0.5]], dtype=np.float32))

    empty = surrogate.evaluate_compression_batch(
        np.empty((0, 6), dtype=np.float32), disp=[0.0, 1.0], chunk_size=0
    )
    assert empty.shape == (0, 2)


def test_compression_surrogate_batch_validates_input_shapes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _ModelWithTorchHooks()

    def fake_load_model_states(path: str):
        return (
            model,
            np.zeros(7),
            np.ones(7),
            np.array([0.0]),
            np.array([1.0]),
        )

    monkeypatch.setattr(compression_evaluate, "load_model_states", fake_load_model_states)
    surrogate = compression_evaluate.Surrogate(str(tmp_path), device="cpu")
    if not hasattr(surrogate, "evaluate_compression_batch"):
        pytest.skip("Batch compression evaluator not available on this branch.")

    with pytest.raises(ValueError, match="Expected parameter array of shape"):
        surrogate.evaluate_compression_batch(np.ones((2, 5), dtype=np.float32), disp=[0.0, 1.0])
    with pytest.raises(ValueError, match="Expected 1D displacement array"):
        surrogate.evaluate_compression_batch(np.ones((2, 6), dtype=np.float32), disp=[[0.0, 1.0]])
    with pytest.raises(ValueError, match="Expected d0 shape"):
        surrogate.evaluate_compression_batch(
            np.ones((2, 6), dtype=np.float32), disp=[0.0, 1.0], d0=np.array([0.0], dtype=np.float32)
        )


def test_indentation_surrogate_batch_evaluates_chunked_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fallback = tmp_path / "microbubble_disp_BEST.pkl"
    fallback.write_text("weights", encoding="utf-8")
    model = _ModelWithTorchHooks()

    def fake_load_model_states(path: str):
        return (
            model,
            np.zeros(7),
            np.ones(7),
            np.array([0.0]),
            np.array([1.0]),
        )

    monkeypatch.setattr(indentation_evaluate, "load_model_states", fake_load_model_states)
    surrogate = indentation_evaluate.Surrogate(str(tmp_path), device="cpu")
    if not hasattr(surrogate, "evaluate_indentation_batch"):
        pytest.skip("Batch indentation evaluator not available on this branch.")

    theta = np.array(
        [
            [10.0, 20.0, 1.0, 2.0, 3.0, 4.0],
            [11.0, 21.0, 1.1, 2.1, 3.1, 4.1],
        ],
        dtype=np.float32,
    )
    d0 = np.array([0.2, 0.0], dtype=np.float32)
    out = surrogate.evaluate_indentation_batch(theta, forces=[-1.0, 2.0], d0=d0, chunk_size=1)

    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, np.array([[0.0, 2.2], [0.0, 2.0]], dtype=np.float32))

    empty = surrogate.evaluate_indentation_batch(
        np.empty((0, 6), dtype=np.float32), forces=[-1.0, 2.0], chunk_size=0
    )
    assert empty.shape == (0, 2)


def test_indentation_surrogate_batch_validates_input_shapes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fallback = tmp_path / "microbubble_disp_BEST.pkl"
    fallback.write_text("weights", encoding="utf-8")
    model = _ModelWithTorchHooks()

    def fake_load_model_states(path: str):
        return (
            model,
            np.zeros(7),
            np.ones(7),
            np.array([0.0]),
            np.array([1.0]),
        )

    monkeypatch.setattr(indentation_evaluate, "load_model_states", fake_load_model_states)
    surrogate = indentation_evaluate.Surrogate(str(tmp_path), device="cpu")
    if not hasattr(surrogate, "evaluate_indentation_batch"):
        pytest.skip("Batch indentation evaluator not available on this branch.")

    with pytest.raises(ValueError, match="Expected parameter array of shape"):
        surrogate.evaluate_indentation_batch(np.ones((2, 5), dtype=np.float32), forces=[0.0, 1.0])
    with pytest.raises(ValueError, match="Expected 1D force array"):
        surrogate.evaluate_indentation_batch(np.ones((2, 6), dtype=np.float32), forces=[[0.0, 1.0]])
    with pytest.raises(ValueError, match="Expected d0 shape"):
        surrogate.evaluate_indentation_batch(
            np.ones((2, 6), dtype=np.float32),
            forces=[0.0, 1.0],
            d0=np.array([0.0], dtype=np.float32),
        )


def test_compression_bnn_surrogate_batch_honors_chunk_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "microbubble_force_BNN.pt"
    artifact.write_text("artifact", encoding="utf-8")
    call_sizes: list[int] = []

    class _FakeBNNPredictor:
        def __init__(self, artifact_path: str, *, device: str = "cpu") -> None:
            assert artifact_path == str(artifact)
            assert device == "cpu"

        def predict_mean_std(
            self,
            inputs: np.ndarray,
            *,
            predictive_mc_samples: int = 32,
            predictive_mc_chunk_size: int = 8,
        ) -> tuple[np.ndarray, np.ndarray]:
            del predictive_mc_samples, predictive_mc_chunk_size
            arr = np.asarray(inputs, dtype=np.float32)
            call_sizes.append(int(arr.shape[0]))
            return arr[:, -1], np.full(arr.shape[0], 0.25, dtype=np.float32)

    monkeypatch.setattr(compression_evaluate_bnn, "VariationalBNNPredictor", _FakeBNNPredictor)
    surrogate = compression_evaluate_bnn.Surrogate(str(tmp_path), device="cpu")

    theta = np.array(
        [
            [10.0, 20.0, 1.0, 2.0, 3.0, 4.0],
            [11.0, 21.0, 1.1, 2.1, 3.1, 4.1],
            [12.0, 22.0, 1.2, 2.2, 3.2, 4.2],
            [13.0, 23.0, 1.3, 2.3, 3.3, 4.3],
            [14.0, 24.0, 1.4, 2.4, 3.4, 4.4],
        ],
        dtype=np.float32,
    )
    d0 = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    mean, std = surrogate.evaluate_compression_batch(theta, disp=[0.0, 1.0, 2.0], d0=d0, chunk_size=2)

    assert call_sizes == [6, 6, 3]
    expected = np.maximum(
        0.0,
        np.asarray([0.0, 1.0, 2.0], dtype=np.float32)[None, :] - d0[:, None],
    )
    np.testing.assert_allclose(mean, expected)
    np.testing.assert_allclose(std, np.full((5, 3), 0.25, dtype=np.float32))


def test_compression_bnn_surrogate_scalar_eval_clips_and_forwards_mc_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "microbubble_force_BNN.pt"
    artifact.write_text("artifact", encoding="utf-8")
    calls = {"n": 0}

    class _FakeBNNPredictor:
        def __init__(self, artifact_path: str, *, device: str = "cpu") -> None:
            assert artifact_path == str(artifact)
            assert device == "cpu"

        def predict_mean_std(
            self,
            inputs: np.ndarray,
            *,
            predictive_mc_samples: int = 32,
            predictive_mc_chunk_size: int = 8,
        ) -> tuple[np.ndarray, np.ndarray]:
            calls["n"] += 1
            arr = np.asarray(inputs, dtype=np.float32)
            assert arr.shape == (3, 7)
            assert predictive_mc_samples == 11
            assert predictive_mc_chunk_size == 5
            return np.asarray([-1.0, 0.5, 2.0], dtype=np.float32), np.full(3, 0.3, dtype=np.float32)

    monkeypatch.setattr(compression_evaluate_bnn, "VariationalBNNPredictor", _FakeBNNPredictor)
    surrogate = compression_evaluate_bnn.Surrogate(str(tmp_path), device="cpu")

    mean, std = surrogate.evaluate_compression(
        [10.0, 20.0, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0],
        predictive_mc_samples=11,
        predictive_mc_chunk_size=5,
    )
    assert calls["n"] == 1
    np.testing.assert_allclose(np.asarray(mean, dtype=np.float32), np.asarray([0.0, 0.5, 2.0], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(std, dtype=np.float32), np.full(3, 0.3, dtype=np.float32))

    mean_empty, std_empty = surrogate.evaluate_compression([10.0, 20.0, 1.0, 2.0, 3.0, 4.0], [])
    assert mean_empty == []
    assert std_empty == []
    assert calls["n"] == 1


def test_indentation_bnn_surrogate_batch_honors_chunk_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "microbubble_displacement_BNN.pt"
    artifact.write_text("artifact", encoding="utf-8")
    call_sizes: list[int] = []

    class _FakeBNNPredictor:
        def __init__(self, artifact_path: str, *, device: str = "cpu") -> None:
            assert artifact_path == str(artifact)
            assert device == "cpu"

        def predict_mean_std(
            self,
            inputs: np.ndarray,
            *,
            predictive_mc_samples: int = 32,
            predictive_mc_chunk_size: int = 8,
        ) -> tuple[np.ndarray, np.ndarray]:
            del predictive_mc_samples, predictive_mc_chunk_size
            arr = np.asarray(inputs, dtype=np.float32)
            call_sizes.append(int(arr.shape[0]))
            return arr[:, -1], np.full(arr.shape[0], 0.125, dtype=np.float32)

    monkeypatch.setattr(indentation_evaluate_bnn, "VariationalBNNPredictor", _FakeBNNPredictor)
    surrogate = indentation_evaluate_bnn.Surrogate(str(tmp_path), device="cpu")

    theta = np.array(
        [
            [10.0, 20.0, 1.0, 2.0, 3.0, 4.0],
            [11.0, 21.0, 1.1, 2.1, 3.1, 4.1],
            [12.0, 22.0, 1.2, 2.2, 3.2, 4.2],
            [13.0, 23.0, 1.3, 2.3, 3.3, 4.3],
            [14.0, 24.0, 1.4, 2.4, 3.4, 4.4],
        ],
        dtype=np.float32,
    )
    d0 = np.array([0.2, 0.0, 1.0, 0.5, -0.1], dtype=np.float32)
    mean, std = surrogate.evaluate_indentation_batch(
        theta,
        forces=[-1.0, 2.0],
        d0=d0,
        chunk_size=2,
    )

    assert call_sizes == [4, 4, 2]
    expected = np.maximum(
        0.0,
        np.asarray([-1.0, 2.0], dtype=np.float32)[None, :] + d0[:, None],
    )
    np.testing.assert_allclose(mean, expected)
    np.testing.assert_allclose(std, np.full((5, 2), 0.125, dtype=np.float32))
