from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from compression.surrogate import evaluate as compression_evaluate
from indentation.surrogate import evaluate as indentation_evaluate


class _CompressionModel:
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, -1:].clone()


class _IndentationModel:
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, -1:].clone()


def test_compression_surrogate_evaluates_and_clips_negative_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def test_indentation_surrogate_uses_fallback_weights_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
