from __future__ import annotations

import warnings
from pathlib import Path

import pytest

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
