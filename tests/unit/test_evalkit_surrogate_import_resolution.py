from __future__ import annotations

import types
from pathlib import Path

from emb.compression.evalkit import posterior_compression
from emb.indentation.evalkit import posterior_indentation


def _fake_module(tag: str):
    module = types.ModuleType(tag)

    class Surrogate:
        def __init__(self, base_dir: str, device: str = "cpu") -> None:
            self.tag = tag
            self.base_dir = base_dir
            self.device = device

    module.Surrogate = Surrogate
    return module


def test_evalkits_resolve_modality_specific_surrogate_modules(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules,
        "emb.compression.surrogate.evaluate",
        _fake_module("compression-dnn"),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "emb.compression.surrogate.evaluate_bnn",
        _fake_module("compression-bnn"),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "emb.indentation.surrogate.evaluate",
        _fake_module("indentation-dnn"),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "emb.indentation.surrogate.evaluate_bnn",
        _fake_module("indentation-bnn"),
    )

    comp_dnn = posterior_compression._build_surrogate("/repo", 2.9, backend="dnn")
    comp_bnn = posterior_compression._build_surrogate("/repo", 2.9, backend="bnn")
    ind_dnn = posterior_indentation._build_surrogate("/repo", 3.2, backend="dnn")
    ind_bnn = posterior_indentation._build_surrogate("/repo", 3.2, backend="bnn")

    assert comp_dnn.tag == "compression-dnn"
    assert comp_bnn.tag == "compression-bnn"
    assert ind_dnn.tag == "indentation-dnn"
    assert ind_bnn.tag == "indentation-bnn"


def test_indentation_surrogate_uses_configured_external_directory(
    monkeypatch, tmp_path: Path
) -> None:
    external_root = tmp_path / "mechanical_surrogates" / "sonovue"
    monkeypatch.setattr(
        posterior_indentation,
        "_load_config",
        lambda _root: {
            "structure": "emb",
            "experiments": [
                {
                    "name": "indentation",
                    "diameters": [3.2],
                    "surrogate_dir": str(external_root),
                }
            ],
        },
    )

    resolved = posterior_indentation._resolve_surrogate_trained_dir("/repo", 3.2)

    assert resolved == external_root / "3.2um" / "trained"
