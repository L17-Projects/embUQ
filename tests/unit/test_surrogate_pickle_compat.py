import pickle
import sys
import types
import warnings
from pathlib import Path

from meso_uq.surrogate.model import MLP, load_model_states
from meso_uq.surrogate.compat import list_serialized_surrogate_aliases


def test_load_model_states_accepts_legacy_learning_model_pickle(tmp_path):
    legacy_learning = types.ModuleType("learning")
    legacy_learning_model = types.ModuleType("learning.model")
    legacy_learning_model.MLP = MLP
    legacy_learning.model = legacy_learning_model
    sys.modules["learning"] = legacy_learning
    sys.modules["learning.model"] = legacy_learning_model

    original_module = MLP.__module__
    MLP.__module__ = "learning.model"
    try:
        path = tmp_path / "legacy.pkl"
        payload = {
            "model": MLP(input_dims=2, output_dims=1, hl_dims=[4]),
            "xshift": [0.0, 0.0],
            "xscale": [1.0, 1.0],
            "yshift": [0.0],
            "yscale": [1.0],
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)
    finally:
        MLP.__module__ = original_module
        sys.modules.pop("learning", None)
        sys.modules.pop("learning.model", None)

    model, xshift, xscale, yshift, yscale = load_model_states(path)

    assert type(model).__name__ == "MLP"
    assert list(xshift) == [0.0, 0.0]
    assert list(xscale) == [1.0, 1.0]
    assert list(yshift) == [0.0]
    assert list(yscale) == [1.0]


def test_serialized_surrogate_alias_manifest_records_legacy_learning_model():
    aliases = list_serialized_surrogate_aliases()

    assert len(aliases) == 1
    alias = aliases[0]
    assert alias.legacy_module == "learning.model"
    assert alias.legacy_attribute == "MLP"
    assert alias.replacement_module == "meso_uq.surrogate.model"
    assert alias.replacement_attribute == "MLP"
    assert alias.artifact_format == "pickle"


def test_deprecated_learning_model_import_warns(monkeypatch):
    monkeypatch.delitem(sys.modules, "learning", raising=False)
    monkeypatch.delitem(sys.modules, "learning.model", raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        import learning.model as legacy_model

    assert legacy_model.MLP is MLP
    assert any("deprecated MesoUQ surrogate compatibility import" in str(item.message) for item in caught)


def test_committed_release_surrogate_pickle_loads_with_compat_loader():
    repo_root = Path(__file__).resolve().parents[2]
    path = (
        repo_root
        / "emb"
        / "compression"
        / "surrogate"
        / "diameters"
        / "3.0um"
        / "trained"
        / "microbubble_force_BEST.pkl"
    )

    model, xshift, xscale, yshift, yscale = load_model_states(path)

    assert type(model).__name__ == "MLP"
    assert len(xshift) == len(xscale) > 0
    assert len(yshift) == len(yscale) > 0
