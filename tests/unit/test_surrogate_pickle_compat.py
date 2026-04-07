import pickle
import sys
import types

from meso_uq.surrogate.model import MLP, load_model_states


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
