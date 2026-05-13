import importlib


def test_public_package_imports_smoke():
    modules = [
        "meso_uq",
        "meso_uq.postprocess.diagnostics",
        "meso_uq.postprocess.maps",
        "meso_uq.postprocess.plots",
        "meso_uq.postprocess.propagation",
        "meso_uq.surrogate.model_selection",
        "meso_uq.sensitivity",
        "meso_uq.config.loader",
        "meso_uq.experiments",
    ]
    for name in modules:
        mod = importlib.import_module(name)
        assert mod is not None, f"failed to import {name}"
