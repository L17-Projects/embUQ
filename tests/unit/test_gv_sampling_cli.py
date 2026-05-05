from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.structures.gv.sampling import artifacts

SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_sampling.py"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _material_arguments() -> list[str]:
    return [
        "ka=1.1",
        "kb=1.2",
        "mu=0.9",
        "b1=0.1",
        "b2=0.2",
        "a3=0.3",
        "a4=0.4",
        "mu_l=0.5",
        "c=0.6",
    ]


def test_sampling_cli_invokes_sample_gv_with_expected_args(tmp_path: Path) -> None:
    module = _load_module("mesouq_test_gv_sampling_cli")
    fixture = {
        "controls": {"tot_force": 600.0, "bpress": -91.0},
        "channels": {
            "tot_force": [500.0, 600.0],
            "force": [1.0, 2.0],
            "displacement": [0.0, 0.1],
        },
    }
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(fixture, sort_keys=True), encoding="utf-8")

    captured = {}

    def _fake_sample_gv(**kwargs):
        captured.update(kwargs)
        return artifacts.GVSamplingArtifactsResult(
            manifest={"campaign_id": kwargs["campaign_id"], "experiment": kwargs["experiment"], "dataset_id": "fake"},
            channels=kwargs.get("controls", {}).copy() if kwargs.get("channels") is None else dict(kwargs["channels"]),
            manifest_path=tmp_path / "manifest.json",
            hdf5_path=tmp_path / "numerical_dataset.h5",
            campaign_id=kwargs["campaign_id"],
            experiment=kwargs["experiment"],
        )

    module.sample_gv = _fake_sample_gv  # type: ignore[assignment]
    output_root = tmp_path / "sampling-run"

    argv = [
        "--selection",
        "gv:stretching",
        "--campaign-id",
        "campaign-cli",
        "--control",
        "tot_force=600",
        "--control",
        "bpress=-91",
        "--radius",
        "2.5",
        "--height",
        "15.0",
        "--fixture-path",
        str(fixture_path),
        "--output-root",
        str(output_root),
        "--dry-run",
    ]
    for item in _material_arguments():
        argv.extend(["--material", item])

    rc = module.main(argv)

    assert rc == 0
    assert captured["campaign_id"] == "campaign-cli"
    assert captured["experiment"] == "stretching"
    assert captured["geometry"] == {"radGV": 2.5, "height": 15.0}
    assert captured["controls"] == {"tot_force": (600.0,), "bpress": -91.0}
    assert captured["runtime_options"].output_root == output_root.resolve()
    assert captured["dry_run"] is True
    assert captured["write_artifacts"] is False
    assert captured["material_parameters"] == {
        "ka": 1.1,
        "kb": 1.2,
        "mu": 0.9,
        "b1": 0.1,
        "b2": 0.2,
        "a3": 0.3,
        "a4": 0.4,
        "mu_l": 0.5,
        "c": 0.6,
    }
