from __future__ import annotations

import importlib.util
import json
import sys
from collections import namedtuple
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dry_run.py"
SMOKE_SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dnn_smoke.py"


def _load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_smoke_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SMOKE_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_help_does_not_load_runtime_or_mirheo(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("mesouq_test_gv_runtime_help")
    calls: list[str] = []

    def _unexpected_import(name: str):
        calls.append(name)
        raise AssertionError(f"Unexpected deferred import during --help: {name}")

    monkeypatch.setattr(module, "_load_experiment_module", _unexpected_import)

    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])

    assert excinfo.value.code == 0
    assert calls == []
    assert "mirheo" not in {name.lower() for name in sys.modules}


def test_non_shear_experiment_does_not_require_experimental_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module("mesouq_test_gv_runtime_non_shear")
    seen: dict[str, object] = {}

    class _DryRun:
        def __init__(self, request):
            self.request = request

        def to_manifest(self) -> dict[str, object]:
            return {
                "status": "ok",
                "experiment": str(self.request["experiment"]),
                "output_root": str(self.request["output_root"]),
                "controls": dict(self.request["controls"]),
            }

    class _Descriptor:
        def plan(self, *, output_root, geometry, controls, include_experimental):
            seen.update(
                {
                    "output_root": output_root,
                    "geometry": geometry,
                    "controls": controls,
                    "include_experimental": include_experimental,
                    "experiment": "stretching",
                }
            )
            return _DryRun(
                {
                    "experiment": "stretching",
                    "output_root": output_root,
                    "controls": controls,
                }
            )

    class _ExperimentModule:
        __name__ = "fake_gv_runtime.stretching"
        DESCRIPTOR = _Descriptor()

    monkeypatch.setattr(module, "_load_experiment_module", lambda _name: _ExperimentModule())
    output_root = tmp_path / "gv-dry-run"

    rc = module.main(
        [
            "--experiment",
            "stretching",
            "--output-root",
            str(output_root),
            "--control",
            "tot_force=0.25",
            "--control",
            "bpress=-0.01",
        ]
    )

    assert rc == 0
    assert seen["experiment"] == "stretching"
    assert seen["include_experimental"] is False
    assert Path(str(seen["output_root"])) == output_root.resolve()
    manifest = json.loads((output_root / "gv_runtime_dry_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "stretching"
    assert manifest["output_root"] == str(output_root.resolve())


def test_real_stretching_cli_writes_manifest_without_mirheo(tmp_path: Path) -> None:
    module = _load_module("mesouq_test_gv_runtime_real_stretching")
    output_root = tmp_path / "real-stretching"

    rc = module.main(
        [
            "--experiment",
            "stretching",
            "--output-root",
            str(output_root),
            "--control",
            "tot_force=750",
        ]
    )

    manifest = json.loads((output_root / "gv_runtime_dry_run_manifest.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert manifest["structure"] == "gv"
    assert manifest["experiment"] == "stretching"
    assert manifest["controls"]["tot_force"] == 750.0
    assert manifest["dataset_id"].startswith("gv:stretching:")
    assert manifest["selection"] == "gv:stretching"
    assert "mirheo" not in {name.lower() for name in sys.modules}


@pytest.mark.parametrize(
    ("experiment_name", "include_experimental"),
    (
        ("stretching", False),
        ("buckling", False),
        ("torsion", False),
        ("eigenmodes", False),
        ("shear_flow", True),
    ),
)
def test_structure_qualified_selection_supports_all_gv_runtime_experiments(
    tmp_path: Path,
    experiment_name: str,
    include_experimental: bool,
) -> None:
    module = _load_module(f"mesouq_test_gv_runtime_selection_{experiment_name}")
    argv = ["--selection", f"gv:{experiment_name}", "--output-root", str(tmp_path / experiment_name)]
    if include_experimental:
        argv.insert(2, "--include-experimental")

    rc = module.main(argv)

    manifest = json.loads(
        ((tmp_path / experiment_name) / "gv_runtime_dry_run_manifest.json").read_text(encoding="utf-8")
    )
    assert rc == 0
    assert manifest["structure"] == "gv"
    assert manifest["experiment"] == experiment_name
    assert manifest["selection"] == f"gv:{experiment_name}"


def test_control_parser_rejects_malformed_unknown_and_non_float_controls() -> None:
    module = _load_module("mesouq_test_gv_runtime_control_errors")
    structure = module.get_structure("gv")

    with pytest.raises(ValueError, match="Expected NAME=VALUE"):
        module._parse_controls(["tot_force"], "stretching", structure)
    with pytest.raises(ValueError, match="Unsupported control"):
        module._parse_controls(["buck=0.2"], "stretching", structure)
    with pytest.raises(ValueError, match="Expected a float"):
        module._parse_controls(["tot_force=not-a-float"], "stretching", structure)


def test_geometry_resolution_supports_id_and_custom_shape() -> None:
    module = _load_module("mesouq_test_gv_runtime_geometry")
    structure = module.get_structure("gv")

    id_args = module.build_parser().parse_args(
        ["--experiment", "stretching", "--geometry-id", "gv_rad2_height14_28"]
    )
    assert module._resolve_geometry(id_args, structure).id == "gv_rad2_height14_28"

    incomplete_args = module.build_parser().parse_args(
        ["--experiment", "stretching", "--radius", "2.5"]
    )
    with pytest.raises(ValueError, match="requires both --radius and --height"):
        module._resolve_geometry(incomplete_args, structure)

    custom_args = module.build_parser().parse_args(
        ["--experiment", "stretching", "--radius", "2.5", "--height", "15.0"]
    )
    custom = module._resolve_geometry(custom_args, structure)
    assert custom.id == "gv_rad2_5_height15"


def test_descriptor_resolution_fallback_and_bad_payload(tmp_path: Path) -> None:
    module = _load_module("mesouq_test_gv_runtime_descriptor_fallback")

    class _Descriptor:
        def plan(self, *, output_root, geometry, controls, include_experimental):
            return {"output_root": output_root, "geometry": geometry}

    class _FactoryModule:
        __name__ = "factory_module"

        @staticmethod
        def build_runtime_descriptor():
            return _Descriptor()

    assert isinstance(module._resolve_runtime_descriptor(_FactoryModule()), _Descriptor)

    class _BadModule:
        __name__ = "bad_module"

    with pytest.raises(AttributeError, match="RuntimeDescriptor-compatible"):
        module._resolve_runtime_descriptor(_BadModule())

    class _BadPayloadDescriptor:
        def plan(self, *, output_root, geometry, controls, include_experimental):
            return ["not", "a", "dict"]

    class _BadPayloadModule:
        __name__ = "bad_payload_module"
        DESCRIPTOR = _BadPayloadDescriptor()

    monkeypatch_name = "mesouq_test_gv_runtime_bad_payload"
    bad_payload_cli = _load_module(monkeypatch_name)
    bad_payload_cli._load_experiment_module = lambda _name: _BadPayloadModule()
    with pytest.raises(TypeError, match="manifest-capable"):
        bad_payload_cli.main(["--experiment", "stretching", "--output-root", str(tmp_path)])


def test_jsonify_normalizes_paths_namedtuples_and_objects(tmp_path: Path) -> None:
    module = _load_module("mesouq_test_gv_runtime_jsonify")
    Pair = namedtuple("Pair", ["path", "values"])

    class Payload:
        def __init__(self) -> None:
            self.root = tmp_path
            self.pair = Pair(tmp_path / "a", [tmp_path / "b"])

    assert module._jsonify(tmp_path) == str(tmp_path)
    assert module._jsonify(Pair(tmp_path / "x", (tmp_path / "y",))) == {
        "path": str(tmp_path / "x"),
        "values": [str(tmp_path / "y")],
    }
    assert module._jsonify(Payload()) == {
        "root": str(tmp_path),
        "pair": {"path": str(tmp_path / "a"), "values": [str(tmp_path / "b")]},
    }


def test_shear_flow_requires_experimental_flag(tmp_path: Path) -> None:
    module = _load_module("mesouq_test_gv_runtime_shear_guard")

    with pytest.raises(SystemExit) as excinfo:
        module.main(
            [
                "--experiment",
                "shear_flow",
                "--output-root",
                str(tmp_path / "out"),
            ]
        )

    assert excinfo.value.code == 2


@pytest.mark.parametrize("selection", ["emb:compression", "gv:stretching:extra"])
def test_structure_qualified_selection_rejects_non_gv_or_malformed_values(
    tmp_path: Path,
    selection: str,
) -> None:
    module = _load_module("mesouq_test_gv_runtime_bad_selection")

    with pytest.raises(SystemExit) as excinfo:
        module.main(["--selection", selection, "--output-root", str(tmp_path / "out")])

    assert excinfo.value.code == 2


def test_structure_qualified_selection_rejects_conflict_with_experiment_value(tmp_path: Path) -> None:
    module = _load_module("mesouq_test_gv_runtime_selection_conflict")

    with pytest.raises(SystemExit) as excinfo:
        module.main(
            [
                "--experiment",
                "stretching",
                "--selection",
                "gv:torsion",
                "--output-root",
                str(tmp_path / "out"),
            ]
        )

    assert excinfo.value.code == 2


@pytest.mark.parametrize(
    ("selection", "control_arg", "expected_control_key", "include_experimental"),
    (
        ("gv:stretching", "tot_force=700", "tot_force", False),
        ("gv:buckling", "buck=0.2", "buck", False),
        ("gv:torsion", "theta=0.04", "theta", False),
        ("gv:eigenmodes", "bpress=-91.0", "bpress", False),
        ("gv:shear_flow", "ptan=0.4", "ptan", True),
    ),
)
def test_offline_smoke_selection_supports_all_gv_experiments_without_mirheo(
    tmp_path: Path,
    selection: str,
    control_arg: str,
    expected_control_key: str,
    include_experimental: bool,
) -> None:
    module = _load_smoke_module(f"mesouq_test_gv_smoke_selection_{selection.replace(':', '_')}")
    argv = [
        "--selection",
        selection,
        "--output-root",
        str(tmp_path / "smoke"),
        "--control",
        control_arg,
        "--num-curves",
        "3",
        "--points-per-curve",
        "3",
    ]
    if include_experimental:
        argv.insert(2, "--include-experimental")

    rc = module.main(argv)

    manifest = json.loads(
        ((tmp_path / "smoke") / "gv_reference_manifest.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        ((tmp_path / "smoke") / "gv_dnn_surrogate_smoke_report.json").read_text(encoding="utf-8")
    )
    assert rc == 0
    assert manifest["selection"] == selection
    assert manifest["structure"] == "gv"
    assert expected_control_key in manifest["controls"]
    assert report["structure"] == "gv"
    assert report["execution_mode"] in {"dry_run_manifest_only", "train_load_evaluate"}
    assert "mirheo" not in {name.lower() for name in sys.modules}


def test_default_output_root_routes_under_runs_and_avoids_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module("mesouq_test_gv_runtime_default_root")

    class _DryRun:
        def __init__(self, output_root: Path):
            self.output_root = output_root

        def to_manifest(self) -> dict[str, str]:
            return {"output_root": str(self.output_root)}

    class _Descriptor:
        def plan(self, *, output_root, geometry, controls, include_experimental):
            return _DryRun(output_root)

    class _ExperimentModule:
        __name__ = "fake_gv_runtime_default_root.buckling"
        DESCRIPTOR = _Descriptor()

    monkeypatch.setattr(module, "_load_experiment_module", lambda _name: _ExperimentModule())
    monkeypatch.setattr(
        module,
        "default_runs_root",
        lambda *_args, **_kwargs: tmp_path / "_runs" / "vega" / "gv_runtime" / "20260430_120000",
    )

    rc = module.main(["--experiment", "buckling"])

    assert rc == 0
    manifest_path = (
        tmp_path
        / "_runs"
        / "vega"
        / "gv_runtime"
        / "20260430_120000"
        / "gv_runtime_dry_run_manifest.json"
    )
    assert manifest_path.is_file()
    assert module.GV_SIMULATION_ROOT not in manifest_path.parents


def test_output_root_inside_gv_simulation_files_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module("mesouq_test_gv_runtime_block_staging")

    class _Descriptor:
        def plan(self, *, output_root, geometry, controls, include_experimental):
            return None

    class _ExperimentModule:
        __name__ = "fake_gv_runtime_block_staging.stretching"
        DESCRIPTOR = _Descriptor()

    monkeypatch.setattr(module, "_load_experiment_module", lambda _name: _ExperimentModule())
    blocked_root = REPO_ROOT / "gv_simulation_files" / "stretching" / "gv" / "generated"

    with pytest.raises(SystemExit) as excinfo:
        module.main(
            [
                "--experiment",
                "stretching",
                "--output-root",
                str(blocked_root),
            ]
        )

    assert excinfo.value.code == 2
