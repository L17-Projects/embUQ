from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.structures.gv.runtime import load_runtime_descriptor
from meso_uq.structures.gv.sampling.planner import (
    GVSamplingPlan,
    build_sampling_plan,
    _coerce_command_list,
    _coerce_request,
    _coerce_runtime_payload,
    _find_generate_command_index,
    _replace_axis_range,
    _resolve_request,
    _coerce_values,
    SamplingCommand,
)
from meso_uq.structures.gv.sampling.failures import GVSamplingPlanError


def _extract_parameter_sweep(command_argv: tuple[str, ...], axis: str) -> tuple[str, str, str] | None:
    for index in range(len(command_argv)):
        if command_argv[index] in {"-p", "--parameter"}:
            if index + 4 < len(command_argv) and command_argv[index + 1] == axis:
                return command_argv[index + 2], command_argv[index + 3], command_argv[index + 4]
    return None


def _as_text(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def _build_runtime(tmp_path: Path):
    return load_runtime_descriptor("stretching").plan(output_root=tmp_path / "runtime")


def test_planner_routes_arithmetic_values_to_generate_span(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    plan = build_sampling_plan(runtime, control_axis="tot_force", values=(500.0, 2500.0, 4500.0))

    assert isinstance(plan, GVSamplingPlan)
    assert plan.linear is True
    assert len(plan.runs) == 1
    run = plan.runs[0]
    assert run.provenance["mode"] == "linear"
    assert run.provenance["steps"] == 3

    generate = run.command_sequence[0]
    tot_force = _extract_parameter_sweep(generate.argv, "tot_force")
    bpress = _extract_parameter_sweep(generate.argv, "bpress")
    assert tot_force == ("500", "4500", "3")
    assert bpress == ("-91", "-100", "1")


def test_planner_stages_non_linear_values_per_run(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    values = (500.0, 1400.0, 2200.0)
    plan = build_sampling_plan(runtime, control_axis="tot_force", values=values)

    assert plan.linear is False
    assert len(plan.runs) == len(values)
    for expected_value, run in zip(values, plan.runs):
        assert run.provenance["mode"] == "exact"
        assert run.provenance["value"] == expected_value
        generate = run.command_sequence[0]
        tot_force = _extract_parameter_sweep(generate.argv, "tot_force")
        assert tot_force == (
            _as_text(expected_value),
            _as_text(expected_value),
            "1",
        )


def test_planner_accepts_request_like_object(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    request = SimpleNamespace(
        runtime_manifest=runtime.to_manifest(),
        control_axis="tot_force",
        values=[500.0, 1000.0],
    )
    plan = build_sampling_plan(request)

    assert isinstance(plan, GVSamplingPlan)
    assert plan.linear is True
    assert plan.values == (500.0, 1000.0)


def test_planner_accepts_runtime_manifest_path(tmp_path: Path) -> None:
    runtime_manifest = _build_runtime(tmp_path).to_manifest()
    runtime_path = tmp_path / "runtime_manifest.json"
    runtime_path.write_text(__import__("json").dumps(runtime_manifest), encoding="utf-8")

    plan = build_sampling_plan(
        {
            "runtime_manifest": str(runtime_path),
            "control_axis": "tot_force",
            "values": [500.0, 1500.0],
        }
    )

    assert isinstance(plan, GVSamplingPlan)
    assert plan.linear is True


def test_planner_rejects_unknown_axis(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    with pytest.raises(GVSamplingPlanError):
        build_sampling_plan(runtime, control_axis="missing", values=(1.0, 2.0))


def test_planner_rejects_invalid_runtime_input_type() -> None:
    with pytest.raises(GVSamplingPlanError, match="Runtime input"):
        build_sampling_plan(12345, control_axis="tot_force", values=(1.0, 2.0))


def test_planner_rejects_negative_timeout_seconds(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)

    with pytest.raises(GVSamplingPlanError, match="timeout_seconds must be positive"):
        build_sampling_plan(
            runtime,
            control_axis="tot_force",
            values=(1.0, 2.0),
            timeout_seconds=0,
        )


def test_planner_rejects_missing_generate_command(tmp_path: Path) -> None:
    with pytest.raises(GVSamplingPlanError, match="does not include a generate.py command"):
        build_sampling_plan(
            {
                "work_dir": str(tmp_path / "runtime"),
                "controls": {"tot_force": 500.0, "bpress": -91.0},
                "commands": [{"argv": ["echo", "hi"], "cwd": str(tmp_path / "runtime")}],
            },
            control_axis="tot_force",
            values=(500.0,),
        )


def test_planner_rejects_duplicate_axis_in_generate_command(tmp_path: Path) -> None:
    with pytest.raises(GVSamplingPlanError, match="Duplicate sweep axis"):
        build_sampling_plan(
            {
                "work_dir": str(tmp_path / "runtime"),
                "controls": {"tot_force": 500.0, "bpress": -91.0},
                "commands": [
                    {
                        "argv": [
                            "python3",
                            "generate.py",
                            "-p",
                            "tot_force",
                            "500",
                            "1500",
                            "2",
                            "-p",
                            "tot_force",
                            "600",
                            "1600",
                            "2",
                        ],
                        "cwd": str(tmp_path / "runtime"),
                    }
                ],
            },
            control_axis="tot_force",
            values=(500.0, 1500.0),
        )


def test_planner_rejects_incomplete_generate_tuple(tmp_path: Path) -> None:
    with pytest.raises(GVSamplingPlanError, match="Invalid generate command: incomplete '-p' sweep tuple."):
        build_sampling_plan(
            {
                "work_dir": str(tmp_path / "runtime"),
                "controls": {"tot_force": 500.0, "bpress": -91.0},
                "commands": [
                    {"argv": ["python3", "generate.py", "-p", "tot_force", "500"], "cwd": str(tmp_path / "runtime")},
                ],
            },
            control_axis="tot_force",
            values=(500.0,),
        )


def test_planner_rejects_control_value_non_numeric(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)

    with pytest.raises(GVSamplingPlanError, match="must be numeric"):
        build_sampling_plan(runtime, control_axis="tot_force", values=("a",))


def test_planner_rejects_empty_values() -> None:
    runtime = _build_runtime(Path("."))
    with pytest.raises(GVSamplingPlanError, match="Sampling values cannot be empty"):
        build_sampling_plan(runtime, control_axis="tot_force", values=())


def test_planner_rejects_non_iterable_values() -> None:
    runtime = _build_runtime(Path("."))
    with pytest.raises(GVSamplingPlanError, match="must be a sequence"):
        _coerce_values("abc")


def test_planner_rejects_non_finite_timeout_value(tmp_path: Path) -> None:
    runtime = _build_runtime(tmp_path)
    with pytest.raises(GVSamplingPlanError, match="timeout_seconds must be an integer"):
        build_sampling_plan(runtime, control_axis="tot_force", values=(1.0, 2.0), timeout_seconds="12")


def test_planner_rejects_runtime_path_that_is_not_a_file(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime_dir"
    runtime_dir.mkdir()
    with pytest.raises(GVSamplingPlanError, match="is not a file"):
        _coerce_runtime_payload(runtime_dir)


def test_planner_rejects_non_object_runtime_file(tmp_path: Path) -> None:
    runtime_path = tmp_path / "not-object.json"
    runtime_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(GVSamplingPlanError, match="must contain a JSON object"):
        _coerce_runtime_payload(runtime_path)


def test_planner_rejects_command_list_missing_argv_and_unsupported_entry() -> None:
    with pytest.raises(GVSamplingPlanError, match="missing argv"):
        _coerce_command_list({"commands": ({"cwd": "."},)})
    with pytest.raises(GVSamplingPlanError, match="Unsupported runtime command entry type"):
        _coerce_command_list({"commands": (123,)})


def test_planner_rejects_non_mapping_runtime_manifest_controls() -> None:
    runtime = {"work_dir": "/tmp", "controls": "bad", "commands": []}
    with pytest.raises(GVSamplingPlanError, match="Runtime manifest is missing"):
        build_sampling_plan(runtime, control_axis="tot_force", values=(500.0,))


def test_planner_replace_axis_range_rejects_missing_axis() -> None:
    with pytest.raises(GVSamplingPlanError, match="does not contain control axis"):
        _replace_axis_range(("python3", "generate.py"), "tot_force", 1.0, 2.0, 3)


def test_planner_build_without_commands_in_manifest() -> None:
    with pytest.raises(GVSamplingPlanError, match="does not contain executable commands"):
        build_sampling_plan({"work_dir": ".", "controls": {"tot_force": 1.0, "bpress": -91.0}}, control_axis="tot_force", values=(1.0,))


def test_planner_request_object_fallback_without_axis_and_values() -> None:
    request = SimpleNamespace(runtime_manifest=_build_runtime(Path(".")).to_manifest())
    with pytest.raises(GVSamplingPlanError, match="A control axis must be provided"):
        build_sampling_plan(request)


def test_planner_resolve_request_requires_values_when_not_provided() -> None:
    request = SimpleNamespace(
        runtime_manifest=_build_runtime(Path(".")).to_manifest(),
        control_axis="tot_force",
    )
    with pytest.raises(GVSamplingPlanError, match="Sampling values must be provided"):
        _resolve_request(request)


def test_planner_find_generate_command_index_returns_none() -> None:
    assert _find_generate_command_index((SamplingCommand((), "."),)) is None
