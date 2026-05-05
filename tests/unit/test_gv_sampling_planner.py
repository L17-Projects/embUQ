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
from meso_uq.structures.gv.sampling.planner import GVSamplingPlan, build_sampling_plan
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
