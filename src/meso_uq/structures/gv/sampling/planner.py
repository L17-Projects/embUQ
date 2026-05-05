from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .failures import GVSamplingPlanError


default_sampling_timeout_seconds = 7200


@dataclass(frozen=True)
class SamplingCommand:
    argv: tuple[str, ...]
    cwd: str
    description: str = ""

    def to_manifest(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": self.cwd,
            "description": self.description,
        }


@dataclass(frozen=True)
class SamplingRun:
    command_sequence: tuple[SamplingCommand, ...]
    controls: Mapping[str, float]
    provenance: Mapping[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "provenance": dict(self.provenance),
            "controls": dict(self.controls),
            "commands": [command.to_manifest() for command in self.command_sequence],
        }


@dataclass(frozen=True)
class GVSamplingPlan:
    runtime_manifest: Mapping[str, Any]
    control_axis: str
    values: tuple[float, ...]
    linear: bool
    timeout_seconds: int
    runs: tuple[SamplingRun, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "runtime_manifest": dict(self.runtime_manifest),
            "control_axis": self.control_axis,
            "values": list(self.values),
            "linear": self.linear,
            "timeout_seconds": self.timeout_seconds,
            "runs": [run.to_manifest() for run in self.runs],
        }


def build_sampling_plan(
    runtime_or_request: object,
    control_axis: str | None = None,
    values: Sequence[float] | None = None,
    *,
    timeout_seconds: int = default_sampling_timeout_seconds,
) -> GVSamplingPlan:
    """Build a sampling plan for one explicit control-axis sweep."""

    runtime_manifest, axis, sweep_values = _resolve_request(
        runtime_or_request,
        control_axis=control_axis,
        values=values,
    )
    _validate_timeout(timeout_seconds)
    normalized_values = _coerce_values(sweep_values)
    if not normalized_values:
        raise GVSamplingPlanError("Sampling values cannot be empty.")

    payload = _coerce_runtime_payload(runtime_manifest)
    manifest_controls = _coerce_runtime_controls(payload)
    if axis not in manifest_controls:
        raise GVSamplingPlanError(f"Control axis {axis!r} not present in runtime controls.")
    commands = _coerce_command_list(payload)
    if not commands:
        raise GVSamplingPlanError("Runtime manifest does not contain executable commands.")

    generate_index = _find_generate_command_index(commands)
    if generate_index is None:
        raise GVSamplingPlanError("Runtime command manifest does not include a generate.py command.")

    linear = _is_linear_spacing(normalized_values)
    if linear:
        run = _build_linear_run(commands, generate_index, payload, axis, normalized_values)
        runs = (run,)
    else:
        base_controls = dict(manifest_controls)
        runs = tuple(
            _build_exact_run(
                commands,
                generate_index,
                base_controls,
                axis,
                value,
            )
            for value in normalized_values
        )

    return GVSamplingPlan(
        runtime_manifest=payload,
        control_axis=axis,
        values=normalized_values,
        linear=linear,
        timeout_seconds=timeout_seconds,
        runs=runs,
    )


def _build_linear_run(
    commands: tuple[SamplingCommand, ...],
    generate_index: int,
    payload: Mapping[str, Any],
    axis: str,
    values: tuple[float, ...],
) -> SamplingRun:
    start = values[0]
    stop = values[-1]
    run_commands = _commands_for_axis_range(commands, generate_index, axis, start, stop, len(values))
    controls = dict(_coerce_runtime_controls(payload))
    controls[axis] = start
    return SamplingRun(
        command_sequence=run_commands,
        controls=controls,
        provenance={
            "axis": axis,
            "mode": "linear",
            "start": start,
            "stop": stop,
            "steps": len(values),
            "values": list(values),
        },
    )


def _build_exact_run(
    commands: tuple[SamplingCommand, ...],
    generate_index: int,
    base_controls: Mapping[str, float],
    axis: str,
    value: float,
) -> SamplingRun:
    run_commands = _commands_for_axis_range(commands, generate_index, axis, value, value, 1)
    controls = dict(base_controls)
    controls[axis] = value
    return SamplingRun(
        command_sequence=run_commands,
        controls=controls,
        provenance={
            "axis": axis,
            "mode": "exact",
            "value": value,
        },
    )


def _commands_for_axis_range(
    commands: tuple[SamplingCommand, ...],
    generate_index: int,
    axis: str,
    start: float,
    stop: float,
    steps: int,
) -> tuple[SamplingCommand, ...]:
    patched: list[SamplingCommand] = []
    for index, command in enumerate(commands):
        if index != generate_index:
            patched.append(command)
            continue
        patched.append(
            SamplingCommand(
                _replace_axis_range(tuple(command.argv), axis, start, stop, steps),
                command.cwd,
                command.description,
            )
        )
    return tuple(patched)


def _replace_axis_range(
    argv: tuple[str, ...],
    axis: str,
    start: float,
    stop: float,
    steps: int,
) -> tuple[str, ...]:
    args = list(argv)
    i = 0
    replaced = False
    while i < len(args):
        if args[i] in {"-p", "--parameter"}:
            if i + 4 >= len(args):
                raise GVSamplingPlanError("Invalid generate command: incomplete '-p' sweep tuple.")
            if args[i + 1] == axis:
                if replaced:
                    raise GVSamplingPlanError(f"Duplicate sweep axis {axis!r} in generate command.")
                args[i + 2] = _format_float(start)
                args[i + 3] = _format_float(stop)
                args[i + 4] = str(int(steps))
                replaced = True
            i += 5
            continue
        i += 1
    if not replaced:
        raise GVSamplingPlanError(f"Runtime generate command does not contain control axis {axis!r}.")
    return tuple(args)


def _coerce_runtime_payload(payload: object) -> dict[str, Any]:
    if isinstance(payload, GVSamplingPlan):
        payload = payload.runtime_manifest
    if isinstance(payload, (str, Path)):
        runtime_path = Path(payload)
        if not runtime_path.exists():
            raise GVSamplingPlanError(f"Runtime manifest path does not exist: {runtime_path}")
        if not runtime_path.is_file():
            raise GVSamplingPlanError(f"Runtime manifest path is not a file: {runtime_path}")
        runtime_manifest = json.loads(runtime_path.read_text(encoding="utf-8"))
        if not isinstance(runtime_manifest, Mapping):
            raise GVSamplingPlanError("Runtime manifest file must contain a JSON object.")
        return dict(runtime_manifest)
    if hasattr(payload, "to_manifest"):
        runtime_manifest = payload.to_manifest()
        if not isinstance(runtime_manifest, Mapping):
            raise GVSamplingPlanError("Runtime manifest .to_manifest() must return a mapping.")
        return dict(runtime_manifest)
    if isinstance(payload, Mapping):
        return dict(payload)
    raise GVSamplingPlanError("Runtime input must be a RuntimeDryRun-like object, request object, or mapping.")


def _coerce_runtime_controls(payload: Mapping[str, Any]) -> dict[str, float]:
    controls = payload.get("controls")
    if not isinstance(controls, Mapping):
        raise GVSamplingPlanError("Runtime manifest is missing 'controls'.")
    normalized: dict[str, float] = {}
    for key, value in controls.items():
        try:
            normalized[str(key)] = float(value)
        except Exception as exc:
            raise GVSamplingPlanError(f"Runtime control value {key!r} must be numeric.") from exc
    return normalized


def _coerce_command_list(payload: Mapping[str, Any]) -> tuple[SamplingCommand, ...]:
    commands = payload.get("commands")
    if commands is None:
        return ()
    parsed: list[SamplingCommand] = []
    for entry in commands:
        if isinstance(entry, SamplingCommand):
            parsed.append(entry)
            continue
        if hasattr(entry, "argv") and hasattr(entry, "cwd"):
            command = SamplingCommand(
                tuple(str(part) for part in getattr(entry, "argv")),
                str(getattr(entry, "cwd")),
                str(getattr(entry, "description", "")),
            )
            parsed.append(command)
            continue
        if isinstance(entry, Mapping):
            argv = entry.get("argv")
            if not isinstance(argv, Sequence):
                raise GVSamplingPlanError("Runtime command entry is missing argv sequence.")
            parsed.append(
                SamplingCommand(
                    tuple(str(part) for part in argv),
                    str(entry.get("cwd", "")),
                    str(entry.get("description", "")),
                )
            )
            continue
        raise GVSamplingPlanError("Unsupported runtime command entry type.")
    return tuple(parsed)


def _find_generate_command_index(commands: tuple[SamplingCommand, ...]) -> int | None:
    for index, command in enumerate(commands):
        if not command.argv:
            continue
        if any(Path(str(token)).name == "generate.py" for token in command.argv):
            return index
    return None

def _coerce_request(
    request: object,
) -> tuple[object | None, str | None, Sequence[float] | None]:
    axis = None
    values = None
    runtime = request

    if isinstance(request, Mapping):
        if "runtime_manifest" in request:
            runtime = request["runtime_manifest"]
        axis = _first_non_none(
            request.get("control_axis"),
            request.get("sweep_axis"),
            request.get("axis"),
        )
        values = _first_non_none(
            request.get("values"),
            request.get("value_sequence"),
            request.get("control_values"),
        )
    else:
        for runtime_field in ("runtime_manifest", "runtime", "run_manifest"):
            if runtime_field and hasattr(request, runtime_field):
                runtime = getattr(request, runtime_field)
                break
        axis = _first_non_none(
            getattr(request, "control_axis", None),
            getattr(request, "sweep_axis", None),
            getattr(request, "axis", None),
        )
        values = _first_non_none(
            getattr(request, "values", None),
            getattr(request, "value_sequence", None),
            getattr(request, "control_values", None),
        )

    return runtime, axis, values


def _resolve_request(
    runtime_or_request: object,
    control_axis: str | None = None,
    values: Sequence[float] | None = None,
) -> tuple[Mapping[str, Any], str, Sequence[float]]:
    runtime_payload: object
    resolved_axis = control_axis
    resolved_values = values
    if values is None and control_axis is None:
        runtime_payload, maybe_axis, maybe_values = _coerce_request(runtime_or_request)
        if resolved_axis is None:
            resolved_axis = maybe_axis
        if resolved_values is None:
            resolved_values = maybe_values
    else:
        runtime_payload = runtime_or_request
        if resolved_axis is None and hasattr(runtime_or_request, "__dict__"):
            _, maybe_axis, maybe_values = _coerce_request(runtime_or_request)
            if maybe_axis is not None:
                resolved_axis = maybe_axis
            if resolved_values is None:
                resolved_values = maybe_values

    if not resolved_axis:
        raise GVSamplingPlanError("A control axis must be provided.")
    if resolved_values is None:
        raise GVSamplingPlanError("Sampling values must be provided.")
    return _coerce_runtime_payload(runtime_payload), str(resolved_axis), resolved_values


def _coerce_values(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise GVSamplingPlanError("Sampling values must be a sequence of numeric values.")
    normalized: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise GVSamplingPlanError(f"Control values must be numeric, got {value!r}.") from exc
        if not math.isfinite(numeric):
            raise GVSamplingPlanError(f"Control value {value!r} must be finite.")
        normalized.append(numeric)
    return tuple(normalized)


def _validate_timeout(timeout_seconds: int) -> None:
    if not isinstance(timeout_seconds, int):
        raise GVSamplingPlanError("timeout_seconds must be an integer.")
    if timeout_seconds <= 0:
        raise GVSamplingPlanError("timeout_seconds must be positive.")


def _is_linear_spacing(values: tuple[float, ...]) -> bool:
    if len(values) <= 1:
        return True
    deltas = [values[index + 1] - values[index] for index in range(len(values) - 1)]
    step0 = deltas[0]
    for delta in deltas[1:]:
        if not math.isclose(delta, step0, rel_tol=1e-12, abs_tol=1e-12):
            return False
    return True


def _first_non_none(*items: object | None) -> object | None:
    for item in items:
        if item is not None:
            return item
    return None


def _format_float(value: float) -> str:
    return f"{value:g}"


__all__ = [
    "SamplingCommand",
    "SamplingRun",
    "GVSamplingPlan",
    "build_sampling_plan",
    "default_sampling_timeout_seconds",
]
