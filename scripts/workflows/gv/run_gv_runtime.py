#!/usr/bin/env python3
"""Render GV runtime jobs from a generated dry-run manifest.

The script performs the same dry-run planning as :mod:`run_gv_dry_run`, then writes
execution assets (``commands.txt`` plus generated scheduler scripts) into the staged
work directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

DRY_RUN_WORKFLOW = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_gv_dry_run.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "_runs" / "gv" / "runtime"
GV_RUNTIME_MANIFEST = "gv_runtime_dry_run_manifest.json"
GV_RUNTIME_RENDER_MANIFEST = "gv_runtime_render_manifest.json"
_KNOWN_ISSUE_BLOCKED_SEVERITIES = {"error", "blocking", "blocked", "critical"}
_GV_VENV_ENV_SCRIPT = str((REPO_ROOT / "_vega" / "gv_venv" / "env.sh").resolve())


def _normalize_known_issues(raw_known_issues: object) -> list[dict[str, Any]]:
    if not isinstance(raw_known_issues, list):
        return []
    normalized: list[dict[str, Any]] = []
    for issue in raw_known_issues:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity", "")).lower()
        normalized.append(
            {
                "id": str(issue.get("id", "")),
                "summary": str(issue.get("summary", "")),
                "evidence": str(issue.get("evidence", "")),
                "severity": severity,
                "classification": "experimental_blocked" if severity in _KNOWN_ISSUE_BLOCKED_SEVERITIES else "observed",
            }
        )
    return normalized


def _build_runtime_stage(runtime_manifest: dict[str, Any]) -> dict[str, Any]:
    is_experimental = bool(runtime_manifest.get("experimental", False))
    known_issues = _normalize_known_issues(runtime_manifest.get("known_issues", []))
    return {
        "experimental": is_experimental,
        "runtime_package": str(runtime_manifest.get("runtime_package", "")),
        "runtime_package_source": str(runtime_manifest.get("source_root", "")),
        "known_issues": known_issues,
        "blocked_issue_count": sum(1 for issue in known_issues if issue.get("classification") == "experimental_blocked"),
    }


def _load_gv_dry_run_module():
    spec = importlib.util.spec_from_file_location("run_gv_dry_run_workflow", DRY_RUN_WORKFLOW)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {DRY_RUN_WORKFLOW}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_GV_DRY_RUN_WORKFLOW = _load_gv_dry_run_module()
RUN_GV_DRY_RUN_MAIN = _GV_DRY_RUN_WORKFLOW.main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="gv", choices=("gv",))
    parser.add_argument(
        "--selection",
        default=None,
        help="Optional structure-qualified selection, for example gv:stretching.",
    )
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--geometry-id", default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument(
        "--control",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable control override, for example --control tot_force=750.",
    )
    parser.add_argument(
        "--material",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Repeatable GV material override, for example --material ka=1.2.",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--platform", choices=("vega", "karolina", "local"), default="vega")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--include-experimental", action="store_true", default=False)
    return parser


def _resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root is not None:
        return Path(args.output_root).expanduser().resolve()
    resolved = DEFAULT_OUTPUT_ROOT
    if args.run_tag is not None:
        resolved = resolved / args.run_tag
    return resolved


def _load_runtime_manifest(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("GV runtime manifest must be a JSON object.")

    required = (
        "structure",
        "experiment",
        "geometry",
        "controls",
        "control_id",
        "dataset_id",
        "work_dir",
        "output_root",
        "commands",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"GV runtime manifest is missing required field(s): {', '.join(missing)}")
    if not isinstance(raw["controls"], dict):
        raise ValueError("GV runtime manifest field 'controls' must be an object.")
    return raw


def _normalize_command_value(command: Any) -> list[str]:
    if isinstance(command, list):
        return [str(item) for item in command]
    if isinstance(command, tuple):
        return [str(item) for item in command]
    raise ValueError(f"Expected a command list in runtime manifest; got {type(command)!r}: {command}")


def _normalize_cwd(cwd: Any, *, work_dir: Path) -> Path:
    if cwd is None:
        return work_dir
    if not isinstance(cwd, str):
        raise ValueError("Runtime manifest command entries require a string cwd.")
    normalized = cwd.replace("{work_dir}", str(work_dir))
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate
    return work_dir / candidate


def _as_command_list(manifest: dict[str, Any]) -> list[tuple[tuple[str, ...], Path]]:
    command_entries = list(manifest.get("commands", [])) + list(manifest.get("analysis_commands", []))
    work_dir = Path(manifest["work_dir"]).expanduser().resolve()
    if not command_entries:
        return []
    normalized: list[tuple[tuple[str, ...], Path]] = []
    for index, command in enumerate(command_entries):
        if not isinstance(command, dict):
            raise ValueError(f"Runtime command #{index} is malformed: expected a mapping.")
        argv = _normalize_command_value(command.get("argv"))
        if not argv:
            raise ValueError(f"Runtime command #{index} is empty.")
        cwd = _normalize_cwd(command.get("cwd", str(work_dir)), work_dir=work_dir)
        normalized.append((tuple(argv), cwd))
    return normalized


def _quoted_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _material_overrides_json(manifest: dict[str, Any]) -> str:
    overrides = manifest.get("material_parameter_overrides", {})
    if not isinstance(overrides, dict) or not overrides:
        return ""
    return json.dumps(overrides, sort_keys=True)


def _write_commands_txt(
    commands: list[tuple[tuple[str, ...], Path]],
    commands_path: Path,
    *,
    material_overrides_json: str = "",
) -> None:
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    with commands_path.open("w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\n")
        handle.write("set -euo pipefail\n")
        handle.write(f"if [[ ! -f {_GV_VENV_ENV_SCRIPT!r} ]]; then\n")
        handle.write("  echo 'Missing required GV runtime environment: _vega/gv_venv/env.sh' >&2\n")
        handle.write("  exit 1\n")
        handle.write("fi\n")
        handle.write(f"source {_GV_VENV_ENV_SCRIPT!r}\n\n")
        if material_overrides_json:
            handle.write(
                "export MESOUQ_GV_MATERIAL_OVERRIDES_JSON="
                f"{shlex.quote(material_overrides_json)}\n\n"
            )
        for command, cwd in commands:
            handle.write(f"(cd {shlex.quote(str(cwd))} && {_quoted_command(command)})\n")
        handle.write("\n")
    commands_path.chmod(0o750)


def _default_sbatch_script(
    *,
    experiment: str,
    geometry: str,
    control_id: str,
    platform: str,
    work_dir: Path,
) -> str:
    script = "#!/usr/bin/env bash\n"
    script += "#SBATCH --job-name=gv-runtime\n"
    script += "#SBATCH --output=gv-runtime-%j.out\n"
    script += "#SBATCH --error=gv-runtime-%j.err\n"
    script += "set -euo pipefail\n"
    script += "cd \"$(dirname \"$0\")\"\n\n"
    script += f'if [[ ! -f {_GV_VENV_ENV_SCRIPT!r} ]]; then\n'
    script += "  echo 'Missing required GV runtime environment: _vega/gv_venv/env.sh' >&2\n"
    script += "  exit 1\n"
    script += "fi\n"
    script += f'source {_GV_VENV_ENV_SCRIPT!r}\n\n'
    script += "if [ -x ./run_all_HPC.sh ]; then\n"
    script += "  bash ./run_all_HPC.sh\n"
    script += "elif [ -x ./run.sh ]; then\n"
    script += "  bash ./run.sh\n"
    script += "else\n"
    script += "  echo 'No runtime entry point found in staged work dir.' >&2\n"
    script += "  exit 1\n"
    script += "fi\n\n"
    script += f"# platform: {platform}\n"
    script += f"# experiment: {experiment}\n"
    script += f"# geometry: {geometry}\n"
    script += f"# control_id: {control_id}\n"
    script += f"# work_dir: {shlex.quote(str(work_dir))}\n"
    return script


def _ensure_scheduler_scripts(
    *,
    command_list: list[tuple[tuple[str, ...], Path]],
    manifest: dict[str, Any],
    platform: str,
) -> list[str]:
    work_dir = Path(manifest["work_dir"]).resolve()
    generated: list[str] = []
    for command, cwd in command_list:
        if not command:
            continue
        if command[0] != "sbatch":
            continue
        if len(command) < 2:
            continue
        scheduler_name = str(command[1])
        scheduler_file = (cwd / scheduler_name).resolve()
        if scheduler_file.exists():
            continue
        scheduler_file.parent.mkdir(parents=True, exist_ok=True)
        scheduler_file.write_text(
            _default_sbatch_script(
                experiment=str(manifest.get("experiment", "gv")),
                geometry=str(manifest.get("geometry", "")),
                control_id=str(manifest.get("control_id", "default")),
                platform=platform,
                work_dir=work_dir,
            ),
            encoding="utf-8",
        )
        scheduler_file.chmod(0o750)
        generated.append(scheduler_name)
    return sorted(set(generated))


def _ensure_generated_directories(work_dir: Path, manifest: dict[str, Any]) -> None:
    for relpath in manifest.get("generated_subdirs", []):
        if not isinstance(relpath, str):
            continue
        (work_dir / relpath).mkdir(parents=True, exist_ok=True)


def _run_commands(
    commands: list[tuple[tuple[str, ...], Path]],
    *,
    dry_run: bool,
    material_overrides_json: str = "",
) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    if dry_run:
        for command, cwd in commands:
            records.append(
                {
                    "argv": list(command),
                    "cwd": str(cwd),
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "skipped-dry-run",
                }
            )
        return records, 0
    for command, cwd in commands:
        env = None
        if material_overrides_json:
            env = dict(os.environ)
            env["MESOUQ_GV_MATERIAL_OVERRIDES_JSON"] = material_overrides_json
        run_kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "capture_output": True,
            "text": True,
            "check": False,
        }
        if env is not None:
            run_kwargs["env"] = env
        proc = subprocess.run(command, **run_kwargs)
        records.append(
            {
                "argv": list(command),
                "cwd": str(cwd),
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "status": "completed" if proc.returncode == 0 else "failed",
            }
        )
        if proc.returncode != 0:
            return records, proc.returncode
    return records, 0


def _to_render_manifest(
    *,
    args: argparse.Namespace,
    runtime_manifest: dict[str, Any],
    manifest_path: Path,
    command_records: list[dict[str, Any]],
    commands_path: Path,
    scheduled_scripts: list[str],
    returncode: int,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "workflow": "gv_runtime",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": args.platform,
        "structure": runtime_manifest["structure"],
        "experiment": runtime_manifest["experiment"],
        "geometry": runtime_manifest["geometry"],
        "control_id": runtime_manifest["control_id"],
        "work_dir": runtime_manifest["work_dir"],
        "output_root": runtime_manifest["output_root"],
        "runtime_manifest": str(manifest_path),
        "commands_txt": str(commands_path),
        "generated_scheduler_scripts": scheduled_scripts,
        "commands": command_records,
        "include_experimental": args.include_experimental,
        "dry_run": dry_run,
        "runtime_stage": _build_runtime_stage(runtime_manifest),
        "returncode": returncode,
    }


def _build_runtime_argv(args: argparse.Namespace) -> list[str]:
    argv: list[str] = ["--structure", args.structure]
    if args.selection is not None:
        argv.extend(["--selection", args.selection])
    elif args.experiment is not None:
        argv.extend(["--experiment", str(args.experiment)])
    if args.geometry_id is not None:
        argv.extend(["--geometry-id", str(args.geometry_id)])
    if args.radius is not None:
        argv.extend(["--radius", str(args.radius)])
    if args.height is not None:
        argv.extend(["--height", str(args.height)])
    for item in args.control:
        argv.extend(["--control", item])
    for item in getattr(args, "material", []):
        argv.extend(["--material", item])
    if args.run_tag is not None:
        argv.extend(["--run-tag", str(args.run_tag)])
    if args.include_experimental:
        argv.append("--include-experimental")
    return argv


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runtime_output_root = _resolve_output_root(args)
    runtime_argv = _build_runtime_argv(args)
    runtime_argv.extend(["--output-root", str(runtime_output_root)])

    runtime_rc = int(RUN_GV_DRY_RUN_MAIN(runtime_argv))
    if runtime_rc != 0:
        raise RuntimeError(f"GV runtime dry-run failed with code {runtime_rc}.")

    manifest_path = runtime_output_root / GV_RUNTIME_MANIFEST
    runtime_manifest = _load_runtime_manifest(manifest_path)

    command_list = _as_command_list(runtime_manifest)
    work_dir = Path(runtime_manifest["work_dir"]).resolve()
    _ensure_generated_directories(work_dir=work_dir, manifest=runtime_manifest)
    platform = args.platform
    material_overrides_json = _material_overrides_json(runtime_manifest)

    commands_path = work_dir / "commands.txt"
    _write_commands_txt(
        commands=command_list,
        commands_path=commands_path,
        material_overrides_json=material_overrides_json,
    )
    generated_scripts = _ensure_scheduler_scripts(
        command_list=command_list,
        manifest=runtime_manifest,
        platform=platform,
    )

    command_records, returncode = _run_commands(
        commands=command_list,
        dry_run=args.dry_run,
        material_overrides_json=material_overrides_json,
    )

    render_manifest_path = runtime_output_root / GV_RUNTIME_RENDER_MANIFEST
    render_manifest = _to_render_manifest(
        args=args,
        runtime_manifest=runtime_manifest,
        manifest_path=manifest_path,
        command_records=command_records,
        commands_path=commands_path,
        scheduled_scripts=generated_scripts,
        returncode=returncode,
        dry_run=args.dry_run,
    )
    render_manifest_path.write_text(json.dumps(render_manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"GV runtime manifest: {manifest_path}")
    print(f"GV runtime render manifest: {render_manifest_path}")
    print(f"GV runtime work dir: {work_dir}")
    if returncode != 0 and not args.dry_run:
        raise RuntimeError(
            f"GV runtime command execution failed with code {returncode}; "
            f"see {render_manifest_path}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
