"""Dependency-light Korali runtime policy and validation helpers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

from meso_uq.configs.policy import FORBIDDEN_PRIVATE_PATHS

KORALI_VENDOR_RELATIVE_ROOT = Path("extern") / "korali"
KORALI_RUNTIME_ENV_KEYS = (
    "MESOUQ_SITE",
    "MESOUQ_SITE_RUNTIME_ROOT",
    "MESOUQ_KORALI_BUILD_ROOT",
    "MESOUQ_KORALI_LIBRARY_PATH",
    "PYTHONPATH",
    "PATH",
)
KORALI_RUNTIME_PLATFORM = "karolina"
KORALI_RUNTIME_SCHEDULER = "slurm"


@dataclass(frozen=True)
class KoraliRuntimeValidationIssue:
    severity: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}


@dataclass(frozen=True)
class KoraliRuntimeValidationReport:
    repo_root: Path
    vendor_root: Path
    platform: str
    scheduler: str
    issues: tuple[KoraliRuntimeValidationIssue, ...]

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(issue.message for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(issue.message for issue in self.issues if issue.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": str(self.repo_root),
            "vendor_root": str(self.vendor_root),
            "platform": self.platform,
            "scheduler": self.scheduler,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def format_text(self) -> str:
        header = (
            f"Korali runtime validation for {self.platform}/{self.scheduler}: "
            f"{'ok' if self.ok else 'failed'}"
        )
        lines = [header, f"  repo_root: {self.repo_root}", f"  vendor_root: {self.vendor_root}"]
        for warning in self.warnings:
            lines.append(f"  warning: {warning}")
        for error in self.errors:
            lines.append(f"  error: {error}")
        return "\n".join(lines)


@dataclass(frozen=True)
class KoraliRuntimeValidationCommand:
    argv: tuple[str, ...]
    cwd: Path
    env: tuple[tuple[str, str], ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "cwd": str(self.cwd),
            "env": {key: value for key, value in self.env},
            "summary": self.summary,
        }


def build_korali_runtime_validation_command(
    repo_root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    vendor_root: str | Path | None = None,
    build_root: str | Path | None = None,
    library_paths: Sequence[str | Path] = (),
    pythonpath_hint: str | None = None,
    path_hint: str | None = None,
    platform: str = KORALI_RUNTIME_PLATFORM,
    scheduler: str = KORALI_RUNTIME_SCHEDULER,
) -> KoraliRuntimeValidationCommand:
    resolved_repo_root = _resolve_path(repo_root)
    resolved_vendor_root = _resolve_path(vendor_root) if vendor_root is not None else resolved_repo_root / KORALI_VENDOR_RELATIVE_ROOT
    resolved_env = _normalize_env(env)
    build_root_value = build_root if build_root is not None else resolved_env.get("MESOUQ_KORALI_BUILD_ROOT")
    library_path_values: Sequence[str | Path]
    if library_paths:
        library_path_values = library_paths
    elif resolved_env.get("MESOUQ_KORALI_LIBRARY_PATH"):
        library_path_values = _split_path_list(resolved_env["MESOUQ_KORALI_LIBRARY_PATH"])
    else:
        library_path_values = ()
    pythonpath_value = pythonpath_hint if pythonpath_hint is not None else resolved_env.get("PYTHONPATH")
    path_value = path_hint if path_hint is not None else resolved_env.get("PATH")

    if build_root_value is not None:
        resolved_env["MESOUQ_KORALI_BUILD_ROOT"] = str(_resolve_path(build_root_value))
    if library_path_values:
        resolved_env["MESOUQ_KORALI_LIBRARY_PATH"] = os.pathsep.join(
            str(_resolve_path(path)) for path in library_path_values
        )
    if pythonpath_value is not None:
        resolved_env["PYTHONPATH"] = pythonpath_value
    if path_value is not None:
        resolved_env["PATH"] = path_value

    argv = (
        sys.executable,
        "-m",
        "meso_uq.platforms.korali_runtime",
        "--repo-root",
        str(resolved_repo_root),
        "--vendor-root",
        str(resolved_vendor_root),
        "--platform",
        platform,
        "--scheduler",
        scheduler,
        "--strict",
        "--json",
    )
    if build_root_value is not None:
        argv += ("--build-root", str(_resolve_path(build_root_value)))
    for library_path in library_path_values:
        argv += ("--library-path", str(_resolve_path(library_path)))
    if pythonpath_value is not None:
        argv += ("--pythonpath", pythonpath_value)
    if path_value is not None:
        argv += ("--path", path_value)
    for key in KORALI_RUNTIME_ENV_KEYS:
        if key in resolved_env:
            argv += (f"--env-{key.lower().replace('_', '-')}", resolved_env[key])

    return KoraliRuntimeValidationCommand(
        argv=argv,
        cwd=resolved_repo_root,
        env=tuple(sorted(resolved_env.items())),
        summary="Validate the Korali runtime contract without importing heavy optional dependencies.",
    )


def validate_korali_runtime_contract(
    repo_root: str | Path,
    *,
    env: Mapping[str, str] | None = None,
    vendor_root: str | Path | None = None,
    build_root: str | Path | None = None,
    library_paths: Sequence[str | Path] = (),
    pythonpath_hint: str | None = None,
    path_hint: str | None = None,
    platform: str = KORALI_RUNTIME_PLATFORM,
    scheduler: str = KORALI_RUNTIME_SCHEDULER,
) -> KoraliRuntimeValidationReport:
    resolved_repo_root = _resolve_path(repo_root)
    resolved_env = _normalize_env(env)
    issues: list[KoraliRuntimeValidationIssue] = []

    site_value = resolved_env.get("MESOUQ_SITE")
    if site_value is None:
        issues.append(
            KoraliRuntimeValidationIssue(
                severity="warning",
                code="site-missing",
                message=(
                    "MESOUQ_SITE was not supplied. Karolina validation still works, but operators should "
                    "record the active site label alongside the runtime snapshot."
                ),
            )
        )
    elif site_value != platform:
        issues.append(
            KoraliRuntimeValidationIssue(
                severity="error",
                code="site-mismatch",
                message=f"MESOUQ_SITE={site_value!r} does not match the validation platform {platform!r}.",
            )
        )

    site_runtime_root = resolved_env.get("MESOUQ_SITE_RUNTIME_ROOT")
    if site_runtime_root is None:
        issues.append(
            KoraliRuntimeValidationIssue(
                severity="warning",
                code="site-runtime-root-missing",
                message=(
                    "MESOUQ_SITE_RUNTIME_ROOT was not supplied. The validator will use the repo-local "
                    "_karolina fallback conventions instead of a staged runtime root."
                ),
            )
        )
    else:
        _validate_path_hint_entries(issues, "MESOUQ_SITE_RUNTIME_ROOT", (site_runtime_root,))

    if not resolved_repo_root.exists():
        issues.append(
            KoraliRuntimeValidationIssue(
                severity="error",
                code="repo-root-missing",
                message=f"repo_root {resolved_repo_root} does not exist.",
            )
        )
    elif not resolved_repo_root.is_dir():
        issues.append(
            KoraliRuntimeValidationIssue(
                severity="error",
                code="repo-root-not-directory",
                message=f"repo_root {resolved_repo_root} is not a directory.",
            )
        )

    expected_vendor_root = resolved_repo_root / KORALI_VENDOR_RELATIVE_ROOT
    resolved_vendor_root = _resolve_path(vendor_root) if vendor_root is not None else expected_vendor_root

    if vendor_root is None:
        if not expected_vendor_root.exists():
            issues.append(
                KoraliRuntimeValidationIssue(
                    severity="error",
                    code="vendor-root-missing",
                    message=(
                        f"Missing vendored Korali source root at {expected_vendor_root}. "
                        "Keep extern/korali checked in as protected source, not generated output."
                    ),
                )
            )
    else:
        if resolved_vendor_root != expected_vendor_root:
            issues.append(
                KoraliRuntimeValidationIssue(
                    severity="error",
                    code="vendor-root-mismatch",
                    message=(
                        f"vendor_root must resolve to {expected_vendor_root}, got {resolved_vendor_root}. "
                        "extern/korali is the protected vendored source root."
                    ),
                )
            )
        if not resolved_vendor_root.exists():
            issues.append(
                KoraliRuntimeValidationIssue(
                    severity="error",
                    code="vendor-root-missing",
                    message=f"vendor_root {resolved_vendor_root} does not exist.",
                )
            )
        elif not resolved_vendor_root.is_dir():
            issues.append(
                KoraliRuntimeValidationIssue(
                    severity="error",
                    code="vendor-root-not-directory",
                    message=f"vendor_root {resolved_vendor_root} is not a directory.",
                )
            )

    _validate_path_like(
        issues,
        "vendor_root",
        resolved_vendor_root,
        must_exist=True,
        private_path_ok=False,
    )

    build_root_value = build_root if build_root is not None else resolved_env.get("MESOUQ_KORALI_BUILD_ROOT")
    if build_root_value is not None:
        resolved_build_root = _resolve_path(build_root_value)
        _validate_path_like(
            issues,
            "build_root",
            resolved_build_root,
            must_exist=True,
            private_path_ok=False,
        )
    else:
        issues.append(
            KoraliRuntimeValidationIssue(
                severity="warning",
                code="build-root-missing",
                message=(
                    "No explicit build root was provided. That is acceptable for source-tree validation, "
                    "but operators should still record the build prefix used for Korali."
                ),
            )
        )

    library_path_values: Sequence[str | Path]
    if library_paths:
        library_path_values = library_paths
    elif resolved_env.get("MESOUQ_KORALI_LIBRARY_PATH"):
        library_path_values = _split_path_list(resolved_env["MESOUQ_KORALI_LIBRARY_PATH"])
    else:
        library_path_values = ()

    if library_path_values:
        for index, library_path in enumerate(library_path_values):
            _validate_path_like(
                issues,
                f"library_paths[{index}]",
                _resolve_path(library_path),
                must_exist=True,
                private_path_ok=False,
            )
    else:
        issues.append(
            KoraliRuntimeValidationIssue(
                severity="warning",
                code="library-paths-missing",
                message=(
                    "No library path hints were supplied. On Karolina, operators should provide the "
                    "runtime library prefix or LD_LIBRARY_PATH/DYLD_LIBRARY_PATH equivalent."
                ),
            )
        )

    pythonpath_value = pythonpath_hint if pythonpath_hint is not None else resolved_env.get("PYTHONPATH")
    if pythonpath_value:
        pythonpath_entries = _split_path_list(pythonpath_value)
        _validate_path_hint_entries(issues, "PYTHONPATH", pythonpath_entries)
        expected_src = str(resolved_repo_root / "src")
        expected_repo = str(resolved_repo_root)
        if expected_src not in pythonpath_entries:
            issues.append(
                KoraliRuntimeValidationIssue(
                    severity="warning",
                    code="pythonpath-missing-src",
                    message=f"PYTHONPATH does not include repo-local src root {expected_src}.",
                )
            )
        if expected_repo not in pythonpath_entries:
            issues.append(
                KoraliRuntimeValidationIssue(
                    severity="warning",
                    code="pythonpath-missing-repo",
                    message=f"PYTHONPATH does not include repo root {expected_repo}.",
                )
            )
    else:
        issues.append(
            KoraliRuntimeValidationIssue(
                severity="warning",
                code="pythonpath-missing",
                message=(
                    "No PYTHONPATH hint was supplied. Validation can still confirm the vendored tree, "
                    "but a bootstrap run should record the repo-local import path order."
                ),
            )
        )

    path_value = path_hint if path_hint is not None else resolved_env.get("PATH")
    if path_value:
        path_entries = _split_path_list(path_value)
        _validate_path_hint_entries(issues, "PATH", path_entries)
    else:
        issues.append(
            KoraliRuntimeValidationIssue(
                severity="warning",
                code="path-missing",
                message=(
                    "No PATH hint was supplied. Validation can still run, but operators should retain "
                    "the executable search path used for Korali and Slurm launch helpers."
                ),
            )
        )

    return KoraliRuntimeValidationReport(
        repo_root=resolved_repo_root,
        vendor_root=resolved_vendor_root,
        platform=platform,
        scheduler=scheduler,
        issues=tuple(issues),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m meso_uq.platforms.korali_runtime",
        description="Validate the dependency-light Korali runtime policy contract.",
    )
    parser.add_argument("--repo-root", required=True, help="Path to the MesoUQ repository root.")
    parser.add_argument(
        "--vendor-root",
        help="Path to the vendored Korali root. Defaults to repo_root/extern/korali.",
    )
    parser.add_argument("--build-root", help="Optional Korali build prefix to validate.")
    parser.add_argument(
        "--library-path",
        action="append",
        default=[],
        help="Optional library-path hint. May be repeated.",
    )
    parser.add_argument(
        "--pythonpath",
        help="Optional PYTHONPATH hint to validate against repo-local imports.",
    )
    parser.add_argument(
        "--path",
        dest="path_hint",
        help="Optional PATH hint to validate against launch helper expectations.",
    )
    parser.add_argument("--platform", default=KORALI_RUNTIME_PLATFORM)
    parser.add_argument("--scheduler", default=KORALI_RUNTIME_SCHEDULER)
    parser.add_argument("--strict", action="store_true", help="Return non-zero when warnings are present.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    for env_key in KORALI_RUNTIME_ENV_KEYS:
        parser.add_argument(
            f"--env-{env_key.lower().replace('_', '-')}",
            dest=f"env_{env_key.lower()}",
            help=f"Optional snapshot of the {env_key} environment value.",
        )

    args = parser.parse_args(argv)
    env = {
        env_key: value
        for env_key, value in (
            (env_key, getattr(args, f"env_{env_key.lower()}")) for env_key in KORALI_RUNTIME_ENV_KEYS
        )
        if value is not None
    }

    report = validate_korali_runtime_contract(
        args.repo_root,
        env=env,
        vendor_root=args.vendor_root,
        build_root=args.build_root,
        library_paths=args.library_path,
        pythonpath_hint=args.pythonpath,
        path_hint=args.path_hint,
        platform=args.platform,
        scheduler=args.scheduler,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.format_text())

    if not report.ok:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


def _normalize_env(env: Mapping[str, str] | None) -> dict[str, str]:
    if env is None:
        return {}
    return {str(key): str(value) for key, value in env.items()}


def _resolve_path(value: str | Path | None) -> Path:
    if value is None:
        raise ValueError("Path value must not be None.")
    return Path(value).expanduser().resolve()


def _split_path_list(value: str) -> tuple[str, ...]:
    return tuple(entry for entry in value.split(os.pathsep) if entry)


def _validate_path_like(
    issues: list[KoraliRuntimeValidationIssue],
    label: str,
    path_value: Path,
    *,
    must_exist: bool,
    private_path_ok: bool,
) -> None:
    path_text = str(path_value)
    if not private_path_ok:
        for private_prefix in FORBIDDEN_PRIVATE_PATHS:
            if private_prefix in path_text:
                issues.append(
                    KoraliRuntimeValidationIssue(
                        severity="error",
                        code=f"{label}-private-path",
                        message=f"{label} uses forbidden private path prefix {private_prefix!r}: {path_text}.",
                    )
                )
                break
    if must_exist and not path_value.exists():
        issues.append(
            KoraliRuntimeValidationIssue(
                severity="error",
                code=f"{label}-missing",
                message=f"{label} does not exist: {path_text}.",
            )
        )


def _validate_path_hint_entries(
    issues: list[KoraliRuntimeValidationIssue],
    label: str,
    entries: Sequence[str],
) -> None:
    for entry in entries:
        for private_prefix in FORBIDDEN_PRIVATE_PATHS:
            if private_prefix in entry:
                issues.append(
                    KoraliRuntimeValidationIssue(
                        severity="error",
                        code=f"{label.lower()}-private-path",
                        message=f"{label} contains forbidden private path prefix {private_prefix!r}: {entry}.",
                    )
                )
                break


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
