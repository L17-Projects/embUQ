from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml

from .parameters import GV_MATERIAL_PARAMETER_NAMES


_MATERIAL_ALIASES: dict[str, str] = {
    "muL": "mu_l",
}
_ZERO_ALLOWED_MATERIAL_PARAMETERS = frozenset({"b1", "b2", "a3", "a4"})


def _canonicalize_name(name: str) -> str:
    return _MATERIAL_ALIASES.get(name, name)


def _assert_finite_positive(name: str, value: object) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Material parameter '{name}' must be numeric.") from exc
    if not isfinite(value_f):
        raise ValueError(f"Material parameter '{name}' must be finite and > 0.")
    if value_f < 0.0:
        raise ValueError(f"Material parameter '{name}' must be finite and > 0.")
    if value_f == 0.0 and name not in _ZERO_ALLOWED_MATERIAL_PARAMETERS:
        raise ValueError(f"Material parameter '{name}' must be finite and > 0.")
    return value_f


def validate_material_parameter_overrides(overrides: Mapping[str, object]) -> dict[str, float]:
    """Validate exactly the nine GV material parameters and return canonical values."""

    canonicalized: dict[str, float] = {}
    seen_canonical: set[str] = set()

    for raw_name, raw_value in overrides.items():
        canonical_name = _canonicalize_name(raw_name)
        if canonical_name in seen_canonical:
            raise ValueError(f"Material parameter '{canonical_name}' is duplicated in overrides.")
        if canonical_name not in GV_MATERIAL_PARAMETER_NAMES:
            raise ValueError(f"Unexpected material parameter '{raw_name}'.")
        canonicalized[canonical_name] = _assert_finite_positive(canonical_name, raw_value)
        seen_canonical.add(canonical_name)

    required = set(GV_MATERIAL_PARAMETER_NAMES)
    provided = set(canonicalized)
    missing = required - provided
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required GV material parameters: {missing_list}")
    extra = provided - required
    if extra:
        extra_list = ", ".join(sorted(extra))
        raise ValueError(f"Unexpected GV material parameters: {extra_list}")

    ordered = {name: canonicalized[name] for name in GV_MATERIAL_PARAMETER_NAMES}
    return ordered


def _material_key_for_serialization(name: str, payload: Mapping[str, object]) -> str:
    if name == "mu_l" and "muL" in payload and "mu_l" not in payload:
        return "muL"
    return name


def _read_yaml_file(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return raw


def _write_yaml_file(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def apply_material_parameter_overrides(payload: Mapping[str, object], overrides: Mapping[str, object]) -> dict[str, object]:
    """Return a new mapping with GV material overrides applied."""

    canonical_overrides = validate_material_parameter_overrides(overrides)

    updated = dict(payload)
    for canonical_name, value in canonical_overrides.items():
        serialized_name = _material_key_for_serialization(canonical_name, payload)
        updated[serialized_name] = value

    return updated


def apply_material_overrides_to_yaml(
    path: Path | str,
    overrides: Mapping[str, object],
) -> dict[str, float]:
    """Apply GV material overrides directly into a YAML file."""

    target = Path(path)
    payload = _read_yaml_file(target)
    updated = apply_material_parameter_overrides(payload, overrides)
    _write_yaml_file(target, updated)
    return validate_material_parameter_overrides(overrides)


def _iter_parameter_paths(root: Path, patterns: Sequence[str]) -> Iterable[Path]:
    for pattern in patterns:
        yield from sorted(root.glob(pattern))


def apply_material_overrides_to_files(
    paths: Sequence[Path | str],
    overrides: Mapping[str, object],
) -> list[Path]:
    """Apply overrides to each file path in ``paths``."""

    canonical_overrides = validate_material_parameter_overrides(overrides)
    updated_files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = _read_yaml_file(path)
        updated = dict(payload)
        for canonical_name, value in canonical_overrides.items():
            updated[_material_key_for_serialization(canonical_name, payload)] = value
        _write_yaml_file(path, updated)
        updated_files.append(path)
    return updated_files


def apply_material_overrides_to_glob(
    root: Path | str,
    overrides: Mapping[str, object],
    *,
    patterns: Sequence[str] = ("parameters.prms*.yaml",),
) -> list[Path]:
    """Apply overrides to matching files under ``root``.

    This helper keeps material aliases explicit per file by preserving ``muL`` when
    it is already present in a target file.
    """

    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root_path}")

    paths = list(_iter_parameter_paths(root_path, patterns))
    if not paths:
        raise FileNotFoundError(f"No parameter files matched under {root_path} for patterns: {patterns}")
    return apply_material_overrides_to_files(paths, overrides)


def _parse_overrides_json(raw: str) -> dict[str, object]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("GV material overrides JSON must decode to a mapping.")
    return payload


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply explicit GV material parameter overrides to YAML files.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="YAML parameter files to update, typically parameter/parameters.prms<simnum>.yaml.",
    )
    parser.add_argument(
        "--overrides-json",
        required=True,
        help="JSON object containing ka, kb, mu, b1, b2, a3, a4, mu_l, and c.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    overrides = _parse_overrides_json(args.overrides_json)
    updated = apply_material_overrides_to_files([Path(path) for path in args.paths], overrides)
    for path in updated:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests.
    raise SystemExit(main())
