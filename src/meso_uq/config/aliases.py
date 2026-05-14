"""Legacy configuration path alias tooling.

This module is intentionally dependency-light.  It keeps a machine-readable
inventory of legacy config roots and resolves repository-relative legacy paths
into canonicalized records for migration tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from meso_uq.configs.policy import FORBIDDEN_PRIVATE_PATHS


@dataclass(frozen=True)
class LegacyConfigPathAlias:
    """An inventory record for a legacy config root and its canonical replacement."""

    legacy_root: str
    canonical_root: str
    description: str
    retirement: str


@dataclass(frozen=True)
class LegacyConfigPathResolution:
    """A resolved legacy config path mapped to a canonical config location."""

    alias: LegacyConfigPathAlias
    requested_path: str
    canonical_path: Path


_BASE_LEGACY_CONFIG_PATH_ALIASES: tuple[LegacyConfigPathAlias, ...] = (
    LegacyConfigPathAlias(
        legacy_root="inference/configs/production",
        canonical_root="inference/configs/production",
        description="full-model production config root",
        retirement="Keep for one migration cycle unless/ until new canonical config APIs replace it.",
    ),
    LegacyConfigPathAlias(
        legacy_root="inference/configs/validation",
        canonical_root="inference/configs/validation",
        description="full-model validation config root",
        retirement="Keep for one migration cycle unless/ until new canonical config APIs replace it.",
    ),
    LegacyConfigPathAlias(
        legacy_root="reduced/configs/production",
        canonical_root="reduced/configs/production",
        description="reduced-model production config root",
        retirement="Keep for one migration cycle unless/ until new canonical config APIs replace it.",
    ),
    LegacyConfigPathAlias(
        legacy_root="reduced/configs/validation",
        canonical_root="reduced/configs/validation",
        description="reduced-model validation config root",
        retirement="Keep for one migration cycle unless/ until new canonical config APIs replace it.",
    ),
    LegacyConfigPathAlias(
        legacy_root="examples/configs",
        canonical_root="examples/configs",
        description="example configuration root (refreshed/archived during migration)",
        retirement="Keep while examples/configs refresh/archive decisions are pending.",
    ),
)


def _available_aliases(project_root: Path) -> tuple[LegacyConfigPathAlias, ...]:
    aliases: list[LegacyConfigPathAlias] = []
    for alias in _BASE_LEGACY_CONFIG_PATH_ALIASES:
        if (project_root / alias.legacy_root).exists():
            aliases.append(alias)
    return tuple(aliases)


def list_legacy_config_path_aliases(
    *,
    project_root: str | Path | None = None,
) -> tuple[LegacyConfigPathAlias, ...]:
    """Return known legacy config path aliases.

    If ``project_root`` is provided, aliases are filtered to roots that currently
    exist under the project root.
    """

    if project_root is None:
        return _BASE_LEGACY_CONFIG_PATH_ALIASES
    return _available_aliases(Path(project_root))


def resolve_legacy_config_path(
    project_root: str | Path,
    legacy_path: str | Path,
) -> LegacyConfigPathResolution:
    """Resolve a legacy config path into a canonical legacy-alias record.

    The function keeps resolution conservative:
    * only repository-relative paths rooted at a known legacy config root are allowed;
    * path traversal via ``..`` is rejected;
    * hard-coded private absolute paths are rejected.
    """

    project_root_path = Path(project_root).resolve()
    candidate = Path(legacy_path)

    if candidate.is_absolute():
        candidate_text = candidate.as_posix()
        if any(forbidden in candidate_text for forbidden in FORBIDDEN_PRIVATE_PATHS):
            raise ValueError(f"Legacy config path uses forbidden private root: {candidate_text}")
        raise ValueError(f"Legacy config path must be repository-relative: {candidate_text}")

    if ".." in candidate.parts:
        raise ValueError(f"Legacy config path contains path traversal: {candidate.as_posix()}")

    normalized = candidate.as_posix()
    if not normalized:
        raise ValueError("Legacy config path must not be empty.")

    aliases = _available_aliases(project_root_path)
    for alias in aliases:
        alias_root = Path(alias.legacy_root)
        if _is_within_legacy_root(candidate, alias_root):
            canonical_path = project_root_path / alias.canonical_root / _relative_to_root(
                candidate,
                alias_root,
            )
            return LegacyConfigPathResolution(
                alias=alias,
                requested_path=normalized,
                canonical_path=canonical_path,
            )

    known_roots = ", ".join(alias.legacy_root for alias in aliases)
    raise ValueError(f"Unknown legacy config path '{normalized}'. Known roots: [{known_roots}]")


def _is_within_legacy_root(candidate: Path, alias_root: Path) -> bool:
    candidate_parts = tuple(candidate.parts)
    alias_parts = tuple(alias_root.parts)
    return candidate_parts[: len(alias_parts)] == alias_parts


def _relative_to_root(candidate: Path, root: Path) -> Path:
    candidate_parts = tuple(candidate.parts)
    root_parts = tuple(root.parts)
    if candidate_parts[: len(root_parts)] != root_parts:
        return candidate
    return Path(*candidate_parts[len(root_parts) :])


__all__ = [
    "LegacyConfigPathAlias",
    "LegacyConfigPathResolution",
    "list_legacy_config_path_aliases",
    "resolve_legacy_config_path",
]
