from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from meso_uq.platforms.site_selector import VALID_MESOUQ_SITES, resolve_hpc_site

_VALID_SITES = set(VALID_MESOUQ_SITES)
_REPO_OUTPUT_ROOT_PREFIXES = ("_runs", "paper_data")
_EXTERNAL_OUTPUT_ROOT_ANCHORS = (*_REPO_OUTPUT_ROOT_PREFIXES, "runs")


def _candidate_output_root(output_root: str | Path, repo_root: Path) -> Path:
    candidate = Path(output_root).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _enforces_canonical_output_root(output_root: Path, repo_root: Path) -> None:
    try:
        relative = output_root.relative_to(repo_root)
    except ValueError:
        if not any(part in output_root.parts for part in _EXTERNAL_OUTPUT_ROOT_ANCHORS):
            raise ValueError(
                "Output roots for runtime artifacts must be rooted under '.../_runs', "
                "'.../runs', or '.../paper_data'."
            )
        return


def detect_hpc_site(*, env: dict[str, str] | None = None, hostname: str | None = None) -> str:
    return resolve_hpc_site(env=env, hostname=hostname, allow_hostname=True, default="vega")


def make_run_tag(now: datetime | None = None) -> str:
    ts = now if now is not None else datetime.now()
    return ts.strftime("%Y%m%d_%H%M%S")


def default_runs_root(
    repo_root: str | Path,
    workflow_name: str,
    *,
    site: str | None = None,
    run_tag: str | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    source_env = env if env is not None else os.environ
    root = Path(repo_root).resolve()
    resolved_site = resolve_hpc_site(cli_site=site, env=source_env, allow_hostname=True, default="vega")
    if resolved_site not in _VALID_SITES:
        raise ValueError(f"Unsupported site '{resolved_site}'. Expected one of: {sorted(_VALID_SITES)}")
    resolved_tag = run_tag if run_tag is not None else make_run_tag()
    runs_root = source_env.get("MESOUQ_RUNS_ROOT", "").strip()
    if runs_root:
        return Path(runs_root).expanduser().resolve() / workflow_name / resolved_tag
    return root / "_runs" / resolved_site / workflow_name / resolved_tag


def ensure_canonical_output_root(
    output_root: str | Path,
    repo_root: str | Path,
    *,
    field_name: str = "--output-root",
) -> Path:
    """
    Return a resolved root that is guaranteed to be under the canonical output trees.

    Canonical trees are:
      - <repo>/_runs/...   (runtime artifacts)
      - <repo>/paper_data/... (durable paper-facing campaign assets)

    We still allow explicitly provided non-repo-root absolute paths if they already carry
    a canonical anchor segment (``_runs``, ``runs``, or ``paper_data``). This keeps site
    scratch overrides explicit and testable.
    """

    repo = Path(repo_root).resolve()
    resolved = _candidate_output_root(output_root, repo)
    _enforces_canonical_output_root(resolved, repo)
    if resolved == repo:
        raise ValueError(f"{field_name} must not resolve to repository root '{repo}'.")
    try:
        relative = resolved.relative_to(repo)
    except ValueError:
        return resolved

    if relative.parts[0] not in _REPO_OUTPUT_ROOT_PREFIXES:
        raise ValueError(
            f"{field_name} = '{resolved}' is not canonical. Use a path under {repo}/_runs "
            f"for runtime outputs or {repo}/paper_data for durable campaign artifacts."
        )
    return resolved
