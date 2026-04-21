from __future__ import annotations

import os
import socket
from datetime import datetime
from pathlib import Path

_VALID_SITES = {"vega", "karolina"}


def detect_hpc_site(*, env: dict[str, str] | None = None, hostname: str | None = None) -> str:
    source_env = env if env is not None else os.environ
    explicit = source_env.get("HPC_SITE", "").strip().lower()
    if explicit in _VALID_SITES:
        return explicit

    host = (hostname if hostname is not None else socket.gethostname()).strip().lower()
    if "karolina" in host or host.startswith("acn"):
        return "karolina"
    if "vega" in host:
        return "vega"

    # Backward-compatible fallback for existing Vega-first logic.
    return "vega"


def make_run_tag(now: datetime | None = None) -> str:
    ts = now if now is not None else datetime.now()
    return ts.strftime("%Y%m%d_%H%M%S")


def default_runs_root(
    repo_root: str | Path,
    workflow_name: str,
    *,
    site: str | None = None,
    run_tag: str | None = None,
) -> Path:
    root = Path(repo_root).resolve()
    resolved_site = site if site is not None else detect_hpc_site()
    if resolved_site not in _VALID_SITES:
        raise ValueError(f"Unsupported site '{resolved_site}'. Expected one of: {sorted(_VALID_SITES)}")
    resolved_tag = run_tag if run_tag is not None else make_run_tag()
    return root / "_runs" / resolved_site / workflow_name / resolved_tag
