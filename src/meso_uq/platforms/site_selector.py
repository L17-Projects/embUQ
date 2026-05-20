"""Canonical MesoUQ HPC site selection helpers."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from typing import Mapping

VALID_MESOUQ_SITES = frozenset({"vega", "karolina"})
SITE_ENV_NAME = "MESOUQ_SITE"
LEGACY_SITE_ENV_NAME = "HPC" "_SITE"


@dataclass(frozen=True)
class SiteSelectionError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def normalize_hpc_site(site: str | None, *, label: str = "site") -> str:
    resolved = (site or "").strip().lower()
    if resolved not in VALID_MESOUQ_SITES:
        expected = ", ".join(sorted(VALID_MESOUQ_SITES))
        raise SiteSelectionError(f"Unsupported {label}={site!r}. Expected one of: {expected}.")
    return resolved


def _host_site(hostname: str | None = None) -> str | None:
    host = (hostname if hostname is not None else socket.gethostname()).strip().lower()
    if "karolina" in host or host.startswith("acn"):
        return "karolina"
    if "vega" in host:
        return "vega"
    return None


def resolve_hpc_site(
    *,
    cli_site: str | None = None,
    env: Mapping[str, str] | None = None,
    default: str | None = "vega",
    hostname: str | None = None,
    allow_hostname: bool = False,
) -> str:
    source_env = env if env is not None else os.environ
    if LEGACY_SITE_ENV_NAME in source_env:
        raise SiteSelectionError(
            f"{LEGACY_SITE_ENV_NAME} is no longer supported. Use --site or {SITE_ENV_NAME}."
        )

    explicit_site = normalize_hpc_site(cli_site, label="--site") if cli_site is not None else None
    env_value = source_env.get(SITE_ENV_NAME)
    env_site = normalize_hpc_site(env_value, label=SITE_ENV_NAME) if env_value else None

    if explicit_site is not None and env_site is not None and explicit_site != env_site:
        raise SiteSelectionError(
            f"Conflicting site selectors: --site={explicit_site!r} and {SITE_ENV_NAME}={env_site!r}."
        )
    if explicit_site is not None:
        return explicit_site
    if env_site is not None:
        return env_site

    if allow_hostname:
        detected = _host_site(hostname)
        if detected is not None:
            return detected

    if default is None:
        raise SiteSelectionError(f"Missing site selector. Pass --site or set {SITE_ENV_NAME}.")
    return normalize_hpc_site(default, label="default site")
