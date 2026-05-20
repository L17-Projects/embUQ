from __future__ import annotations

import argparse
import os

_VALID_SITES = {"vega", "karolina"}
_LEGACY_SITE_ENV = "HPC" "_SITE"


def add_site_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--site",
        choices=sorted(_VALID_SITES),
        default=None,
        help="Optional MesoUQ HPC site selector; must match MESOUQ_SITE when both are set.",
    )


def validate_site_argument(parser: argparse.ArgumentParser, cli_site: str | None) -> str | None:
    if _LEGACY_SITE_ENV in os.environ:
        parser.error(f"{_LEGACY_SITE_ENV} is no longer supported. Use --site or MESOUQ_SITE.")
    env_site = os.environ.get("MESOUQ_SITE")
    if env_site and env_site not in _VALID_SITES:
        parser.error(f"Unsupported MESOUQ_SITE={env_site!r}; expected one of: karolina, vega.")
    if cli_site and env_site and cli_site != env_site:
        parser.error(f"Conflicting site selectors: --site={cli_site} and MESOUQ_SITE={env_site}.")
    return cli_site or env_site
