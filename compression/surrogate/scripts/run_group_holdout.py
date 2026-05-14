#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.core import Modality
from meso_uq.surrogate.emb_workflows import best_artifact_suffix, run_emb_group_holdout_cli


def _best_artifact_suffix(family: str) -> str:
    return best_artifact_suffix(family)


def main(argv: list[str] | None = None):
    return run_emb_group_holdout_cli(Modality.COMPRESSION, argv=argv)


if __name__ == "__main__":
    main()
