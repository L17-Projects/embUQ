#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.core import Modality
from meso_uq.surrogate.emb_workflows import run_emb_dnn_training_cli


def read_compression_training_table(*args, **kwargs):
    from meso_uq.surrogate.cli import read_compression_training_table as _reader

    return _reader(*args, **kwargs)


def train_tabular_surrogate(*args, **kwargs):
    from meso_uq.surrogate.cli import train_tabular_surrogate as _trainer

    return _trainer(*args, **kwargs)


def main(argv: list[str] | None = None):
    return run_emb_dnn_training_cli(
        Modality.COMPRESSION,
        argv=argv,
        reader=read_compression_training_table,
        trainer=train_tabular_surrogate,
    )


if __name__ == "__main__":
    main()
