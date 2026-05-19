from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
TRIM_SCRIPT = REPO_ROOT / "gv" / "eigenmodes" / "src" / "analysis" / "trim_eigenmodes.py"


def _load_trim_module():
    spec = importlib.util.spec_from_file_location("mesouq_test_gv_trim_eigenmodes", TRIM_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_trim_eigenmodes_frequency_window_rejects_low_frequency_modes_and_orders_paper_modes(tmp_path: Path) -> None:
    module = _load_trim_module()
    output = tmp_path / "output"
    output.mkdir()
    # With kBT=1, these map to frequencies 1, 4, 2, 5, 3.  The paper
    # window should select by frequency threshold and write final modes in
    # ascending paper-frequency order rather than raw covariance order.
    np.savetxt(output / "eigvalues.txt", np.array([1.0, 1.0 / 16.0, 0.25, 1.0 / 25.0, 1.0 / 9.0]))
    np.savetxt(output / "eigvectors.txt", np.arange(5.0 * 3.0).reshape(5, 3))

    rc = module.main(
        [
            "--mode-count",
            "3",
            "--policy",
            "frequency-min",
            "--min-frequency",
            "2.5",
            "--output-dir",
            str(output),
            "--parameter-file",
            str(tmp_path / "missing.yaml"),
        ]
    )

    manifest = json.loads((output / "mode_window_manifest.json").read_text(encoding="utf-8"))
    assert rc == 0
    assert np.loadtxt(output / "eigvalues_new.txt").tolist() == [1.0 / 9.0, 1.0 / 16.0, 1.0 / 25.0]
    assert np.loadtxt(output / "eigvectors_new.txt").tolist() == [[12.0, 13.0, 14.0], [3.0, 4.0, 5.0], [9.0, 10.0, 11.0]]
    assert manifest["mode_window_policy"] == "frequency-min"
    assert manifest["raw_eigenpair_count"] == 5
    assert manifest["final_mode_count"] == 3
    assert manifest["final_mode_indices"] == [0, 1, 2]
    assert manifest["selected_paper_mode_indices"] == [0, 1, 2]
    assert manifest["selected_raw_mode_indices"] == [4, 1, 3]
    assert manifest["rejected_low_frequency_count"] == 2


def test_trim_eigenmodes_explicit_indices_are_manifested(tmp_path: Path) -> None:
    module = _load_trim_module()
    output = tmp_path / "output"
    output.mkdir()
    np.savetxt(output / "eigvalues.txt", np.array([10.0, 9.0, 8.0, 7.0]))

    module.main(
        [
            "--mode-count",
            "2",
            "--policy",
            "explicit-indices",
            "--explicit-indices",
            "1,3",
            "--output-dir",
            str(output),
        ]
    )

    manifest = json.loads((output / "mode_window_manifest.json").read_text(encoding="utf-8"))
    assert np.loadtxt(output / "eigvalues_new.txt").tolist() == [9.0, 7.0]
    assert manifest["selected_raw_mode_indices"] == [1, 3]
    assert manifest["selected_paper_mode_indices"] == [0, 1]
