from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_uqdpd_group_holdout_adapter_reads_staged_mesouq_layout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "papers" / "huq_emb" / "uqdpd_generate_reduced_story_assets.py"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        group_holdout_root = tmp_root / "group_holdout"
        figures_dir = tmp_root / "figures"
        holdout_leaf = group_holdout_root / "compression" / "2.9um" / "dnn"
        holdout_leaf.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        (holdout_leaf / "summary.json").write_text(
            json.dumps(
                {
                    "modality": "compression",
                    "surrogate_family": "dnn",
                    "diameter_um": "2.9",
                    "best_architecture": "microbubble_force_BEST",
                    "best_median_curve_rel_l2_pct": 12.5,
                    "representative_curve_rel_l2_pct": 12.5,
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame([{"curve_id": 7, "rel_l2_pct": 12.5}]).to_csv(
            holdout_leaf / "per_curve_metrics.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {"disp": 0.0, "F_true": 1.0, "F_pred": 1.5, "pred_std": 0.0},
                {"disp": 1.0, "F_true": 2.0, "F_pred": 2.5, "pred_std": 0.0},
            ]
        ).to_csv(holdout_leaf / "representative_curve.csv", index=False)

        previous_group_holdout = os.environ.get("MESOUQ_PAPER_GROUP_HOLDOUT_ROOT")
        previous_figures = os.environ.get("HUQ_PAPER_FIGURES_DIR")
        os.environ["MESOUQ_PAPER_GROUP_HOLDOUT_ROOT"] = str(group_holdout_root)
        os.environ["HUQ_PAPER_FIGURES_DIR"] = str(figures_dir)
        try:
            module = _load_module(script_path, "uqdpd_group_holdout_adapter_test")
        finally:
            if previous_group_holdout is None:
                os.environ.pop("MESOUQ_PAPER_GROUP_HOLDOUT_ROOT", None)
            else:
                os.environ["MESOUQ_PAPER_GROUP_HOLDOUT_ROOT"] = previous_group_holdout
            if previous_figures is None:
                os.environ.pop("HUQ_PAPER_FIGURES_DIR", None)
            else:
                os.environ["HUQ_PAPER_FIGURES_DIR"] = previous_figures

        metrics = module.load_best_holdout_curve_metrics("compression", "2.9")
        assert metrics["rel_l2_pct"].tolist() == [12.5]

        x_dpd, y_true_dpd, y_pred_dpd = module.load_representative_holdout_curve(
            "compression", "2.9"
        )
        assert x_dpd.tolist() == [0.0, 1.0]
        assert y_true_dpd.tolist() == [1.0, 2.0]
        assert y_pred_dpd.tolist() == [1.5, 2.5]

        summary = module.load_group_holdout_summary()
        assert summary.loc[0, "surrogate_family"] == "dnn"
        assert (figures_dir / "surrogate_group_holdout_summary.csv").exists()
