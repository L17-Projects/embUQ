from pathlib import Path

import pandas as pd

from meso_uq.postprocess.maps import extract_map_from_directory, load_posterior_samples
from meso_uq.postprocess.plots import plot_d0_correlations, plot_posterior_marginals, plot_validation_overlay


def _write_latest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """
{
  "Variables": [
    {"Name": "Yt"},
    {"Name": "kb"},
    {"Name": "d0"},
    {"Name": "[Sigma]"}
  ],
  "Results": {
    "Posterior Sample Database": [
      [1.0, 2.0, 0.1, 0.01],
      [1.5, 2.5, 0.2, 0.02],
      [0.9, 1.8, 0.05, 0.03]
    ],
    "Posterior Sample LogLikelihood Database": [-10.0, -5.0, -12.0],
    "Posterior Sample LogPrior Database": [-1.0, -0.5, -0.8]
  }
}
""".strip(),
        encoding="utf-8",
    )


def test_map_extraction_and_plotting_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "results_phase_3b" / "compression_2.1um"
    _write_latest(run_dir / "latest")

    df = load_posterior_samples(run_dir)
    assert list(df.columns) == ["Yt", "kb", "d0", "sigma", "logLikelihood", "logPrior", "logPosterior"]
    assert len(df) == 3

    map_df = extract_map_from_directory(run_dir, output_csv=str(tmp_path / "map.csv"))
    assert len(map_df) == 1
    assert float(map_df.iloc[0]["logPosterior"]) == -5.5

    samples_csv = tmp_path / "samples.csv"
    df.to_csv(samples_csv, index=False)

    ref_csv = tmp_path / "reference.csv"
    pred_csv = tmp_path / "prediction.csv"
    pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 1.5, 2.0]}).to_csv(ref_csv, index=False)
    pd.DataFrame({"x": [0.0, 1.0, 2.0], "mean": [1.1, 1.4, 1.9]}).to_csv(pred_csv, index=False)

    overlay_png = tmp_path / "overlay.png"
    d0_png = tmp_path / "d0.png"
    marg_png = tmp_path / "marginals.png"

    plot_validation_overlay(str(ref_csv), str(pred_csv), str(overlay_png), x_col="x", y_ref_col="y", y_pred_col="mean")
    plot_d0_correlations(str(samples_csv), str(d0_png), d0_col="d0")
    plot_posterior_marginals(str(samples_csv), str(marg_png))

    assert overlay_png.exists()
    assert d0_png.exists()
    assert marg_png.exists()


def test_validation_overlay_accepts_headerless_reference_csv(tmp_path: Path) -> None:
    ref_csv = tmp_path / "reference.csv"
    pred_csv = tmp_path / "prediction.csv"
    overlay_png = tmp_path / "overlay.png"

    ref_csv.write_text("0.2,288.4\n0.26,467.4\n0.34,576.8\n", encoding="utf-8")
    pd.DataFrame({"x": [0.2, 0.26, 0.34], "mean": [236.1, 350.4, 556.3]}).to_csv(pred_csv, index=False)

    plot_validation_overlay(str(ref_csv), str(pred_csv), str(overlay_png))

    assert overlay_png.exists()
