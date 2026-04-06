from pathlib import Path

import pandas as pd

from meso_uq.surrogate.model_selection import grid_search_tabular_surrogate


def test_grid_search_tabular_surrogate_smoke(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "Yt": [1.0, 1.1, 0.9, 1.2, 0.8, 1.05, 0.95, 1.15, 0.85, 1.0],
            "kb": [2.0, 2.1, 1.9, 2.2, 1.8, 2.05, 1.95, 2.15, 1.85, 2.0],
            "b1": [0.1] * 10,
            "b2": [0.2] * 10,
            "a3": [0.3] * 10,
            "a4": [0.4] * 10,
            "disp": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.1, 0.3, 0.5, 0.7],
            "F": [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 0.2, 0.6, 1.0, 1.4],
        }
    )

    result = grid_search_tabular_surrogate(
        df,
        input_cols=["Yt", "kb", "b1", "b2", "a3", "a4", "disp"],
        target_col="F",
        output_dir=tmp_path,
        widths=[8, 16],
        depths=[1, 2],
        batch_size=4,
        lr=1e-3,
        max_epoch=5,
    )

    leaderboard = result["leaderboard"]
    assert len(leaderboard) == 4
    assert (tmp_path / "leaderboard.csv").exists()
    assert (tmp_path / "best_model.json").exists()
    assert Path(result["best"]["model_path"]).exists()
