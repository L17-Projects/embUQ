from pathlib import Path

from papers.huq_emb import uqdpd_generate_reduced_story_assets as assets


def test_load_map_parameters_accepts_current_meso_uq_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "compression_2.1um.csv"
    csv_path.write_text(
        ",".join(
            [
                "Yt",
                "kb",
                "b1",
                "b2",
                "a3",
                "a4",
                "d0",
                "sigma",
                "logLikelihood",
                "logPrior",
                "logPosterior",
            ]
        )
        + "\n"
        + ",".join(
            [
                "1.0",
                "2.0",
                "3.0",
                "4.0",
                "5.0",
                "6.0",
                "7.0",
                "8.0",
                "9.0",
                "10.0",
                "11.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(assets, "map_phase3b_path", lambda modality, model_kind, diameter: csv_path)

    data = assets.load_map_parameters("compression", "full", "2.1")

    assert data["parameters"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert data["sigma"] == 8.0
    assert data["logLikelihood"] == 9.0
    assert data["logPrior"] == 10.0
    assert data["logPosterior"] == 11.0
