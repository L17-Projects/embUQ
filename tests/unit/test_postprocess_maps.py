"""Tests for meso_uq.postprocess.maps — MAP extraction from Korali state files."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from meso_uq.postprocess.maps import (
    _find_latest_state,
    _load_json,
    extract_map_from_directory,
    load_posterior_samples,
)


# ---------------------------------------------------------------------------
# _find_latest_state
# ---------------------------------------------------------------------------


def test_find_latest_state_prefers_latest_file(tmp_path: Path) -> None:
    (tmp_path / "latest").write_text("{}")
    (tmp_path / "genLatest.json").write_text("{}")
    assert _find_latest_state(tmp_path) == tmp_path / "latest"


def test_find_latest_state_falls_back_to_genLatest(tmp_path: Path) -> None:
    (tmp_path / "genLatest.json").write_text("{}")
    assert _find_latest_state(tmp_path) == tmp_path / "genLatest.json"


def test_find_latest_state_falls_back_to_sorted_json(tmp_path: Path) -> None:
    (tmp_path / "gen00000001.json").write_text("{}")
    (tmp_path / "gen00000002.json").write_text("{}")
    assert _find_latest_state(tmp_path) == tmp_path / "gen00000002.json"


def test_find_latest_state_returns_none_for_empty_dir(tmp_path: Path) -> None:
    assert _find_latest_state(tmp_path) is None


# ---------------------------------------------------------------------------
# _load_json
# ---------------------------------------------------------------------------


def test_load_json_round_trips(tmp_path: Path) -> None:
    data = {"key": [1, 2, 3]}
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    assert _load_json(path) == data


# ---------------------------------------------------------------------------
# load_posterior_samples
# ---------------------------------------------------------------------------


def _write_state(run_dir: Path, state: dict) -> Path:
    path = run_dir / "latest"
    path.write_text(json.dumps(state))
    return path


def _minimal_state(
    samples: list[list[float]],
    loglike: list[float],
    logprior: list[float] | None = None,
    var_names: list[str] | None = None,
) -> dict:
    variables = [{"Name": n} for n in (var_names or ["Yt", "kb"])]
    state: dict = {
        "Variables": variables,
        "Results": {
            "Posterior Sample Database": samples,
            "Posterior Sample LogLikelihood Database": loglike,
        },
    }
    if logprior is not None:
        state["Results"]["Posterior Sample LogPrior Database"] = logprior
    return state


def test_load_posterior_samples_basic(tmp_path: Path) -> None:
    samples = [[1.0, 2.0], [3.0, 4.0]]
    loglike = [-10.0, -5.0]
    logprior = [-1.0, -2.0]
    _write_state(tmp_path, _minimal_state(samples, loglike, logprior))

    df = load_posterior_samples(tmp_path)

    assert list(df.columns) == ["Yt", "kb", "logLikelihood", "logPrior", "logPosterior"]
    assert len(df) == 2
    assert df["logPosterior"].iloc[0] == pytest.approx(-11.0)
    assert df["logPosterior"].iloc[1] == pytest.approx(-7.0)


def test_load_posterior_samples_without_logprior(tmp_path: Path) -> None:
    samples = [[1.0, 2.0]]
    loglike = [-5.0]
    _write_state(tmp_path, _minimal_state(samples, loglike))

    df = load_posterior_samples(tmp_path)
    assert "logPrior" not in df.columns
    assert df["logPosterior"].iloc[0] == pytest.approx(-5.0)


def test_load_posterior_samples_sigma_rename(tmp_path: Path) -> None:
    state = _minimal_state([[1.0, 2.0]], [-5.0], var_names=["Yt", "[Sigma]"])
    _write_state(tmp_path, state)

    df = load_posterior_samples(tmp_path)
    assert "sigma" in df.columns
    assert "[Sigma]" not in df.columns


def test_load_posterior_samples_raises_on_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No Korali state JSON found"):
        load_posterior_samples(tmp_path)


def test_load_posterior_samples_raises_on_missing_databases(tmp_path: Path) -> None:
    _write_state(tmp_path, {"Results": {}})
    with pytest.raises(ValueError, match="does not contain posterior"):
        load_posterior_samples(tmp_path)


def test_load_posterior_samples_unnamed_variable_fallback(tmp_path: Path) -> None:
    state = {
        "Variables": [{}],
        "Results": {
            "Posterior Sample Database": [[1.0]],
            "Posterior Sample LogLikelihood Database": [-5.0],
        },
    }
    _write_state(tmp_path, state)
    df = load_posterior_samples(tmp_path)
    assert "var_0" in df.columns


# ---------------------------------------------------------------------------
# extract_map_from_directory
# ---------------------------------------------------------------------------


def test_extract_map_selects_argmax_logposterior(tmp_path: Path) -> None:
    samples = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    loglike = [-10.0, -3.0, -8.0]
    logprior = [-1.0, -1.0, -1.0]
    _write_state(tmp_path, _minimal_state(samples, loglike, logprior))

    map_row = extract_map_from_directory(tmp_path)
    assert map_row["Yt"].iloc[0] == pytest.approx(3.0)
    assert map_row["kb"].iloc[0] == pytest.approx(4.0)


def test_extract_map_writes_csv(tmp_path: Path) -> None:
    samples = [[1.0, 2.0]]
    loglike = [-5.0]
    _write_state(tmp_path, _minimal_state(samples, loglike))

    csv_path = tmp_path / "output" / "map.csv"
    map_row = extract_map_from_directory(tmp_path, output_csv=str(csv_path))

    assert csv_path.exists()
    df_read = pd.read_csv(csv_path)
    assert df_read["Yt"].iloc[0] == pytest.approx(1.0)
