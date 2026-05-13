import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _find_latest_state(run_dir):
    run_dir = Path(run_dir)
    candidates = [run_dir / "latest", run_dir / "genLatest.json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    json_candidates = sorted(run_dir.glob("*.json"))
    return json_candidates[-1] if json_candidates else None


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_korali_state(run_dir):
    state_path = _find_latest_state(run_dir)
    if state_path is None:
        raise FileNotFoundError(f"No Korali state JSON found under {run_dir}")
    return state_path, _load_json(state_path)


def _variable_names_from_state(state):
    var_names = []
    for var in state.get("Variables", []):
        name = var.get("Name", f"var_{len(var_names)}")
        if name == "[Sigma]":
            name = "sigma"
        var_names.append(name)
    return var_names


def _samples_to_dataframe(sample_db, state):
    if sample_db is None:
        raise ValueError("Expected sample database to be present in Korali state.")
    if not sample_db:
        return pd.DataFrame()

    first = sample_db[0]
    if isinstance(first, (list, tuple)):
        matrix = sample_db
        sample_width = len(first)
    else:
        matrix = [[entry] for entry in sample_db]
        sample_width = 1

    var_names = _variable_names_from_state(state)
    while len(var_names) < sample_width:
        var_names.append(f"var_{len(var_names)}")
    return pd.DataFrame(matrix, columns=var_names[:sample_width])


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def load_posterior_samples(run_dir):
    state_path, state = load_korali_state(run_dir)
    results = state.get("Results", {})
    sample_db = results.get("Posterior Sample Database")
    loglike_db = results.get("Posterior Sample LogLikelihood Database")
    logprior_db = results.get("Posterior Sample LogPrior Database")
    if sample_db is None or loglike_db is None:
        raise ValueError(f"State at {state_path} does not contain posterior sample/loglike databases")
    df = _samples_to_dataframe(sample_db, state)
    df["logLikelihood"] = loglike_db
    if logprior_db is not None:
        df["logPrior"] = logprior_db
        df["logPosterior"] = df["logLikelihood"] + df["logPrior"]
    else:
        df["logPosterior"] = df["logLikelihood"]
    return df


def load_chain_leader_samples(run_dir):
    state_path, state = load_korali_state(run_dir)
    solver = state.get("Solver", {})
    results = state.get("Results", {})

    chain_db = _first_present(
        solver,
        ("Chain Leaders", "Chain Leaders Database"),
    )
    if chain_db is None:
        chain_db = _first_present(
            results,
            ("Chain Leaders", "Chain Leaders Database"),
        )
    if chain_db is None:
        raise ValueError(f"State at {state_path} does not contain chain leader databases")

    loglike_db = _first_present(
        solver,
        ("Chain Leaders LogLikelihoods", "Chain Leaders LogLikelihood Database"),
    )
    if loglike_db is None:
        loglike_db = _first_present(
            results,
            ("Chain Leaders LogLikelihoods", "Chain Leaders LogLikelihood Database"),
        )
    logprior_db = _first_present(
        solver,
        ("Chain Leaders LogPriors", "Chain Leaders LogPrior Database"),
    )
    if logprior_db is None:
        logprior_db = _first_present(
            results,
            ("Chain Leaders LogPriors", "Chain Leaders LogPrior Database"),
        )

    df = _samples_to_dataframe(chain_db, state)
    if loglike_db is not None:
        df["logLikelihood"] = loglike_db
    if logprior_db is not None:
        df["logPrior"] = logprior_db
    if "logLikelihood" in df.columns and "logPrior" in df.columns:
        df["logPosterior"] = df["logLikelihood"] + df["logPrior"]
    elif "logLikelihood" in df.columns:
        df["logPosterior"] = df["logLikelihood"]
    return df


def extract_map_from_directory(run_dir, output_csv=None):
    df = load_posterior_samples(run_dir)
    idx = int(np.nanargmax(df["logPosterior"].to_numpy()))
    map_row = df.iloc[[idx]].copy()
    if output_csv is not None:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        map_row.to_csv(output_csv, index=False)
    return map_row
