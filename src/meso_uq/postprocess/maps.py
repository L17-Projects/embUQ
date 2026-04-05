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


def load_posterior_samples(run_dir):
    state_path = _find_latest_state(run_dir)
    if state_path is None:
        raise FileNotFoundError(f"No Korali state JSON found under {run_dir}")
    state = _load_json(state_path)
    results = state.get("Results", {})
    sample_db = results.get("Posterior Sample Database")
    loglike_db = results.get("Posterior Sample LogLikelihood Database")
    logprior_db = results.get("Posterior Sample LogPrior Database")
    if sample_db is None or loglike_db is None:
        raise ValueError(f"State at {state_path} does not contain posterior sample/loglike databases")
    var_names = []
    for var in state.get("Variables", []):
        name = var.get("Name", f"var_{len(var_names)}")
        if name == "[Sigma]":
            name = "sigma"
        var_names.append(name)
    df = pd.DataFrame(sample_db, columns=var_names[: len(sample_db[0])])
    df["logLikelihood"] = loglike_db
    if logprior_db is not None:
        df["logPrior"] = logprior_db
        df["logPosterior"] = df["logLikelihood"] + df["logPrior"]
    else:
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
