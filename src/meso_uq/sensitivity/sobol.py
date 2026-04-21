import numpy as np
import pandas as pd
import torch
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample


def _evaluate_model(model, X, xshift, xscale, yshift, yscale):
    X_std = (X - xshift) / xscale
    with torch.no_grad():
        X_tensor = torch.tensor(X_std, dtype=torch.float32)
        y_std = model(X_tensor).numpy()
    y = y_std * yscale[0] + yshift[0]
    return y.flatten()


def _evaluate_predictor_mean(
    predictor,
    X,
    *,
    predictive_mc_samples,
    predictive_mc_chunk_size,
):
    mean, _ = predictor.predict_mean_std(
        np.asarray(X, dtype=np.float32),
        predictive_mc_samples=int(predictive_mc_samples),
        predictive_mc_chunk_size=int(predictive_mc_chunk_size),
    )
    return np.asarray(mean, dtype=np.float64).flatten()


def run_sobol_over_axis(
    *,
    model=None,
    xshift=None,
    xscale=None,
    yshift=None,
    yscale=None,
    predictor=None,
    problem,
    fixed_axis_name,
    fixed_axis_values,
    evaluate_columns,
    n_samples=1024,
    calc_second_order=False,
    predictive_mc_samples=64,
    predictive_mc_chunk_size=8,
):
    if predictor is None:
        required = {
            "model": model,
            "xshift": xshift,
            "xscale": xscale,
            "yshift": yshift,
            "yscale": yscale,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                "run_sobol_over_axis requires either predictor=... or deterministic "
                f"model/scaling payload. Missing: {joined}"
            )

    rows = []
    sampled = sobol_sample.sample(problem, n_samples, calc_second_order=calc_second_order)
    param_names = problem["names"]
    for axis_value in fixed_axis_values:
        df_inputs = pd.DataFrame(sampled, columns=param_names)
        df_inputs[fixed_axis_name] = axis_value
        X = df_inputs[evaluate_columns].to_numpy(float)
        if predictor is None:
            Y = _evaluate_model(
                model,
                X,
                np.array(xshift),
                np.array(xscale),
                np.array(yshift),
                np.array(yscale),
            )
        else:
            Y = _evaluate_predictor_mean(
                predictor,
                X,
                predictive_mc_samples=predictive_mc_samples,
                predictive_mc_chunk_size=predictive_mc_chunk_size,
            )
        Si = sobol_analyze.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)
        for i, name in enumerate(param_names):
            rows.append({"axis": axis_value, "parameter": name, "index_type": "S1", "value": Si["S1"][i], "confidence": Si["S1_conf"][i]})
            rows.append({"axis": axis_value, "parameter": name, "index_type": "ST", "value": Si["ST"][i], "confidence": Si["ST_conf"][i]})
    return pd.DataFrame(rows)
