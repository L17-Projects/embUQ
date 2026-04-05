import numpy as np
import pandas as pd
import torch
from SALib.analyze import sobol
from SALib.sample import saltelli


def _evaluate_model(model, X, xshift, xscale, yshift, yscale):
    X_std = (X - xshift) / xscale
    with torch.no_grad():
        X_tensor = torch.tensor(X_std, dtype=torch.float32)
        y_std = model(X_tensor).numpy()
    y = y_std * yscale[0] + yshift[0]
    return y.flatten()


def run_sobol_over_axis(*, model, xshift, xscale, yshift, yscale, problem, fixed_axis_name, fixed_axis_values, evaluate_columns, n_samples=1024, calc_second_order=False):
    rows = []
    sampled = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)
    param_names = problem["names"]
    for axis_value in fixed_axis_values:
        df_inputs = pd.DataFrame(sampled, columns=param_names)
        df_inputs[fixed_axis_name] = axis_value
        X = df_inputs[evaluate_columns].to_numpy(float)
        Y = _evaluate_model(model, X, np.array(xshift), np.array(xscale), np.array(yshift), np.array(yscale))
        Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)
        for i, name in enumerate(param_names):
            rows.append({"axis": axis_value, "parameter": name, "index_type": "S1", "value": Si["S1"][i], "confidence": Si["S1_conf"][i]})
            rows.append({"axis": axis_value, "parameter": name, "index_type": "ST", "value": Si["ST"][i], "confidence": Si["ST_conf"][i]})
    return pd.DataFrame(rows)
