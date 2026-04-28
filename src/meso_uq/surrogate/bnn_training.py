from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from .bnn import build_variational_components, make_artifact_payload, _resolve_torch_device
from .cli import make_tensors, seed_training_runtime, split_row_indices
from .model import load_model_states


def _resolve_obs_noise_prior_scale(
    *, obs_noise_prior_scale: float, obs_noise: float | None
) -> float:
    resolved = float(obs_noise_prior_scale)
    if obs_noise is not None:
        legacy = float(obs_noise)
        if legacy <= 0:
            raise ValueError("obs_noise must be > 0.")
        if not np.isclose(resolved, 1.0) and not np.isclose(resolved, legacy):
            raise ValueError(
                "Received conflicting values for obs_noise_prior_scale and obs_noise. "
                "Use one, or provide the same value for both."
            )
        resolved = legacy
    if resolved <= 0:
        raise ValueError("obs_noise_prior_scale must be > 0.")
    return resolved


def _split_like_dnn(
    Xz: torch.Tensor,
    yz: torch.Tensor,
    X_phys: np.ndarray,
    y_phys: np.ndarray,
    *,
    seed: int | None,
) -> Dict[str, Any]:
    n = int(len(Xz))
    if n < 2:
        raise ValueError("Need at least 2 samples to create train/validation split.")
    perm, n_val = split_row_indices(n, val_fraction=0.10, seed=seed)

    Xz_perm = Xz[perm]
    yz_perm = yz[perm]
    idx = perm.detach().cpu().numpy()
    X_phys_perm = X_phys[idx]
    y_phys_perm = y_phys[idx]

    return {
        "X_train": Xz_perm[n_val:],
        "y_train": yz_perm[n_val:],
        "X_val": Xz_perm[:n_val],
        "y_val": yz_perm[:n_val],
        "X_val_phys": X_phys_perm[:n_val],
        "y_val_phys": y_phys_perm[:n_val],
        "n_val": int(n_val),
        "n_train": int(n - n_val),
    }


def _predictive_mean_std(
    pyro: Any,
    model: Any,
    guide: Any,
    x: torch.Tensor,
    *,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictive = pyro.infer.Predictive(model, guide=guide, num_samples=int(num_samples), return_sites=("_RETURN",))
    samples = predictive(x)["_RETURN"]
    if samples.ndim == 3 and samples.shape[-1] == 1:
        samples = samples.squeeze(-1)
    if samples.ndim != 2:
        raise ValueError(f"Unexpected predictive sample shape: {tuple(samples.shape)}")
    mean = samples.mean(dim=0)
    std = samples.std(dim=0, unbiased=False)
    return mean.detach().cpu().numpy(), std.detach().cpu().numpy()


def _snapshot_pyro_params(pyro: Any) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for name, param in pyro.get_param_store().named_parameters():
        snap[str(name)] = param.detach().cpu().clone()
    return snap


def _snapshot_module_state(module: Any) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for name, tensor in module.state_dict().items():
        snap[str(name)] = torch.as_tensor(tensor).detach().cpu().clone()
    return snap


def _restore_pyro_params(pyro: Any, snapshot: Dict[str, torch.Tensor], device: torch.device) -> None:
    store = pyro.get_param_store()
    for name, value in snapshot.items():
        if name not in store._params:
            raise KeyError(f"Missing Pyro parameter '{name}' while restoring checkpoint.")
        store._params[name] = torch.nn.Parameter(
            torch.as_tensor(value, dtype=torch.float32, device=device)
        )


def _evaluate_dnn_rmse(
    *,
    dnn_model_path: str,
    X_val_phys: np.ndarray,
    y_val_phys: np.ndarray,
    device: torch.device,
) -> float:
    model, xshift, xscale, yshift, yscale = load_model_states(dnn_model_path)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    xshift_arr = np.asarray(xshift, dtype=np.float64)
    xscale_arr = np.asarray(xscale, dtype=np.float64)
    yshift_arr = np.asarray(yshift, dtype=np.float64)
    yscale_arr = np.asarray(yscale, dtype=np.float64)
    Xz = (X_val_phys - xshift_arr) / xscale_arr
    Xz_t = torch.as_tensor(Xz, dtype=torch.float32, device=device)
    with torch.inference_mode():
        pred_norm = model(Xz_t).detach().cpu().numpy().reshape(-1)
    pred = pred_norm * yscale_arr[0] + yshift_arr[0]
    pred = np.maximum(0.0, pred)
    return float(np.sqrt(np.mean(np.square(pred - y_val_phys.reshape(-1)))))


def train_tabular_bnn_surrogate(
    df,
    *,
    input_cols: Sequence[str],
    target_col: str,
    out_path: str,
    dnn_reference_path: str,
    width: int = 64,
    depth: int = 3,
    prior_scale: float = 1.0,
    obs_noise_prior_scale: float = 1.0,
    obs_noise: float | None = None,
    batch_size: int = 512,
    lr: float = 1e-3,
    max_steps: int = 2500,
    eval_every: int = 25,
    predictive_mc_samples: int = 64,
    max_walltime_seconds: int = 1200,
    seed: int | None = None,
    parity_tol: float = 1.20,
    require_parity: bool = True,
    device: str = "cpu",
    report_path: str | None = None,
) -> Dict[str, Any]:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1.")
    if eval_every < 1:
        raise ValueError("eval_every must be >= 1.")
    if predictive_mc_samples < 1:
        raise ValueError("predictive_mc_samples must be >= 1.")
    if max_walltime_seconds < 1:
        raise ValueError("max_walltime_seconds must be >= 1.")
    if parity_tol <= 0:
        raise ValueError("parity_tol must be > 0.")
    resolved_obs_noise_prior_scale = _resolve_obs_noise_prior_scale(
        obs_noise_prior_scale=float(obs_noise_prior_scale),
        obs_noise=obs_noise,
    )
    seed_training_runtime(seed)

    Xz, yz, x_mu, x_sd, y_mu, y_sd = make_tensors(df, list(input_cols), target_col)
    X_phys = df[list(input_cols)].to_numpy(float)
    y_phys = df[[target_col]].to_numpy(float).reshape(-1)
    split = _split_like_dnn(Xz, yz, X_phys, y_phys, seed=seed)

    X_train = split["X_train"]
    y_train = split["y_train"]
    X_val = split["X_val"]
    X_val_phys = split["X_val_phys"]
    y_val_phys = split["y_val_phys"]

    device_t = _resolve_torch_device(device)
    X_train = X_train.to(device_t)
    y_train = y_train.to(device_t)
    X_val = X_val.to(device_t)

    pyro, base_model, model, guide = build_variational_components(
        input_dim=len(input_cols),
        width=int(width),
        depth=int(depth),
        prior_scale=float(prior_scale),
        obs_noise_prior_scale=resolved_obs_noise_prior_scale,
        device=device_t,
    )
    if seed is not None:
        pyro.set_rng_seed(int(seed))
    pyro.clear_param_store()
    svi = pyro.infer.SVI(
        model,
        guide,
        pyro.optim.Adam({"lr": float(lr)}),
        loss=pyro.infer.Trace_ELBO(),
    )

    best_state = _snapshot_pyro_params(pyro)
    best_guide_state = _snapshot_module_state(guide)
    best_val_rmse = float("inf")
    best_step = 0
    train_losses: list[float] = []
    val_rmse_history: list[float] = []
    start = time.monotonic()
    step = 0
    stop_reason = "max_steps"

    while step < int(max_steps):
        train_loss = float(svi.step(X_train, y_train))
        step += 1
        train_losses.append(train_loss / max(1, len(X_train)))

        if step == 1 or step % int(eval_every) == 0:
            mean_norm, _ = _predictive_mean_std(
                pyro, model, guide, X_val, num_samples=int(predictive_mc_samples)
            )
            mean_phys = np.maximum(0.0, mean_norm * y_sd[0] + y_mu[0])
            val_rmse = float(np.sqrt(np.mean(np.square(mean_phys - y_val_phys))))
            val_rmse_history.append(val_rmse)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_step = step
                best_state = _snapshot_pyro_params(pyro)
                best_guide_state = _snapshot_module_state(guide)

        elapsed = time.monotonic() - start
        if elapsed >= int(max_walltime_seconds):
            stop_reason = "walltime"
            break

    _restore_pyro_params(pyro, best_state, device_t)
    guide.load_state_dict(best_guide_state, strict=True)

    mean_norm, std_norm = _predictive_mean_std(
        pyro,
        model,
        guide,
        X_val,
        num_samples=int(predictive_mc_samples),
    )
    mean_phys = np.maximum(0.0, mean_norm * y_sd[0] + y_mu[0])
    std_phys = np.maximum(0.0, std_norm * y_sd[0])
    bnn_rmse = float(np.sqrt(np.mean(np.square(mean_phys - y_val_phys))))

    dnn_rmse = _evaluate_dnn_rmse(
        dnn_model_path=dnn_reference_path,
        X_val_phys=X_val_phys,
        y_val_phys=y_val_phys,
        device=device_t,
    )
    parity_ratio = float(bnn_rmse / dnn_rmse) if dnn_rmse > 0 else float("inf")
    parity_passed = bool(parity_ratio <= float(parity_tol))
    if require_parity and not parity_passed:
        raise RuntimeError(
            "BNN parity check failed: "
            f"bnn_rmse={bnn_rmse:.6g}, dnn_rmse={dnn_rmse:.6g}, ratio={parity_ratio:.6g}, "
            f"tol={float(parity_tol):.6g}"
        )

    training_summary = {
        "step_count": int(step),
        "best_step": int(best_step),
        "best_val_rmse": float(best_val_rmse),
        "final_val_rmse": float(bnn_rmse),
        "elapsed_seconds": float(time.monotonic() - start),
        "stop_reason": stop_reason,
        "train_loss_last": float(train_losses[-1]) if train_losses else None,
        "parity_dnn_rmse": float(dnn_rmse),
        "parity_bnn_rmse": float(bnn_rmse),
        "parity_ratio": float(parity_ratio),
        "parity_tol": float(parity_tol),
        "parity_passed": bool(parity_passed),
    }

    payload = make_artifact_payload(
        xshift=x_mu,
        xscale=x_sd,
        yshift=y_mu,
        yscale=y_sd,
        input_dim=len(input_cols),
        width=int(width),
        depth=int(depth),
        prior_scale=float(prior_scale),
        obs_noise_prior_scale=resolved_obs_noise_prior_scale,
        pyro_param_values=best_state,
        base_model_state_dict={
            name: tensor.detach().cpu().clone()
            for name, tensor in base_model.state_dict().items()
        },
        guide_state_dict=best_guide_state,
        training_summary=training_summary,
    )

    out_file = Path(out_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(out_file))

    reload_result: dict[str, float | bool] | None = None
    try:
        from .bnn import VariationalBNNPredictor

        predictor = VariationalBNNPredictor(str(out_file), device=device)
        reload_mean, reload_std = predictor.predict_mean_std(
            np.asarray(X_val_phys, dtype=np.float32),
            predictive_mc_samples=int(predictive_mc_samples),
            predictive_mc_chunk_size=max(1, min(int(predictive_mc_samples), 8)),
        )
        reload_mean = np.maximum(0.0, np.asarray(reload_mean, dtype=np.float64))
        reload_std = np.maximum(0.0, np.asarray(reload_std, dtype=np.float64))
        reload_rmse = float(np.sqrt(np.mean(np.square(reload_mean - y_val_phys))))
        reload_degradation_abs = float(reload_rmse - bnn_rmse)
        reload_degradation_rel = (
            float(reload_degradation_abs / bnn_rmse) if bnn_rmse > 0 else float("inf")
        )
        reload_result = {
            "val_rmse": float(reload_rmse),
            "degradation_abs": float(reload_degradation_abs),
            "degradation_rel": float(reload_degradation_rel),
            "pred_std_mean": float(np.mean(reload_std)),
            "passed_rel_tol_0p05": bool(np.isfinite(reload_degradation_rel) and reload_degradation_rel <= 0.05),
        }
    except Exception as exc:  # pragma: no cover - exercised in integration usage
        reload_result = {
            "val_rmse": float("nan"),
            "degradation_abs": float("nan"),
            "degradation_rel": float("nan"),
            "pred_std_mean": float("nan"),
            "passed_rel_tol_0p05": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    result = {
        "out": str(out_file),
        "training": training_summary,
        "validation": {
            "n_train": split["n_train"],
            "n_val": split["n_val"],
            "val_target_mean": float(np.mean(y_val_phys)),
            "val_predictive_std_mean": float(np.mean(std_phys)),
        },
        "reload": reload_result,
    }
    if report_path is not None:
        report_file = Path(report_path).resolve()
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    loss_hist_path = out_file.with_name(f"{out_file.stem}_loss_hist.json")
    loss_hist = {
        "train_loss": train_losses,
        "val_rmse": val_rmse_history,
    }
    loss_hist_path.write_text(json.dumps(loss_hist), encoding="utf-8")
    return result
