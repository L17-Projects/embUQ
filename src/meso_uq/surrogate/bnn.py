from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import torch

from .model import MLP

_BNN_ARTIFACT_FORMAT = "mesouq_bnn_v1"


def _resolve_torch_device(device: str) -> torch.device:
    normalized = device.strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    if normalized == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "Requested CUDA for BNN surrogate, but CUDA is unavailable. Falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        normalized = "cpu"
    return torch.device(normalized)


def _require_pyro() -> Any:
    try:
        import pyro  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "BNN backend requires optional dependency 'pyro-ppl'. "
            "Install with: pip install -e '.[bnn]'"
        ) from exc
    return pyro


def build_variational_components(
    *,
    input_dim: int,
    width: int,
    depth: int,
    prior_scale: float,
    obs_noise: float,
    device: torch.device,
) -> Tuple[Any, MLP, Callable[..., Any], Any]:
    pyro = _require_pyro()
    if input_dim < 1:
        raise ValueError("input_dim must be >= 1.")
    if width < 1:
        raise ValueError("width must be >= 1.")
    if depth < 1:
        raise ValueError("depth must be >= 1.")
    if prior_scale <= 0:
        raise ValueError("prior_scale must be > 0.")
    if obs_noise <= 0:
        raise ValueError("obs_noise must be > 0.")

    base_model = MLP(input_dims=input_dim, output_dims=1, hl_dims=[width] * depth).to(device)

    def model(x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        priors = {}
        for name, param in base_model.named_parameters():
            priors[name] = pyro.distributions.Normal(
                torch.zeros_like(param),
                prior_scale * torch.ones_like(param),
            ).to_event(param.dim())
        lifted = pyro.random_module("module", base_model, priors)()
        mean = lifted(x).squeeze(-1)
        with pyro.plate("data", x.shape[0]):
            pyro.sample(
                "obs",
                pyro.distributions.Normal(mean, obs_noise),
                obs=None if y is None else y.squeeze(-1),
            )
        return mean.unsqueeze(-1)

    guide = pyro.infer.autoguide.AutoDiagonalNormal(model)
    return pyro, base_model, model, guide


def make_artifact_payload(
    *,
    xshift: list[float],
    xscale: list[float],
    yshift: list[float],
    yscale: list[float],
    input_dim: int,
    width: int,
    depth: int,
    prior_scale: float,
    obs_noise: float,
    pyro_param_values: Dict[str, torch.Tensor],
    training_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    serialized_params: Dict[str, torch.Tensor] = {}
    for name, tensor in pyro_param_values.items():
        serialized_params[str(name)] = torch.as_tensor(tensor).detach().cpu()
    payload: Dict[str, Any] = {
        "format": _BNN_ARTIFACT_FORMAT,
        "input_dim": int(input_dim),
        "width": int(width),
        "depth": int(depth),
        "prior_scale": float(prior_scale),
        "obs_noise": float(obs_noise),
        "output_site": "_RETURN",
        "xshift": list(xshift),
        "xscale": list(xscale),
        "yshift": list(yshift),
        "yscale": list(yscale),
        "pyro_param_values": serialized_params,
    }
    if training_summary is not None:
        payload["training"] = training_summary
    return payload


def _coerce_draws(draws: Any, n_points: int, *, device: torch.device) -> torch.Tensor:
    draws_t = torch.as_tensor(draws, dtype=torch.float32, device=device)
    if draws_t.ndim == 3 and draws_t.shape[-1] == 1:
        draws_t = draws_t.squeeze(-1)
    if draws_t.ndim == 1:
        if n_points != 1:
            raise ValueError(f"Expected {n_points} predictive points, got scalar draws shape.")
        draws_t = draws_t.unsqueeze(1)
    if draws_t.ndim != 2:
        raise ValueError(f"Expected predictive draws with 2 dims [samples, points], got {draws_t.shape}.")
    if draws_t.shape[1] == n_points:
        return draws_t
    if draws_t.shape[0] == n_points:
        return draws_t.transpose(0, 1)
    raise ValueError(
        f"Could not align predictive draws shape {draws_t.shape} to expected point count {n_points}."
    )


class VariationalBNNPredictor:
    """Shared runtime for variational BNN surrogate prediction."""

    def __init__(self, artifact_path: str, *, device: str = "cpu") -> None:
        artifact = Path(artifact_path)
        if not artifact.exists():
            raise FileNotFoundError(f"BNN artifact not found: {artifact}")

        self.device = _resolve_torch_device(device)
        payload = torch.load(str(artifact), map_location=self.device)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected BNN artifact payload to be a dict, got {type(payload).__name__}.")

        self._predictive_fn = payload.get("predictive")
        self._model = payload.get("model")
        self._guide = payload.get("guide")
        self._output_site = str(payload.get("output_site", "_RETURN"))
        self._predictive_cls = None
        if str(payload.get("format", "")) == _BNN_ARTIFACT_FORMAT:
            self._predictive_fn = self._predictive_from_format_v1(payload)
        elif not callable(self._predictive_fn):
            if self._model is None or self._guide is None:
                raise KeyError(
                    "BNN artifact must provide either callable 'predictive', "
                    "both 'model' and 'guide', or format='mesouq_bnn_v1'."
                )
            pyro = _require_pyro()
            self._predictive_cls = pyro.infer.Predictive

        self._xshift_t = torch.as_tensor(payload["xshift"], dtype=torch.float32, device=self.device)
        self._xscale_t = torch.as_tensor(payload["xscale"], dtype=torch.float32, device=self.device)
        self._yshift_t = torch.as_tensor(payload["yshift"], dtype=torch.float32, device=self.device)
        self._yscale_t = torch.as_tensor(payload["yscale"], dtype=torch.float32, device=self.device)

    def _predictive_from_format_v1(self, payload: Dict[str, Any]) -> Callable[..., Any]:
        pyro, _, model, guide = build_variational_components(
            input_dim=int(payload["input_dim"]),
            width=int(payload["width"]),
            depth=int(payload["depth"]),
            prior_scale=float(payload["prior_scale"]),
            obs_noise=float(payload["obs_noise"]),
            device=self.device,
        )
        pyro.clear_param_store()
        with torch.inference_mode():
            guide(
                torch.zeros((1, int(payload["input_dim"])), dtype=torch.float32, device=self.device),
                torch.zeros((1, 1), dtype=torch.float32, device=self.device),
            )
        store = pyro.get_param_store()
        expected_names = set(store._params.keys())
        param_constraints = {
            name: store._constraints[name] for name in expected_names
        }
        loaded_names = set(payload["pyro_param_values"].keys())
        missing = sorted(expected_names - loaded_names)
        unexpected = sorted(loaded_names - expected_names)
        if missing:
            raise KeyError(
                "Missing Pyro parameters while loading BNN artifact: " + ", ".join(missing)
            )
        if unexpected:
            raise KeyError(
                "Unexpected Pyro parameters while loading BNN artifact: " + ", ".join(unexpected)
            )
        param_values = {
            name: torch.as_tensor(value, dtype=torch.float32, device=self.device).detach().clone()
            for name, value in payload["pyro_param_values"].items()
        }

        def _restore_param_store() -> None:
            pyro.clear_param_store()
            local_store = pyro.get_param_store()
            for name, value in param_values.items():
                unconstrained = value.clone().detach()
                unconstrained.requires_grad_(True)
                local_store._params[name] = unconstrained
                local_store._constraints[name] = param_constraints[name]
                local_store._param_to_name[unconstrained] = name

        _restore_param_store()
        output_site = str(payload.get("output_site", "_RETURN"))

        def _predictive(inputs: torch.Tensor, num_samples: int) -> Any:
            # Pyro parameter store is process-global. Restore this predictor's
            # parameters on each call so multiple diameter-specific models do not
            # overwrite each other.
            _restore_param_store()
            predictive = pyro.infer.Predictive(
                model,
                guide=guide,
                num_samples=num_samples,
                return_sites=(output_site,),
            )
            samples = predictive(inputs)
            if output_site not in samples:
                available = ", ".join(sorted(samples.keys()))
                raise KeyError(
                    f"Configured BNN output site '{output_site}' not found in predictive output. "
                    f"Available sites: {available}"
                )
            return samples[output_site]

        return _predictive

    def _run_predictive_chunk(self, inputs: torch.Tensor, num_samples: int) -> torch.Tensor:
        n_points = int(inputs.shape[0])

        if callable(self._predictive_fn):
            draws = self._call_predictive_fn(self._predictive_fn, inputs, num_samples)
            return _coerce_draws(draws, n_points, device=self.device)

        assert self._predictive_cls is not None
        predictive = self._predictive_cls(
            self._model,
            guide=self._guide,
            num_samples=num_samples,
            return_sites=(self._output_site,),
        )
        samples = predictive(inputs)
        if self._output_site not in samples:
            available = ", ".join(sorted(samples.keys()))
            raise KeyError(
                f"Configured BNN output site '{self._output_site}' not found in predictive output. "
                f"Available sites: {available}"
            )
        return _coerce_draws(samples[self._output_site], n_points, device=self.device)

    @staticmethod
    def _call_predictive_fn(
        fn: Callable[..., Any], inputs: torch.Tensor, num_samples: int
    ) -> Any:
        try:
            return fn(inputs, num_samples=num_samples)
        except TypeError:
            return fn(inputs, num_samples)

    def predict_mean_std(
        self,
        inputs: np.ndarray,
        *,
        predictive_mc_samples: int,
        predictive_mc_chunk_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if predictive_mc_samples < 1:
            raise ValueError("predictive_mc_samples must be >= 1.")
        if predictive_mc_chunk_size < 1:
            raise ValueError("predictive_mc_chunk_size must be >= 1.")

        x = torch.as_tensor(inputs, dtype=torch.float32, device=self.device)
        if x.ndim != 2:
            raise ValueError(f"Expected surrogate inputs with shape [points, features], got {x.shape}.")
        x_norm = (x - self._xshift_t) / self._xscale_t

        chunks: list[torch.Tensor] = []
        remaining = predictive_mc_samples
        with torch.inference_mode():
            while remaining > 0:
                n_chunk = min(predictive_mc_chunk_size, remaining)
                chunk = self._run_predictive_chunk(x_norm, n_chunk)
                chunks.append(chunk)
                remaining -= n_chunk

        draws = torch.cat(chunks, dim=0)
        draws = draws * self._yscale_t + self._yshift_t
        mean = draws.mean(dim=0)
        std = draws.std(dim=0, unbiased=False)
        return mean.detach().cpu().numpy(), std.detach().cpu().numpy()
