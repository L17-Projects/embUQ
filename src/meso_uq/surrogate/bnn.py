from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


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

        if not callable(self._predictive_fn):
            if self._model is None or self._guide is None:
                raise KeyError(
                    "BNN artifact must provide either callable 'predictive' or both 'model' and 'guide'."
                )
            pyro = _require_pyro()
            self._predictive_cls = pyro.infer.Predictive
        else:
            self._predictive_cls = None

        self._xshift_t = torch.as_tensor(payload["xshift"], dtype=torch.float32, device=self.device)
        self._xscale_t = torch.as_tensor(payload["xscale"], dtype=torch.float32, device=self.device)
        self._yshift_t = torch.as_tensor(payload["yshift"], dtype=torch.float32, device=self.device)
        self._yscale_t = torch.as_tensor(payload["yscale"], dtype=torch.float32, device=self.device)

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
