#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


DEFAULT_MODE_COUNT = 30
DEFAULT_PAPER_MIN_FREQUENCY = 22.5
SUPPORTED_POLICIES = ("frequency-min", "raw-head", "explicit-indices")


def _parse_indices(raw_value: str) -> tuple[int, ...]:
    text = str(raw_value).strip()
    if not text:
        raise ValueError("Explicit eigenmode indices must not be empty.")
    indices = tuple(int(item.strip()) for item in text.replace(";", ",").split(",") if item.strip())
    if not indices:
        raise ValueError("Explicit eigenmode indices must contain at least one index.")
    if any(index < 0 for index in indices):
        raise ValueError("Explicit eigenmode indices must be non-negative.")
    if len(set(indices)) != len(indices):
        raise ValueError("Explicit eigenmode indices must not contain duplicates.")
    return indices


def _read_kbt(parameter_path: Path) -> float:
    if not parameter_path.is_file():
        return 1.0
    payload = yaml.load(parameter_path.read_bytes(), Loader=yaml.CLoader)
    if not isinstance(payload, dict):
        return 1.0
    return float(payload.get("kbt", 1.0))


def _select_indices(
    eigenvalues: np.ndarray,
    *,
    kbt: float,
    mode_count: int,
    policy: str,
    min_frequency: float,
    explicit_indices: Sequence[int],
) -> tuple[np.ndarray, dict[str, object]]:
    if mode_count <= 0:
        raise ValueError("--mode-count must be positive.")
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported eigenmode window policy {policy!r}; expected one of {SUPPORTED_POLICIES}.")
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Eigenmode eigenvalues must all be positive before trimming.")

    frequencies = np.sqrt(kbt / eigenvalues)
    if policy == "raw-head":
        selected = np.arange(min(mode_count, eigenvalues.size), dtype=int)
        details = {
            "policy_rationale": "Legacy/debug policy: take the first raw covariance modes without a paper-mode filter.",
            "rejected_low_frequency_count": 0,
        }
    elif policy == "frequency-min":
        eligible = np.flatnonzero(frequencies >= float(min_frequency))
        if eligible.size < mode_count:
            raise ValueError(
                "Not enough eigenmodes satisfy the paper frequency window: "
                f"need {mode_count}, found {eligible.size} with omega >= {min_frequency}."
            )
        selected = eligible[np.argsort(frequencies[eligible], kind="stable")][:mode_count]
        details = {
            "policy_rationale": (
                "Paper Figure 8 reports oscillatory shell modes. Modes below the configured "
                "frequency floor are retained in raw outputs but excluded from the paper window "
                "as slow global/large-scale fluctuation modes."
            ),
            "min_frequency_tau_inv": float(min_frequency),
            "rejected_low_frequency_count": int(np.count_nonzero(frequencies < float(min_frequency))),
        }
    else:
        selected = np.asarray(tuple(explicit_indices), dtype=int)
        if selected.size < mode_count:
            raise ValueError(
                f"Explicit eigenmode window supplies {selected.size} modes, but {mode_count} are required."
            )
        if np.max(selected) >= eigenvalues.size:
            raise ValueError(
                f"Explicit eigenmode index {int(np.max(selected))} exceeds raw mode count {eigenvalues.size}."
            )
        selected = selected[:mode_count]
        details = {
            "policy_rationale": "Operator-supplied raw mode indices; intended for forensic replay only.",
            "explicit_indices": selected.astype(int).tolist(),
            "rejected_low_frequency_count": int(np.min(selected)) if selected.size else 0,
        }

    return selected, details


def build_manifest(
    *,
    eigenvalues: np.ndarray,
    kbt: float,
    selected_indices: np.ndarray,
    policy: str,
    mode_count: int,
    details: dict[str, object],
) -> dict[str, object]:
    selected_eigenvalues = eigenvalues[selected_indices]
    selected_frequencies = np.sqrt(np.sort(kbt / selected_eigenvalues)[: selected_eigenvalues.size])
    raw_frequencies = np.sqrt(kbt / eigenvalues)
    return {
        "schema": "mesouq.gv.eigenmodes.mode_window.v1",
        "raw_eigenpair_count": int(eigenvalues.size),
        "final_mode_count": int(selected_indices.size),
        "requested_mode_count": int(mode_count),
        "mode_window_policy": policy,
        "selected_raw_mode_indices": selected_indices.astype(int).tolist(),
        "final_mode_indices": list(range(int(selected_indices.size))),
        "selected_paper_mode_indices": list(range(int(selected_indices.size))),
        "raw_frequency_tau_inv_minmax": [float(np.min(raw_frequencies)), float(np.max(raw_frequencies))],
        "final_frequency_tau_inv_minmax": [
            float(np.min(selected_frequencies)),
            float(np.max(selected_frequencies)),
        ],
        **details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trim raw GV eigenpairs to the configured paper-mode window.")
    parser.add_argument("--mode-count", type=int, default=int(os.environ.get("MESOUQ_GV_EIGENMODES_MODE_COUNT", DEFAULT_MODE_COUNT)))
    parser.add_argument(
        "--policy",
        default=os.environ.get("MESOUQ_GV_EIGENMODES_MODE_WINDOW_POLICY", "frequency-min"),
        choices=SUPPORTED_POLICIES,
    )
    parser.add_argument(
        "--min-frequency",
        type=float,
        default=float(os.environ.get("MESOUQ_GV_EIGENMODES_MODE_MIN_FREQUENCY", DEFAULT_PAPER_MIN_FREQUENCY)),
    )
    parser.add_argument(
        "--explicit-indices",
        default=os.environ.get("MESOUQ_GV_EIGENMODES_MODE_INDICES", ""),
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--parameter-file", default="../parameter/parameters00001.yaml")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    eigenvalue_path = output_dir / "eigvalues.txt"
    eigenvector_path = output_dir / "eigvectors.txt"
    eigenvalues = np.atleast_1d(np.loadtxt(eigenvalue_path, dtype=float))
    if eigenvalues.size == 0:
        raise ValueError(f"No raw eigenvalues found in {eigenvalue_path}.")
    kbt = _read_kbt(Path(args.parameter_file))
    explicit_indices = _parse_indices(args.explicit_indices) if args.explicit_indices else ()
    selected_indices, details = _select_indices(
        eigenvalues,
        kbt=kbt,
        mode_count=int(args.mode_count),
        policy=str(args.policy),
        min_frequency=float(args.min_frequency),
        explicit_indices=explicit_indices,
    )

    np.savetxt(output_dir / "eigvalues_new.txt", eigenvalues[selected_indices])
    if eigenvector_path.is_file():
        eigenvectors = np.atleast_2d(np.loadtxt(eigenvector_path, dtype=float))
        if eigenvectors.shape[0] < eigenvalues.size:
            raise ValueError(
                f"Raw eigenvector count {eigenvectors.shape[0]} is smaller than eigenvalue count {eigenvalues.size}."
            )
        np.savetxt(output_dir / "eigvectors_new.txt", eigenvectors[selected_indices])

    manifest = build_manifest(
        eigenvalues=eigenvalues,
        kbt=kbt,
        selected_indices=selected_indices,
        policy=str(args.policy),
        mode_count=int(args.mode_count),
        details=details,
    )
    (output_dir / "mode_window_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "Selected "
        f"{manifest['final_mode_count']} eigenmodes from {manifest['raw_eigenpair_count']} raw eigenpairs "
        f"using policy {manifest['mode_window_policy']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
