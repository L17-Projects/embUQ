#!/usr/bin/env python3
"""Replay and validate the frozen UQ_EMB acoustic polynomial surrogates.

The accepted family predicts squared resonance frequency with one free-intercept
quadratic per exact diameter.  It is fitted to the highest 28 of 32 physical
resonance labels with unweighted residuals in frequency space.  This script is
deliberately bounded: it neither interpolates diameters nor evaluates outside
the frozen ``0 <= k_a <= k_a,max`` support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import LinearConstraint, minimize


sys.dont_write_bytecode = True

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))

from replay_provenance import replay_receipt_provenance  # noqa: E402


SCHEMA_VERSION = "mesouq.uq_emb.acoustic_polynomial_replay.v1"
FIT_POINT_COUNT = 28
TOTAL_POINT_COUNT = 32
HELD_LOW_KA_POINT_COUNT = TOTAL_POINT_COUNT - FIT_POINT_COUNT
MINIMUM_SQUARED_FREQUENCY_MHZ2 = 1.0e-12
RESPONSE = "frequency_mhz_squared"
FORMULA = "sqrt(a0_mhz2 + a1_mhz2_per_dpd*ka + a2_mhz2_per_dpd2*ka^2)"
LABEL_FREQUENCY_COLUMN = "mass_corrected_driven_peak_frequency_mhz"


class ReplayMismatchError(RuntimeError):
    """Raised when recomputed values do not satisfy the frozen tolerance."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parse_float(row: Mapping[str, str], key: str, context: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context}: missing or invalid {key!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context}: {key!r} must be finite")
    return value


def load_labels(path: Path) -> list[dict[str, Any]]:
    """Load the accepted physical labels needed by the bounded fit."""

    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    required = {"agent", "diameter_um", "ka_dpd", LABEL_FREQUENCY_COLUMN}
    if not raw_rows:
        raise ValueError(f"Physical label table is empty: {path}")
    missing = required.difference(raw_rows[0])
    if missing:
        raise ValueError(f"Physical label table is missing columns: {sorted(missing)}")

    labels: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows, start=2):
        context = f"{path}:{index}"
        agent = str(row.get("agent") or "").strip().lower()
        if not agent:
            raise ValueError(f"{context}: missing agent")
        labels.append(
            {
                "agent": agent,
                "diameter_um": _parse_float(row, "diameter_um", context),
                "ka_dpd": _parse_float(row, "ka_dpd", context),
                "frequency_mhz": _parse_float(row, LABEL_FREQUENCY_COLUMN, context),
            }
        )
    return labels


def squared_design_matrix(scaled_ka: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones_like(scaled_ka), scaled_ka, scaled_ka**2))


def squared_constraints() -> LinearConstraint:
    """Require q(0)>0, q'(0)>=0, and q'(1)>=0 for q(u)=a+b*u+c*u^2."""

    matrix = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 2.0]],
        dtype=np.float64,
    )
    lower = np.asarray(
        [MINIMUM_SQUARED_FREQUENCY_MHZ2, 0.0, 0.0], dtype=np.float64
    )
    return LinearConstraint(matrix, lower, np.full(lower.shape, np.inf))


def feasible_squared_seed(scaled_ka: np.ndarray, frequency_mhz: np.ndarray) -> np.ndarray:
    matrix = squared_design_matrix(scaled_ka)
    initial = np.linalg.lstsq(matrix, np.square(frequency_mhz), rcond=None)[0]
    constraint = squared_constraints()
    if np.all(constraint.A @ initial >= constraint.lb - 1.0e-12):
        return np.asarray(initial, dtype=np.float64)
    intercept = max(float(np.min(np.square(frequency_mhz))) * 0.5, 1.0e-8)
    slope = max(
        float((np.max(np.square(frequency_mhz)) - intercept) / np.max(scaled_ka)),
        1.0e-8,
    )
    return np.asarray([intercept, slope, 0.0], dtype=np.float64)


def fit_squared_frequency(
    scaled_ka: np.ndarray, frequency_mhz: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Fit the accepted free-intercept quadratic with frequency-space residuals."""

    initial = feasible_squared_seed(scaled_ka, frequency_mhz)
    matrix = squared_design_matrix(scaled_ka)
    constraint = squared_constraints()

    def objective(coefficients: np.ndarray) -> float:
        squared_prediction = matrix @ coefficients
        if np.any(squared_prediction <= 0.0):
            return 1.0e30
        residual = np.sqrt(squared_prediction) - frequency_mhz
        return float(residual @ residual)

    def gradient(coefficients: np.ndarray) -> np.ndarray:
        squared_prediction = matrix @ coefficients
        if np.any(squared_prediction <= 0.0):
            return np.zeros_like(coefficients)
        prediction = np.sqrt(squared_prediction)
        residual = prediction - frequency_mhz
        return matrix.T @ (residual / prediction)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        constraints=[constraint],
        method="SLSQP",
        options={"ftol": 1.0e-14, "maxiter": 4000},
    )
    if not result.success:
        raise RuntimeError(f"Constrained squared-frequency fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64), not np.allclose(
        result.x, initial, rtol=1.0e-9, atol=1.0e-11
    )


def predict_frequency_mhz(raw_coefficients: np.ndarray, ka_dpd: np.ndarray) -> np.ndarray:
    ka = np.asarray(ka_dpd, dtype=np.float64)
    squared = np.polynomial.polynomial.polyval(ka, raw_coefficients)
    if np.any(~np.isfinite(squared)) or np.any(squared < -1.0e-10):
        raise ValueError("Squared-frequency surrogate returned an invalid prediction")
    return np.sqrt(np.maximum(squared, 0.0))


def leave_one_out_rmse(scaled_ka: np.ndarray, frequency_mhz: np.ndarray) -> float:
    residuals: list[float] = []
    for index in range(scaled_ka.size):
        coefficients, _ = fit_squared_frequency(
            np.delete(scaled_ka, index), np.delete(frequency_mhz, index)
        )
        squared_prediction = squared_design_matrix(
            np.asarray([scaled_ka[index]])
        ) @ coefficients
        prediction = float(np.sqrt(np.maximum(0.0, squared_prediction))[0])
        residuals.append(prediction - float(frequency_mhz[index]))
    return float(np.sqrt(np.mean(np.square(residuals))))


def _group_labels(labels: Iterable[Mapping[str, Any]]) -> dict[tuple[str, float], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        grouped[(str(row["agent"]), float(row["diameter_um"]))].append(dict(row))
    for key, rows in grouped.items():
        rows.sort(key=lambda row: float(row["ka_dpd"]))
        ka = np.asarray([float(row["ka_dpd"]) for row in rows], dtype=np.float64)
        if len(rows) != TOTAL_POINT_COUNT:
            raise ValueError(
                f"{key[0]} {key[1]:g} um has {len(rows)} labels; expected {TOTAL_POINT_COUNT}"
            )
        if np.any(np.diff(ka) <= 0.0):
            raise ValueError(f"{key[0]} {key[1]:g} um has non-increasing ka labels")
    return grouped


def fit_replay(labels: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Fit every exact-diameter polynomial and retain policy/provenance-ready metadata."""

    rows: list[dict[str, Any]] = []
    for (agent, diameter), group in sorted(_group_labels(labels).items()):
        ka = np.asarray([float(row["ka_dpd"]) for row in group], dtype=np.float64)
        frequency = np.asarray(
            [float(row["frequency_mhz"]) for row in group], dtype=np.float64
        )
        upper = float(ka[-1])
        fit_ka = ka[-FIT_POINT_COUNT:]
        fit_frequency = frequency[-FIT_POINT_COUNT:]
        scaled = fit_ka / upper
        scaled_coefficients, constraint_active = fit_squared_frequency(scaled, fit_frequency)
        raw_coefficients = np.asarray(
            [
                scaled_coefficients[0],
                scaled_coefficients[1] / upper,
                scaled_coefficients[2] / upper**2,
            ],
            dtype=np.float64,
        )
        dense_ka = np.linspace(0.0, upper, 4097)
        dense_squared = np.polynomial.polynomial.polyval(dense_ka, raw_coefficients)
        dense_derivative = raw_coefficients[1] + 2.0 * raw_coefficients[2] * dense_ka
        if (
            np.any(~np.isfinite(dense_squared))
            or np.any(dense_squared <= 0.0)
            or np.any(dense_derivative < -1.0e-10)
        ):
            raise RuntimeError(f"Positive-monotone replay gate failed at {agent} {diameter:g} um")
        prediction = predict_frequency_mhz(raw_coefficients, fit_ka)
        rows.append(
            {
                "agent": agent,
                "diameter_um": diameter,
                "response": RESPONSE,
                "formula": FORMULA,
                "degree": 2,
                "fit_point_count": FIT_POINT_COUNT,
                "held_low_ka_point_count": HELD_LOW_KA_POINT_COUNT,
                "simulated_ka_min_dpd": float(ka[0]),
                "simulated_ka_max_dpd": upper,
                "fit_ka_min_dpd": float(fit_ka[0]),
                "fit_ka_max_dpd": upper,
                "scaled_coefficient_c0_mhz2": float(scaled_coefficients[0]),
                "scaled_coefficient_c1_mhz2": float(scaled_coefficients[1]),
                "scaled_coefficient_c2_mhz2": float(scaled_coefficients[2]),
                "a0_mhz2": float(raw_coefficients[0]),
                "a1_mhz2_per_dpd": float(raw_coefficients[1]),
                "a2_mhz2_per_dpd2": float(raw_coefficients[2]),
                "frequency_at_ka_zero_mhz": float(math.sqrt(raw_coefficients[0])),
                "fit_rmse_mhz": float(np.sqrt(np.mean(np.square(prediction - fit_frequency)))),
                "leave_one_out_rmse_mhz": leave_one_out_rmse(scaled, fit_frequency),
                "constraint_active": bool(constraint_active),
                "evaluation_ka_min_dpd": 0.0,
                "evaluation_ka_max_dpd": upper,
                "dense_validation_point_count": int(dense_ka.size),
            }
        )
    return rows


def _index_csv(path: Path) -> dict[tuple[str, float], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[tuple[str, float], dict[str, str]] = {}
    for row in rows:
        context = f"{path} row {len(indexed) + 2}"
        key = (str(row.get("agent") or "").lower(), _parse_float(row, "diameter_um", context))
        if key in indexed:
            raise ValueError(f"Accepted coefficient table has duplicate row for {key}")
        indexed[key] = row
    return indexed


def _index_bank(path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    agent = str(payload.get("agent") or "").lower()
    if not agent or payload.get("schema") != "meso_uq.emb_dpd_frequency_polynomial_bank.v1":
        raise ValueError(f"Unexpected frozen bank schema: {path}")
    indexed: dict[tuple[str, float], dict[str, Any]] = {}
    for entry in payload.get("emulators", []):
        key = (agent, float(entry["diameter_um"]))
        if key in indexed:
            raise ValueError(f"Frozen bank has duplicate emulator for {key}")
        indexed[key] = entry
    return indexed


def _bank_file_receipts(bank_dir: Path) -> list[dict[str, str]]:
    paths = sorted(bank_dir.glob("*_approved_free_intercept_squared_frequency_bank.json"))
    if not paths:
        raise FileNotFoundError(f"No approved squared-frequency banks found in {bank_dir}")
    return [{"path": str(path), "sha256": sha256_file(path)} for path in paths]


def _allclose(
    actual: np.ndarray, expected: np.ndarray, *, rtol: float, atol: float
) -> tuple[bool, float]:
    difference = np.abs(np.asarray(actual) - np.asarray(expected))
    return bool(np.allclose(actual, expected, rtol=rtol, atol=atol)), float(difference.max())


def compare_against_frozen(
    replay_rows: list[Mapping[str, Any]],
    labels: list[Mapping[str, Any]],
    accepted_fits: Path,
    accepted_bank_dir: Path,
    *,
    coefficient_rtol: float,
    coefficient_atol: float,
    frequency_rtol: float,
    frequency_atol: float,
) -> dict[str, Any]:
    """Compare recomputed coefficients and in-support predictions to frozen artifacts."""

    accepted_rows = _index_csv(accepted_fits)
    banks: dict[tuple[str, float], dict[str, Any]] = {}
    bank_receipts = _bank_file_receipts(accepted_bank_dir)
    for receipt in bank_receipts:
        path = Path(receipt["path"])
        banks.update(_index_bank(path))
    grouped = _group_labels(labels)
    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    for replay in replay_rows:
        key = (str(replay["agent"]), float(replay["diameter_um"]))
        accepted = accepted_rows.get(key)
        bank = banks.get(key)
        if accepted is None or bank is None:
            failures.append(f"Missing frozen comparison artifact for {key[0]} {key[1]:g} um")
            continue
        if accepted.get("response") != RESPONSE or accepted.get("degree") != "2":
            failures.append(f"Frozen coefficient policy differs for {key[0]} {key[1]:g} um")
            continue
        if (
            accepted.get("fit_point_count") != str(FIT_POINT_COUNT)
            or accepted.get("held_low_ka_point_count") != str(HELD_LOW_KA_POINT_COUNT)
        ):
            failures.append(f"Frozen fit-window policy differs for {key[0]} {key[1]:g} um")
            continue
        expected_csv = np.asarray(
            [
                _parse_float(accepted, "a0_mhz2", f"accepted {key}"),
                _parse_float(accepted, "a1_mhz2_per_dpd", f"accepted {key}"),
                _parse_float(accepted, "a2_mhz2_per_dpd2", f"accepted {key}"),
            ]
        )
        actual = np.asarray(
            [replay["a0_mhz2"], replay["a1_mhz2_per_dpd"], replay["a2_mhz2_per_dpd2"]],
            dtype=np.float64,
        )
        csv_match, csv_max_abs = _allclose(
            actual, expected_csv, rtol=coefficient_rtol, atol=coefficient_atol
        )
        bank_polynomial = bank.get("polynomial") or {}
        if (
            bank_polynomial.get("response_encoding") != RESPONSE
            or bank_polynomial.get("degree") != 2
        ):
            failures.append(f"Frozen bank policy differs for {key[0]} {key[1]:g} um")
            continue
        expected_bank = np.asarray(bank_polynomial.get("coefficients") or [], dtype=np.float64)
        bank_match, bank_max_abs = _allclose(
            actual, expected_bank, rtol=coefficient_rtol, atol=coefficient_atol
        )
        group = grouped[key]
        ka = np.asarray([row["ka_dpd"] for row in group], dtype=np.float64)
        support = np.asarray(bank.get("ka_bounds_dpd") or [], dtype=np.float64)
        if support.shape != (2,) or support[0] != 0.0 or support[1] != float(ka[-1]):
            failures.append(f"Frozen bank support differs from replay policy for {key[0]} {key[1]:g} um")
            continue
        grid = np.unique(np.concatenate((ka, np.linspace(support[0], support[1], 4097))))
        actual_frequency = predict_frequency_mhz(actual, grid)
        csv_frequency = predict_frequency_mhz(expected_csv, grid)
        bank_frequency = predict_frequency_mhz(expected_bank, grid)
        csv_prediction_match, csv_prediction_max_abs = _allclose(
            actual_frequency,
            csv_frequency,
            rtol=frequency_rtol,
            atol=frequency_atol,
        )
        bank_prediction_match, bank_prediction_max_abs = _allclose(
            actual_frequency,
            bank_frequency,
            rtol=frequency_rtol,
            atol=frequency_atol,
        )
        ok = csv_match and bank_match and csv_prediction_match and bank_prediction_match
        if not ok:
            failures.append(
                f"Tolerance mismatch for {key[0]} {key[1]:g} um "
                f"(csv coefficients {csv_max_abs:.3e}, bank coefficients {bank_max_abs:.3e}, "
                f"csv frequency {csv_prediction_max_abs:.3e}, bank frequency {bank_prediction_max_abs:.3e})"
            )
        comparisons.append(
            {
                "agent": key[0],
                "diameter_um": key[1],
                "csv_coefficients_match": csv_match,
                "bank_coefficients_match": bank_match,
                "csv_prediction_match": csv_prediction_match,
                "bank_prediction_match": bank_prediction_match,
                "csv_coefficient_max_abs_difference": csv_max_abs,
                "bank_coefficient_max_abs_difference": bank_max_abs,
                "csv_prediction_max_abs_difference_mhz": csv_prediction_max_abs,
                "bank_prediction_max_abs_difference_mhz": bank_prediction_max_abs,
                "comparison_grid_point_count": int(grid.size),
            }
        )
    extra_csv = sorted(set(accepted_rows).difference((str(row["agent"]), float(row["diameter_um"])) for row in replay_rows))
    extra_banks = sorted(set(banks).difference((str(row["agent"]), float(row["diameter_um"])) for row in replay_rows))
    if extra_csv:
        failures.append(f"Frozen coefficient rows not replayed: {extra_csv}")
    if extra_banks:
        failures.append(f"Frozen bank entries not replayed: {extra_banks}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "replayed_diameter_count": len(replay_rows),
        "comparison_count": len(comparisons),
        "tolerances": {
            "coefficient_rtol": coefficient_rtol,
            "coefficient_atol": coefficient_atol,
            "frequency_rtol": frequency_rtol,
            "frequency_atol_mhz": frequency_atol,
        },
        "comparisons": comparisons,
        "accepted_banks": bank_receipts,
        "failures": failures,
    }


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("No replay rows to write")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_replay_banks(
    output_dir: Path, replay_rows: list[Mapping[str, Any]], provenance: Mapping[str, Any]
) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for agent in sorted({str(row["agent"]) for row in replay_rows}):
        entries = []
        for row in replay_rows:
            if row["agent"] != agent:
                continue
            entries.append(
                {
                    "diameter_um": row["diameter_um"],
                    "ka_bounds_dpd": [row["evaluation_ka_min_dpd"], row["evaluation_ka_max_dpd"]],
                    "polynomial": {
                        "basis": "ascending_power_of_affine_ka",
                        "coefficient_order": "ascending_power",
                        "coefficients": [row["a0_mhz2"], row["a1_mhz2_per_dpd"], row["a2_mhz2_per_dpd2"]],
                        "degree": 2,
                        "input_transform": {"type": "affine", "offset_dpd": 0.0, "scale_dpd": 1.0},
                        "method": "affine_low_order_horner",
                        "response_encoding": RESPONSE,
                    },
                    "validation": {
                        "fit_point_count": FIT_POINT_COUNT,
                        "held_low_ka_point_count": HELD_LOW_KA_POINT_COUNT,
                        "calibration_ka_bounds_dpd": [row["fit_ka_min_dpd"], row["fit_ka_max_dpd"]],
                        "evaluation_ka_bounds_dpd": [row["evaluation_ka_min_dpd"], row["evaluation_ka_max_dpd"]],
                        "positive_squared_frequency_on_evaluation_support": True,
                        "monotonic_non_decreasing_frequency_on_evaluation_support": True,
                    },
                }
            )
        payload = {
            "schema": "meso_uq.emb_dpd_frequency_polynomial_replay_bank.v1",
            "status": "replayed_not_accepted_artifact",
            "agent": agent,
            "response": RESPONSE,
            "policy": {
                "fit_policy": "highest_28_of_32_per_exact_diameter_unweighted_frequency_residuals",
                "degree": 2,
                "free_intercept": True,
                "outside_declared_support_evaluation_allowed": False,
                "fallback_allowed": False,
            },
            "provenance": dict(provenance),
            "emulators": entries,
        }
        payload["integrity"] = {"algorithm": "sha256", "canonical_payload_sha256": canonical_json_sha256(payload)}
        target = output_dir / f"replayed_{agent}_polynomial_bank.json"
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        outputs.append({"path": str(target), "sha256": sha256_file(target)})
    return outputs


def replay(
    *,
    artifact_root: Path,
    input_labels: Path | None = None,
    accepted_fits: Path | None = None,
    accepted_bank_dir: Path | None = None,
    output_dir: Path | None = None,
    report_path: Path | None = None,
    dry_run: bool = False,
    coefficient_rtol: float = 1.0e-8,
    coefficient_atol: float = 1.0e-13,
    frequency_rtol: float = 1.0e-8,
    frequency_atol: float = 1.0e-8,
) -> dict[str, Any]:
    """Recompute, compare, and optionally materialize bounded replay artifacts."""

    artifact_root = artifact_root.expanduser().resolve()
    labels_path = (input_labels or artifact_root / "physical_labels/physical_resonance_labels.csv").expanduser().resolve()
    fits_path = (accepted_fits or artifact_root / "polynomial_fits/free_intercept_squared_frequency_fits.csv").expanduser().resolve()
    banks_path = (accepted_bank_dir or artifact_root / "frozen_banks").expanduser().resolve()
    for path in (labels_path, fits_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required acoustic replay input is missing: {path}")
    if not banks_path.is_dir():
        raise FileNotFoundError(f"Required frozen bank directory is missing: {banks_path}")
    if output_dir is not None and dry_run:
        raise ValueError("--output-dir cannot be used with --dry-run")
    if output_dir is not None:
        output_dir = output_dir.expanduser().resolve()
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Refusing to write into non-empty replay output: {output_dir}")

    labels = load_labels(labels_path)
    replay_rows = fit_replay(labels)
    comparison = compare_against_frozen(
        replay_rows,
        labels,
        fits_path,
        banks_path,
        coefficient_rtol=coefficient_rtol,
        coefficient_atol=coefficient_atol,
        frequency_rtol=frequency_rtol,
        frequency_atol=frequency_atol,
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "artifact_root": str(artifact_root),
        "input_labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
        "accepted_coefficients": {"path": str(fits_path), "sha256": sha256_file(fits_path)},
        "accepted_bank_directory": str(banks_path),
        "accepted_bank_files": _bank_file_receipts(banks_path),
        "replay_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "policy": {
            "response": RESPONSE,
            "degree": 2,
            "free_intercept": True,
            "fit_point_selection": "highest_28_of_32_ka_values",
            "residual_space": "frequency_mhz_unweighted",
            "constraints": ["positive_squared_frequency", "monotonic_non_decreasing_frequency"],
            "diameter_interpolation_allowed": False,
            "fallback_allowed": False,
            "outside_declared_support_evaluation_allowed": False,
        },
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": comparison["status"],
        "execution_provenance": replay_receipt_provenance(
            repo_root=REPO_ROOT,
            runner=Path(__file__),
            consumed_paths=[artifact_root, labels_path, fits_path, banks_path],
        ),
        "provenance": provenance,
        "comparison": comparison,
        "outputs": [],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=False)
        csv_path = output_dir / "replayed_free_intercept_squared_frequency_fits.csv"
        _write_csv(csv_path, replay_rows)
        report["outputs"].append({"path": str(csv_path), "sha256": sha256_file(csv_path)})
        report["outputs"].extend(_write_replay_banks(output_dir, replay_rows, provenance))
    if report_path is not None:
        report_path = report_path.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif output_dir is not None:
        target = output_dir / "acoustic_polynomial_replay_report.json"
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["outputs"].append({"path": str(target), "sha256": sha256_file(target)})
    if comparison["status"] != "PASS":
        raise ReplayMismatchError("Acoustic polynomial replay failed: " + "; ".join(comparison["failures"]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--input", dest="input_labels", type=Path)
    parser.add_argument("--accepted-fits", type=Path)
    parser.add_argument("--accepted-bank-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path, help="Write the JSON report to this explicit path.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print only; do not create replay outputs.")
    parser.add_argument("--coefficient-rtol", type=float, default=1.0e-8)
    parser.add_argument("--coefficient-atol", type=float, default=1.0e-13)
    parser.add_argument("--frequency-rtol", type=float, default=1.0e-8)
    parser.add_argument("--frequency-atol", type=float, default=1.0e-8)
    args = parser.parse_args()
    try:
        report = replay(
            artifact_root=args.artifact_root,
            input_labels=args.input_labels,
            accepted_fits=args.accepted_fits,
            accepted_bank_dir=args.accepted_bank_dir,
            output_dir=args.output_dir,
            report_path=args.report,
            dry_run=args.dry_run,
            coefficient_rtol=args.coefficient_rtol,
            coefficient_atol=args.coefficient_atol,
            frequency_rtol=args.frequency_rtol,
            frequency_atol=args.frequency_atol,
        )
    except (FileNotFoundError, ValueError, ReplayMismatchError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "replayed_diameter_count": report["comparison"]["replayed_diameter_count"],
                "comparison_count": report["comparison"]["comparison_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
