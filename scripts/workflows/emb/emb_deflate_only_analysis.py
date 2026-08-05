"""Qualified early-ringdown analysis shared by EMB deflate-only campaigns."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import run_emb_free_shell_breathing_protocol as protocol


MIN_OBSERVED_CYCLES = 3.0
MIN_FIT_R2 = 0.95
MAX_FFT_DAMPED_RELATIVE_DELTA = 0.15
MIN_AMPLITUDE_TO_RESIDUAL = 5.0
EXTREMUM_MIN_RESIDUAL_MULTIPLIER = 1.5
MIN_MODE_PURITY_FRAMES = 20
MAX_COHERENT_MODE_RATIO = 0.20
MAX_RADIAL_RESIDUAL_RATIO = 0.25


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_radius_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = _read_csv(path)
    if not rows or "time_dpd" not in rows[0] or "rms_radius_dpd" not in rows[0]:
        raise RuntimeError(f"Invalid RMS-radius time series: {path}")
    time_dpd = np.asarray([float(row["time_dpd"]) for row in rows], dtype=np.float64)
    radius_dpd = np.asarray([float(row["rms_radius_dpd"]) for row in rows], dtype=np.float64)
    if time_dpd.size < 24 or not np.all(np.isfinite(time_dpd)) or not np.all(np.isfinite(radius_dpd)):
        raise RuntimeError(f"Insufficient or nonfinite RMS-radius time series: {path}")
    return time_dpd, radius_dpd


def observed_cycles(
    time_dpd: np.ndarray,
    radius_dpd: np.ndarray,
    damped: dict[str, Any],
) -> dict[str, Any]:
    coefficients = damped.get("coefficients") or []
    if len(coefficients) < 6:
        return {"cycles": 0.0, "qualified_extrema_count": 0, "threshold_dpd": None}
    shifted = time_dpd - float(time_dpd[0])
    baseline = float(coefficients[0]) + float(coefficients[1]) * shifted
    centered = radius_dpd - baseline
    residual_rms = float(damped.get("residual_rms", math.nan))
    amplitude = float(damped.get("amplitude", math.nan))
    thresholds = [
        EXTREMUM_MIN_RESIDUAL_MULTIPLIER * residual_rms,
        0.01 * amplitude,
    ]
    threshold = max(value for value in thresholds if math.isfinite(value) and value >= 0.0)
    extrema: list[tuple[float, float, str]] = []
    for index in range(1, centered.size - 1):
        value = float(centered[index])
        if value > centered[index - 1] and value >= centered[index + 1] and value >= threshold:
            kind = "max"
        elif value < centered[index - 1] and value <= centered[index + 1] and value <= -threshold:
            kind = "min"
        else:
            continue
        item = (float(time_dpd[index]), value, kind)
        if not extrema or kind != extrema[-1][2]:
            extrema.append(item)
        elif (kind == "max" and value > extrema[-1][1]) or (kind == "min" and value < extrema[-1][1]):
            extrema[-1] = item
    return {
        "cycles": max(0.0, 0.5 * float(len(extrema) - 1)),
        "qualified_extrema_count": len(extrema),
        "threshold_dpd": threshold,
        "threshold_residual_multiplier": EXTREMUM_MIN_RESIDUAL_MULTIPLIER,
    }


def select_early_ringdown_window(metric: dict[str, Any]) -> dict[str, Any]:
    time_series_text = metric.get("time_series_csv")
    if not isinstance(time_series_text, str) or not Path(time_series_text).is_file():
        return {"status": "missing_time_series", "accepted": False}
    try:
        time_dpd, radius_dpd = load_radius_series(Path(time_series_text))
    except Exception as exc:
        return {
            "status": "invalid_time_series",
            "accepted": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    minimum_index = max(24, int(math.ceil(0.03 * time_dpd.size)))
    candidate_indices = sorted(
        set(np.linspace(minimum_index, time_dpd.size - 1, 100, dtype=np.int64).tolist())
    )
    candidates: list[dict[str, Any]] = []
    for end_index in candidate_indices:
        end_dpd = float(time_dpd[end_index])
        fit = protocol.estimate_frequency(
            time_dpd,
            radius_dpd,
            transient_cut_fraction=0.0,
            fit_start_dpd=0.0,
            fit_end_dpd=end_dpd,
        )
        damped = fit.get("damped_sine") or {}
        if damped.get("status") != "ok":
            continue
        mask = time_dpd <= end_dpd
        visibility = observed_cycles(time_dpd[mask], radius_dpd[mask], damped)
        r2 = damped.get("r2")
        fft_delta = fit.get("fft_damped_relative_delta")
        amplitude_to_residual = damped.get("amplitude_to_residual_std")
        finite_quality = all(
            value is not None and math.isfinite(float(value))
            for value in (r2, fft_delta, amplitude_to_residual)
        )
        accepted = bool(
            finite_quality
            and float(r2) >= MIN_FIT_R2
            and float(fft_delta) <= MAX_FFT_DAMPED_RELATIVE_DELTA
            and float(amplitude_to_residual) >= MIN_AMPLITUDE_TO_RESIDUAL
            and float(visibility["cycles"]) >= MIN_OBSERVED_CYCLES
        )
        score = (
            (1000.0 if accepted else 0.0)
            + 100.0 * float(r2 if r2 is not None else -1.0)
            - 20.0 * float(fft_delta if fft_delta is not None else 10.0)
            + min(10.0, float(visibility["cycles"]))
            + min(20.0, float(amplitude_to_residual if amplitude_to_residual is not None else 0.0)) / 10.0
        )
        candidates.append(
            {
                "status": "accepted" if accepted else "candidate_gate_failed",
                "accepted": accepted,
                "score": score,
                "fit_start_dpd": 0.0,
                "fit_end_dpd": end_dpd,
                "fit_end_index": int(end_index),
                "fit_sample_count": int(np.count_nonzero(mask)),
                "pilot_visible_cycles": visibility["cycles"],
                "pilot_qualified_extrema_count": visibility["qualified_extrema_count"],
                "pilot_extrema_threshold_dpd": visibility["threshold_dpd"],
                "pilot_extrema_threshold_residual_multiplier": visibility[
                    "threshold_residual_multiplier"
                ],
                "fit": fit,
            }
        )
    if not candidates:
        return {"status": "no_fitted_candidates", "accepted": False}
    selected = max(candidates, key=lambda candidate: float(candidate["score"]))
    selected["candidate_count"] = len(candidates)
    selected["selection_policy"] = (
        "maximize documented score; accepted candidates receive a 1000-point priority; "
        "all windows begin at release and candidate ends span the retained record"
    )
    return selected


def selected_window_mode_purity(
    metric: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    result_text = metric.get("result_json")
    if not isinstance(result_text, str) or not Path(result_text).is_file():
        return {"status": "missing_result_json", "mode_purity_clean_l0_dominant": False}
    try:
        result = json.loads(Path(result_text).read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "invalid_result_json",
            "mode_purity_clean_l0_dominant": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    mode_csv_text = result.get("mode_purity_csv")
    if not isinstance(mode_csv_text, str) or not Path(mode_csv_text).is_file():
        return {"status": "missing_mode_purity_csv", "mode_purity_clean_l0_dominant": False}
    fit_start = selected.get("fit_start_dpd")
    fit_end = selected.get("fit_end_dpd")
    if fit_start is None or fit_end is None:
        return {"status": "missing_selected_window", "mode_purity_clean_l0_dominant": False}
    try:
        rows = [
            row
            for row in _read_csv(Path(mode_csv_text))
            if float(fit_start) <= float(row["time_dpd"]) <= float(fit_end)
        ]
        return protocol.summarize_mode_purity_rows(rows)
    except Exception as exc:
        return {
            "status": "mode_purity_summary_failed",
            "mode_purity_clean_l0_dominant": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def qualified_early_window_row(metric: dict[str, Any]) -> dict[str, Any]:
    case = metric.get("case") or {}
    selected = select_early_ringdown_window(metric)
    mode_purity = selected_window_mode_purity(metric, selected)
    fit = selected.get("fit") or {}
    damped = fit.get("damped_sine") or {}
    raw_frequency = damped.get("frequency_mhz") if damped.get("status") == "ok" else None
    mass_scale = case.get("mass_scale")
    corrected_frequency = (
        None
        if raw_frequency is None or mass_scale is None
        else float(raw_frequency) * math.sqrt(float(mass_scale))
    )
    raw_damping = damped.get("damping_dpd_inv")
    corrected_undamped_frequency = None
    raw_undamped_frequency = None
    damping_ratio = None
    mass_weighted_damping = None
    if raw_frequency is not None and raw_damping is not None:
        omega_damped_dpd = 2.0 * math.pi * float(raw_frequency) * 1.0e6 * protocol.UNIT_TIME_SECONDS
        omega_undamped_dpd = math.sqrt(omega_damped_dpd**2 + float(raw_damping) ** 2)
        raw_undamped_frequency = omega_undamped_dpd / (
            2.0 * math.pi * protocol.UNIT_TIME_SECONDS * 1.0e6
        )
        damping_ratio = float(raw_damping) / omega_undamped_dpd
        if mass_scale is not None:
            corrected_undamped_frequency = raw_undamped_frequency * math.sqrt(float(mass_scale))
            mass_weighted_damping = float(raw_damping) * float(mass_scale)
    finite_accepted = bool(metric.get("finite_observable") and metric.get("finite_volume"))
    frame_count = int(mode_purity.get("frame_count") or 0)
    l1 = mode_purity.get("l1_spatiotemporal_rms_over_l0_half_amplitude")
    l2 = mode_purity.get("l2_spatiotemporal_rms_over_l0_half_amplitude")
    radial = mode_purity.get("radial_residual_spatiotemporal_rms_over_l0_half_amplitude")
    tangential = mode_purity.get("tangential_spatiotemporal_rms_over_l0_half_amplitude")
    mode_accepted = bool(
        mode_purity.get("status") == "ok"
        and frame_count >= MIN_MODE_PURITY_FRAMES
        and l1 is not None and float(l1) <= MAX_COHERENT_MODE_RATIO
        and l2 is not None and float(l2) <= MAX_COHERENT_MODE_RATIO
        and radial is not None and float(radial) <= MAX_RADIAL_RESIDUAL_RATIO
        and tangential is not None and float(tangential) <= MAX_COHERENT_MODE_RATIO
    )
    fit_accepted = bool(selected.get("accepted"))
    accepted = bool(finite_accepted and fit_accepted and mode_accepted)
    if accepted:
        status = "accepted"
    elif not finite_accepted:
        status = "finite_observable_gate_failed"
    elif fit_accepted:
        status = "mode_purity_gate_failed"
    else:
        status = selected.get("status")
    return {
        "case_index": case.get("case_index"),
        "agent": case.get("agent"),
        "surface_node_id": case.get("surface_node_id"),
        "diameter_um": case.get("diameter_um"),
        "ka": case.get("ka"),
        "kb": case.get("kb"),
        "mass_scale": mass_scale,
        "deflation_percent": case.get("deflation_percent"),
        "status": status,
        "accepted": accepted,
        "finite_accepted": finite_accepted,
        "finite_observable": metric.get("finite_observable"),
        "finite_volume": metric.get("finite_volume"),
        "fit_signal_accepted": fit_accepted,
        "mode_purity_accepted": mode_accepted,
        "fit_start_dpd": selected.get("fit_start_dpd"),
        "fit_end_dpd": selected.get("fit_end_dpd"),
        "fit_sample_count": selected.get("fit_sample_count"),
        "candidate_count": selected.get("candidate_count"),
        "raw_frequency_mhz": raw_frequency,
        "mass_corrected_frequency_mhz": corrected_frequency,
        "raw_undamped_frequency_mhz": raw_undamped_frequency,
        "mass_corrected_undamped_frequency_mhz": corrected_undamped_frequency,
        "raw_damping_lambda_dpd_inv": raw_damping,
        "mass_weighted_damping_lambda_dpd_inv": mass_weighted_damping,
        "damping_ratio": damping_ratio,
        "fit_amplitude_dpd": damped.get("amplitude"),
        "damped_sine_r2": damped.get("r2"),
        "amplitude_to_residual_std": damped.get("amplitude_to_residual_std"),
        "fft_damped_relative_delta": fit.get("fft_damped_relative_delta"),
        "visible_alternating_cycles": selected.get("pilot_visible_cycles"),
        "qualified_extrema_count": selected.get("pilot_qualified_extrema_count"),
        "extrema_threshold_dpd": selected.get("pilot_extrema_threshold_dpd"),
        "mode_purity_status": mode_purity.get("status"),
        "mode_purity_gate_basis": "spatiotemporal_rms_over_selected_fit_window",
        "mode_purity_frame_count": frame_count,
        "mode_purity_minimum_frame_count": MIN_MODE_PURITY_FRAMES,
        "mode_purity_ratio_threshold": MAX_COHERENT_MODE_RATIO,
        "radial_residual_ratio_threshold": MAX_RADIAL_RESIDUAL_RATIO,
        "l1_over_l0": l1,
        "l2_over_l0": l2,
        "radial_residual_over_l0": radial,
        "tangential_over_l0": tangential,
        "l1_max_over_l0": mode_purity.get("l1_rms_max_over_l0_half_amplitude"),
        "l2_max_over_l0": mode_purity.get("l2_rms_max_over_l0_half_amplitude"),
        "radial_residual_max_over_l0": mode_purity.get(
            "radial_residual_rms_max_over_l0_half_amplitude"
        ),
        "tangential_max_over_l0": mode_purity.get(
            "tangential_rms_max_over_l0_half_amplitude"
        ),
        "l1_p95_over_l0": mode_purity.get("l1_temporal_p95_over_l0_half_amplitude"),
        "l2_p95_over_l0": mode_purity.get("l2_temporal_p95_over_l0_half_amplitude"),
        "radial_residual_p95_over_l0": mode_purity.get(
            "radial_residual_temporal_p95_over_l0_half_amplitude"
        ),
        "tangential_p95_over_l0": mode_purity.get(
            "tangential_temporal_p95_over_l0_half_amplitude"
        ),
        "time_series_csv": metric.get("time_series_csv"),
        "result_json": metric.get("result_json"),
        "selected_fit_json": json.dumps(selected, sort_keys=True),
        "full_record_status": metric.get("status"),
        "full_record_frequency_mhz": metric.get("raw_frequency_mhz"),
        "full_record_r2": metric.get("damped_sine_r2"),
        "full_record_visible_cycles": metric.get("visible_alternating_cycles"),
        "selected_fit": selected,
    }
