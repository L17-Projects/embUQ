from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from meso_uq.inference import emb_resonance
from meso_uq.inference.emb_frequency_emulator import (
    DENSE_VALIDATION_MIN_POINTS,
    DIAMETER_TOLERANCE_UM,
    DpdFrequencyEmulator,
    DpdFrequencyEmulatorBank,
    FREQUENCY_EMULATOR_BANK_SCHEMA,
    FIT_PRIMARY_LABEL_ADMISSION_POLICY,
    NONMONOTONE_FREQUENCY_POLICY,
    fit_frequency_emulator,
    leave_one_ka_out_metrics,
)


def _pchip_emulator(diameter_um: float, *, lower: float = 500.0, upper: float = 42000.0):
    ka = np.linspace(lower, upper, 4, dtype=np.float64)
    frequency = np.sqrt(4.0 + diameter_um / 10.0 + 3.0e-5 * (ka - lower))
    emulator = fit_frequency_emulator(
        diameter_um=diameter_um,
        ka_dpd=ka,
        frequency_mhz=frequency,
        ka_bounds_dpd=(lower, upper),
        provenance={"campaign": f"d{diameter_um:g}", "inference_ready": True},
        validation={
            "label_admission_policy": FIT_PRIMARY_LABEL_ADMISSION_POLICY,
            "fit_primary_gate_passed": True,
            "frequency_labels_monotone": True,
            "nonmonotone_frequency_policy": NONMONOTONE_FREQUENCY_POLICY,
            "dense_grid_point_count": DENSE_VALIDATION_MIN_POINTS,
            "dense_grid_finite_positive": True,
        },
    )
    return emulator


def _bank() -> DpdFrequencyEmulatorBank:
    return DpdFrequencyEmulatorBank(
        agent="sonovue",
        emulators=[_pchip_emulator(5.8), _pchip_emulator(3.2)],
        conditions={"fixed_kb_dpd": 7850.288865935088},
        provenance={
            "git_commit": "synthetic",
            "inference_ready": True,
            "source_datasets": [{"path": "/fixture/source.json", "sha256": "a" * 64}],
        },
        validation={
            "label_admission_policy": FIT_PRIMARY_LABEL_ADMISSION_POLICY,
            "nonmonotone_frequency_policy": NONMONOTONE_FREQUENCY_POLICY,
            "all_diameters_positive_and_monotonic": True,
            "all_dense_grids_finite_positive": True,
            "frequency_labels_monotone_by_diameter": True,
            "nonmonotone_frequency_label_diameters_um": [],
            "production_quality_gate_passed": True,
        },
    )


def _refresh_integrity(payload: dict) -> None:
    unsigned = dict(payload)
    unsigned.pop("integrity", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["integrity"] = {
        "algorithm": "sha256",
        "canonical_payload_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def test_bank_round_trip_sorts_diameters_broadcasts_and_validates_hash(tmp_path: Path) -> None:
    path = _bank().write(tmp_path / "sonovue_frequency_emulator_bank.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    bank = DpdFrequencyEmulatorBank.load(path)

    batch = bank.predict_for_diameters_mhz(
        np.asarray([500.0, 42000.0]).reshape(-1, 1),
        np.asarray([3.2, 5.8]).reshape(1, -1),
    )

    assert payload["schema"] == FREQUENCY_EMULATOR_BANK_SCHEMA
    assert payload["runtime_contract"]["ka_clipping_allowed"] is False
    assert payload["runtime_contract"]["ka_extrapolation_allowed"] is False
    assert payload["runtime_contract"]["diameter_interpolation_allowed"] is False
    assert payload["runtime_contract"]["analytical_fallback_allowed"] is False
    assert bank.diameters_um == (3.2, 5.8)
    assert bank.ka_bounds_dpd == (500.0, 42000.0)
    assert batch.shape == (2, 2)
    assert np.all(np.isfinite(batch))
    assert np.all(batch > 0.0)

    payload["emulators"][0]["pchip"]["interval_coefficients"][0][0] += 1.0
    with pytest.raises(ValueError, match="integrity hash does not match"):
        DpdFrequencyEmulatorBank.from_mapping(payload)


def test_bank_requires_exact_diameter_and_rejects_ka_extrapolation() -> None:
    bank = _bank()

    tolerated = bank.predict_for_diameters_mhz(10000.0, 3.2 + 0.5 * DIAMETER_TOLERANCE_UM)
    exact = bank.predict_for_diameters_mhz(10000.0, 3.2)
    assert tolerated == pytest.approx(exact)

    with pytest.raises(ValueError, match="No exact DPD frequency emulator"):
        bank.predict_for_diameters_mhz(10000.0, 3.2 + 1.5 * DIAMETER_TOLERANCE_UM)
    with pytest.raises(ValueError, match="outside the supported range"):
        bank.predict_for_diameters_mhz(499.999999, 3.2)
    with pytest.raises(ValueError, match="outside the supported range"):
        bank.predict_for_diameters_mhz(42000.000001, 3.2)


def test_endpoints_are_inclusive_and_scalar_shapes_are_preserved() -> None:
    bank = _bank()

    lower = bank.predict_for_diameters_mhz(500.0, 3.2)
    upper = bank.predict_for_diameters_mhz(42000.0, 3.2)
    scalar = bank.emulator_for_diameter(3.2).predict_mhz(500.0)

    assert lower.shape == ()
    assert scalar.shape == ()
    assert float(lower) == pytest.approx(float(scalar))
    assert float(upper) > float(lower)


def test_population_batch_matches_per_diameter_pchip_without_row_loop() -> None:
    bank = _bank()
    ka = np.asarray([500.0, 10000.0, 30000.0, 42000.0]).reshape(-1, 1)
    diameters = np.asarray([3.2, 5.8]).reshape(1, -1)

    predicted = bank.predict_for_diameters_mhz(ka, diameters)
    expected = np.column_stack(
        [
            bank.emulator_for_diameter(3.2).predict_mhz(ka.reshape(-1)),
            bank.emulator_for_diameter(5.8).predict_mhz(ka.reshape(-1)),
        ]
    )

    assert predicted.shape == (4, 2)
    assert predicted == pytest.approx(expected)


def test_bank_rejects_duplicate_diameters_and_disjoint_ka_support() -> None:
    with pytest.raises(ValueError, match="duplicate diameter"):
        DpdFrequencyEmulatorBank(
            agent="definity",
            emulators=[_pchip_emulator(2.1), _pchip_emulator(2.1 + 5.0e-10)],
        )
    with pytest.raises(ValueError, match="no common ka support"):
        DpdFrequencyEmulatorBank(
            agent="definity",
            emulators=[
                _pchip_emulator(2.1, lower=0.0, upper=100.0),
                _pchip_emulator(2.9, lower=100.0, upper=200.0),
            ],
        )


def test_bank_schema_runtime_contract_and_recorded_common_support_are_strict() -> None:
    payload = _bank().to_mapping()
    payload["coordinates"]["diameter_mode"] = "interpolated"
    _refresh_integrity(payload)
    with pytest.raises(ValueError, match="diameter_mode must be exact"):
        DpdFrequencyEmulatorBank.from_mapping(payload)

    payload = _bank().to_mapping()
    payload["coordinates"]["common_ka_bounds_dpd"] = [600.0, 42000.0]
    _refresh_integrity(payload)
    with pytest.raises(ValueError, match="does not match its entries"):
        DpdFrequencyEmulatorBank.from_mapping(payload)

    payload = _bank().to_mapping()
    payload["integrity"].pop("canonical_payload_sha256")
    with pytest.raises(ValueError, match="integrity hash is invalid"):
        DpdFrequencyEmulatorBank.from_mapping(payload)

    payload = _bank().to_mapping()
    payload["schema"] = "meso_uq.emb_dpd_frequency_emulator_bank.v1"
    with pytest.raises(ValueError, match="Unsupported frequency emulator bank schema"):
        DpdFrequencyEmulatorBank.from_mapping(payload)


def _nonmonotone_bank() -> DpdFrequencyEmulatorBank:
    emulator = fit_frequency_emulator(
        diameter_um=5.8,
        ka_dpd=np.asarray([500.0, 2000.0, 8000.0, 42000.0]),
        frequency_mhz=np.asarray([0.58, 0.46, 0.84, 2.1]),
        validation={
            "label_admission_policy": FIT_PRIMARY_LABEL_ADMISSION_POLICY,
            "fit_primary_gate_passed": True,
            "frequency_labels_monotone": False,
            "nonmonotone_frequency_policy": NONMONOTONE_FREQUENCY_POLICY,
            "dense_grid_point_count": DENSE_VALIDATION_MIN_POINTS,
            "dense_grid_finite_positive": True,
        },
        provenance={"inference_ready": True},
    )
    return DpdFrequencyEmulatorBank(
        agent="sonovue",
        emulators=[emulator],
        conditions={"fixed_kb_dpd": 7850.288865935088},
        provenance={
            "inference_ready": True,
            "source_datasets": [{"path": "/fixture/fit-primary.json", "sha256": "b" * 64}],
        },
        validation={
            "label_admission_policy": FIT_PRIMARY_LABEL_ADMISSION_POLICY,
            "nonmonotone_frequency_policy": NONMONOTONE_FREQUENCY_POLICY,
            "all_diameters_positive_and_monotonic": False,
            "all_dense_grids_finite_positive": True,
            "frequency_labels_monotone_by_diameter": False,
            "nonmonotone_frequency_label_diameters_um": [5.8],
            "production_quality_gate_passed": True,
        },
    )


def test_loader_allows_explicit_fit_primary_dense_positive_nonmonotone_curve() -> None:
    payload = _nonmonotone_bank().to_mapping()

    loaded = DpdFrequencyEmulatorBank.from_mapping(payload)

    assert loaded.validation["frequency_labels_monotone_by_diameter"] is False
    assert loaded.validation["nonmonotone_frequency_label_diameters_um"] == [5.8]
    assert np.all(loaded.emulator_for_diameter(5.8).predict_mhz([500.0, 2000.0]) > 0.0)


def test_nonmonotone_frequency_compatibility_reports_all_supported_branches() -> None:
    emulator = _nonmonotone_bank().emulator_for_diameter(5.8)

    roots = emulator.frequency_roots_ka_dpd(0.55)
    branches = emulator.frequency_compatible_branches_ka_dpd([0.50, 0.60])

    assert len(roots) == 2
    assert len(branches) == 2
    assert all(emulator.ka_bounds_dpd[0] <= value <= emulator.ka_bounds_dpd[1] for value in roots)
    assert all(lower < upper for lower, upper in branches)


def test_loader_rejects_false_monotone_declaration_for_nonmonotone_curve() -> None:
    payload = _nonmonotone_bank().to_mapping()
    payload["emulators"][0]["validation"]["frequency_labels_monotone"] = True
    payload["validation"]["frequency_labels_monotone_by_diameter"] = True
    payload["validation"]["nonmonotone_frequency_label_diameters_um"] = []
    payload["validation"]["all_diameters_positive_and_monotonic"] = True
    _refresh_integrity(payload)

    with pytest.raises(ValueError, match="contradicts its PCHIP knots"):
        DpdFrequencyEmulatorBank.from_mapping(payload)


def test_loader_rejects_missing_source_hashes_and_dense_positive_failure() -> None:
    payload = _nonmonotone_bank().to_mapping()
    payload["provenance"]["source_datasets"] = []
    _refresh_integrity(payload)
    with pytest.raises(ValueError, match="hashed source datasets"):
        DpdFrequencyEmulatorBank.from_mapping(payload)

    payload = _nonmonotone_bank().to_mapping()
    payload["emulators"][0]["pchip"]["interval_coefficients"][0] = [1.0, -0.01, 0.0, 0.0]
    _refresh_integrity(payload)
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        DpdFrequencyEmulatorBank.from_mapping(payload)


def test_fit_serializes_scipy_pchip_coefficients_for_frequency_squared() -> None:
    ka = np.asarray([500.0, 8000.0, 21000.0, 42000.0], dtype=np.float64)
    frequency = np.asarray([2.0, 2.7, 3.4, 4.1], dtype=np.float64)

    fitted = fit_frequency_emulator(
        diameter_um=3.4,
        ka_dpd=ka,
        frequency_mhz=frequency,
        ka_bounds_dpd=(500.0, 42000.0),
    )
    reference = PchipInterpolator(ka, frequency**2, extrapolate=False)
    query = np.asarray([500.0, 6000.0, 18000.0, 42000.0])

    assert fitted.predict_mhz(ka) == pytest.approx(frequency, abs=1.0e-12)
    assert fitted.predict_frequency_squared_mhz2(query) == pytest.approx(reference(query))
    assert fitted.knots_ka_dpd.flags.writeable is False
    assert fitted.interval_coefficients.flags.writeable is False


def test_leave_one_ka_out_excludes_endpoints() -> None:
    ka = np.asarray([500.0, 8000.0, 21000.0, 42000.0], dtype=np.float64)
    frequency = np.asarray([2.0, 2.7, 3.4, 4.1], dtype=np.float64)

    metrics = leave_one_ka_out_metrics(
        diameter_um=3.2,
        ka_dpd=ka,
        frequency_mhz=frequency,
        ka_bounds_dpd=(500.0, 42000.0),
    )

    assert [metric.held_out_ka_dpd for metric in metrics] == pytest.approx([8000.0, 21000.0])
    assert all(metric.point_count == 1 for metric in metrics)
    assert all(np.isfinite(metric.rmse_mhz) for metric in metrics)


def test_hbi_frequency_callbacks_keep_scalar_and_batch_shapes(tmp_path: Path) -> None:
    path = _bank().write(tmp_path / "sonovue_frequency_emulator_bank.json")
    config = {
        "resonance": {
            "agent": "sonovue",
            "evaluator": {
                "mode": "artifact_emulator_bank",
                "artifact_path": str(path),
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "expected_fixed_kb_dpd": 7850.288865935088,
            },
        }
    }
    emb_resonance._load_frequency_emulator_bank.cache_clear()

    scalar = emb_resonance.predict_resonance_frequencies_mhz(
        [3.2, 5.8],
        ka_dpd=10000.0,
        config=config,
        project_root=tmp_path,
    )
    batch = emb_resonance.predict_resonance_frequencies_mhz_batch(
        [3.2, 5.8],
        ka_dpd=np.asarray([10000.0, 20000.0, 30000.0]),
        config=config,
        project_root=tmp_path,
    )

    assert isinstance(scalar, list)
    assert len(scalar) == 2
    assert batch.shape == (3, 2)
    assert batch[0, :] == pytest.approx(scalar)
