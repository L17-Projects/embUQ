from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from meso_uq.inference.emb_frequency_emulator import (
    DENSE_VALIDATION_MIN_POINTS,
    DIAMETER_TOLERANCE_UM,
    DpdPolynomialFrequencyEmulator,
    DpdPolynomialFrequencyEmulatorBank,
    FREQUENCY_RESPONSE_ENCODING,
    FREQUENCY_SQUARED_RESPONSE_ENCODING,
)


def _entry(
    diameter: float,
    *,
    degree: int,
    response: str,
    coefficients: list[float],
) -> DpdPolynomialFrequencyEmulator:
    return DpdPolynomialFrequencyEmulator(
        diameter_um=diameter,
        ka_bounds_dpd=(0.0, 20.0),
        coefficients=coefficients,
        degree=degree,
        response_encoding=response,
        ka_offset_dpd=0.0,
        ka_scale_dpd=20.0,
        provenance={"inference_ready": True},
        validation={
            "dense_grid_point_count": DENSE_VALIDATION_MIN_POINTS,
            "dense_grid_finite_positive": True,
        },
    )


def _bank() -> DpdPolynomialFrequencyEmulatorBank:
    return DpdPolynomialFrequencyEmulatorBank(
        agent="sonovue",
        emulators=[
            _entry(
                2.6,
                degree=1,
                response=FREQUENCY_RESPONSE_ENCODING,
                coefficients=[4.0, 1.0],
            ),
            _entry(
                3.2,
                degree=2,
                response=FREQUENCY_SQUARED_RESPONSE_ENCODING,
                coefficients=[9.0, 2.0, 1.0],
            ),
        ],
        conditions={"fixed_kb_dpd": 7850.288865935088},
        provenance={
            "source_datasets": [{"path": "/fixture/source.csv", "sha256": "a" * 64}],
        },
        validation={"all_dense_grids_finite_positive": True},
    )


def _refresh_integrity(payload: dict) -> None:
    unsigned = dict(payload)
    unsigned.pop("integrity", None)
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload["integrity"] = {
        "algorithm": "sha256",
        "canonical_payload_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def test_mixed_degree_and_response_bank_uses_vectorized_horner_dispatch() -> None:
    bank = _bank()
    ka = np.asarray([[0.0], [10.0], [20.0]])
    diameters = np.asarray([[2.6, 3.2]])

    predicted = bank.predict_for_diameters_mhz(ka, diameters)

    assert predicted.shape == (3, 2)
    assert predicted[:, 0] == pytest.approx([4.0, 4.5, 5.0])
    assert predicted[:, 1] == pytest.approx(np.sqrt([9.0, 10.25, 12.0]))
    assert bank.to_mapping()["runtime_contract"]["population_evaluation"] == (
        "numpy_exact_dispatch_gather_affine_horner"
    )


def test_exact_diameter_and_support_fail_closed_without_clipping_or_fallback() -> None:
    bank = _bank()
    tolerated = bank.predict_for_diameters_mhz(10.0, 2.6 + 0.5 * DIAMETER_TOLERANCE_UM)
    assert float(tolerated) == pytest.approx(4.5)
    with pytest.raises(ValueError, match="No exact polynomial"):
        bank.predict_for_diameters_mhz(10.0, 2.6 + 2.0 * DIAMETER_TOLERANCE_UM)
    with pytest.raises(ValueError, match="outside the supported range"):
        bank.predict_for_diameters_mhz(-1.0, 2.6)
    with pytest.raises(ValueError, match="outside the supported range"):
        bank.predict_for_diameters_mhz(21.0, 2.6)

    values, support = bank.predict_with_support_mhz(
        np.asarray([-1.0, 0.0, 20.0, 21.0]),
        np.asarray([2.6, 2.6, 3.2, 3.2]),
    )
    assert support.tolist() == [False, True, True, False]
    assert values[[0, 3]].tolist() == [0.0, 0.0]
    assert np.all(values[support] > 0.0)


def test_loader_rejects_tamper_fallback_and_nonpositive_polynomial() -> None:
    payload = _bank().to_mapping()
    payload["emulators"][0]["polynomial"]["coefficients"][0] += 1.0
    with pytest.raises(ValueError, match="integrity hash does not match"):
        DpdPolynomialFrequencyEmulatorBank.from_mapping(payload)

    payload = _bank().to_mapping()
    payload["runtime_contract"]["pchip_fallback_allowed"] = True
    _refresh_integrity(payload)
    with pytest.raises(ValueError, match="pchip_fallback_allowed must be false"):
        DpdPolynomialFrequencyEmulatorBank.from_mapping(payload)

    with pytest.raises(ValueError, match="non-positive or non-finite"):
        _entry(
            4.0,
            degree=1,
            response=FREQUENCY_RESPONSE_ENCODING,
            coefficients=[1.0, -2.0],
        )


def test_loader_allows_benchmark_polynomial_bank_without_production_source_hashes() -> None:
    payload = _bank().to_mapping()
    payload["provenance"] = {"benchmark_only": True}
    payload["validation"]["deterministic_fixture_gate_passed"] = True
    _refresh_integrity(payload)

    loaded = DpdPolynomialFrequencyEmulatorBank.from_mapping(payload)

    assert loaded.provenance["benchmark_only"] is True


def test_degree_two_frequency_squared_roots_use_raw_response_before_sqrt() -> None:
    emulator = DpdPolynomialFrequencyEmulator(
        diameter_um=1.3,
        ka_bounds_dpd=(0.0, 10.0),
        coefficients=[1.0, 2.0, 1.0],
        degree=2,
        response_encoding=FREQUENCY_SQUARED_RESPONSE_ENCODING,
        ka_offset_dpd=0.0,
        ka_scale_dpd=1.0,
    )

    assert emulator.frequency_roots_ka_dpd(3.0) == pytest.approx((2.0,))
    assert float(emulator.predict_mhz(2.0)) == pytest.approx(3.0)
