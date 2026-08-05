from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from meso_uq.inference import emb_resonance as resonance
from meso_uq.inference.emb_frequency_emulator import (
    DENSE_VALIDATION_MIN_POINTS,
    DpdFrequencyEmulator,
    DpdFrequencyEmulatorBank,
    FIT_PRIMARY_LABEL_ADMISSION_POLICY,
    NONMONOTONE_FREQUENCY_POLICY,
)
from tests.support.korali_reference_normal import (
    korali_reference_normal_loglikelihood,
    read_vendored_reference_normal_source,
)


def _heterogeneous_sonovue_bank(path: Path) -> DpdFrequencyEmulatorBank:
    emulators = []
    specs = (
        (2.6, (24139.240506329115, 42000.0), (2.6, 3.8)),
        (3.2, (17698.49246231156, 42000.0), (1.55, 3.3)),
        (4.0, (17819.095477386934, 42000.0), (1.2, 2.4)),
    )
    for diameter, bounds, frequencies in specs:
        lower, upper = bounds
        lower_f, upper_f = frequencies
        emulators.append(
            DpdFrequencyEmulator(
                diameter_um=diameter,
                ka_bounds_dpd=bounds,
                coefficients=np.asarray(
                    [lower_f * lower_f, upper_f * upper_f - lower_f * lower_f]
                ),
                degree=1,
                provenance={"inference_ready": False, "bank_side_release_ready": True},
                validation={
                    "label_admission_policy": FIT_PRIMARY_LABEL_ADMISSION_POLICY,
                    "fit_primary_gate_passed": True,
                    "frequency_labels_monotone": True,
                    "nonmonotone_frequency_policy": NONMONOTONE_FREQUENCY_POLICY,
                    "dense_grid_point_count": DENSE_VALIDATION_MIN_POINTS,
                    "dense_grid_finite_positive": True,
                },
            )
        )
    bank = DpdFrequencyEmulatorBank(
        agent="sonovue",
        emulators=emulators,
        conditions={
            "fixed_kb_dpd": 7850.288865935088,
            "frequency_label": "mass_corrected_driven_peak_frequency_mhz",
        },
        provenance={
            "inference_ready": False,
            "bank_side_release_ready": True,
            "source_datasets": [
                {"path": "/fixture/heterogeneous-sonovue.json", "sha256": "b" * 64}
            ],
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
    bank.write(path)
    return bank


def _bank_config(path: Path) -> dict:
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
    report = path.with_suffix(".report.json")
    report.write_text('{"status": "bank_side_passed"}\n', encoding="utf-8")
    gate = path.with_suffix(".independent_go.json")
    gate.write_text(
        json.dumps(
            {
                "schema": resonance.PCHIP_INDEPENDENT_GO_SCHEMA,
                "status": "GO",
                "independent_audit": True,
                "agent": "sonovue",
                "bank_artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bank_build_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "gates": {
                    name: True for name in resonance.REQUIRED_PCHIP_INDEPENDENT_GO_GATES
                },
            }
        ),
        encoding="utf-8",
    )
    config["resonance"]["evaluator"].update(
        {
            "bank_build_report_path": str(report),
            "bank_build_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "independent_go_path": str(gate),
            "independent_go_sha256": hashlib.sha256(gate.read_bytes()).hexdigest(),
        }
    )
    return config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_runtime_config_cache_follows_environment_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = tmp_path / "first.yaml"
    second_path = tmp_path / "second.yaml"
    first_path.write_text("marker: first\n", encoding="utf-8")
    second_path.write_text("marker: second\n", encoding="utf-8")
    resonance._load_runtime_config.cache_clear()

    monkeypatch.setenv("HUQ_INFERENCE_CONFIG", str(first_path))
    assert resonance._load_runtime_config()["marker"] == "first"

    monkeypatch.setenv("HUQ_INFERENCE_CONFIG", str(second_path))
    assert resonance._load_runtime_config()["marker"] == "second"


@pytest.mark.parametrize(
    ("diameter_um", "unsupported_ka_dpd", "observed_frequency_mhz"),
    (
        (2.6, 23528.7275961778, 3.1),
        (3.2, 17000.0, 2.1),
        (4.0, 17750.0, 1.6),
    ),
)
def test_emulator_bank_hard_invalid_rows_are_negative_infinity_in_reference_likelihood(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diameter_um: float,
    unsupported_ka_dpd: float,
    observed_frequency_mhz: float,
) -> None:
    read_vendored_reference_normal_source(_repo_root())
    artifact_path = tmp_path / "heterogeneous_sonovue_bank.json"
    _heterogeneous_sonovue_bank(artifact_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_bank_config(artifact_path)), encoding="utf-8")
    monkeypatch.setenv("HUQ_INFERENCE_CONFIG", str(config_path))
    resonance._load_runtime_config.cache_clear()
    resonance._load_frequency_emulator_bank.cache_clear()

    sample = {"Parameters": [unsupported_ka_dpd, 7850.0, 0.0, 0.1]}
    batch = {
        "Batch Parameters": [
            [unsupported_ka_dpd, 7850.0, 0.0, 0.1],
            [26000.0, 7850.0, 0.0, 0.1],
        ]
    }

    resonance.compute_emb_resonance(sample, [diameter_um], diameter_um)
    resonance.compute_emb_resonance_batch(batch, [diameter_um], diameter_um)

    assert sample["Reference Evaluations"] == pytest.approx(
        batch["Batch Reference Evaluations"][0]
    )
    assert sample["Standard Deviation"] == pytest.approx(batch["Batch Standard Deviation"][0])
    assert sample["Standard Deviation"] == pytest.approx([0.0])

    actual_observation_likelihood = korali_reference_normal_loglikelihood(
        [observed_frequency_mhz],
        [sample["Reference Evaluations"][0]],
        [sample["Standard Deviation"][0]],
    )
    zero_match_likelihood = korali_reference_normal_loglikelihood(
        [0.0],
        [sample["Reference Evaluations"][0]],
        [sample["Standard Deviation"][0]],
    )

    assert actual_observation_likelihood == -math.inf
    assert math.isnan(zero_match_likelihood)


def test_emulator_bank_3p2_root_stays_finite_in_reference_likelihood(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_vendored_reference_normal_source(_repo_root())
    artifact_path = tmp_path / "heterogeneous_sonovue_bank.json"
    _heterogeneous_sonovue_bank(artifact_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_bank_config(artifact_path)), encoding="utf-8")
    monkeypatch.setenv("HUQ_INFERENCE_CONFIG", str(config_path))
    resonance._load_runtime_config.cache_clear()
    resonance._load_frequency_emulator_bank.cache_clear()

    ka_3p2_root = 23528.7275961778
    sample = {"Parameters": [ka_3p2_root, 7850.0, 0.0, 0.1]}
    batch = {
        "Batch Parameters": [
            [ka_3p2_root, 7850.0, 0.0, 0.1],
            [30000.0, 7850.0, 0.0, 0.1],
        ]
    }

    resonance.compute_emb_resonance(sample, [2.6, 3.2, 4.0], 2.6)
    resonance.compute_emb_resonance_batch(batch, [2.6, 3.2, 4.0], 2.6)

    assert sample["Reference Evaluations"] == pytest.approx(
        batch["Batch Reference Evaluations"][0]
    )
    assert sample["Standard Deviation"] == pytest.approx(batch["Batch Standard Deviation"][0])
    assert sample["Standard Deviation"][0] == pytest.approx(0.0)
    assert sample["Standard Deviation"][1] > 0.0
    assert sample["Standard Deviation"][2] > 0.0

    supported_likelihood = korali_reference_normal_loglikelihood(
        [2.1],
        [sample["Reference Evaluations"][1]],
        [sample["Standard Deviation"][1]],
    )
    unsupported_likelihood = korali_reference_normal_loglikelihood(
        [3.1],
        [sample["Reference Evaluations"][0]],
        [sample["Standard Deviation"][0]],
    )

    assert math.isfinite(supported_likelihood)
    assert unsupported_likelihood == -math.inf
