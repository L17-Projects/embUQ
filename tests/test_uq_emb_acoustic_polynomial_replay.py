from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPLAY_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/workflows/emb/uq_emb/replay_acoustic_polynomial_surrogates.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "uq_emb_acoustic_polynomial_replay", REPLAY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _labels() -> list[dict[str, object]]:
    ka = np.linspace(100.0, 3200.0, 32)
    coefficients = np.asarray([1.0, 0.003, 2.0e-6])
    frequency = np.sqrt(np.polynomial.polynomial.polyval(ka, coefficients))
    return [
        {
            "agent": "sonovue",
            "diameter_um": 2.6,
            "ka_dpd": float(value),
            "frequency_mhz": float(response),
        }
        for value, response in zip(ka, frequency, strict=True)
    ]


def _write_artifacts(module, root: Path, labels: list[dict[str, object]]) -> None:
    physical = root / "physical_labels"
    fits = root / "polynomial_fits"
    banks = root / "frozen_banks"
    physical.mkdir(parents=True)
    fits.mkdir()
    banks.mkdir()

    with (physical / "physical_resonance_labels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "agent",
                "diameter_um",
                "ka_dpd",
                "mass_corrected_driven_peak_frequency_mhz",
            ],
        )
        writer.writeheader()
        for row in labels:
            writer.writerow(
                {
                    "agent": row["agent"],
                    "diameter_um": row["diameter_um"],
                    "ka_dpd": row["ka_dpd"],
                    "mass_corrected_driven_peak_frequency_mhz": row["frequency_mhz"],
                }
            )

    replay = module.fit_replay(labels)[0]
    with (fits / "free_intercept_squared_frequency_fits.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "agent",
                "diameter_um",
                "response",
                "degree",
                "fit_point_count",
                "held_low_ka_point_count",
                "a0_mhz2",
                "a1_mhz2_per_dpd",
                "a2_mhz2_per_dpd2",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                key: replay[key]
                for key in (
                    "agent",
                    "diameter_um",
                    "response",
                    "degree",
                    "fit_point_count",
                    "held_low_ka_point_count",
                    "a0_mhz2",
                    "a1_mhz2_per_dpd",
                    "a2_mhz2_per_dpd2",
                )
            }
        )

    bank = {
        "schema": "meso_uq.emb_dpd_frequency_polynomial_bank.v1",
        "agent": "sonovue",
        "emulators": [
            {
                "diameter_um": replay["diameter_um"],
                "ka_bounds_dpd": [0.0, replay["evaluation_ka_max_dpd"]],
                "polynomial": {
                    "response_encoding": "frequency_mhz_squared",
                    "degree": 2,
                    "coefficients": [
                        replay["a0_mhz2"],
                        replay["a1_mhz2_per_dpd"],
                        replay["a2_mhz2_per_dpd2"],
                    ]
                },
            }
        ],
    }
    (banks / "sonovue_approved_free_intercept_squared_frequency_bank.json").write_text(
        json.dumps(bank), encoding="utf-8"
    )


def test_fit_replays_known_free_intercept_squared_frequency_family() -> None:
    module = _module()
    result = module.fit_replay(_labels())

    assert len(result) == 1
    row = result[0]
    assert row["fit_point_count"] == 28
    assert row["held_low_ka_point_count"] == 4
    assert row["response"] == "frequency_mhz_squared"
    assert np.allclose(
        [row["a0_mhz2"], row["a1_mhz2_per_dpd"], row["a2_mhz2_per_dpd2"]],
        [1.0, 0.003, 2.0e-6],
        rtol=1.0e-9,
        atol=1.0e-11,
    )


def test_replay_materializes_bounded_receipt_and_banks(tmp_path: Path) -> None:
    module = _module()
    labels = _labels()
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, labels)

    report = module.replay(artifact_root=artifact_root, output_dir=tmp_path / "output")

    assert report["status"] == "PASS"
    assert (tmp_path / "output/replayed_free_intercept_squared_frequency_fits.csv").is_file()
    assert (tmp_path / "output/replayed_sonovue_polynomial_bank.json").is_file()
    saved = json.loads(
        (tmp_path / "output/acoustic_polynomial_replay_report.json").read_text()
    )
    assert saved["comparison"]["replayed_diameter_count"] == 1
    assert saved["provenance"]["policy"]["fallback_allowed"] is False


def test_replay_fails_clearly_on_frozen_bank_tolerance_mismatch(tmp_path: Path) -> None:
    module = _module()
    labels = _labels()
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, labels)
    bank_path = (
        artifact_root
        / "frozen_banks/sonovue_approved_free_intercept_squared_frequency_bank.json"
    )
    bank = json.loads(bank_path.read_text())
    bank["emulators"][0]["polynomial"]["coefficients"][1] *= 1.5
    bank_path.write_text(json.dumps(bank), encoding="utf-8")

    with pytest.raises(module.ReplayMismatchError, match="Tolerance mismatch"):
        module.replay(artifact_root=artifact_root, dry_run=True)
