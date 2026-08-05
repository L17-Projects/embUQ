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


def test_replay_materializes_bounded_receipt_and_banks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
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


@pytest.mark.parametrize("target_kind", ("output", "report"))
def test_replay_rejects_outputs_inside_immutable_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    kwargs = {"artifact_root": artifact_root, "dry_run": target_kind == "report"}
    forbidden = artifact_root / "replay"
    if target_kind == "output":
        kwargs["output_dir"] = forbidden
    else:
        kwargs["report_path"] = forbidden / "report.json"

    with pytest.raises(ValueError, match="outside the immutable artifact root"):
        module.replay(**kwargs)

    assert not forbidden.exists()


def test_replay_rejects_report_collision_with_materialized_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="must not overwrite a replay output"):
        module.replay(
            artifact_root=artifact_root,
            output_dir=output_dir,
            report_path=output_dir
            / "replayed_free_intercept_squared_frequency_fits.csv",
        )

    assert not output_dir.exists()


def test_replay_rejects_report_hardlink_to_locked_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    labels = artifact_root / "physical_labels/physical_resonance_labels.csv"
    original = labels.read_bytes()
    report = tmp_path / "report.json"
    report.hardlink_to(labels)

    with pytest.raises(ValueError, match="existing multi-link file"):
        module.replay(
            artifact_root=artifact_root,
            report_path=report,
            dry_run=True,
        )

    assert labels.read_bytes() == original


def test_replay_rejects_report_hardlink_to_accepted_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    fits = artifact_root / "polynomial_fits/free_intercept_squared_frequency_fits.csv"
    original = fits.read_bytes()
    report = tmp_path / "report.json"
    report.hardlink_to(fits)

    with pytest.raises(ValueError, match="existing multi-link file"):
        module.replay(
            artifact_root=artifact_root,
            report_path=report,
            dry_run=True,
        )

    assert fits.read_bytes() == original


def test_replay_rejects_report_hardlink_to_external_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    external_banks = tmp_path / "external_banks"
    external_banks.mkdir()
    source_bank = next((artifact_root / "frozen_banks").iterdir())
    bank = external_banks / source_bank.name
    bank.write_bytes(source_bank.read_bytes())
    original = bank.read_bytes()
    report = tmp_path / "report.json"
    report.hardlink_to(bank)

    with pytest.raises(ValueError, match="existing multi-link file"):
        module.replay(
            artifact_root=artifact_root,
            accepted_bank_dir=external_banks,
            report_path=report,
            dry_run=True,
        )

    assert bank.read_bytes() == original


def test_replay_rejects_output_inside_external_bank_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    external_banks = tmp_path / "external_banks"
    external_banks.mkdir()
    source_bank = next((artifact_root / "frozen_banks").iterdir())
    (external_banks / source_bank.name).write_bytes(source_bank.read_bytes())
    output_dir = external_banks / "replay"

    with pytest.raises(ValueError, match="accepted bank directory"):
        module.replay(
            artifact_root=artifact_root,
            accepted_bank_dir=external_banks,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_replay_rejects_symlinked_artifact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    alias = tmp_path / "artifact_alias"
    alias.symlink_to(artifact_root, target_is_directory=True)
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="symlinked path or ancestor"):
        module.replay(artifact_root=alias, output_dir=output_dir)

    assert not output_dir.exists()


def test_replay_rejects_parent_traversal_in_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    output_dir = tmp_path / "intermediate" / ".." / "output"

    with pytest.raises(ValueError, match="parent traversal"):
        module.replay(artifact_root=artifact_root, output_dir=output_dir)

    assert not (tmp_path / "output").exists()


def test_replay_rejects_report_as_output_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    report = tmp_path / "out"
    output_dir = report / "replay"

    with pytest.raises(ValueError, match="must not replace the output directory"):
        module.replay(
            artifact_root=artifact_root,
            output_dir=output_dir,
            report_path=report,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("target_kind", ("output", "report"))
def test_replay_rejects_outputs_in_enclosing_locked_root_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    locked_root = tmp_path / "frozen_runtime_dependencies_202607"
    artifact_root = locked_root / "acoustic_surrogates"
    _write_artifacts(module, artifact_root, _labels())
    target = locked_root / "sibling_output"
    kwargs = {"artifact_root": artifact_root, "dry_run": target_kind == "report"}
    if target_kind == "output":
        kwargs["output_dir"] = target
    else:
        kwargs["report_path"] = target / "report.json"

    with pytest.raises(ValueError, match="outside immutable artifact roots"):
        module.replay(**kwargs)

    assert not target.exists()


def test_replay_rejects_existing_directory_as_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    report = tmp_path / "report"
    report.mkdir()

    with pytest.raises(ValueError, match="report path must be a file"):
        module.replay(artifact_root=artifact_root, report_path=report, dry_run=True)


def test_replay_rejects_report_collision_with_generated_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
    artifact_root = tmp_path / "artifacts"
    _write_artifacts(module, artifact_root, _labels())
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="must not overwrite a replay output"):
        module.replay(
            artifact_root=artifact_root,
            output_dir=output_dir,
            report_path=output_dir / "replayed_sonovue_polynomial_bank.json",
        )

    assert not output_dir.exists()


def test_replay_fails_clearly_on_frozen_bank_tolerance_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "replay_receipt_provenance", lambda **_kwargs: {})
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
