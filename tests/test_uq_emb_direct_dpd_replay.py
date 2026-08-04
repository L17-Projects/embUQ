from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/workflows/emb/uq_emb/materialize_direct_dpd_replay.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("uq_emb_direct_dpd_replay", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _accepted_root(tmp_path: Path) -> Path:
    root = tmp_path / "accepted"
    sonovue_cases = []
    for index, (bubble, diameter) in enumerate((("d4", 3.2), ("d5", 3.4), ("d6", 5.8)), start=1):
        symbol = f"sonovue_{diameter:.2f}um".replace(".", "p")
        state_path = root / f"sonovue/hbi/results_phase_3b/indentation_{diameter:.1f}um/genLatest.json"
        _write_json(state_path, {"bubble": bubble})
        acoustic_map = {
            "agent": "SonoVue", "symbol": symbol, "modality": "indentation",
            "dataset": f"indentation_{diameter:.1f}um", "diameter_um": diameter,
            "radius_dpd": diameter * 2.0, "radius_source": "established_breathing_protocol",
            "source_label": "accepted_sonovue_50k", "sample_index": index,
            "sample_count": 50000, "ka": 1000.0 + index, "kb": 2000.0 + index,
            "d0": 0.1, "sigma": 0.04, "legacy_yt": 10000.0,
            "log_likelihood": 2.0, "log_prior": -1.0, "log_posterior": 1.0,
            "state_path": "/historical/state",
        }
        map_payload = {
            "diameter_um": diameter,
            "parameters": [1000.0 + index, 2000.0 + index, 0.1],
            "parameter_names": ["Yt", "kb", "d0"],
            "sample_id": index,
            "logPosterior": 1.0,
            "source": {"sigma": 0.04, "logLikelihood": 2.0, "logPrior": -1.0, "state_path": "/accepted/state"},
        }
        _write_json(root / f"sonovue/direct_dpd/inputs/force_spectroscopy/indentation_{diameter:.1f}um_map.json", map_payload)
        _write_csv(root / f"sonovue/direct_dpd/inputs/breathing_frequency/{symbol}.csv", {"symbol": symbol, "ka": 1.0})
        sonovue_cases.append({"diameter_um": diameter, "symbol": symbol})
        _write_json(
            root / f"sonovue/direct_dpd/acoustic/{bubble}/manifest.json",
            {"results": [{"symbol": symbol, "map": acoustic_map, "setup_protocol": {
                "dt": 0.0001, "equil_steps": 10000, "relax_steps": 120000, "sample_every": 150,
                "trajectory_capture": "particle-dump", "excitation_mode": "prestrain", "initial_radius_scale": 0.95,
                "post_deflation_ramp_steps": 500, "post_deflation_hold_steps": 500,
                "post_deflation_hold_update_every_steps": 10, "solvent_mode": "full", "water_shell_fsi_scale": 0.4,
                "membrane_mass_scale": 160.0, "lim_mu_policy": "derive-from-ka", "primary_observable": "rms_radius_dpd", "fit_end_dpd": 0.25,
            }}]},
        )
        _write_json(
            root / f"sonovue/direct_dpd/mechanical/{bubble}/provenance.json",
            {"protocol": {
                "mpi_ranks": 2, "force_grid_points": 15,
                "force_grid_extension_fraction": 0.1, "production_steps": 20000,
                "equilibration_steps": 40000, "retry_attempt": 0,
            }},
        )
    _write_json(root / "sonovue/direct_dpd/inputs/manifest.json", {"cases": sonovue_cases})
    for bubble, diameter in (("d1", 2.1), ("d2", 2.9), ("d3", 3.0)):
        dataset = f"compression_{diameter:.1f}um"
        state_path = root / f"definity/hbi/results_phase_3b/{dataset}/genLatest.json"
        _write_json(state_path, {"bubble": bubble})
        _write_json(
            root / f"definity/direct_dpd/mechanical/{bubble}/map_workflow/map_phase3b/phase3b_map_manifest.json",
            {"experiment": "compression", "datasets": {dataset: {
                "Yt": 1000.0, "kb": 300.0, "d0": 0.1, "sigma": 0.04,
                "logLikelihood": 2.0, "logPrior": -1.0, "logPosterior": 1.0,
                "diameter_um": diameter, "run_dir": "/accepted/state", "output_csv": "",
            }}},
        )
        _write_json(
            root / f"definity/direct_dpd/mechanical/{bubble}/map_workflow/map_mirheo/map_mirheo_manifest.json",
            {"n_displacements": 15, "mpi_ranks": 2, "timeout_seconds": 1800, "max_retries": 1},
        )
        acoustic = root / f"definity/direct_dpd/acoustic/{bubble}"
        symbol = f"definity_{diameter:.1f}um".replace(".", "p")
        setup = {
            "bubble": {
                "agent": "Definity", "symbol": symbol, "modality": "compression",
                "dataset": dataset, "diameter_um": diameter, "radius_dpd": diameter * 2.0,
                "radius_source": "frozen_diameter_um_over_0p5",
                "source_label": "attempt081_seed00_50k", "sample_index": 1,
                "sample_count": 50000, "ka": 1000.0, "kb": 389.2, "d0": 0.1,
                "sigma": 0.04, "legacy_yt": 10000.0, "log_likelihood": 2.0,
                "log_prior": -1.0, "log_posterior": 1.0, "state_path": "/historical/state",
            },
            "protocol": {
                "dt": 0.0001, "equil_steps": 40000, "relax_steps": 120000,
                "sample_every": 150, "trajectory_capture": "particle-dump",
                "excitation_mode": "prestrain", "initial_radius_scale": 0.95,
                "post_deflation_ramp_steps": 500, "post_deflation_hold_steps": 5000,
                "post_deflation_hold_update_every_steps": 10, "solvent_mode": "full",
                "water_shell_fsi_scale": 0.4, "membrane_mass_scale": 160.0,
                "lim_mu_policy": "derive-from-ka", "primary_observable": "rms_radius_dpd",
                "fit_end_dpd": None, "particle_checker_every": 100,
            },
        }
        setup_path = (
            acoustic / "fullfluid_campaign/production/definity" / symbol
            / "ka-index-000" / symbol / "seed-000/setup_manifest.json"
        )
        if bubble == "d2":
            setup_path = acoustic / "../d2_near_map_0p1482pct/setup_manifest.json"
        _write_json(setup_path.resolve(), setup)
    return root


def _accepted_manifest(tmp_path: Path, root: Path) -> Path:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "accepted.files.json"
    _write_json(
        manifest,
        {
            "schema_version": "1.0",
            "paper_id": "UQ_EMB",
            "artifact_set_id": "accepted-production-outputs-202607",
            "artifact_set_dir": root.name,
            "locked": True,
            "file_count": len(files),
            "logical_size_bytes": sum(item["size_bytes"] for item in files),
            "files": files,
        },
    )
    return manifest


def test_materialize_direct_dpd_replay_writes_six_static_pairs(tmp_path: Path) -> None:
    module = _module()
    accepted_root = _accepted_root(tmp_path)
    accepted_manifest = _accepted_manifest(tmp_path, accepted_root)
    plan = module.materialize_direct_dpd_replay(
        accepted_root=accepted_root,
        accepted_manifest=accepted_manifest,
        output_root=tmp_path / "replay",
        site="karolina",
        python_bin="/usr/bin/python3.11",
    )

    assert plan["mode"] == "static_dry_run_only"
    assert plan["submission"] == "not performed"
    assert len(plan["bubbles"]) == 6
    assert plan["runtime_activation"]["required_before_execution"] is True
    assert "mesouq_activate_site_env karolina" in plan["runtime_activation"]["shared_activation"]
    assert plan["runtime_source_hashes"]
    assert plan["accepted_artifact_manifest_sha256"] == hashlib.sha256(
        accepted_manifest.read_bytes()
    ).hexdigest()
    assert (tmp_path / "replay/direct_dpd_replay_plan.json").is_file()
    d1 = plan["bubbles"][0]
    assert d1["mechanical"]["command"][0] == "/usr/bin/python3.11"
    assert d1["mechanical"]["command"][d1["mechanical"]["command"].index("--profile") + 1] == "validation"
    assert d1["acoustic"]["status"] == "reconstructed_exact_protocol_command"
    assert "--particle-dump-root" in d1["acoustic"]["command"]
    d2 = plan["bubbles"][1]
    assert d2["acoustic"]["command"] is not None
    assert d2["acoustic"]["status"] == "user_accepted_near_map_reconstructed_exact_protocol_command"
    d4 = plan["bubbles"][3]
    assert d4["acoustic"]["status"] == "reconstructed_exact_protocol_command"
    assert "--trajectory-capture" in d4["acoustic"]["command"]
    assert "--extend-range" in d4["mechanical"]["command"]
    assert (tmp_path / "replay/d4/acoustic/inputs/sonovue_3p20um.csv").is_file()


def test_materializer_rejects_conflicting_site_environment(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setenv("MESOUQ_SITE", "vega")
    try:
        module._resolve_site("karolina")
    except ValueError as exc:
        assert "Conflicting site selectors" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected conflicting site selectors to fail")


def test_materializer_requires_explicit_site(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    monkeypatch.delenv("MESOUQ_SITE", raising=False)

    try:
        module._resolve_site(None)
    except ValueError as exc:
        assert "Missing site selector" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected a missing site selector to fail")


def test_materializer_rejects_source_mutated_after_manifest(tmp_path: Path) -> None:
    module = _module()
    accepted_root = _accepted_root(tmp_path)
    accepted_manifest = _accepted_manifest(tmp_path, accepted_root)
    source = accepted_root / "sonovue/direct_dpd/inputs/manifest.json"
    source.write_text("{}\n", encoding="utf-8")

    try:
        module.materialize_direct_dpd_replay(
            accepted_root=accepted_root,
            accepted_manifest=accepted_manifest,
            output_root=tmp_path / "replay",
            site="karolina",
            python_bin="/usr/bin/python3.11",
        )
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected a mutated accepted source to fail")
