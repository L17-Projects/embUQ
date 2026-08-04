from __future__ import annotations

import csv
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
            {"results": [{"symbol": symbol, "setup_protocol": {
                "dt": 0.0001, "equil_steps": 10000, "relax_steps": 120000, "sample_every": 150,
                "trajectory_capture": "particle-dump", "excitation_mode": "prestrain", "initial_radius_scale": 0.95,
                "post_deflation_ramp_steps": 500, "post_deflation_hold_steps": 500,
                "post_deflation_hold_update_every_steps": 10, "solvent_mode": "full", "water_shell_fsi_scale": 0.4,
                "membrane_mass_scale": 160.0, "lim_mu_policy": "derive-from-ka", "primary_observable": "rms_radius_dpd", "fit_end_dpd": 0.25,
            }}]},
        )
    _write_json(root / "sonovue/direct_dpd/inputs/manifest.json", {"cases": sonovue_cases})
    for bubble, diameter in (("d1", 2.1), ("d2", 2.9), ("d3", 3.0)):
        dataset = f"compression_{diameter:.1f}um"
        _write_json(
            root / f"definity/direct_dpd/mechanical/{bubble}/map_workflow/map_phase3b/phase3b_map_manifest.json",
            {"experiment": "compression", "datasets": {dataset: {
                "Yt": 1000.0, "kb": 300.0, "d0": 0.1, "sigma": 0.04,
                "logLikelihood": 2.0, "logPrior": -1.0, "logPosterior": 1.0,
                "diameter_um": diameter, "run_dir": "/accepted/state", "output_csv": "",
            }}},
        )
        acoustic = root / f"definity/direct_dpd/acoustic/{bubble}"
        if bubble == "d2":
            _write_json(
                root / "definity/direct_dpd/acoustic/d2_near_map_0p1482pct/setup_manifest.json",
                {"near_map": True},
            )
        else:
            _write_csv(acoustic / "accepted_map_values.csv", {"diameter_um": diameter, "ka": 1000.0})
            _write_csv(acoustic / "fullfluid_campaign/design/production_design.csv", {"phase": "production", "case_index": 0})
    return root


def test_materialize_direct_dpd_replay_writes_six_static_pairs(tmp_path: Path) -> None:
    module = _module()
    plan = module.materialize_direct_dpd_replay(
        accepted_root=_accepted_root(tmp_path),
        output_root=tmp_path / "replay",
        site="karolina",
        python_bin="/usr/bin/python3.11",
    )

    assert plan["mode"] == "static_dry_run_only"
    assert plan["submission"] == "not performed"
    assert len(plan["bubbles"]) == 6
    assert plan["runtime_source_hashes"]
    assert (tmp_path / "replay/direct_dpd_replay_plan.json").is_file()
    d1 = plan["bubbles"][0]
    assert d1["mechanical"]["command"][0] == "/usr/bin/python3.11"
    assert d1["acoustic"]["status"] == "exact_protocol_command"
    assert "--particle-staging-root" in d1["acoustic"]["command"]
    d2 = plan["bubbles"][1]
    assert d2["acoustic"]["command"] is None
    assert d2["acoustic"]["status"] == "provenance_incomplete_near_map_only"
    d4 = plan["bubbles"][3]
    assert d4["acoustic"]["status"] == "candidate_provenance_incomplete"
    assert "--trajectory-capture" in d4["acoustic"]["command"]
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
