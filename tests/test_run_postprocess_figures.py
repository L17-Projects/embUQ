from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_postprocess_figures_stops_on_comparison_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        Path("scripts/platforms/vega/run_postprocess_figures.py"),
        "run_postprocess_figures_failfast_test",
    )

    calls: list[str] = []

    def fake_run(cmd, label):  # noqa: ANN001
        del cmd
        calls.append(label)
        if label == "Family comparison + BNN decision":
            return 1
        return 0

    monkeypatch.setattr(module, "_run", fake_run)

    rc = module.main(["--run-root", str(tmp_path)])

    assert rc == 1
    assert calls == [
        "Holdout L2 figure",
        "Sensitivity figure",
        "Family comparison + BNN decision",
    ]


def test_run_postprocess_figures_runs_all_steps_on_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module(
        Path("scripts/platforms/vega/run_postprocess_figures.py"),
        "run_postprocess_figures_success_test",
    )

    calls: list[str] = []

    def fake_run(cmd, label):  # noqa: ANN001
        del cmd
        calls.append(label)
        return 0

    monkeypatch.setattr(module, "_run", fake_run)

    rc = module.main(["--run-root", str(tmp_path)])

    assert rc == 0
    assert calls == [
        "Holdout L2 figure",
        "Sensitivity figure",
        "Family comparison + BNN decision",
        "UQ_DPD parity report",
        "Final report",
    ]


def _write_lane_manifest(path: Path) -> None:
    payload = {
        "lane": "compression:full-model:production",
        "stage_statuses": [
            {"stage": "phase1", "status": "passed"},
            {"stage": "phase2", "status": "passed"},
            {"stage": "phase3b", "status": "passed"},
            {"stage": "propagation_phase3b", "status": "passed"},
            {"stage": "map_phase3b", "status": "passed"},
            {"stage": "map_mirheo", "status": "passed"},
        ],
        "artifacts": {
            "phase1_map_manifest": str(path.parent / "phase1_map_manifest.json"),
            "phase3b_map_manifest": str(path.parent / "phase3b_map_manifest.json"),
            "map_mirheo_manifest": str(path.parent / "map_mirheo_manifest.json"),
            "phase3b_propagation_root": str(path.parent / "propagation_phase3b"),
        },
        "job_manifests": [str(path.parent / "job_phase1.json")],
        "policy": {"status": "pass", "violations": []},
    }
    for artifact in payload["artifacts"].values():
        artifact_path = Path(artifact)
        if artifact_path.suffix:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("{}", encoding="utf-8")
        else:
            artifact_path.mkdir(parents=True, exist_ok=True)
    Path(payload["job_manifests"][0]).write_text("{}", encoding="utf-8")
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_required_assets(module, run_root: Path) -> None:
    figures_main = run_root / "figures" / "main"
    figures_supp = run_root / "figures" / "supplementary"
    tables = run_root / "tables"
    figures_main.mkdir(parents=True, exist_ok=True)
    figures_supp.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for name in module.MANDATORY_MAIN_FIGURES:
        (figures_main / name).write_text("main", encoding="utf-8")
    for name in module.MANDATORY_SUPPLEMENTARY_FIGURES:
        (figures_supp / name).write_text("supp", encoding="utf-8")
    for name in module.MANDATORY_TABLES:
        (tables / name).write_text("table", encoding="utf-8")


def test_run_postprocess_figures_emits_release_manifest_and_passes_when_complete(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module(
        Path("scripts/platforms/vega/run_postprocess_figures.py"),
        "run_postprocess_figures_manifest_pass_test",
    )

    def fake_run(cmd, label):  # noqa: ANN001
        del cmd, label
        return 0

    monkeypatch.setattr(module, "_run", fake_run)
    _write_required_assets(module, tmp_path)
    lane_manifest = tmp_path / "manifests" / "lanes" / "emb.compression.json"
    lane_manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_lane_manifest(lane_manifest)

    rc = module.main(
        [
            "--run-root",
            str(tmp_path),
            "--emit-release-manifest",
            "--lane-manifest",
            str(lane_manifest),
        ]
    )
    assert rc == 0
    manifest_path = tmp_path / "manifests" / "paper_release_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["release_status"] == "PASS"


def test_run_postprocess_figures_release_gate_fails_on_missing_assets(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module(
        Path("scripts/platforms/vega/run_postprocess_figures.py"),
        "run_postprocess_figures_manifest_fail_test",
    )

    def fake_run(cmd, label):  # noqa: ANN001
        del cmd, label
        return 0

    monkeypatch.setattr(module, "_run", fake_run)
    lane_manifest = tmp_path / "manifests" / "lanes" / "emb.compression.json"
    lane_manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_lane_manifest(lane_manifest)

    rc = module.main(
        [
            "--run-root",
            str(tmp_path),
            "--emit-release-manifest",
            "--lane-manifest",
            str(lane_manifest),
        ]
    )
    assert rc == 1
    manifest_path = tmp_path / "manifests" / "paper_release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["release_status"] == "FAIL"
    assert payload["hard_failures"]


def test_manifest_only_requires_emit_release_manifest(tmp_path: Path) -> None:
    module = _load_module(
        Path("scripts/platforms/vega/run_postprocess_figures.py"),
        "run_postprocess_figures_manifest_only_validation_test",
    )
    try:
        module.main(["--run-root", str(tmp_path), "--manifest-only"])
    except ValueError as exc:
        assert "--manifest-only requires --emit-release-manifest" in str(exc)
    else:
        raise AssertionError("Expected ValueError for --manifest-only without --emit-release-manifest")
