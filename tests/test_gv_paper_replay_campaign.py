from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from meso_uq.structures.gv.paper_replay import (
    MANIFEST_SCHEMA_VERSION,
    GVPaperReplayCampaignManifest,
    GVPaperReplayComparisonPacket,
    GVPaperReplayDataRange,
    GVPaperReplayFiniteCheck,
    GVPaperReplayGitHead,
    GVPaperReplayLaneRecord,
    build_comparison_packet_skeletons,
    build_fixture_lane_record,
    load_lane_record_fixture,
    resolve_campaign_root,
    validate_campaign_manifest,
    write_campaign_manifest,
)
from meso_uq.structures.gv.paper_replay import campaign as campaign_module


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "workflows" / "gv" / "run_paper_figure_replay.py"


def _load_script_module(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _campaign_root(tmp_path: Path, campaign_id: str = "paper-fixture") -> Path:
    return tmp_path / "_runs" / "gv" / "figure_replay" / campaign_id


def _source_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "sources" / "figure_reference.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("%PDF-1.4 fixture\n", encoding="utf-8")
    return path


def _lane_record(tmp_path: Path) -> GVPaperReplayLaneRecord:
    return build_fixture_lane_record(
        repo_root=tmp_path,
        campaign_root=_campaign_root(tmp_path),
        lane="stretching",
        source_pdfs=(_source_pdf(tmp_path),),
        fixture_mode=True,
    )


def test_manifest_schema_and_write_round_trip(tmp_path: Path) -> None:
    lane = _lane_record(tmp_path)
    manifest = GVPaperReplayCampaignManifest(
        campaign_id="paper-fixture",
        campaign_root=_campaign_root(tmp_path),
        generated_at_utc="2026-05-05T12:00:00+00:00",
        schema_version=MANIFEST_SCHEMA_VERSION,
        git_head=GVPaperReplayGitHead(commit="deadbeef", branch="feature/test", dirty_worktree=False),
        dry_run=False,
        fixture_mode=True,
        source_pdfs=lane.source_pdfs,
        lanes=(lane,),
        comparison_packets=build_comparison_packet_skeletons((lane,)),
    )

    manifest_path = write_campaign_manifest(manifest=manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert payload["campaign_id"] == "paper-fixture"
    assert payload["git_head"]["commit"] == "deadbeef"
    assert payload["lanes"][0]["material_parameters"]["a3"] == 0.0
    assert payload["validation"]["status"] == "passed"


def test_finite_validation_rejects_non_finite_data_ranges() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        GVPaperReplayDataRange(name="response_y", minimum=0.0, maximum=float("nan"))

    with pytest.raises(ValueError, match="must be finite"):
        GVPaperReplayFiniteCheck(
            name="response_channels",
            passed=False,
            finite_ratio=float("inf"),
            nonfinite_count=1,
        )


def test_campaign_low_level_validators_reject_invalid_inputs(tmp_path: Path) -> None:
    assert campaign_module._run_git(tmp_path, "status") is None
    assert campaign_module.collect_git_head(REPO_ROOT).branch

    with pytest.raises(ValueError, match="non-empty"):
        campaign_module._normalize_identifier("", field_name="lane")
    with pytest.raises(ValueError, match="path traversal"):
        campaign_module._normalize_identifier("../bad", field_name="lane")
    with pytest.raises(ValueError, match="finite"):
        campaign_module._ensure_finite(float("nan"), field_name="finite")
    with pytest.raises(ValueError, match="minimum"):
        GVPaperReplayDataRange(name="response", minimum=2.0, maximum=1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        GVPaperReplayFiniteCheck(name="finite", passed=False, finite_ratio=1.5, nonfinite_count=1)
    with pytest.raises(ValueError, match=">= 0"):
        GVPaperReplayFiniteCheck(name="finite", passed=False, finite_ratio=0.5, nonfinite_count=-1)
    with pytest.raises(ValueError, match="PDF"):
        GVPaperReplayLaneRecord(
            lane="stretching",
            experiment="stretching",
            mode="fixture",
            source_pdfs=(tmp_path / "source.txt",),
            runtime_commands=(("python", "run.py"),),
            material_parameters={
                "ka": 1.0,
                "kb": 0.2,
                "mu": 0.5,
                "b1": 0.0,
                "b2": 0.0,
                "a3": 0.0,
                "a4": 0.0,
                "mu_l": 0.4,
                "c": 0.0,
            },
            geometry={"radius": 2.0, "height": 14.28},
            controls={"tot_force": [1.0]},
            output_paths={"summary": _campaign_root(tmp_path) / "lanes" / "stretching" / "summary.json"},
            plot_paths=(_campaign_root(tmp_path) / "lanes" / "stretching" / "preview.txt",),
            validation_status="passed",
            data_ranges=(GVPaperReplayDataRange(name="response_x", minimum=0.0, maximum=1.0),),
            finite_checks=(GVPaperReplayFiniteCheck(name="response_channels", passed=True, finite_ratio=1.0, nonfinite_count=0),),
        )
    with pytest.raises(ValueError, match="at least one sweep"):
        GVPaperReplayLaneRecord(
            lane="stretching",
            experiment="stretching",
            mode="fixture",
            source_pdfs=(_source_pdf(tmp_path),),
            runtime_commands=(("python", "run.py"),),
            material_parameters={
                "ka": 1.0,
                "kb": 0.2,
                "mu": 0.5,
                "b1": 0.0,
                "b2": 0.0,
                "a3": 0.0,
                "a4": 0.0,
                "mu_l": 0.4,
                "c": 0.0,
            },
            geometry={"radius": 2.0, "height": 14.28},
            controls={"tot_force": []},
            output_paths={"summary": _campaign_root(tmp_path) / "lanes" / "stretching" / "summary.json"},
            plot_paths=(_campaign_root(tmp_path) / "lanes" / "stretching" / "preview.txt",),
            validation_status="passed",
            data_ranges=(GVPaperReplayDataRange(name="response_x", minimum=0.0, maximum=1.0),),
            finite_checks=(GVPaperReplayFiniteCheck(name="response_channels", passed=True, finite_ratio=1.0, nonfinite_count=0),),
        )


def test_collect_git_head_records_clean_worktree_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {
        ("rev-parse", "HEAD"): "deadbeef\n",
        ("rev-parse", "--abbrev-ref", "HEAD"): "feature/test\n",
        ("status", "--porcelain"): "",
    }

    def fake_run(command, **_kwargs):
        args = tuple(command[1:])
        return SimpleNamespace(returncode=0, stdout=outputs[args])

    monkeypatch.setattr(campaign_module.subprocess, "run", fake_run)

    git_head = campaign_module.collect_git_head(tmp_path)

    assert git_head.commit == "deadbeef"
    assert git_head.branch == "feature/test"
    assert git_head.dirty_worktree is False


def test_campaign_private_helpers_cover_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        campaign_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )
    assert campaign_module._run_git(tmp_path, "status") is None

    class ManifestLike:
        def to_manifest(self):
            return {"path": tmp_path / "artifact.txt"}

    assert campaign_module._manifest_safe(ManifestLike()) == {
        "path": str(tmp_path / "artifact.txt")
    }


def test_output_paths_must_live_under_runs_root(tmp_path: Path) -> None:
    source_pdf = _source_pdf(tmp_path)
    bad_output = tmp_path / "outside" / "summary.json"
    bad_output.parent.mkdir(parents=True, exist_ok=True)
    bad_output.write_text("{}", encoding="utf-8")

    lane = GVPaperReplayLaneRecord(
        lane="stretching",
        experiment="stretching",
        mode="fixture",
        source_pdfs=(source_pdf,),
        runtime_commands=(("python", "run_gv_runtime.py"),),
        material_parameters={
            "ka": 1.0,
            "kb": 0.2,
            "mu": 0.5,
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
            "mu_l": 0.4,
            "c": 0.0,
        },
        geometry={"radius": 2.0, "height": 14.28},
        controls={"tot_force": 1.0, "bpress": 0.0},
        output_paths={"summary": bad_output},
        plot_paths=(_campaign_root(tmp_path) / "lanes" / "stretching" / "plots" / "preview.txt",),
        validation_status="passed",
        data_ranges=(GVPaperReplayDataRange(name="response_x", minimum=0.0, maximum=1.0),),
        finite_checks=(GVPaperReplayFiniteCheck(name="response_channels", passed=True, finite_ratio=1.0, nonfinite_count=0),),
    )
    manifest = GVPaperReplayCampaignManifest(
        campaign_id="paper-fixture",
        campaign_root=_campaign_root(tmp_path),
        generated_at_utc="2026-05-05T12:00:00+00:00",
        schema_version=MANIFEST_SCHEMA_VERSION,
        git_head=GVPaperReplayGitHead(commit="deadbeef", branch="feature/test", dirty_worktree=False),
        dry_run=False,
        fixture_mode=True,
        source_pdfs=(source_pdf,),
        lanes=(lane,),
        comparison_packets=(),
    )

    with pytest.raises(ValueError, match="must live under"):
        validate_campaign_manifest(manifest)


def test_campaign_manifest_validation_reports_duplicate_and_finite_failures(tmp_path: Path) -> None:
    lane = _lane_record(tmp_path)
    failing_lane = GVPaperReplayLaneRecord(
        lane="stretching",
        experiment="stretching",
        mode="fixture",
        source_pdfs=lane.source_pdfs,
        runtime_commands=(("python", "run.py"),),
        material_parameters=lane.material_parameters,
        geometry=lane.geometry,
        controls=lane.controls,
        output_paths=lane.output_paths,
        plot_paths=lane.plot_paths,
        validation_status="passed",
        data_ranges=lane.data_ranges,
        finite_checks=(GVPaperReplayFiniteCheck(name="response_channels", passed=False, finite_ratio=0.5, nonfinite_count=1),),
    )
    duplicate_manifest = GVPaperReplayCampaignManifest(
        campaign_id="paper-fixture",
        campaign_root=_campaign_root(tmp_path),
        generated_at_utc="2026-05-05T12:00:00+00:00",
        schema_version=MANIFEST_SCHEMA_VERSION,
        git_head=GVPaperReplayGitHead(commit="deadbeef", branch="feature/test", dirty_worktree=False),
        dry_run=False,
        fixture_mode=True,
        source_pdfs=lane.source_pdfs,
        lanes=(lane, failing_lane),
        comparison_packets=(),
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_campaign_manifest(duplicate_manifest)

    missing_pdf = tmp_path / "sources" / "missing.pdf"
    missing_pdf.parent.mkdir(parents=True, exist_ok=True)
    failed_manifest = GVPaperReplayCampaignManifest(
        campaign_id="paper-fixture",
        campaign_root=_campaign_root(tmp_path),
        generated_at_utc="2026-05-05T12:00:00+00:00",
        schema_version=MANIFEST_SCHEMA_VERSION,
        git_head=GVPaperReplayGitHead(commit="deadbeef", branch="feature/test", dirty_worktree=False),
        dry_run=False,
        fixture_mode=True,
        source_pdfs=(missing_pdf,),
        lanes=(failing_lane,),
        comparison_packets=(),
    )
    validation = validate_campaign_manifest(failed_manifest)
    assert validation["status"] == "failed"
    assert validation["finite_failure_count"] == 1
    assert validation["missing_source_pdfs"] == [str(missing_pdf.resolve())]


def test_comparison_packet_skeleton_marks_qualitative_review_ready(tmp_path: Path) -> None:
    lane = _lane_record(tmp_path)
    packets = build_comparison_packet_skeletons((lane,))

    assert len(packets) == 1
    payload = packets[0].to_manifest()
    assert payload["lane"] == "stretching"
    assert payload["status"] == "ready_for_qualitative_review"
    assert payload["metrics"]["l2_error"] is None
    assert "qualitative comparison" in payload["notes"][0].lower()

    packet = GVPaperReplayComparisonPacket(
        lane="stretching",
        status="pending",
        reference_paths=(lane.source_pdfs[0],),
        replay_paths=lane.plot_paths,
        metric_placeholders=("qualitative",),
        notes=("fixture",),
    )
    assert packet.to_manifest()["metrics"] == {"qualitative": None}


def test_resolve_campaign_root_restricts_output_location(tmp_path: Path) -> None:
    resolved = resolve_campaign_root(repo_root=tmp_path, campaign_id="campaign-001")
    assert resolved == tmp_path / "_runs" / "gv" / "figure_replay" / "campaign-001"
    explicit = resolve_campaign_root(
        repo_root=tmp_path,
        campaign_id="campaign-001",
        output_root=tmp_path / "_runs" / "gv" / "figure_replay" / "campaign-001",
    )
    assert explicit == tmp_path / "_runs" / "gv" / "figure_replay" / "campaign-001"

    with pytest.raises(ValueError, match="must live under"):
        resolve_campaign_root(
            repo_root=tmp_path,
            campaign_id="campaign-001",
            output_root=tmp_path / "elsewhere" / "campaign-001",
        )

    with pytest.raises(ValueError, match="leaf directory"):
        resolve_campaign_root(
            repo_root=tmp_path,
            campaign_id="campaign-001",
            output_root=tmp_path / "_runs" / "gv" / "figure_replay" / "other",
        )


def test_load_lane_record_fixture_round_trips_json_payload(tmp_path: Path) -> None:
    lane = _lane_record(tmp_path)
    payload = lane.to_manifest()
    payload["controls"]["tot_force"] = [1.0, 2.0]
    fixture_path = tmp_path / "lane_fixture.json"
    fixture_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_lane_record_fixture(fixture_path)

    assert loaded.lane == lane.lane
    assert loaded.controls["tot_force"] == [1.0, 2.0]
    assert loaded.runtime_commands == lane.runtime_commands

    bad_fixture_path = tmp_path / "bad_fixture.json"
    bad_fixture_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_lane_record_fixture(bad_fixture_path)


def test_cli_fixture_mode_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module("mesouq_test_gv_paper_replay_cli")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        module,
        "collect_git_head",
        lambda _repo_root: GVPaperReplayGitHead(commit="cafebabe", branch="feature/test", dirty_worktree=False),
    )

    source_pdf = _source_pdf(tmp_path)
    rc = module.main(
        [
            "--campaign-id",
            "fixture-cli",
            "--fixture-mode",
            "--lane",
            "stretching",
            "--source-pdf",
            str(source_pdf),
        ]
    )

    manifest_path = tmp_path / "_runs" / "gv" / "figure_replay" / "fixture-cli" / "campaign_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["git_head"]["commit"] == "cafebabe"
    assert payload["lanes"][0]["lane"] == "stretching"
    assert payload["comparison_packets"][0]["status"] == "ready_for_qualitative_review"


def test_cli_reports_invalid_campaign_root_and_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module("mesouq_test_gv_paper_replay_cli_errors")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    with pytest.raises(SystemExit):
        module.main(["--campaign-id", "bad/id", "--fixture-mode"])

    with pytest.raises(SystemExit):
        module.main(["--campaign-id", "fixture-cli", "--fixture-mode", "--lane", "bad-lane"])


def test_replay_cli_private_helpers_cover_error_and_fallback_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("mesouq_test_gv_paper_replay_helpers")
    args = SimpleNamespace(lane=["stretching", "bad-lane"])
    with pytest.raises(ValueError, match="Unsupported"):
        module._resolve_lanes(args)

    assert module._resolve_source_pdfs(["~/missing.pdf"])[0].name == "missing.pdf"
    assert module._jsonable({"array": np.array([1.0]), "flag": np.bool_(True)}) == {
        "array": {"shape": [1], "dtype": "float64"},
        "flag": True,
    }
    with pytest.raises(ValueError, match="empty"):
        module._channel_arrays({"bad": []})
    with pytest.raises(ValueError, match="no finite"):
        module._data_ranges({"bad": np.array([np.nan])})

    provenance_result = SimpleNamespace(
        provenance={
            "lane_plan": {"controls": {"theta": 0.03}},
            "work_dirs": [tmp_path / "_runs" / "work"],
            "runtime_manifests": [{"control_id": "theta_0_03"}],
        },
        geometry={"radGV": 2.0, "height": 14.28},
        material_parameters={"ka": 1.0},
        manifest={"status": "passed"},
        channels={"gamma": np.array([0.0])},
    )
    attr_result = SimpleNamespace(
        controls={"theta": 0.03},
        geometry={"radGV": 2.0, "height": 14.28},
        material_parameters={"ka": 1.0},
        channels={"gamma": np.array([0.0])},
    )

    assert module._lane_controls(provenance_result) == {"theta": 0.03}
    assert module._lane_controls(attr_result) == {"theta": 0.03}
    assert module._lane_geometry(attr_result) == {"radGV": 2.0, "height": 14.28}
    assert module._lane_material_parameters(attr_result) == {"ka": 1.0}
    assert module._lane_work_dirs(provenance_result) == (str(tmp_path / "_runs" / "work"),)
    assert module._lane_runtime_ids(provenance_result) == ("theta_0_03",)
    assert module._manifest_without_full_channels(provenance_result)["channels"] == {"gamma": {"shape": [1]}}

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    assert module._slurm_job_ids() == ("123",)

    for helper in (module._lane_controls, module._lane_geometry, module._lane_material_parameters):
        with pytest.raises(ValueError):
            helper(object())


def test_operational_lane_record_writes_finite_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module("mesouq_test_gv_paper_replay_operational")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    source_pdf = _source_pdf(tmp_path)
    si_pdf = tmp_path / "sources" / "si.pdf"
    si_pdf.write_text("%PDF-1.4 fixture SI\n", encoding="utf-8")

    profile = SimpleNamespace(
        material_values=lambda: {
            "ka": 1.0,
            "kb": 2.0,
            "mu": 3.0,
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
            "mu_l": 4.0,
            "c": 5.0,
        },
        geometry=SimpleNamespace(values=lambda: {"radGV": 2.0, "height": 14.28}),
        provenance=SimpleNamespace(
            paper_pdf_path=str(source_pdf),
            si_pdf_path=str(si_pdf),
            to_dict=lambda: {"source": "fixture"},
        ),
    )
    monkeypatch.setattr(module, "load_gv_paper_replay_profile", lambda: profile)
    monkeypatch.setattr(module, "validate_gv_paper_replay_profile", lambda *_args, **_kwargs: None)

    def fake_plan(**kwargs):
        return SimpleNamespace(
            campaign_id=kwargs["campaign_id"],
            geometry_radius=kwargs["geometry_radius"],
            geometry_height=kwargs["geometry_height"],
            material_parameters=kwargs["material_parameters"],
            controls={"tot_force": (1.0, 2.0), "bpress": -91.0},
        )

    def fake_run(plan):
        plot_path = Path("_runs/gv/figure_replay/op/lane/plots/stretching.png")
        return SimpleNamespace(
            plan=plan,
            manifest={"dataset_id": "fixture"},
            channels={
                "displacement": np.array([0.0, 0.1]),
                "force": np.array([0.0, 2.0]),
            },
            summary={"sample_count": 2},
            raw_sample_result={"work_dirs": [tmp_path / "_runs" / "work"]},
            plot_path=plot_path,
        )

    monkeypatch.setattr(module, "plan_stretching_paper_replay_lane", fake_plan)
    monkeypatch.setattr(module, "run_stretching_paper_replay_lane", fake_run)
    start_cwd = Path.cwd()

    record = module._run_operational_lane(
        lane="stretching",
        campaign_id="op",
        campaign_root=tmp_path / "_runs" / "gv" / "figure_replay" / "op",
        source_pdfs=(source_pdf, si_pdf),
        runtime_command=("python", "run_paper_figure_replay.py"),
    )

    assert Path.cwd() == start_cwd
    assert record.mode == "operational"
    assert record.controls["tot_force"] == [1.0, 2.0]
    assert record.runtime_ids == ()
    assert "work_dir_000" in record.output_paths
    assert record.finite_checks[0].passed is True
    assert record.output_paths["summary"].is_file()
    payload = json.loads(record.output_paths["summary"].read_text(encoding="utf-8"))
    assert payload["data_ranges"][0]["name"] == "displacement"
    assert payload["work_dirs"]


@pytest.mark.parametrize(
    ("lane", "plan_name", "run_name", "channels", "controls", "plot_name"),
    [
        (
            "torsion",
            "plan_torsion_paper_replay_lane",
            "run_torsion_paper_replay_lane",
            {"gamma": np.array([0.0, 0.1]), "sigma_phi_r": np.array([0.0, 2.0])},
            {"theta": (0.01, 0.03)},
            "torsion.png",
        ),
        (
            "buckling",
            "plan_buckling_paper_replay_lane",
            "run_buckling_paper_replay_lane",
            {"bpress": np.array([-91.0, -94.0]), "relative_volume": np.array([1.0, 0.95])},
            {"buck": 0.75, "bpress": (-91.0, -94.0)},
            None,
        ),
        (
            "eigenmodes",
            "plan_eigenmodes_paper_replay_lane",
            "run_eigenmodes_paper_replay_lane",
            {"mode_index": np.array([0.0, 1.0]), "eigenvalues": np.array([1.0, 4.0])},
            {"bpress": -91.0},
            None,
        ),
    ],
)
def test_operational_lane_dispatch_covers_all_non_stretching_lanes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane: str,
    plan_name: str,
    run_name: str,
    channels: dict[str, np.ndarray],
    controls: dict[str, object],
    plot_name: str | None,
) -> None:
    module = _load_script_module(f"mesouq_test_gv_paper_replay_{lane}")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    source_pdf = _source_pdf(tmp_path)
    si_pdf = tmp_path / "sources" / "si.pdf"
    si_pdf.write_text("%PDF-1.4 fixture SI\n", encoding="utf-8")
    profile = SimpleNamespace(
        material_values=lambda: {
            "ka": 1.0,
            "kb": 2.0,
            "mu": 3.0,
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
            "mu_l": 4.0,
            "c": 5.0,
        },
        geometry=SimpleNamespace(values=lambda: {"radGV": 2.0, "height": 14.28}),
        provenance=SimpleNamespace(to_dict=lambda: {"source": "fixture"}),
    )
    monkeypatch.setattr(module, "load_gv_paper_replay_profile", lambda: profile)
    monkeypatch.setattr(module, "validate_gv_paper_replay_profile", lambda *_args, **_kwargs: None)

    def fake_plan(**kwargs):
        return SimpleNamespace(
            experiment=lane,
            figure_id=f"fixture-{lane}",
            geometry={"radGV": kwargs.get("radGV", kwargs.get("geometry_radius", 2.0)), "height": kwargs.get("height", kwargs.get("geometry_height", 14.28))},
            geometry_radius=kwargs.get("geometry_radius", kwargs.get("radGV", 2.0)),
            geometry_height=kwargs.get("geometry_height", kwargs.get("height", 14.28)),
            material_parameters=kwargs["material_parameters"],
            controls=controls,
            mode_count=2,
            mapping_assumptions=("fixture mapping",),
            to_manifest=lambda: {"experiment": lane, "controls": controls},
        )

    def fake_run(plan):
        return SimpleNamespace(
            plan=plan,
            manifest={"dataset_id": f"gv__{lane}__fixture"},
            channels=channels,
            summary={"sample_count": 2},
            raw_sample_result={
                "work_dirs": [tmp_path / "_runs" / "work"],
                "runtime_manifests": [{"dataset_id": f"gv__{lane}__fixture"}],
            },
            plot_path=Path(f"_runs/gv/figure_replay/op/lanes/{lane}/plots/{plot_name}") if plot_name else None,
        )

    monkeypatch.setattr(module, plan_name, fake_plan)
    monkeypatch.setattr(module, run_name, fake_run)
    if lane == "buckling":
        monkeypatch.setattr(
            module,
            "plot_buckling_paper_replay",
            lambda _result, *, output_path: Path(output_path),
        )
    if lane == "eigenmodes":
        monkeypatch.setattr(
            module,
            "plot_eigenmodes_paper_replay",
            lambda _result, *, output_path: Path(output_path),
        )

    record = module._run_operational_lane(
        lane=lane,
        campaign_id="op",
        campaign_root=tmp_path / "_runs" / "gv" / "figure_replay" / "op",
        source_pdfs=(source_pdf, si_pdf),
        runtime_command=("python", "run_paper_figure_replay.py"),
    )

    assert record.lane == lane
    assert record.runtime_ids == (f"gv__{lane}__fixture",)
    assert record.slurm_job_ids == ()
    assert all(check.passed for check in record.finite_checks)
    assert record.output_paths["summary"].is_file()
    assert record.plot_paths


def test_operational_lane_rejects_unsupported_lane_and_restores_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("mesouq_test_gv_paper_replay_bad_lane")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    profile = SimpleNamespace(
        material_values=lambda: {
            "ka": 1.0,
            "kb": 2.0,
            "mu": 3.0,
            "b1": 0.0,
            "b2": 0.0,
            "a3": 0.0,
            "a4": 0.0,
            "mu_l": 4.0,
            "c": 5.0,
        },
        geometry=SimpleNamespace(values=lambda: {"radGV": 2.0, "height": 14.28}),
        provenance=SimpleNamespace(to_dict=lambda: {"source": "fixture"}),
    )
    monkeypatch.setattr(module, "load_gv_paper_replay_profile", lambda: profile)
    monkeypatch.setattr(module, "validate_gv_paper_replay_profile", lambda *_args, **_kwargs: None)
    start_cwd = Path.cwd()

    with pytest.raises(ValueError, match="Unsupported"):
        module._run_operational_lane(
            lane="unknown",
            campaign_id="op",
            campaign_root=tmp_path / "_runs" / "gv" / "figure_replay" / "op",
            source_pdfs=(_source_pdf(tmp_path),),
            runtime_command=("python", "run_paper_figure_replay.py"),
        )

    assert Path.cwd() == start_cwd
