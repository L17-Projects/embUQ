from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from meso_uq.active_learning import Candidate
from meso_uq.structures.gv.active_learning_handoff import (
    GV_ACTIVE_LEARNING_HANDOFF_SCHEMA_VERSION,
    build_gv_active_learning_launch_handoff,
    render_gv_active_learning_launch_handoff,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "active_learning" / "gv_selected_candidates.example.yaml"
EXAMPLE_HANDOFF_MANIFEST = (
    REPO_ROOT / "configs" / "active_learning" / "gv_selected_candidates_handoff_manifest.example.json"
)

_VALID_MATERIAL_PARAMETERS = {
    "ka": 1.1,
    "kb": 1.2,
    "mu": 0.9,
    "b1": 0.1,
    "b2": 0.2,
    "a3": 0.3,
    "a4": 0.4,
    "mu_l": 0.5,
    "c": 0.6,
}


def _selected_candidates() -> list[Candidate]:
    return [
        Candidate(
            candidate_id="gv-al-stretch-001",
            parameters={
                "gv_launch": {
                    "experiment": "stretching",
                    "controls": {"tot_force": [500.0, 750.0]},
                }
            },
            metadata={"acquisition_score": 0.81},
        ),
        Candidate(
            candidate_id="gv-al-buckling-002",
            parameters={
                "gv_launch": {
                    "experiment": "buckling",
                    "controls": {"buck": [0.8, 0.9]},
                }
            },
            metadata={"acquisition_score": 0.79},
        ),
    ]


def _defaults() -> dict[str, object]:
    return {
        "geometry": {"radGV": 2.0, "height": 14.28},
        "material_parameters": _VALID_MATERIAL_PARAMETERS,
        "controls": {"bpress": -91.0},
    }


def test_build_handoff_converts_selected_candidates_to_launch_requests() -> None:
    handoff = build_gv_active_learning_launch_handoff(
        _selected_candidates(),
        campaign_root="_runs/gv/active_learning/iter-000",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"linear_issue": "MES-185"},
        defaults=_defaults(),
        batch_id="iter-000",
    )

    assert handoff.campaign_root == Path("_runs/gv/active_learning/iter-000")
    assert [request.experiment for request in handoff.launch_requests] == ["stretching", "buckling"]
    assert [request.output_root for request in handoff.launch_requests] == [
        Path("_runs/gv/active_learning/iter-000/gv-al-stretch-001"),
        Path("_runs/gv/active_learning/iter-000/gv-al-buckling-002"),
    ]
    assert handoff.launch_requests[0].sweep.axis == "tot_force"
    assert handoff.launch_requests[1].sweep.axis == "buck"
    assert handoff.launch_requests[0].fixed_controls == {"bpress": -91.0}
    assert handoff.launch_requests[1].fixed_controls == {"bpress": -91.0}
    assert handoff.launch_requests[0].provenance_tags["source"] == "active_learning_selected_candidates"
    assert handoff.launch_requests[0].provenance_tags["active_learning_candidate_id"] == "gv-al-stretch-001"

    manifest = handoff.to_manifest()
    assert manifest["schema_version"] == GV_ACTIVE_LEARNING_HANDOFF_SCHEMA_VERSION
    assert manifest["submission"] == {"submitted": False, "submission_commands": []}
    assert manifest["candidates"][0]["expected_hdf5_datasets"]["campaign"]["dataset_id"].startswith(
        "gv:stretching:"
    )
    assert manifest["candidates"][1]["expected_hdf5_datasets"]["runs"][1]["hdf5_path"].endswith(
        "numerical_dataset.h5"
    )


def test_handoff_preserves_candidate_controls_without_default_controls() -> None:
    defaults = dict(_defaults())
    defaults.pop("controls")
    handoff = build_gv_active_learning_launch_handoff(
        _selected_candidates()[:1],
        campaign_root="_runs/gv/active_learning/iter-000",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"linear_issue": "MES-185"},
        defaults=defaults,
        batch_id="iter-000",
    )

    request = handoff.launch_requests[0]
    assert request.sweep.axis == "tot_force"
    assert request.sweep.values == (500.0, 750.0)
    assert request.fixed_controls == {}


def test_render_handoff_uses_launch_renderer_without_submission(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    handoff = build_gv_active_learning_launch_handoff(
        _selected_candidates(),
        campaign_root="_runs/gv/active_learning/iter-000",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"linear_issue": "MES-186"},
        defaults=_defaults(),
    )

    rendered = render_gv_active_learning_launch_handoff(
        handoff,
        platforms=("karolina", "vega"),
    )

    assert len(rendered) == 2
    for campaign in rendered:
        payload = json.loads(campaign.manifest_path.read_text(encoding="utf-8"))
        assert payload["submission"] == {"submitted": False, "submission_commands": []}
        assert [script["submits_jobs"] for script in payload["scheduler_scripts"]] == [False, False]
        assert payload["expected_hdf5_datasets"]["campaign"]["hdf5_path"].endswith("numerical_dataset.h5")
        assert campaign.campaign_dir.is_dir()
        assert all(script.script_path.is_file() for script in campaign.scheduler_scripts)


def test_handoff_rejects_scheduler_owned_candidate_fields() -> None:
    candidate = Candidate(
        candidate_id="bad-scheduler-owner",
        parameters={
                "gv_launch": {
                    "experiment": "stretching",
                    "controls": {"tot_force": [500.0]},
                    "platform": "vega",
            }
        },
    )

    with pytest.raises(ValueError, match="adapter-owned launch fields: platform"):
        build_gv_active_learning_launch_handoff(
            [candidate],
            campaign_root="_runs/gv/active_learning/iter-000",
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
            provenance_tags={"linear_issue": "MES-185"},
            defaults=_defaults(),
        )


def test_handoff_rejects_mixed_default_geometry_and_candidate_rad_height() -> None:
    candidate = Candidate(
        candidate_id="mixed-geometry",
        parameters={
                "gv_launch": {
                    "experiment": "stretching",
                    "controls": {"tot_force": [500.0]},
                    "radGV": 3.0,
                "height": 15.0,
            }
        },
    )

    with pytest.raises(ValueError, match="mixes 'geometry' with explicit 'radGV'/'height'"):
        build_gv_active_learning_launch_handoff(
            [candidate],
            campaign_root="_runs/gv/active_learning/iter-000",
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
            provenance_tags={"linear_issue": "MES-185"},
            defaults=_defaults(),
        )


def test_handoff_rejects_duplicate_path_safe_candidate_directories() -> None:
    candidates = [
        Candidate(
            candidate_id="candidate/a",
            parameters={
                "gv_launch": {
                    "experiment": "stretching",
                    "controls": {"tot_force": [500.0]},
                }
            },
        ),
        Candidate(
            candidate_id="candidate_a",
            parameters={
                "gv_launch": {
                    "experiment": "stretching",
                    "controls": {"tot_force": [750.0]},
                }
            },
        ),
    ]

    with pytest.raises(ValueError, match="duplicate campaign directories"):
        build_gv_active_learning_launch_handoff(
            candidates,
            campaign_root="_runs/gv/active_learning/iter-000",
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
            provenance_tags={"linear_issue": "MES-185"},
            defaults=_defaults(),
        )


def test_handoff_rejects_shear_flow_through_existing_launch_validation() -> None:
    candidate = Candidate(
        candidate_id="deferred-shear",
        parameters={
            "gv_launch": {
                "experiment": "shear_flow",
                "controls": {"ptan": [0.2], "afsi": 1.0, "bpress": -91.0},
            }
        },
    )

    with pytest.raises(ValueError, match="not supported for sampling"):
        build_gv_active_learning_launch_handoff(
            [candidate],
            campaign_root="_runs/gv/active_learning/iter-000",
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
            provenance_tags={"linear_issue": "MES-185"},
            defaults=_defaults(),
        )


def test_gv_selected_candidates_example_config_converts_to_launch_requests() -> None:
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    spec = payload["spec"]

    handoff = build_gv_active_learning_launch_handoff(
        spec["selected_candidates"],
        campaign_root=spec["campaign_root"],
        platform=spec["platform"],
        walltime=spec["walltime"],
        gpu_count=spec["gpu_count"],
        provenance_tags=spec["provenance_tags"],
        defaults=spec["defaults"],
        batch_id=payload["metadata"]["id"],
    )

    assert len(handoff.launch_requests) == 2
    assert {request.experiment for request in handoff.launch_requests} == {"stretching", "buckling"}
    assert all(request.experiment != "shear_flow" for request in handoff.launch_requests)


def test_gv_selected_candidates_expected_manifest_matches_builder() -> None:
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    spec = payload["spec"]

    handoff = build_gv_active_learning_launch_handoff(
        spec["selected_candidates"],
        campaign_root=spec["campaign_root"],
        platform=spec["platform"],
        walltime=spec["walltime"],
        gpu_count=spec["gpu_count"],
        provenance_tags=spec["provenance_tags"],
        defaults=spec["defaults"],
        batch_id=payload["metadata"]["id"],
    )

    expected = json.loads(EXAMPLE_HANDOFF_MANIFEST.read_text(encoding="utf-8"))
    assert handoff.to_manifest() == expected
