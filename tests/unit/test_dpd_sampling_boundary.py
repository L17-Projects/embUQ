from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meso_uq.active_learning import Candidate
from meso_uq.active_learning.emb_34um_dpd_adapter import (
    EMB_34UM_DPD_SCHEMA_VERSION,
    EMB_34UM_RETRY_LIMIT,
)
from meso_uq.dpd_sampling.boundary import (
    build_dpd_sampling_batch_request,
    build_and_render_dpd_sampling_batch,
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


def _gv_candidates() -> list[Candidate]:
    return [
        Candidate(
            candidate_id="gv-sampling-000",
            parameters={
                "gv_launch": {
                    "experiment": "stretching",
                    "controls": {"tot_force": [500.0, 750.0]},
                }
            },
        ),
        Candidate(
            candidate_id="gv-sampling-001",
            parameters={
                "gv_launch": {
                    "experiment": "buckling",
                    "controls": {"buck": [0.8, 0.9]},
                }
            },
        ),
    ]


def _gv_defaults() -> dict[str, object]:
    return {
        "geometry": {"radGV": 2.0, "height": 14.28},
        "material_parameters": _VALID_MATERIAL_PARAMETERS,
        "controls": {"bpress": -91.0},
    }


def _run_boundary_root(tmp_path: Path, run_id: str, iteration: str | int) -> Path:
    return tmp_path / "_runs" / "active_learning" / run_id / "iterations" / f"iter_{int(iteration):04d}"


def _gv_tagged_candidate() -> Candidate:
    return Candidate(
        candidate_id="gv-tagged",
        parameters={
            "family": "gv",
            "dpd_family": "gv",
            "experiment": "stretching",
            "controls": {"tot_force": [500.0, 750.0]},
        },
    )


def _metadata_candidate(candidate_id: str) -> tuple[Candidate, dict[str, object]]:
    metadata = {
        "acquisition_score": 0.93,
        "selection_batch_id": "al-batch-007",
        "selection_timestamp": "2026-05-19T13:00:00Z",
    }
    return (
        Candidate(
            candidate_id=candidate_id,
            parameters={
                "gv_launch": {
                    "experiment": "stretching",
                    "controls": {"tot_force": [500.0, 750.0]},
                }
            },
            metadata=metadata,
        ),
        metadata,
    )


def test_build_and_render_gv_batch_writes_deterministic_manifests_and_plots(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-gv-01", iteration=0)
    result = build_and_render_dpd_sampling_batch(
        _gv_candidates(),
        run_id="run-gv-01",
        iteration="0",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"source_issue": "MES-17"},
        defaults=_gv_defaults(),
        campaign_root=root,
        batch_id="batch-gv",
        platforms="karolina",
    )

    assert result.campaign_root == root
    assert result.batch_request.family == "gv"
    assert result.batch_request.platform == "karolina"

    batch_manifest = json.loads((root / "dpd_sampling_batch_manifest.json").read_text(encoding="utf-8"))
    assert batch_manifest["schema_version"] == "meso_uq.dpd_sampling.batch_request.v1"
    assert batch_manifest["submission"] == {"submitted": False, "submission_commands": []}

    for candidate in result.batch_request.candidate_manifests:
        candidate_manifest_path = candidate.output_root / "dpd_sampling_candidate_manifest.json"
        assert candidate_manifest_path.is_file(), candidate_manifest_path
        candidate_payload = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
        assert candidate_payload["submission"]["submitted"] is False
        assert candidate_payload["submission"]["submission_commands"] == []
        assert candidate_payload["normalized_payload"]["experiment"] in {"stretching", "buckling"}
        assert candidate_payload["normalized_payload"]["output_root"] == candidate.output_root.as_posix()
        rendered_payload = candidate_payload["rendered_payload"]
        assert rendered_payload["campaign_dir"] == candidate.output_root.as_posix()
        assert rendered_payload["output_root"] == candidate.output_root.as_posix()
        for script in rendered_payload["scheduler_scripts"]:
            script_path = Path(script["script_path"])
            assert script_path.is_absolute()
            assert script_path.is_file()
            script_text = script_path.read_text(encoding="utf-8")
            assert f"CAMPAIGN_DIR={candidate.output_root.as_posix()}" in script_text
            assert f"#SBATCH --output={candidate.output_root.as_posix()}/logs/" in script_text
            assert f"#SBATCH --error={candidate.output_root.as_posix()}/logs/" in script_text
            assert f"CAMPAIGN_DIR={candidate.candidate_id}" not in script_text

    validation_payload = json.loads((root / "dpd_sampling_validation_report.json").read_text(encoding="utf-8"))
    assert validation_payload["schema_version"] == "meso_uq.dpd_sampling.validation_report.v1"
    assert validation_payload["family_distribution"] == {"gv": 2}
    assert validation_payload["runtime_distribution"] == {"00:30:00:karolina:1": 2}
    assert validation_payload["expected_hdf5_refs"], validation_payload
    for candidate in result.batch_request.candidate_manifests:
        candidate_token = candidate.candidate_id
        refs = [
            ref
            for ref in validation_payload["expected_hdf5_refs"]
            if f"/{candidate_token}/" in Path(ref["hdf5_path"]).as_posix()
        ]
        assert refs, candidate_token
        for ref in refs:
            hdf5_path = Path(ref["hdf5_path"]).as_posix()
            assert f"/{candidate_token}/datasets/" in hdf5_path
            assert f"/{candidate_token}/{candidate_token}/" not in hdf5_path

    plot_path = root / "dpd_sampling_validation_plot.png"
    plot_sidecar_path = root / "dpd_sampling_validation_plot.png.json"
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0
    assert plot_sidecar_path.exists()
    plot_sidecar = json.loads(plot_sidecar_path.read_text(encoding="utf-8"))
    assert plot_sidecar["family_distribution"] == {"gv": 2}
    assert "expected_hdf5_refs" in plot_sidecar
    assert isinstance(plot_sidecar["expected_hdf5_refs"], list)

    assert result.plot_paths == (plot_path,)
    assert result.plot_sidecar_paths == (plot_sidecar_path,)
    assert result.rendered_manifest_paths[-1] == root / "dpd_sampling_validation_report.json"


def test_build_and_render_gv_batch_supports_flat_explicit_family_payload_markers(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-gv-tagged", iteration=0)
    result = build_and_render_dpd_sampling_batch(
        [_gv_tagged_candidate()],
        run_id="run-gv-tagged",
        iteration="0",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"source_issue": "MES-17"},
        defaults=_gv_defaults(),
        campaign_root=root,
        batch_id="batch-gv-tagged",
        platforms=("karolina",),
    )

    candidate_manifest_path = root / "gv-tagged" / "dpd_sampling_candidate_manifest.json"
    assert candidate_manifest_path.is_file()
    candidate_payload = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    assert "family" not in candidate_payload["normalized_payload"]
    assert "dpd_family" not in candidate_payload["normalized_payload"]
    assert result.batch_request.family == "gv"


def test_build_and_render_gv_batch_supports_tagged_material_parameters_with_defaults(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-gv-tagged-material", iteration=0)
    defaults = {
        "experiment": "stretching",
        "geometry": {"radGV": 2.0, "height": 14.28},
        "controls": {"tot_force": [500.0, 750.0]},
    }
    result = build_and_render_dpd_sampling_batch(
        [
            Candidate(
                candidate_id="gv-tagged-material",
                parameters={
                    "family": "gv",
                    "material_parameters": _VALID_MATERIAL_PARAMETERS,
                },
            )
        ],
        run_id="run-gv-tagged-material",
        iteration="0",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"source_issue": "MES-17"},
        defaults=defaults,
        campaign_root=root,
        batch_id="batch-gv-tagged-material",
        platforms=("karolina",),
    )

    candidate_manifest_path = root / "gv-tagged-material" / "dpd_sampling_candidate_manifest.json"
    assert candidate_manifest_path.is_file()
    candidate_payload = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    normalized_payload = candidate_payload["normalized_payload"]
    assert "family" not in normalized_payload
    assert normalized_payload["material_parameters"] == _VALID_MATERIAL_PARAMETERS
    assert normalized_payload["experiment"] == "stretching"
    assert result.batch_request.family == "gv"


def test_build_and_render_gv_batch_preserves_candidate_metadata_in_manifests_and_sidecars(tmp_path: Path) -> None:
    candidate, metadata = _metadata_candidate("gv-meta")
    root = _run_boundary_root(tmp_path, run_id="run-gv-metadata", iteration=0)
    build_and_render_dpd_sampling_batch(
        [candidate],
        run_id="run-gv-metadata",
        iteration="0",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        defaults=_gv_defaults(),
        campaign_root=root,
        batch_id="batch-gv-metadata",
    )

    candidate_manifest = json.loads((root / "gv-meta" / "dpd_sampling_candidate_manifest.json").read_text(encoding="utf-8"))
    assert candidate_manifest["active_learning_metadata"] == metadata
    batch_manifest = json.loads((root / "dpd_sampling_batch_manifest.json").read_text(encoding="utf-8"))
    assert batch_manifest["candidate_manifests"][0]["active_learning_metadata"] == metadata
    sidecar = json.loads((root / "dpd_sampling_validation_plot.png.json").read_text(encoding="utf-8"))
    assert sidecar["candidate_lineage"] == [{"candidate_id": "gv-meta", "active_learning_metadata": metadata}]
    validation_payload = json.loads((root / "dpd_sampling_validation_report.json").read_text(encoding="utf-8"))
    assert validation_payload["candidate_lineage"] == [{"candidate_id": "gv-meta", "active_learning_metadata": metadata}]


def test_build_and_render_gv_batch_supports_dpd_family_material_parameters_with_defaults(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-gv-dpd-family-material", iteration=0)
    defaults = {
        "experiment": "stretching",
        "geometry": {"radGV": 2.0, "height": 14.28},
        "controls": {"tot_force": [500.0, 750.0]},
    }
    build_and_render_dpd_sampling_batch(
        [
            Candidate(
                candidate_id="gv-dpd-family-material",
                parameters={
                    "dpd_family": "gv",
                    "material_parameters": _VALID_MATERIAL_PARAMETERS,
                },
            )
        ],
        run_id="run-gv-dpd-family-material",
        iteration="0",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"source_issue": "MES-17"},
        defaults=defaults,
        campaign_root=root,
        batch_id="batch-gv-dpd-family-material",
        platforms=("karolina",),
    )

    candidate_payload = json.loads(
        (root / "gv-dpd-family-material" / "dpd_sampling_candidate_manifest.json").read_text(encoding="utf-8")
    )
    normalized_payload = candidate_payload["normalized_payload"]
    assert "dpd_family" not in normalized_payload
    assert normalized_payload["material_parameters"] == _VALID_MATERIAL_PARAMETERS


def test_reject_reduced_emb_payload_and_emit_no_artifacts(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-emb-reduced", iteration=0)
    reduced_cases = [
        (
            "emb-reduced-vector",
            {
                "family": "emb",
                "experiment": "stretching",
                "material_parameters": [1.0, 2.0],
            },
        ),
        (
            "emb-parameter-vector",
            {
                "family": "emb",
                "experiment": "stretching",
                "parameter_vector": [1.0, 2.0, 3.0],
            },
        ),
    ]
    for candidate_id, parameters in reduced_cases:
        with pytest.raises(ValueError, match="full expanded EMB payload is required"):
            build_and_render_dpd_sampling_batch(
                [Candidate(candidate_id=candidate_id, parameters=parameters)],
                run_id="run-emb-reduced",
                iteration="0",
                platform="vega",
                walltime="00:30:00",
                gpu_count=1,
                campaign_root=root,
            )
    assert not root.exists()


@pytest.mark.parametrize(
    ("candidate_id", "parameters"),
    [
        (
            "emb-tagged-envelope-root-vector",
            {
                "family": "emb",
                "parameter_vector": [1.0, 2.0, 3.0],
                "emb_launch": {
                    "experiment": "stretching",
                    "notes": "expanded launch payload",
                },
            },
        ),
        (
            "emb-untagged-envelope-root-vector",
            {
                "reduced_parameter_vector": [1.0, 2.0, 3.0],
                "emb_launch": {
                    "experiment": "stretching",
                    "notes": "expanded launch payload",
                },
            },
        ),
    ],
)
def test_reject_emb_launch_envelope_with_root_reduced_vectors(
    tmp_path: Path,
    candidate_id: str,
    parameters: dict[str, object],
) -> None:
    root = _run_boundary_root(tmp_path, run_id=f"run-{candidate_id}", iteration=0)

    with pytest.raises(ValueError, match="full expanded EMB payload is required"):
        build_and_render_dpd_sampling_batch(
            [Candidate(candidate_id=candidate_id, parameters=parameters)],
            run_id=f"run-{candidate_id}",
            iteration="0",
            platform="vega",
            walltime="00:30:00",
            gpu_count=1,
            campaign_root=root,
        )

    assert not root.exists()


def test_build_and_render_gv_preserves_candidate_controls_without_default_controls(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-gv-candidate-controls", iteration=0)
    defaults = _gv_defaults()
    defaults.pop("controls")
    result = build_and_render_dpd_sampling_batch(
        _gv_candidates()[:1],
        run_id="run-gv-candidate-controls",
        iteration="0",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        provenance_tags={"source_issue": "MES-17"},
        defaults=defaults,
        campaign_root=root,
        batch_id="batch-gv-candidate-controls",
        platforms=("karolina",),
    )

    request = result.batch_request._gv_handoff.launch_requests[0]
    assert request.sweep.axis == "tot_force"
    assert request.sweep.values == (500.0, 750.0)
    assert request.fixed_controls == {}


def test_render_emb_placeholder_manifesting_is_non_production_and_parseable(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-emb-01", iteration=0)
    result = build_and_render_dpd_sampling_batch(
        [
            Candidate(
                candidate_id="emb-001",
                parameters={
                    "family": "emb",
                    "experiment": "stretching",
                    "notes": "emb-placehoder",
                },
            )
        ],
        run_id="run-emb-01",
        iteration=0,
        platform="vega",
        walltime="00:30:00",
        gpu_count=1,
        campaign_root=root,
    )

    assert result.batch_request.family == "emb"
    assert result.batch_request.platform == "vega"

    manifest_payload = result.rendered_manifest_paths
    assert manifest_payload, manifest_payload
    emb_manifest = root / "emb" / "emb-001" / "dpd_sampling_candidate_manifest.json"
    assert emb_manifest.exists()
    emb_payload = json.loads(emb_manifest.read_text(encoding="utf-8"))
    assert emb_payload["normalized_payload"]["placeholder"] is True
    assert emb_payload["submission"]["submission_commands"] == []
    assert "no production Mirheo parity" in emb_payload["rendered_payload"]["notes"]
    assert "candidate_id" in emb_payload["rendered_payload"]

    validation_payload = json.loads((root / "dpd_sampling_validation_report.json").read_text(encoding="utf-8"))
    assert validation_payload["family_distribution"] == {"emb": 1}

    sidecar = json.loads((root / "dpd_sampling_validation_plot.png.json").read_text(encoding="utf-8"))
    assert sidecar["submission"]["submitted"] is False
    assert sidecar["platform_distribution"] == {"vega": 1}


def test_build_and_render_emb_34um_full_request_manifest_contains_full_request_payload(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-emb-full-01", iteration=0)
    result = build_and_render_dpd_sampling_batch(
        [
            Candidate(
                candidate_id="emb-full-01",
                parameters={
                    "family": "emb",
                    "experiment": "indentation",
                    "Yt": 1.2e6,
                    "kb": 5000.0,
                },
            )
        ],
        run_id="run-emb-full-01",
        iteration="0",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        campaign_root=root,
    )

    candidate_root = root / "emb" / "emb-full-01"
    candidate_payload_path = candidate_root / "dpd_sampling_candidate_manifest.json"
    assert candidate_payload_path.is_file()

    candidate_payload = json.loads(candidate_payload_path.read_text(encoding="utf-8"))
    rendered_payload = candidate_payload["rendered_payload"]
    request_payload = rendered_payload["request_payload"]
    assert candidate_payload["normalized_payload"]["schema_version"] == EMB_34UM_DPD_SCHEMA_VERSION
    assert request_payload["schema_version"] == EMB_34UM_DPD_SCHEMA_VERSION
    assert request_payload["request_type"] == "emb_34um_full_force_sweep"
    assert request_payload["retry_limit"] == EMB_34UM_RETRY_LIMIT
    assert request_payload["platform"] == "karolina"
    assert request_payload["platform_defaults"]["platform"] == "karolina"
    assert request_payload["platform_defaults"]["walltime"] == "00:30:00"
    assert request_payload["platform_defaults"]["gpu_count"] == 1
    assert request_payload["fingerprint"]["radp"] == 6.8
    assert request_payload["expected_output_paths"]["request_manifest"].endswith("emb_34um_request_manifest.json")
    assert rendered_payload["expected_hdf5_datasets"][0]["manifest_path"].endswith("emb_34um_request_manifest.json")
    assert result.batch_request.family == "emb"
    assert result.rendered_manifest_paths


def test_build_and_render_emb_34um_canary_request_is_marked(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-emb-canary-01", iteration=0)
    build_and_render_dpd_sampling_batch(
        [
            Candidate(
                candidate_id="emb-canary-01",
                parameters={
                    "family": "emb",
                    "experiment": "indentation",
                    "Yt": 1.2e6,
                    "kb": 5000.0,
                    "canary": True,
                },
            )
        ],
        run_id="run-emb-canary-01",
        iteration="0",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        campaign_root=root,
    )

    candidate_payload_path = root / "emb" / "emb-canary-01" / "dpd_sampling_candidate_manifest.json"
    payload = json.loads(candidate_payload_path.read_text(encoding="utf-8"))
    rendered_payload = payload["rendered_payload"]["request_payload"]
    assert "canary" in rendered_payload
    assert rendered_payload["canary"]["enabled"] is True
    assert rendered_payload["canary"]["point_count"] == 3
    assert len(rendered_payload["canary"]["point_indices"]) == 3
    assert len(rendered_payload["force_grid"]) == 3


def test_build_and_render_emb_34um_rejects_out_of_bounds_dimensions(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-emb-oob", iteration=0)
    with pytest.raises(ValueError, match="outside bounds"):
        build_and_render_dpd_sampling_batch(
            [
                Candidate(
                    candidate_id="emb-oob",
                    parameters={
                        "family": "emb",
                        "experiment": "indentation",
                        "Yt": 1000.0,
                        "kb": 5000.0,
                    },
                )
            ],
            run_id="run-emb-oob",
            iteration="0",
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
            campaign_root=root,
        )


def test_gv_relative_campaign_root_does_not_nest_rendered_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path("_runs") / "active_learning" / "relative-gv" / "iterations" / "iter_0000"

    build_and_render_dpd_sampling_batch(
        _gv_candidates()[:1],
        run_id="relative-gv",
        iteration=0,
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        defaults=_gv_defaults(),
        campaign_root=root,
        platforms="karolina",
    )

    candidate_manifest = root / "gv-sampling-000" / "dpd_sampling_candidate_manifest.json"
    nested_manifest = root / root / "gv-sampling-000" / "dpd_sampling_candidate_manifest.json"
    assert candidate_manifest.is_file()
    assert not nested_manifest.exists()
    candidate_payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    for ref in candidate_payload["expected_hdf5_datasets"]:
        hdf5_path = Path(ref["hdf5_path"]).as_posix()
        assert f"{root.as_posix()}/gv-sampling-000/datasets/" in hdf5_path
        assert f"gv-sampling-000/{root.as_posix()}/" not in hdf5_path


def test_build_and_render_gv_uses_iterable_platform_argument(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-gv-platform-iter", iteration=0)
    build_and_render_dpd_sampling_batch(
        _gv_candidates()[:1],
        run_id="run-gv-platform-iter",
        iteration=0,
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        defaults=_gv_defaults(),
        campaign_root=root,
        platforms=(platform for platform in ("karolina", "vega")),
    )

    candidate_root = root / "gv-sampling-000"
    scripts = candidate_root / "scripts"
    assert scripts.is_dir()
    assert (scripts / "karolina").is_dir()
    assert (scripts / "vega").is_dir()


def test_build_gv_request_rejects_absolute_campaign_root_under_repo_source() -> None:
    with pytest.raises(ValueError, match="repository source roots"):
        build_dpd_sampling_batch_request(
            _gv_candidates()[:1],
            run_id="bad-gv-root",
            iteration=0,
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
            defaults=_gv_defaults(),
            campaign_root=REPO_ROOT / "src" / "dpd-sampling-bad",
        )


def test_build_and_render_rejects_explicit_empty_platforms(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-gv-empty-platforms", iteration=0)
    with pytest.raises(ValueError, match="platforms must include at least one value"):
        build_and_render_dpd_sampling_batch(
            _gv_candidates()[:1],
            run_id="run-gv-empty-platforms",
            iteration=0,
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
            defaults=_gv_defaults(),
            campaign_root=root,
            platforms=(),
        )


def test_nested_and_legacy_gv_payloads_are_accepted() -> None:
    defaulted = _gv_defaults()
    nested = build_dpd_sampling_batch_request(
        [
            Candidate(
                candidate_id="nested-gv",
                parameters={
                    "gv_launch": {
                        "experiment": "buckling",
                        "controls": {"buck": [0.8, 0.9]},
                    }
                },
            )
        ],
        run_id="run-gv-nested",
        iteration="0000",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        defaults=defaulted,
    )
    assert nested.family == "gv"

    legacy = build_dpd_sampling_batch_request(
        [
            Candidate(
                candidate_id="legacy-gv",
                parameters={"experiment": "stretching", "controls": {"tot_force": [500.0]}},
            )
        ],
        run_id="run-gv-legacy",
        iteration="0000",
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        defaults=defaulted,
    )
    assert legacy.family == "gv"


def test_gv_render_honors_overwrite_flag(tmp_path: Path) -> None:
    root = _run_boundary_root(tmp_path, run_id="run-gv-overwrite", iteration=0)
    kwargs = dict(
        selected_candidates=_gv_candidates()[:1],
        run_id="run-gv-overwrite",
        iteration=0,
        platform="karolina",
        walltime="00:30:00",
        gpu_count=1,
        defaults=_gv_defaults(),
        campaign_root=root,
        platforms="karolina",
    )

    build_and_render_dpd_sampling_batch(**kwargs)
    build_and_render_dpd_sampling_batch(**kwargs, overwrite=True)

    assert (root / "gv-sampling-000" / "dpd_sampling_candidate_manifest.json").is_file()


def test_reject_mixed_family_batch_request() -> None:
    with pytest.raises(ValueError, match="Mixed candidate families"):
        build_dpd_sampling_batch_request(
            [
                Candidate(
                    candidate_id="mixed-gv",
                    parameters={"gv_launch": {"experiment": "stretching", "controls": {"tot_force": [500.0]}}},
                ),
                Candidate(
                    candidate_id="mixed-emb",
                    parameters={"family": "emb", "experiment": "stretching"},
                ),
            ],
            run_id="mixed",
            iteration="0000",
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
        )


def test_reject_scheduler_owned_fields_in_candidate_payload() -> None:
    with pytest.raises(ValueError, match="scheduler-owned fields"):
        build_dpd_sampling_batch_request(
            [
                Candidate(
                    candidate_id="bad-scheduler",
                    parameters={
                        "gv_launch": {
                            "experiment": "stretching",
                            "controls": {"tot_force": [500.0]},
                            "platform": "vega",
                        }
                    },
                )
            ],
            run_id="bad",
            iteration=0,
            platform="karolina",
            walltime="00:30:00",
            gpu_count=1,
            defaults=_gv_defaults(),
        )


def test_reject_platforms_outside_whitelist() -> None:
    with pytest.raises(ValueError, match="not supported for DPD sampling"):
        build_dpd_sampling_batch_request(
            _gv_candidates(),
            run_id="bad-platform",
            iteration=0,
            platform="generic",
            walltime="00:30:00",
            gpu_count=1,
            defaults=_gv_defaults(),
        )


def test_dpd_sampling_import_does_not_import_matplotlib_or_scheduler_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    code = """
import sys

import meso_uq.dpd_sampling
loaded = [name for name in ('torch', 'pyro', 'matplotlib', 'mpi4py', 'mirheo', 'korali', 'slurm') if name in sys.modules]
assert loaded == []
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
