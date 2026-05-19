from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from meso_uq.active_learning import CandidateGenerationConfig, CandidateParameterDimension
from meso_uq.active_learning.acquisition import (
    ACQUISITION_PLOT_FILENAME,
    ACQUISITION_PLOT_SIDECAR_FILENAME,
    ACQUISITION_REPORT_FILENAME,
    ACQUISITION_MANIFEST_FILENAME,
    _build_fallback_png,
)
from meso_uq.active_learning.candidate_generation import (
    CANDIDATE_GENERATION_MANIFEST_FILENAME,
    CANDIDATE_GENERATION_PLOT_FILENAME,
    CANDIDATE_GENERATION_PLOT_SIDECAR_FILENAME,
    CANDIDATE_GENERATION_REPORT_FILENAME,
    _FALLBACK_PNG,
)
from meso_uq.active_learning.cli import build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
_AL_ENTRYPOINT_PATTERN = re.compile(
    r"""(?xm)
    ^\s*mesouq-al\s*=\s*["']meso_uq\.active_learning\.cli:main["']\s*$
    """
)


def _gv_candidate_generation_config() -> CandidateGenerationConfig:
    return CandidateGenerationConfig(
        family="gv",
        experiment="stretching",
        dimensions=(
            CandidateParameterDimension(
                name="ka",
                path=("material_parameters", "ka"),
                lower=0.8,
                upper=1.2,
                posterior_mean=1.0,
                posterior_std=0.1,
            ),
        ),
        count=3,
        strategy="prior_exploration",
        seed=17,
        candidate_prefix="al-test",
        payload_template={
            "geometry": {"radGV": 2.0, "height": 14.28},
            "material_parameters": {
                "ka": 1.1,
                "kb": 1.2,
                "mu": 0.9,
                "b1": 0.1,
                "b2": 0.2,
                "a3": 0.3,
                "a4": 0.4,
                "mu_l": 0.5,
                "c": 0.6,
            },
            "controls": {"tot_force": [500.0, 750.0]},
        },
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]):
    rc = main(argv)
    captured = capsys.readouterr()
    return rc, captured.out.strip()


def _has_meso_uq_al_entrypoint() -> bool:
    in_scripts_section = False
    for line in PYPROJECT_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_scripts_section = stripped == "[project.scripts]"
            continue
        if in_scripts_section and _AL_ENTRYPOINT_PATTERN.match(line):
            return True
    return False


def _read_png_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    return data


def test_build_parser_exposes_active_learning_subcommands() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    assert "validate" in help_text
    assert "generate" in help_text
    assert "acquire" in help_text
    assert "Dry-run/render-only mode is always active" in help_text


def test_validate_generation_config_returns_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "generation_config.json"
    config_path.write_text(json.dumps(_gv_candidate_generation_config().as_dict()), encoding="utf-8")

    rc, payload_json = _run_cli(["validate", str(config_path)], capsys)
    payload = json.loads(payload_json)

    assert rc == 0
    assert payload["command"] == "validate"
    assert payload["kind"] == "generation"
    assert payload["result"]["experiment"] == "stretching"


def test_validate_candidate_inputs_returns_nonzero_for_invalid_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad_candidates = tmp_path / "candidates.yaml"
    bad_candidates.write_text("not-json", encoding="utf-8")

    rc, payload_json = _run_cli(["validate", str(bad_candidates)], capsys)

    assert rc != 0
    assert payload_json == ""


def test_generate_command_writes_candidate_generation_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "generation_config.yaml"
    config_path.write_text(
        json.dumps(_gv_candidate_generation_config().as_dict()),
        encoding="utf-8",
    )

    output_root = tmp_path / "al_outputs"
    run_id = "cli-generate"
    iteration = 2

    rc, payload_json = _run_cli(
        [
            "generate",
            str(config_path),
            "--run-id",
            run_id,
            "--iteration",
            str(iteration),
            "--output-root",
            str(output_root),
        ],
        capsys,
    )
    payload = json.loads(payload_json)

    assert rc == 0
    assert payload["command"] == "generate"
    artifact_dir = output_root / run_id / "iterations" / f"iter_{iteration:04d}"
    assert payload["candidate_count"] == 3
    assert payload["artifacts"]["manifest_path"] == str(artifact_dir / CANDIDATE_GENERATION_MANIFEST_FILENAME)
    assert payload["artifacts"]["report_path"] == str(artifact_dir / CANDIDATE_GENERATION_REPORT_FILENAME)
    assert payload["artifacts"]["plot_path"] == str(artifact_dir / CANDIDATE_GENERATION_PLOT_FILENAME)
    assert payload["artifacts"]["plot_sidecar_path"] == str(
        artifact_dir / CANDIDATE_GENERATION_PLOT_SIDECAR_FILENAME
    )
    assert payload["artifacts"]["validation_report_path"] == str(
        artifact_dir / "candidate_validation_report.json"
    )
    assert Path(payload["artifacts"]["manifest_path"]).is_file()
    assert Path(payload["artifacts"]["report_path"]).is_file()
    assert Path(payload["artifacts"]["plot_path"]).is_file()
    assert Path(payload["artifacts"]["plot_sidecar_path"]).is_file()
    assert Path(payload["artifacts"]["validation_report_path"]).is_file()

    validation = _read_json(artifact_dir / "candidate_validation_report.json")
    assert validation["candidate_count"] == 3
    assert validation["rejected_candidate_ids"] == []

    generation_manifest = _read_json(artifact_dir / CANDIDATE_GENERATION_MANIFEST_FILENAME)
    assert generation_manifest["candidate_count"] == 3


def test_acquire_command_writes_artifacts_and_plot_metadata(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(
        json.dumps(
            (
                {
                    "candidate_id": "first",
                    "parameters": {"family": "gv", "uncertainty": 0.81},
                },
                {
                    "candidate_id": "second",
                    "parameters": {"family": "gv", "uncertainty": 0.42},
                },
            )
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "acquire_outputs"
    run_id = "cli-acquire"
    iteration = 4

    rc, payload_json = _run_cli(
        [
            "acquire",
            str(candidates_path),
            "--policy",
            "uncertainty",
            "--run-id",
            run_id,
            "--iteration",
            str(iteration),
            "--output-root",
            str(output_root),
        ],
        capsys,
    )
    payload = json.loads(payload_json)

    assert rc == 0
    assert payload["command"] == "acquire"
    artifact_dir = output_root / run_id / "iterations" / f"iter_{iteration:04d}"
    assert payload["artifacts"]["manifest_path"] == str(artifact_dir / ACQUISITION_MANIFEST_FILENAME)
    assert payload["artifacts"]["report_path"] == str(artifact_dir / ACQUISITION_REPORT_FILENAME)
    assert payload["artifacts"]["plot_path"] == str(artifact_dir / ACQUISITION_PLOT_FILENAME)
    assert payload["artifacts"]["plot_sidecar_path"] == str(artifact_dir / ACQUISITION_PLOT_SIDECAR_FILENAME)
    assert Path(payload["artifacts"]["manifest_path"]).is_file()
    assert Path(payload["artifacts"]["report_path"]).is_file()
    assert Path(payload["artifacts"]["plot_path"]).is_file()
    assert Path(payload["artifacts"]["plot_sidecar_path"]).is_file()

    report = _read_json(artifact_dir / ACQUISITION_PLOT_SIDECAR_FILENAME)
    assert report["plot"] == ACQUISITION_PLOT_FILENAME


def test_generate_command_no_plot_writes_fallback_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "generation_config.yaml"
    config_path.write_text(
        json.dumps(_gv_candidate_generation_config().as_dict()),
        encoding="utf-8",
    )

    output_root = tmp_path / "al_outputs_no_plot"
    run_id = "cli-generate-no-plot"
    iteration = 1

    rc, payload_json = _run_cli(
        [
            "generate",
            str(config_path),
            "--run-id",
            run_id,
            "--iteration",
            str(iteration),
            "--output-root",
            str(output_root),
            "--no-plot",
        ],
        capsys,
    )
    payload = json.loads(payload_json)

    assert rc == 0
    plot_path = Path(payload["artifacts"]["plot_path"])
    plot_sidecar_path = Path(payload["artifacts"]["plot_sidecar_path"])
    validation_report_path = Path(payload["artifacts"]["validation_report_path"])

    assert plot_path.is_file()
    assert plot_sidecar_path.is_file()
    assert validation_report_path.is_file()
    assert _read_png_bytes(plot_path) == _FALLBACK_PNG

    plot_sidecar = _read_json(plot_sidecar_path)
    assert plot_sidecar["plot"] == str(plot_path)

    validation_report = _read_json(validation_report_path)
    assert validation_report["plot"] == CANDIDATE_GENERATION_PLOT_FILENAME
    assert validation_report["plot_sidecar"] == CANDIDATE_GENERATION_PLOT_SIDECAR_FILENAME


def test_acquire_command_no_plot_writes_fallback_plot_and_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(
        json.dumps(
            (
                {
                    "candidate_id": "first",
                    "parameters": {"family": "gv", "uncertainty": 0.31},
                },
                {
                    "candidate_id": "second",
                    "parameters": {"family": "gv", "uncertainty": 0.28},
                },
            )
        ),
        encoding="utf-8",
    )

    output_root = tmp_path / "acquire_outputs_no_plot"
    run_id = "cli-acquire-no-plot"
    iteration = 3

    rc, payload_json = _run_cli(
        [
            "acquire",
            str(candidates_path),
            "--policy",
            "uncertainty",
            "--run-id",
            run_id,
            "--iteration",
            str(iteration),
            "--output-root",
            str(output_root),
            "--no-plot",
        ],
        capsys,
    )
    payload = json.loads(payload_json)

    assert rc == 0
    plot_path = Path(payload["artifacts"]["plot_path"])
    plot_sidecar_path = Path(payload["artifacts"]["plot_sidecar_path"])
    manifest_path = Path(payload["artifacts"]["manifest_path"])
    report_path = Path(payload["artifacts"]["report_path"])

    assert plot_path.is_file()
    assert plot_sidecar_path.is_file()
    assert manifest_path.is_file()
    assert report_path.is_file()

    plot_sidecar = _read_json(plot_sidecar_path)
    assert plot_sidecar["plot"] == ACQUISITION_PLOT_FILENAME
    assert plot_path.read_bytes() == _build_fallback_png(plot_sidecar)


def test_al_cli_entrypoint_is_declared_in_pyproject_metadata() -> None:
    assert _has_meso_uq_al_entrypoint()
