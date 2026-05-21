from __future__ import annotations

import json
from pathlib import Path

from meso_uq.structures.gv.launch import (
    render_gv_launch_campaigns,
    validate_gv_launch_request,
)


_VALID_MATERIAL_PARAMETERS = {
    "ka": 1.1,
    "kb": 2.2,
    "mu": 3.3,
    "b1": 4.4,
    "b2": 5.5,
    "a3": 6.6,
    "a4": 7.7,
    "mu_l": 8.8,
    "c": 9.9,
}

_GEOMETRY = {"radGV": 2.0, "height": 14.28}
_LANE_CONTROLS = {
    "stretching": {"tot_force": (500.0, 750.0), "bpress": -91.0},
    "buckling": {"buck": (0.2, 0.4), "bpress": -91.0},
    "torsion": {"theta": (0.05, 0.1)},
    "eigenmodes": {"bpress": (-91.0, -90.0)},
}
_LANE_WALLTIMES = {
    "stretching": "01:00:00",
    "buckling": "02:00:00",
    "torsion": "01:30:00",
    "eigenmodes": "03:00:00",
}


def _requests_for(platform: str):
    return tuple(
        validate_gv_launch_request(
            experiment=experiment,
            material_parameters=_VALID_MATERIAL_PARAMETERS,
            geometry=_GEOMETRY,
            controls=controls,
            platform=platform,
            output_root=f"_runs/gv/launch_validation/{platform}/{experiment}",
            walltime=_LANE_WALLTIMES[experiment],
            gpu_count=1,
            provenance_tags={
                "linear_issue": "MES-182" if platform == "karolina" else "MES-183",
                "lane": experiment,
                "validation": "render-only",
            },
        )
        for experiment, controls in _LANE_CONTROLS.items()
    )


def _scheduler_submission_commands(script: str) -> list[str]:
    commands = {"sbatch", "srun", "qsub", "qstat", "qdel", "scancel"}
    matches: list[str] = []
    for line in script.splitlines():
        stripped = line.strip().lower()
        if not stripped or stripped.startswith("#"):
            continue
        command = stripped.split(maxsplit=1)[0]
        if command in commands:
            matches.append(line)
    return matches


def _assert_generated_files_payload(payload: dict[str, object]) -> None:
    generated_files = payload["generated_files"]
    assert generated_files == [
        payload["campaign_manifest_path"],
        *payload["generated_script_paths"],
    ]
    assert all(Path(path).is_file() for path in generated_files)


def test_karolina_render_validation_covers_four_non_shear_gv_lanes(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    rendered = render_gv_launch_campaigns(_requests_for("karolina"), platforms="karolina")

    assert [item.request.experiment for item in rendered] == [
        "stretching",
        "buckling",
        "torsion",
        "eigenmodes",
    ]
    for item in rendered:
        payload = json.loads(item.manifest_path.read_text(encoding="utf-8"))
        script_info = payload["scheduler_scripts"][0]
        script = Path(script_info["script_path"]).read_text(encoding="utf-8")

        assert payload["platform"] == "karolina"
        assert payload["provenance"]["tags"]["validation"] == "render-only"
        assert payload["provenance"]["git"]["head"]
        assert payload["provenance"]["git"]["commit"]
        assert payload["output_root"].startswith("_runs/gv/launch_validation/karolina/")
        assert payload["expected_hdf5_datasets"]["campaign"]["hdf5_path"].startswith(
            "_runs/gv/launch_validation/karolina/"
        )
        assert script_info["runtime_environment_path"] == (
            "${MESOUQ_GV_ENV_SCRIPT:-${MESOUQ_SITE_RUNTIME_ROOT}/env/env.sh}"
        )
        assert script_info["runtime_script"] == "scripts/platforms/hpc/run_gv_runtime.py"
        assert script_info["gpu_resource_directives"] == ["#SBATCH --gpus=1"]
        assert script_info["operator_checks"]
        _assert_generated_files_payload(payload)

        assert "#SBATCH --account=eu-26-17" in script
        assert "#SBATCH --partition=qgpu" in script
        assert "#SBATCH --gpus=1" in script
        assert 'source "${REPO_ROOT}/scripts/platforms/karolina/env_karolina.sh"' in script
        assert "scripts/platforms/hpc/run_gv_runtime.py" in script
        assert "--site karolina" in script
        assert _scheduler_submission_commands(script) == []


def test_vega_render_validation_covers_four_non_shear_gv_lanes(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    rendered = render_gv_launch_campaigns(_requests_for("vega"), platforms="vega")

    assert [item.request.experiment for item in rendered] == [
        "stretching",
        "buckling",
        "torsion",
        "eigenmodes",
    ]
    for item in rendered:
        payload = json.loads(item.manifest_path.read_text(encoding="utf-8"))
        script_info = payload["scheduler_scripts"][0]
        script = Path(script_info["script_path"]).read_text(encoding="utf-8")

        assert payload["platform"] == "vega"
        assert payload["walltime"] == _LANE_WALLTIMES[item.request.experiment]
        assert payload["provenance"]["tags"]["validation"] == "render-only"
        assert payload["provenance"]["git"]["head"]
        assert payload["provenance"]["git"]["commit"]
        assert payload["output_root"].startswith("_runs/gv/launch_validation/vega/")
        assert script_info["runtime_environment_path"] == (
            "${MESOUQ_GV_ENV_SCRIPT:-${REPO_ROOT}/_vega/env/env.sh}"
        )
        assert script_info["runtime_script"] == "scripts/platforms/hpc/run_gv_runtime.py"
        assert script_info["gpu_resource_directives"] == ["#SBATCH --gres=gpu:1"]
        assert any("maintenance" in check for check in script_info["operator_checks"])
        _assert_generated_files_payload(payload)

        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "module purge" in script
        assert "_vega/env/env.sh" in script
        assert "scripts/platforms/hpc/run_gv_runtime.py" in script
        assert "--site vega" in script
        assert _scheduler_submission_commands(script) == []
