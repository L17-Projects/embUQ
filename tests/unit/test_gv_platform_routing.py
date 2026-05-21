from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


_VALID_MATERIAL_PARAMETER_OVERRIDES = {
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


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gv_platform_vega_runtime_wrapper_forwards_platform_flag(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/vega/run_gv_runtime.py"),
        "gv_platform_vega_runtime_wrapper_test",
    )
    captured: list[list[str]] = []

    def _fake_call(cmd, **_kwargs) -> int:
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)

    rc = module.main(["--selection", "gv:stretching"])

    assert rc == 0
    assert captured
    rendered = " ".join(captured[0])
    assert "scripts/platforms/hpc/run_gv_runtime.py" in rendered
    assert "--site vega" in rendered
    assert "--selection gv:stretching" in rendered


def test_gv_platform_karolina_runtime_wrapper_forwards_platform_flag(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/karolina/run_gv_runtime.py"),
        "gv_platform_karolina_runtime_wrapper_test",
    )
    captured: list[list[str]] = []

    def _fake_call(cmd) -> int:
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)

    rc = module.main(["--selection", "gv:torsion"])

    assert rc == 0
    assert captured
    joined = " ".join(captured[0])
    assert "scripts/platforms/hpc/run_gv_runtime.py" in joined
    assert "--site karolina" in joined
    assert "--selection gv:torsion" in joined


def test_gv_platform_hpc_runtime_wrapper_dispatches_by_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_gv_runtime.py"),
        "gv_platform_hpc_runtime_wrapper_test",
    )
    captured: list[list[str]] = []

    def _fake_call(cmd, **_kwargs) -> int:
        captured.append(list(cmd))
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)
    monkeypatch.setenv("MESOUQ_SITE", "karolina")

    rc = module.main(["--selection", "gv:shear_flow"])

    assert rc == 0
    assert captured
    joined = " ".join(captured[0])
    assert "scripts/workflows/gv/run_gv_runtime.py" in joined
    assert "--platform karolina" in joined
    assert "--selection gv:shear_flow" in joined


def test_gv_platform_hpc_runtime_wrapper_rejects_unknown_site(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/platforms/hpc/run_gv_runtime.py"),
        "gv_platform_hpc_runtime_wrapper_unknown_site_test",
    )
    monkeypatch.setenv("MESOUQ_SITE", "unknown")

    assert module.main(["--selection", "gv:stretching"]) == 2


def test_gv_runtime_rendering_targets_staged_work_dir_and_generates_scheduler(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_rendering_workflow_test",
    )

    def _fake_dry_run(argv: list[str]) -> int:
        parsed_root = Path(argv[argv.index("--output-root") + 1]).expanduser().resolve()
        parsed_root.mkdir(parents=True, exist_ok=True)
        work_dir = (
            parsed_root
            / "shear_flow"
            / "gv_rad2_height14_28"
            / "ptan_0_4__afsi_0__bpress_-91"
            / "work"
        )
        manifest = {
            "structure": "gv",
            "experiment": "shear_flow",
            "geometry": "gv_rad2_height14_28",
            "controls": {
                "ptan": 0.4,
                "afsi": 0.0,
                "bpress": -91.0,
            },
            "control_id": "ptan_0_4__afsi_0__bpress_-91",
            "dataset_id": "gv:shear_flow:gv_rad2_height14_28:ptan_0_4__afsi_0__bpress_-91",
            "output_root": str(parsed_root),
            "work_dir": str(work_dir),
            "commands": [
                {
                    "argv": ["python3", "generate.py", "--object", "gv", "--parallel", "--first"],
                    "cwd": "{work_dir}",
                },
                {"argv": ["sbatch", "run_HPC.sbatch"], "cwd": "{work_dir}"},
            ],
            "analysis_commands": [
                {"argv": ["bash", "commands.txt"], "cwd": "analysis"},
            ],
            "generated_subdirs": ["logs", "analysis", "mesh"],
            "material_parameter_overrides": {
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
            "runtime_package": "mirheoOBMD",
        }
        (parsed_root / "gv_runtime_dry_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return 0

    monkeypatch.setattr(module, "RUN_GV_DRY_RUN_MAIN", _fake_dry_run)

    captured_runs: list[tuple[tuple[str, ...], str]] = []

    def _run(commands, **kwargs) -> None:
        captured_runs.append((tuple(commands), kwargs.get("cwd", "")))
        raise AssertionError("runtime commands should not execute under --dry-run")

    monkeypatch.setattr(module.subprocess, "run", _run)

    rc = module.main(
        [
            "--selection",
            "gv:shear_flow",
            "--output-root",
            str(tmp_path / "runtime"),
            "--include-experimental",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert not captured_runs
    runtime_output_root = tmp_path / "runtime"
    manifest = json.loads((runtime_output_root / "gv_runtime_render_manifest.json").read_text(encoding="utf-8"))
    work_dir = runtime_output_root / "shear_flow" / "gv_rad2_height14_28" / "ptan_0_4__afsi_0__bpress_-91" / "work"
    assert work_dir.is_dir()
    commands_txt = work_dir / "commands.txt"
    assert commands_txt.is_file()
    assert (work_dir / "run_HPC.sbatch").is_file()
    assert manifest["dry_run"] is True
    assert manifest["commands"][0]["status"] == "skipped-dry-run"
    assert manifest["generated_scheduler_scripts"] == ["run_HPC.sbatch"]
    contents = commands_txt.read_text(encoding="utf-8")
    assert "generate.py" in contents
    assert "run_HPC.sbatch" in contents
    assert "_vega/env/env.sh" in contents
    assert "_vega/mirheo/env.sh" not in contents
    assert "Missing required GV runtime environment" in contents
    assert "MESOUQ_GV_MATERIAL_OVERRIDES_JSON" in contents
    assert '"mu_l": 0.5' in contents
    assert "source" in contents

    scheduler_contents = (work_dir / "run_HPC.sbatch").read_text(encoding="utf-8")
    assert "_vega/env/env.sh" in scheduler_contents
    assert "_vega/mirheo/env.sh" not in scheduler_contents
    assert "Missing required GV runtime environment" in scheduler_contents


def test_gv_runtime_resolves_karolina_env_script_from_site_runtime_root(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_karolina_env_resolution_test",
    )
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("MESOUQ_SITE_RUNTIME_ROOT", str(runtime_root))

    env_script = module._resolve_gv_env_script("karolina")

    assert env_script == (runtime_root / "env" / "env.sh").resolve()


def test_gv_runtime_resolves_explicit_env_script_override(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_explicit_env_resolution_test",
    )
    override = tmp_path / "custom" / "gv-env.sh"
    monkeypatch.setenv("MESOUQ_GV_ENV_SCRIPT", str(override))

    env_script = module._resolve_gv_env_script("karolina")

    assert env_script == override.resolve()


def test_gv_runtime_karolina_default_output_root_uses_scratch_runs_root(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_karolina_output_root_test",
    )
    runs_root = tmp_path / "runs"
    monkeypatch.setenv("MESOUQ_RUNS_ROOT", str(runs_root))
    args = Namespace(output_root=None, platform="karolina", run_tag="tag1")

    output_root = module._resolve_output_root(args)

    assert output_root == (runs_root / "gv" / "runtime" / "tag1").resolve()


def test_gv_runtime_karolina_default_output_root_falls_back_to_repo_runs(monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_karolina_repo_output_root_test",
    )
    monkeypatch.delenv("MESOUQ_RUNS_ROOT", raising=False)
    args = Namespace(output_root=None, platform="karolina", run_tag=None)

    output_root = module._resolve_output_root(args)

    assert output_root == module.REPO_ROOT / "_runs" / "karolina" / "gv" / "runtime"


def test_gv_dry_run_material_parser_validates_overrides() -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_dry_run.py"),
        "gv_dry_run_material_parser_validation_test",
    )
    raw_overrides = [f"{name}={value}" for name, value in _VALID_MATERIAL_PARAMETER_OVERRIDES.items()]

    parsed_overrides = module._parse_material_overrides(raw_overrides)
    assert parsed_overrides == _VALID_MATERIAL_PARAMETER_OVERRIDES
    assert module._parse_material_overrides([]) is None

    with pytest.raises(ValueError, match="Expected NAME=VALUE"):
        module._parse_material_overrides(["ka1.1"])
    with pytest.raises(ValueError, match="duplicated"):
        module._parse_material_overrides(["ka=1.1", "ka=1.2"])
    with pytest.raises(ValueError, match="Missing required GV material parameters"):
        module._parse_material_overrides(["ka=1.1", "kb=1.2"])
    with pytest.raises(ValueError, match="Unexpected material parameter"):
        module._parse_material_overrides(["ka=1.1", "oops=2.0"])


@pytest.mark.parametrize("experiment_name", ("stretching", "buckling", "torsion"))
def test_gv_generated_mirheo_jobs_source_runtime_environment_and_preserve_materials(
    experiment_name: str,
) -> None:
    generate_script = Path("gv") / experiment_name / "src" / "generate.py"
    contents = generate_script.read_text(encoding="utf-8")

    assert "MESOUQ_GV_ENV_SCRIPT" in contents
    assert "MESOUQ_SITE_RUNTIME_ROOT" in contents
    assert "f'_{site}'" in contents
    assert "_vega' / 'env' / 'env.sh" not in contents
    assert "source {shlex.quote(env_script)}" in contents
    assert "MESOUQ_GV_MATERIAL_OVERRIDES_JSON" in contents
    assert "MESOUQ_GV_MIRHEO_MODULE" in contents
    if experiment_name == "buckling":
        assert "MESOUQ_GV_BUCKLING_FLUID_MODE" in contents
        assert "MESOUQ_GV_BUCKLING_FLUID_STABILIZATION" in contents
        assert "MESOUQ_GV_BUCKLING_PIN_OBJECT" in contents
        assert "MESOUQ_GV_BUCKLING_ODPD_AMP_SCALE" in contents
    assert "bash commands.txt" in contents
    assert "num_mpi_ranks = int(os.environ.get('MESOUQ_GV_MPI_RANKS', str(num_gpus + 1)))" in contents
    assert "#SBATCH --ntasks-per-node={num_mpi_ranks}" in contents
    assert "f'{num_mpi_ranks}'" in contents
    assert "module load Python/3.10.8-GCCcore-12.2.0" in contents
    assert "module load OpenMPI/4.1.4-GCC-12.2.0" in contents
    assert "module load CUDA/12.2.2" in contents


def test_gv_eigenmodes_runtime_allocates_postprocess_rank_by_default() -> None:
    generate_contents = Path("gv/eigenmodes/src/generate.py").read_text(encoding="utf-8")
    run_contents = Path("gv/eigenmodes/src/run.sh").read_text(encoding="utf-8")

    assert "MESOUQ_GV_ENV_SCRIPT" in generate_contents
    assert "MESOUQ_GV_MATERIAL_OVERRIDES_JSON" in generate_contents
    assert "MESOUQ_GV_MIRHEO_MODULE" in generate_contents
    assert "MESOUQ_GV_EIGENMODES_MPI_RANKS" in generate_contents
    assert "def _default_mpi_ranks" in generate_contents
    assert "return 2 * product" in generate_contents
    assert "str(_default_mpi_ranks(domain_ranks))" in generate_contents
    assert "MESOUQ_GV_PAPER_EXACT" in generate_contents
    assert "'numsteps': 40000000" in generate_contents
    assert "'numsteps_eq': 500000" in generate_contents
    assert "'stslik': 200000" in generate_contents
    assert "'numsteps': 4000000" in generate_contents
    assert "'stslik': 20000" in generate_contents
    assert "MESOUQ_GV_EIGENMODES_PROFILE" in generate_contents
    assert "MESOUQ_GV_EIGENMODES_MODE_WINDOW_POLICY" in generate_contents
    assert "MESOUQ_GV_EIGENMODES_MODE_MIN_FREQUENCY" in generate_contents
    assert "MESOUQ_GV_EIGENMODES_MODE_COUNT" in generate_contents
    assert "'configured_final_mode_count': mode_count" in generate_contents
    assert "'selected_paper_mode_indices': list(range(mode_count))" in generate_contents
    assert "MESOUQ_GV_EIGENMODES_NUMSTEPS" in generate_contents
    assert "MESOUQ_GV_EIGENMODES_STSLIK" in generate_contents
    assert "MESOUQ_GV_EIGENMODES_DOMAIN_RANKS" in generate_contents
    assert "domain_ranks = _format_domain_ranks(" in generate_contents
    assert "'gamma_dpd_gas': 3.0" in generate_contents
    assert "MESOUQ_GV_MPI_RANKS" not in generate_contents
    assert "nranks=${3:-${MESOUQ_GV_EIGENMODES_MPI_RANKS:-2}}" in run_contents
    assert "domain_ranks=${4:-${MESOUQ_GV_EIGENMODES_DOMAIN_RANKS:-1,1,1}}" in run_contents
    assert '--domain-ranks "${domain_ranks}"' in run_contents
    assert 'mpirun --bind-to none -np "${nranks}"' in run_contents
    equil_contents = Path("gv/eigenmodes/src/equil.py").read_text(encoding="utf-8")
    assert "def _parse_domain_ranks" in equil_contents
    assert "MESOUQ_GV_EIGENMODES_DOMAIN_RANKS" in equil_contents
    assert "parser.add_argument(\n    '--domain-ranks'" in equil_contents


@pytest.mark.parametrize("experiment_name", ("stretching", "buckling", "torsion"))
def test_gv_mirheo_launchers_disable_openmpi_binding_on_vega(experiment_name: str) -> None:
    run_script = Path("gv") / experiment_name / "src" / "run.sh"
    contents = run_script.read_text(encoding="utf-8")

    assert "nranks=${3:-${MESOUQ_GV_MPI_RANKS:-2}}" in contents
    assert "mpirun --bind-to none" in contents
    assert "-np ${nranks}" in contents
    if experiment_name == "buckling":
        assert "MESOUQ_OPENMPI_LIB_DIR" in contents
        assert "export LD_LIBRARY_PATH=" in contents
        assert "-x LD_LIBRARY_PATH" in contents
        assert "-x MESOUQ_GV_BUCKLING_FLUID_STABILIZATION" in contents


def test_gv_eigenmodes_analysis_accepts_restart_backed_trajectory() -> None:
    analysis_root = Path("gv/eigenmodes/src/analysis")

    combine_contents = (analysis_root / "combine.py").read_text(encoding="utf-8")
    assert "combine_restart_position_files" in combine_contents
    assert "emb.PV-*.h5" in combine_contents
    assert "../restart" in combine_contents

    initial_contents = (analysis_root / "initial.py").read_text(encoding="utf-8")
    assert "emb_0000000.xyz" in initial_contents
    assert "sim{simnum}eq" in initial_contents
    assert "Using equilibrated eigenmode reference frame" in initial_contents

    all_analysis_contents = (analysis_root / "all_analysis.py").read_text(encoding="utf-8")
    assert "np.linalg.svd(trj_np, full_matrices=False)" in all_analysis_contents
    assert "reference_positions = av" in all_analysis_contents


def test_gv_dry_run_plan_receives_material_overrides_only_when_provided() -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_dry_run.py"),
        "gv_dry_run_material_plan_call_test",
    )
    captured: dict[str, object] = {}

    class _Descriptor:
        def plan(self, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return {"material_parameter_overrides": kwargs.get("material_parameter_overrides", {})}

    request = {
        "geometry": "gv_rad2_height14_28",
        "controls": {},
        "output_root": Path("unused"),
        "include_experimental": False,
    }

    module._plan_runtime_dry_run(
        _Descriptor(),
        {**request, "material_parameter_overrides": _VALID_MATERIAL_PARAMETER_OVERRIDES},
    )
    assert captured["material_parameter_overrides"] == _VALID_MATERIAL_PARAMETER_OVERRIDES

    module._plan_runtime_dry_run(_Descriptor(), {**request, "material_parameter_overrides": None})
    assert "material_parameter_overrides" not in captured


def test_gv_runtime_builds_and_forwards_material_arguments_to_dry_run(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_material_forwarding_test",
    )
    captured_argv: list[str] = []

    def _fake_dry_run(argv: list[str]) -> int:
        captured_argv.extend(argv)
        parsed_root = Path(argv[argv.index("--output-root") + 1]).expanduser().resolve()
        parsed_root.mkdir(parents=True, exist_ok=True)
        work_dir = (
            parsed_root
            / "stretching"
            / "gv_rad2_height14_28"
            / "theta_0_03"
            / "work"
        )
        overrides = {}
        for index, item in enumerate(argv):
            if item == "--material":
                name, raw_value = argv[index + 1].split("=", 1)
                overrides[name] = float(raw_value)
        manifest = {
            "structure": "gv",
            "experiment": "stretching",
            "geometry": "gv_rad2_height14_28",
            "controls": {"theta": 0.03},
            "control_id": "theta_0_03",
            "dataset_id": "gv:stretching:gv_rad2_height14_28:theta_0_03",
            "output_root": str(parsed_root),
            "work_dir": str(work_dir),
            "commands": [{"argv": ["bash", "run.sh"], "cwd": "{work_dir}"}],
            "material_parameter_overrides": overrides,
        }
        work_dir.mkdir(parents=True, exist_ok=True)
        (parsed_root / "gv_runtime_dry_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return 0

    monkeypatch.setattr(module, "RUN_GV_DRY_RUN_MAIN", _fake_dry_run)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    material_args = [f"{name}={value}" for name, value in _VALID_MATERIAL_PARAMETER_OVERRIDES.items()]
    material_flags: list[str] = []
    for item in material_args:
        material_flags.extend(["--material", item])

    rc = module.main(
        [
            "--selection",
            "gv:stretching",
            "--output-root",
            str(tmp_path / "runtime"),
            "--control",
            "theta=0.03",
            *material_flags,
            "--dry-run",
        ]
    )

    assert rc == 0
    assert captured_argv.count("--material") == len(material_args)
    for item in material_args:
        assert item in captured_argv
    runtime_output_root = tmp_path / "runtime"
    manifest = json.loads((runtime_output_root / "gv_runtime_render_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    commands_txt = (
        runtime_output_root
        / "stretching"
        / "gv_rad2_height14_28"
        / "theta_0_03"
        / "work"
        / "commands.txt"
    )
    contents = commands_txt.read_text(encoding="utf-8")
    assert "MESOUQ_GV_MATERIAL_OVERRIDES_JSON" in contents


def test_gv_runtime_workflow_helper_edge_cases(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_rendering_workflow_helper_edges_test",
    )

    args = Namespace(
        structure="gv",
        selection=None,
        experiment="torsion",
        geometry_id="gv_rad2_height14_28",
        radius=2.0,
        height=14.28,
        control=["theta=0.03"],
        output_root=None,
        run_tag="tagged-run",
        platform="vega",
        include_experimental=True,
    )

    assert module._resolve_output_root(args).name == "tagged-run"
    assert module._build_runtime_argv(args) == [
        "--structure",
        "gv",
        "--experiment",
        "torsion",
        "--geometry-id",
        "gv_rad2_height14_28",
        "--radius",
        "2.0",
        "--height",
        "14.28",
        "--control",
        "theta=0.03",
        "--run-tag",
        "tagged-run",
        "--include-experimental",
    ]

    explicit_root = tmp_path / "explicit-root"
    args.output_root = str(explicit_root)
    assert module._resolve_output_root(args) == explicit_root.resolve()

    args.selection = "gv:torsion"
    assert module._build_runtime_argv(args)[:4] == ["--structure", "gv", "--selection", "gv:torsion"]

    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        module._load_runtime_manifest(list_json)


def test_gv_runtime_command_execution_exports_material_overrides(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_material_override_env_test",
    )
    captured_env: list[str] = []

    def _fake_run(command, cwd, env, capture_output, text, check):
        captured_env.append(env["MESOUQ_GV_MATERIAL_OVERRIDES_JSON"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    records, returncode = module._run_commands(
        commands=[(("bash", "run.sh"), tmp_path)],
        dry_run=False,
        material_overrides_json='{"ka": 1.1}',
    )

    assert returncode == 0
    assert records[0]["status"] == "completed"
    assert captured_env == ['{"ka": 1.1}']

    missing_json = tmp_path / "missing.json"
    missing_json.write_text(json.dumps({"structure": "gv"}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        module._load_runtime_manifest(missing_json)

    bad_controls_json = tmp_path / "bad-controls.json"
    bad_controls_json.write_text(
        json.dumps(
            {
                "structure": "gv",
                "experiment": "torsion",
                "geometry": "gv_rad2_height14_28",
                "controls": [],
                "control_id": "theta_0_03",
                "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0_03",
                "work_dir": str(tmp_path / "work"),
                "output_root": str(tmp_path),
                "commands": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="controls"):
        module._load_runtime_manifest(bad_controls_json)


def test_gv_runtime_command_execution_propagates_runtime_env_overrides(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_site_env_override_test",
    )
    captured_env: list[dict[str, str]] = []

    def _fake_run(command, cwd, env, capture_output, text, check):
        captured_env.append(dict(env))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    records, returncode = module._run_commands(
        commands=[(("python3", "generate.py"), tmp_path)],
        dry_run=False,
        material_overrides_json='{"ka": 1.1}',
        env_overrides={
            "MESOUQ_SITE": "karolina",
            "MESOUQ_GV_ENV_SCRIPT": "/scratch/mesouq/runtime/env/env.sh",
        },
    )

    assert returncode == 0
    assert records[0]["status"] == "completed"
    assert captured_env[0]["MESOUQ_SITE"] == "karolina"
    assert captured_env[0]["MESOUQ_GV_ENV_SCRIPT"] == "/scratch/mesouq/runtime/env/env.sh"
    assert captured_env[0]["MESOUQ_GV_MATERIAL_OVERRIDES_JSON"] == '{"ka": 1.1}'


def test_gv_runtime_command_execution_without_material_overrides_does_not_pass_env(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_material_override_absent_env_test",
    )
    captured_kwargs: list[dict[str, object]] = []

    def _fake_run(command, **kwargs):
        captured_kwargs.append(dict(kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    records, returncode = module._run_commands(
        commands=[(("bash", "run.sh"), tmp_path)],
        dry_run=False,
    )

    assert returncode == 0
    assert records[0]["status"] == "completed"
    assert "env" not in captured_kwargs[0]


def test_gv_runtime_workflow_passes_karolina_env_to_subprocesses(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_karolina_subprocess_env_test",
    )
    output_root = tmp_path / "runtime"
    work_dir = output_root / "torsion" / "gv_rad2_height14_28" / "theta_0_03" / "work"
    manifest = {
        "structure": "gv",
        "experiment": "torsion",
        "geometry": "gv_rad2_height14_28",
        "controls": {"theta": 0.03},
        "control_id": "theta_0_03",
        "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0_03",
        "output_root": str(output_root),
        "work_dir": str(work_dir),
        "commands": [{"argv": ["python3", "generate.py"], "cwd": "{work_dir}"}],
    }

    def _fake_dry_run(argv: list[str]) -> int:
        output_root.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        (output_root / module.GV_RUNTIME_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        return 0

    captured_env: list[dict[str, str]] = []

    def _fake_run(command, *, cwd, env, capture_output, text, check):
        captured_env.append(dict(env))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module, "RUN_GV_DRY_RUN_MAIN", _fake_dry_run)
    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.delenv("MESOUQ_SITE", raising=False)
    monkeypatch.delenv("MESOUQ_GV_ENV_SCRIPT", raising=False)

    rc = module.main(["--selection", "gv:torsion", "--platform", "karolina", "--output-root", str(output_root)])

    assert rc == 0
    assert len(captured_env) == 1
    expected_env_script = str(module._resolve_gv_env_script("karolina"))
    assert captured_env[0]["MESOUQ_SITE"] == "karolina"
    assert captured_env[0]["MESOUQ_GV_ENV_SCRIPT"] == expected_env_script


def test_gv_runtime_render_manifest_records_classified_known_issues(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_known_issues_render_test",
    )

    manifest = {
        "structure": "gv",
        "experiment": "shear_flow",
        "geometry": "gv_rad2_height14_28",
        "control_id": "theta_0_03",
        "controls": {"theta": 0.03},
        "dataset_id": "gv:shear_flow:gv_rad2_height14_28:theta_0_03",
        "output_root": str(tmp_path / "runtime"),
        "work_dir": str(tmp_path / "runtime" / "work"),
        "geometry_spec": {"id": "gv_rad2_height14_28", "parameters": {"radius": 2.0, "height": 14.28}},
        "source_root": str(tmp_path / "sources" / "shear_flow"),
        "runtime_package": "mirheoOBMD",
        "experimental": True,
        "known_issues": [
            {
                "id": "bouncer_overflow",
                "summary": "Observed triangle overlap candidates.",
                "evidence": "gv/shear_flow/src/fixtures/bouncer.txt",
                "severity": "Blocking",
            },
            {
                "id": "artifact_stale",
                "summary": "Reference artifact timestamp drift.",
                "evidence": "artifacts/state.csv",
                "severity": "warning",
            },
        ],
        "commands": [],
        "analysis_commands": [],
        "generated_subdirs": [],
    }
    manifest_path = tmp_path / "runtime" / module.GV_RUNTIME_MANIFEST

    def _fake_dry_run(argv: list[str]) -> int:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        (manifest_path.parent / "work").mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return 0

    monkeypatch.setattr(module, "RUN_GV_DRY_RUN_MAIN", _fake_dry_run)
    def _forbidden_run(*args, **kwargs) -> None:
        raise AssertionError("runtime commands should not execute under --dry-run")

    monkeypatch.setattr(module.subprocess, "run", _forbidden_run)

    rc = module.main(["--selection", "gv:torsion", "--output-root", str(tmp_path / "runtime"), "--dry-run"])
    assert rc == 0

    rendered = json.loads((tmp_path / "runtime" / module.GV_RUNTIME_RENDER_MANIFEST).read_text(encoding="utf-8"))
    runtime_stage = rendered["runtime_stage"]
    assert runtime_stage["experimental"] is True
    assert runtime_stage["runtime_package"] == "mirheoOBMD"
    assert runtime_stage["runtime_package_source"] == str(manifest["source_root"])
    assert runtime_stage["blocked_issue_count"] == 1
    assert runtime_stage["known_issues"] == [
        {
            "id": "bouncer_overflow",
            "summary": "Observed triangle overlap candidates.",
            "evidence": "gv/shear_flow/src/fixtures/bouncer.txt",
            "severity": "blocking",
            "classification": "experimental_blocked",
        },
        {
            "id": "artifact_stale",
            "summary": "Reference artifact timestamp drift.",
            "evidence": "artifacts/state.csv",
            "severity": "warning",
            "classification": "observed",
        },
    ]
    assert module._normalize_known_issues("not-a-list") == []
    assert module._normalize_known_issues(["not-a-dict"]) == []

    assert module._normalize_command_value(("python3", "generate.py")) == ["python3", "generate.py"]
    with pytest.raises(ValueError, match="Expected a command list"):
        module._normalize_command_value("python3 generate.py")

    work_dir = tmp_path / "work"
    assert module._normalize_cwd(None, work_dir=work_dir) == work_dir
    assert module._normalize_cwd("analysis", work_dir=work_dir) == work_dir / "analysis"
    with pytest.raises(ValueError, match="string cwd"):
        module._normalize_cwd(3, work_dir=work_dir)

    empty_manifest = {"commands": [], "analysis_commands": [], "work_dir": str(work_dir)}
    assert module._as_command_list(empty_manifest) == []

    malformed_manifest = {"commands": ["python3 generate.py"], "work_dir": str(work_dir)}
    with pytest.raises(ValueError, match="malformed"):
        module._as_command_list(malformed_manifest)

    empty_command_manifest = {"commands": [{"argv": []}], "work_dir": str(work_dir)}
    with pytest.raises(ValueError, match="is empty"):
        module._as_command_list(empty_command_manifest)

    scheduled = module._ensure_scheduler_scripts(
        command_list=[
            ((), work_dir),
            (("python3", "generate.py"), work_dir),
            (("sbatch",), work_dir),
        ],
        manifest={"work_dir": str(work_dir)},
        platform="vega",
        gv_env_script=Path("/tmp/env/env.sh"),
    )
    assert scheduled == []

    existing_scheduler = work_dir / "existing.sbatch"
    existing_scheduler.parent.mkdir(parents=True, exist_ok=True)
    existing_scheduler.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    assert module._ensure_scheduler_scripts(
        command_list=[(("sbatch", "existing.sbatch"), work_dir)],
        manifest={"work_dir": str(work_dir)},
        platform="vega",
        gv_env_script=Path("/tmp/env/env.sh"),
    ) == []

    module._ensure_generated_directories(work_dir, {"generated_subdirs": ["logs", 3]})
    assert (work_dir / "logs").is_dir()

    calls: list[tuple[tuple[str, ...], str]] = []

    def _fake_run(command, *, cwd, capture_output, text, check):
        calls.append((tuple(command), cwd))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    records, rc = module._run_commands([(("python3", "generate.py"), work_dir)], dry_run=False)
    assert rc == 0
    assert records == [
        {
            "argv": ["python3", "generate.py"],
            "cwd": str(work_dir),
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "status": "completed",
        }
    ]
    assert calls == [(("python3", "generate.py"), str(work_dir))]

    def _fake_failed_run(command, *, cwd, capture_output, text, check):
        return SimpleNamespace(returncode=17, stdout="", stderr="failed")

    monkeypatch.setattr(module.subprocess, "run", _fake_failed_run)
    records, rc = module._run_commands([(("bash", "run.sh"), work_dir)], dry_run=False)
    assert rc == 17
    assert records[0]["status"] == "failed"


def test_gv_runtime_workflow_raises_for_failed_dry_run(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_rendering_workflow_failed_dry_run_test",
    )
    monkeypatch.setattr(module, "RUN_GV_DRY_RUN_MAIN", lambda _argv: 2)

    with pytest.raises(RuntimeError, match="dry-run failed"):
        module.main(["--selection", "gv:torsion", "--output-root", str(tmp_path)])


def test_gv_runtime_workflow_raises_for_failed_command(tmp_path, monkeypatch) -> None:
    module = _load_module(
        Path("scripts/workflows/gv/run_gv_runtime.py"),
        "gv_runtime_rendering_workflow_failed_command_test",
    )

    def _fake_dry_run(argv: list[str]) -> int:
        parsed_root = Path(argv[argv.index("--output-root") + 1]).expanduser().resolve()
        parsed_root.mkdir(parents=True, exist_ok=True)
        work_dir = parsed_root / "torsion" / "gv_rad2_height14_28" / "theta_0_03" / "work"
        manifest = {
            "structure": "gv",
            "experiment": "torsion",
            "geometry": "gv_rad2_height14_28",
            "controls": {"theta": 0.03},
            "control_id": "theta_0_03",
            "dataset_id": "gv:torsion:gv_rad2_height14_28:theta_0_03",
            "output_root": str(parsed_root),
            "work_dir": str(work_dir),
            "commands": [{"argv": ["bash", "run.sh"], "cwd": "{work_dir}"}],
        }
        (parsed_root / "gv_runtime_dry_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return 0

    monkeypatch.setattr(module, "RUN_GV_DRY_RUN_MAIN", _fake_dry_run)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=3, stdout="", stderr="boom"),
    )

    with pytest.raises(RuntimeError, match="command execution failed"):
        module.main(["--selection", "gv:torsion", "--output-root", str(tmp_path / "runtime")])
