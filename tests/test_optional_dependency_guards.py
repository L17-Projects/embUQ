from __future__ import annotations

from importlib import import_module
import sys
from pathlib import Path
from types import SimpleNamespace

from tests.support.optional_dependencies import (
    guard_cuda,
    guard_hpc,
    guard_korali,
    guard_mirheo,
    guard_mpi,
    guard_operational,
    guard_optional_module,
    guard_pyro,
    guard_slurm,
    module_is_available,
    module_runtime_usable,
    slurm_binary_available,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REQUIRED_MARKERS = (
    "gpu",
    "hpc",
    "slurm",
    "cuda",
    "mpi",
    "mirheo",
    "korali",
    "pyro",
    "operational",
    "slow",
    "integration",
)


def _configured_pytest_markers() -> set[str]:
    lines = PYPROJECT.read_text(encoding="utf-8").splitlines()
    marker_lines: list[str] = []
    in_markers_block = False
    for raw_line in lines:
        line = raw_line.strip()
        if not in_markers_block and line == "markers = [":
            in_markers_block = True
            continue
        if in_markers_block and line == "]":
            break
        if in_markers_block:
            marker_lines.append(line.strip('", '))

    declared: set[str] = set()
    for marker in marker_lines:
        if not marker:
            continue
        declared.add(marker.split(":", 1)[0].strip())
    return declared


def test_optional_dependency_marker_contract_declares_required_governance_tokens() -> None:
    declared = _configured_pytest_markers()
    missing = [name for name in REQUIRED_MARKERS if name not in declared]
    assert not missing, f"Missing pytest marker declarations: {missing}"


def test_support_module_import_is_light() -> None:
    before = set(sys.modules)
    import_module("tests.support.optional_dependencies")
    after = set(sys.modules)
    loaded_optional_modules = [
        name
        for name in ("pyro", "mpi4py", "mirheo", "korali")
        if name in after - before
    ]
    assert not loaded_optional_modules


def test_optional_module_guard_reports_missing_module_actionably() -> None:
    guard = guard_optional_module(
        "pyro",
        label="Pyro",
        purpose="Pyro-based tests",
        install_hint="the `mesouq[bnn]` extra",
        find_spec=lambda name: None,
    )
    assert guard.available is False
    assert guard.runtime_usable is False
    assert "Pyro-based tests" in guard.skip_message
    assert "pyro" in guard.skip_message
    assert "mesouq[bnn]" in guard.skip_message


def test_optional_module_guard_reports_runtime_import_failure() -> None:
    def fake_import(name: str):  # noqa: ANN001
        raise ModuleNotFoundError(f"cannot import {name}")

    guard = guard_mpi(find_spec=lambda name: object(), import_module=fake_import)
    assert guard.available is True
    assert guard.runtime_usable is False
    assert "mpi4py" in guard.fail_message
    assert "cannot import mpi4py" in guard.fail_message


def test_cuda_guard_distinguishes_requested_from_unavailable_without_torch_at_collection() -> None:
    requested_guard = guard_cuda(requested=True, import_module=lambda name: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    assert requested_guard.available is False
    assert requested_guard.runtime_usable is False
    assert "requested" in requested_guard.fail_message
    assert "unavailable" in requested_guard.fail_message

    def probe_should_not_run(_cuda_api):  # noqa: ANN001
        raise AssertionError("probe should not run when CUDA is not requested")

    cpu_guard = guard_cuda(
        requested=False,
        import_module=lambda name: (_ for _ in ()).throw(AssertionError("torch import should not be needed")),
        probe=probe_should_not_run,
    )
    assert cpu_guard.available is True
    assert cpu_guard.runtime_usable is True
    assert "not requested" in cpu_guard.skip_message


def test_slurm_detection_uses_path_and_env_simulation() -> None:
    def fake_which(binary: str, path: str | None = None) -> str | None:  # noqa: ANN001
        if binary == "sbatch" and path and "/opt/slurm/bin" in path:
            return "/opt/slurm/bin/sbatch"
        return None

    available, message = slurm_binary_available(env={"PATH": "/opt/slurm/bin:/usr/bin"}, which=fake_which)
    assert available is True
    assert "sbatch" in message

    missing, missing_message = slurm_binary_available(env={"PATH": "/usr/bin"}, which=fake_which)
    assert missing is False
    assert "Could not find" in missing_message

    guard = guard_slurm(env={"PATH": "/opt/slurm/bin:/usr/bin"}, which=fake_which)
    assert guard.available is True
    assert guard.runtime_usable is True


def test_korali_guard_distinguishes_source_present_from_runtime_usable(tmp_path) -> None:
    vendor_root = tmp_path / "extern" / "korali"
    vendor_root.mkdir(parents=True)

    guard_source_only = guard_korali(
        tmp_path,
        vendor_root=vendor_root,
        import_module=lambda name: (_ for _ in ()).throw(ModuleNotFoundError("korali missing")),
    )
    assert guard_source_only.source_present is True
    assert guard_source_only.runtime_usable is False
    assert "source is present" in guard_source_only.skip_message
    assert "runtime import is not usable" in guard_source_only.skip_message

    guard_runtime_only = guard_korali(
        tmp_path / "missing-root",
        vendor_root=tmp_path / "missing-root" / "extern" / "korali",
        import_module=lambda name: SimpleNamespace(__name__=name),
    )
    assert guard_runtime_only.source_present is False
    assert guard_runtime_only.runtime_usable is True
    assert "vendored source tree is missing" in guard_runtime_only.skip_message


def test_hpc_and_operational_guards_compose_dependency_messages() -> None:
    slurm_guard = guard_slurm(requested=False)
    mpi_guard = guard_mpi(
        find_spec=lambda name: object(),
        import_module=lambda name: SimpleNamespace(__name__=name),
    )
    hpc_guard = guard_hpc(requested=True, slurm=slurm_guard, mpi=mpi_guard)
    assert hpc_guard.available is True
    assert hpc_guard.runtime_usable is True

    failing_gpu = guard_cuda(requested=True, import_module=lambda name: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    operational_guard = guard_operational(
        requested=True,
        hpc=hpc_guard,
        gpu=failing_gpu,
        pyro=guard_pyro(
            find_spec=lambda name: object(),
            import_module=lambda name: SimpleNamespace(__name__=name),
        ),
        mirheo=guard_mirheo(
            find_spec=lambda name: object(),
            import_module=lambda name: SimpleNamespace(__name__=name),
        ),
        korali=SimpleNamespace(source_present=True, runtime_usable=True),
    )
    assert operational_guard.available is False
    assert operational_guard.runtime_usable is False
    assert "gpu" in operational_guard.fail_message


def test_hpc_and_operational_guards_do_not_probe_when_not_requested() -> None:
    def should_not_probe(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("optional runtime probe should not run")

    hpc_guard = guard_hpc(
        requested=False,
        which=should_not_probe,
        find_spec=should_not_probe,
        import_module=should_not_probe,
    )
    assert hpc_guard.available is True
    assert hpc_guard.runtime_usable is True

    operational_guard = guard_operational(
        requested=False,
        which=should_not_probe,
        find_spec=should_not_probe,
        import_module=should_not_probe,
        cuda_probe=should_not_probe,
    )
    assert operational_guard.available is True
    assert operational_guard.runtime_usable is True


def test_optional_dependency_helpers_cover_present_and_missing_simulations() -> None:
    guard = guard_mirheo(
        find_spec=lambda name: object() if name == "mirheo" else None,
        import_module=lambda name: SimpleNamespace(__name__=name),
    )
    assert guard.available is True
    assert guard.runtime_usable is True

    assert module_is_available("mirheo", find_spec=lambda name: object()) is True
    assert module_is_available("missing_mod", find_spec=lambda name: None) is False

    usable, usable_message = module_runtime_usable("mirheo", import_module=lambda name: SimpleNamespace(__name__=name))
    assert usable is True
    assert "imported successfully" in usable_message

    not_usable, not_usable_message = module_runtime_usable(
        "mirheo",
        import_module=lambda name: (_ for _ in ()).throw(ImportError("boom")),
    )
    assert not_usable is False
    assert "ImportError" in not_usable_message
