from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module as _import_module
from importlib import util as importlib_util
from pathlib import Path
from typing import Callable, Mapping
import os
import shutil


@dataclass(frozen=True)
class DependencyGuard:
    name: str
    requested: bool
    available: bool
    runtime_usable: bool
    skip_message: str
    fail_message: str


@dataclass(frozen=True)
class KoraliGuard:
    source_present: bool
    runtime_usable: bool
    skip_message: str
    fail_message: str


def module_is_available(
    module_name: str,
    *,
    find_spec: Callable[[str], object | None] | None = None,
) -> bool:
    if find_spec is None:
        find_spec = importlib_util.find_spec
    return find_spec(module_name) is not None


def module_runtime_usable(
    module_name: str,
    *,
    import_module: Callable[[str], object] | None = None,
) -> tuple[bool, str]:
    if import_module is None:
        import_module = _import_module
    try:
        import_module(module_name)
    except Exception as exc:  # pragma: no cover - defensive guard text.
        return False, f"importing {module_name!r} failed with {exc.__class__.__name__}: {exc}"
    return True, f"{module_name!r} imported successfully."


def _module_guard_message(
    *,
    name: str,
    label: str,
    purpose: str,
    available: bool,
    runtime_usable: bool,
    install_hint: str,
    skip_marker: str,
) -> tuple[str, str]:
    if available and runtime_usable:
        ok = f"{label} is available and importable for {purpose}."
        return ok, ok
    if available and not runtime_usable:
        message = (
            f"{purpose} found {label}, but importing {name!r} failed. "
            f"Check the runtime environment: {install_hint}."
        )
        return message, message
    message = (
        f"{purpose} requires {label} ({name!r}), but it is not importable. "
        f"Install {install_hint} or skip with -m 'not {skip_marker}'."
    )
    return message, message


def guard_optional_module(
    module_name: str,
    *,
    label: str,
    purpose: str,
    install_hint: str,
    skip_marker: str | None = None,
    find_spec: Callable[[str], object | None] | None = None,
    import_module: Callable[[str], object] | None = None,
) -> DependencyGuard:
    if find_spec is None:
        find_spec = importlib_util.find_spec
    if import_module is None:
        import_module = _import_module
    available = module_is_available(module_name, find_spec=find_spec)
    runtime_usable, runtime_note = (
        module_runtime_usable(module_name, import_module=import_module) if available else (False, "")
    )
    skip_message, fail_message = _module_guard_message(
        name=module_name,
        label=label,
        purpose=purpose,
        available=available,
        runtime_usable=runtime_usable,
        install_hint=install_hint if not runtime_note else f"{install_hint}; {runtime_note}",
        skip_marker=module_name if skip_marker is None else skip_marker,
    )
    return DependencyGuard(
        name=module_name,
        requested=True,
        available=available,
        runtime_usable=runtime_usable,
        skip_message=skip_message,
        fail_message=fail_message,
    )


def guard_pyro(
    *,
    find_spec: Callable[[str], object | None] | None = None,
    import_module: Callable[[str], object] | None = None,
) -> DependencyGuard:
    return guard_optional_module(
        "pyro",
        label="Pyro",
        purpose="Pyro-based tests",
        install_hint="the `mesouq[bnn]` extra",
        skip_marker="pyro",
        find_spec=find_spec,
        import_module=import_module,
    )


def guard_mpi(
    *,
    find_spec: Callable[[str], object | None] | None = None,
    import_module: Callable[[str], object] | None = None,
) -> DependencyGuard:
    return guard_optional_module(
        "mpi4py",
        label="mpi4py",
        purpose="MPI-based tests",
        install_hint="the `mesouq[mpi]` extra",
        skip_marker="mpi",
        find_spec=find_spec,
        import_module=import_module,
    )


def guard_mirheo(
    *,
    find_spec: Callable[[str], object | None] | None = None,
    import_module: Callable[[str], object] | None = None,
) -> DependencyGuard:
    return guard_optional_module(
        "mirheo",
        label="Mirheo",
        purpose="Mirheo-backed tests",
        install_hint="the Mirheo runtime bootstrap and site module path",
        skip_marker="mirheo",
        find_spec=find_spec,
        import_module=import_module,
    )


def korali_source_present(repo_root: Path | str, vendor_root: Path | str | None = None) -> bool:
    repo_path = Path(repo_root)
    source_root = repo_path / "extern" / "korali" if vendor_root is None else Path(vendor_root)
    return source_root.is_dir()


def korali_runtime_usable(
    *,
    import_module: Callable[[str], object] | None = None,
) -> tuple[bool, str]:
    return module_runtime_usable("korali", import_module=import_module)


def guard_korali(
    repo_root: Path | str,
    vendor_root: Path | str | None = None,
    *,
    import_module: Callable[[str], object] | None = None,
) -> KoraliGuard:
    repo_path = Path(repo_root)
    source_root = repo_path / "extern" / "korali" if vendor_root is None else Path(vendor_root)
    source_present = source_root.is_dir()
    runtime_usable, runtime_note = korali_runtime_usable(import_module=import_module)

    if source_present and runtime_usable:
        message = f"Korali source is present at {source_root} and the runtime import is usable."
        return KoraliGuard(source_present=True, runtime_usable=True, skip_message=message, fail_message=message)

    if source_present and not runtime_usable:
        skip_message = (
            f"Korali source is present at {source_root}, but the runtime import is not usable: {runtime_note}. "
            "Check the Korali build/install root and PYTHONPATH hints."
        )
        fail_message = skip_message
        return KoraliGuard(
            source_present=True,
            runtime_usable=False,
            skip_message=skip_message,
            fail_message=fail_message,
        )

    if runtime_usable:
        skip_message = (
            f"Korali imports successfully, but the vendored source tree is missing from {source_root}. "
            "This test lane expects the repo-local extern/korali checkout to be present."
        )
        fail_message = skip_message
        return KoraliGuard(
            source_present=False,
            runtime_usable=True,
            skip_message=skip_message,
            fail_message=fail_message,
        )

    skip_message = (
        f"Korali source is missing from {source_root} and the runtime import is not usable: {runtime_note}. "
        "Restore the vendored source tree or bootstrap the Korali runtime environment."
    )
    fail_message = skip_message
    return KoraliGuard(source_present=False, runtime_usable=False, skip_message=skip_message, fail_message=fail_message)


def cuda_runtime_available(
    *,
    requested: bool = True,
    import_module: Callable[[str], object] | None = None,
    probe: Callable[[object], bool] | None = None,
) -> tuple[bool, str]:
    if not requested:
        return True, "CUDA was not requested; CPU execution is acceptable."

    if import_module is None:
        import_module = _import_module

    try:
        torch_module = import_module("torch")
    except Exception as exc:  # pragma: no cover - defensive guard text.
        return False, f"CUDA was requested, but importing 'torch' failed with {exc.__class__.__name__}: {exc}"

    cuda_api = getattr(torch_module, "cuda", None)
    if cuda_api is None:
        return False, "CUDA was requested, but the imported torch module has no cuda attribute."

    if probe is not None:
        try:
            available = bool(probe(cuda_api))
        except Exception as exc:  # pragma: no cover - defensive guard text.
            return False, f"CUDA probe failed with {exc.__class__.__name__}: {exc}"
        return available, "CUDA probe reported availability." if available else "CUDA probe reported that CUDA is unavailable."

    is_available = getattr(cuda_api, "is_available", None)
    if is_available is None:
        return False, "CUDA was requested, but torch.cuda.is_available() is missing."

    try:
        available = bool(is_available())
    except Exception as exc:  # pragma: no cover - defensive guard text.
        return False, f"CUDA availability check failed with {exc.__class__.__name__}: {exc}"

    if available:
        return True, "torch.cuda.is_available() reported a usable CUDA runtime."
    return False, "torch.cuda.is_available() reported that CUDA is unavailable."


def guard_cuda(
    *,
    requested: bool = True,
    import_module: Callable[[str], object] | None = None,
    probe: Callable[[object], bool] | None = None,
) -> DependencyGuard:
    available, runtime_note = cuda_runtime_available(
        requested=requested,
        import_module=import_module,
        probe=probe,
    )
    if requested and not available:
        message = f"CUDA is requested, but unavailable: {runtime_note}"
        return DependencyGuard(
            name="cuda",
            requested=requested,
            available=False,
            runtime_usable=False,
            skip_message=message,
            fail_message=message,
        )
    if not requested:
        message = runtime_note
        return DependencyGuard(
            name="cuda",
            requested=requested,
            available=True,
            runtime_usable=True,
            skip_message=message,
            fail_message=message,
        )
    message = runtime_note
    return DependencyGuard(
        name="cuda",
        requested=requested,
        available=True,
        runtime_usable=True,
        skip_message=message,
        fail_message=message,
    )


def slurm_binary_available(
    *,
    env: Mapping[str, str] | None = None,
    which: Callable[..., str | None] | None = None,
) -> tuple[bool, str]:
    if which is None:
        which = shutil.which
    env_map = os.environ if env is None else env
    path = env_map.get("PATH")
    sbatch = which("sbatch", path=path) if path is not None else which("sbatch")
    srun = which("srun", path=path) if path is not None else which("srun")
    if sbatch is not None:
        return True, f"Found sbatch at {sbatch}."
    if srun is not None:
        return True, f"Found srun at {srun}."
    return False, "Could not find sbatch or srun on PATH."


def guard_slurm(
    *,
    requested: bool = True,
    env: Mapping[str, str] | None = None,
    which: Callable[..., str | None] | None = None,
) -> DependencyGuard:
    available, runtime_note = slurm_binary_available(env=env, which=which)
    if requested and not available:
        message = f"Slurm is requested, but unavailable: {runtime_note}"
        return DependencyGuard(
            name="slurm",
            requested=requested,
            available=False,
            runtime_usable=False,
            skip_message=message,
            fail_message=message,
        )
    if not requested:
        message = "Slurm was not requested; local execution is acceptable."
        return DependencyGuard(
            name="slurm",
            requested=requested,
            available=True,
            runtime_usable=True,
            skip_message=message,
            fail_message=message,
        )
    message = runtime_note
    return DependencyGuard(
        name="slurm",
        requested=requested,
        available=True,
        runtime_usable=True,
        skip_message=message,
        fail_message=message,
    )


def guard_gpu(
    *,
    requested: bool = True,
    import_module: Callable[[str], object] | None = None,
    probe: Callable[[object], bool] | None = None,
) -> DependencyGuard:
    cuda_guard = guard_cuda(requested=requested, import_module=import_module, probe=probe)
    if requested and not cuda_guard.runtime_usable:
        message = f"GPU execution is requested, but CUDA is unavailable: {cuda_guard.skip_message}"
        return DependencyGuard(
            name="gpu",
            requested=requested,
            available=False,
            runtime_usable=False,
            skip_message=message,
            fail_message=message,
        )
    if not requested:
        message = "GPU execution was not requested; CPU execution is acceptable."
        return DependencyGuard(
            name="gpu",
            requested=requested,
            available=True,
            runtime_usable=True,
            skip_message=message,
            fail_message=message,
        )
    message = f"GPU execution is available through CUDA: {cuda_guard.skip_message}"
    return DependencyGuard(
        name="gpu",
        requested=requested,
        available=True,
        runtime_usable=True,
        skip_message=message,
        fail_message=message,
    )


def guard_hpc(
    *,
    requested: bool = True,
    slurm: DependencyGuard | None = None,
    mpi: DependencyGuard | None = None,
    env: Mapping[str, str] | None = None,
    which: Callable[..., str | None] | None = None,
    find_spec: Callable[[str], object | None] | None = None,
    import_module: Callable[[str], object] | None = None,
) -> DependencyGuard:
    if not requested:
        message = "HPC execution was not requested; local execution is acceptable."
        return DependencyGuard(
            name="hpc",
            requested=requested,
            available=True,
            runtime_usable=True,
            skip_message=message,
            fail_message=message,
        )

    slurm_guard = slurm or guard_slurm(requested=requested, env=env, which=which)
    mpi_guard = mpi or guard_mpi(find_spec=find_spec, import_module=import_module)
    if requested and (not slurm_guard.runtime_usable or not mpi_guard.runtime_usable):
        missing = [guard.fail_message for guard in (slurm_guard, mpi_guard) if not guard.runtime_usable]
        message = "HPC execution is requested, but unavailable: " + "; ".join(missing)
        return DependencyGuard(
            name="hpc",
            requested=requested,
            available=False,
            runtime_usable=False,
            skip_message=message,
            fail_message=message,
        )
    message = "HPC execution is available: Slurm and mpi4py checks passed."
    return DependencyGuard(
        name="hpc",
        requested=requested,
        available=True,
        runtime_usable=True,
        skip_message=message,
        fail_message=message,
    )


def guard_operational(
    *,
    requested: bool = True,
    hpc: DependencyGuard | None = None,
    gpu: DependencyGuard | None = None,
    pyro: DependencyGuard | None = None,
    mirheo: DependencyGuard | None = None,
    korali: KoraliGuard | None = None,
    repo_root: Path | str | None = None,
    vendor_root: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    which: Callable[..., str | None] | None = None,
    find_spec: Callable[[str], object | None] | None = None,
    import_module: Callable[[str], object] | None = None,
    cuda_probe: Callable[[object], bool] | None = None,
) -> DependencyGuard:
    if not requested:
        message = "Operational execution was not requested; lightweight execution is acceptable."
        return DependencyGuard(
            name="operational",
            requested=requested,
            available=True,
            runtime_usable=True,
            skip_message=message,
            fail_message=message,
        )

    repo_root_path = Path.cwd() if repo_root is None else Path(repo_root)
    hpc_guard = hpc or guard_hpc(
        requested=requested,
        env=env,
        which=which,
        find_spec=find_spec,
        import_module=import_module,
    )
    gpu_guard = gpu or guard_gpu(requested=requested, import_module=import_module, probe=cuda_probe)
    pyro_guard = pyro or guard_pyro(find_spec=find_spec, import_module=import_module)
    mirheo_guard = mirheo or guard_mirheo(find_spec=find_spec, import_module=import_module)
    korali_guard = korali or guard_korali(repo_root_path, vendor_root=vendor_root, import_module=import_module)

    required_guards: list[tuple[str, bool]] = [
        ("hpc", hpc_guard.runtime_usable),
        ("gpu", gpu_guard.runtime_usable),
        ("pyro", pyro_guard.runtime_usable),
        ("mirheo", mirheo_guard.runtime_usable),
        ("korali", korali_guard.runtime_usable),
    ]
    if requested and not all(usable for _, usable in required_guards):
        missing = [name for name, usable in required_guards if not usable]
        message = f"Operational execution is requested, but the following runtime checks failed: {', '.join(missing)}."
        return DependencyGuard(
            name="operational",
            requested=requested,
            available=False,
            runtime_usable=False,
            skip_message=message,
            fail_message=message,
        )
    message = "Operational execution is available: HPC, GPU, Pyro, Mirheo, and Korali checks passed."
    return DependencyGuard(
        name="operational",
        requested=requested,
        available=True,
        runtime_usable=True,
        skip_message=message,
        fail_message=message,
    )
