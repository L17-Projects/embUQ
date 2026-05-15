from __future__ import annotations

import ctypes
import ctypes.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tests.support.native_cuda_psi import (
    KERNEL_ENTRYPOINT,
    ConditionalPrior,
    SubProblemSamples,
    batch_log_likelihood,
    read_vendored_psi_native_cuda_kernel_source,
    subproblem_log_likelihood,
)


class NativeCudaRuntimeUnavailable(RuntimeError):
    """Raised when the CUDA driver or NVRTC runtime is not available."""


@dataclass(frozen=True)
class NativeCudaPsiRuntimeResult:
    subproblem_log_likelihoods: tuple[tuple[float, ...], ...]
    batch_log_likelihoods: tuple[float, ...]
    compute_capability: tuple[int, int]


def _load_library(*names: str) -> ctypes.CDLL:
    attempted: list[str] = []
    for name in names:
        candidates = [ctypes.util.find_library(name), name]
        if not name.startswith("lib"):
            candidates.extend((f"lib{name}.so", f"lib{name}.so.1", f"lib{name}.so.12"))
        for candidate in candidates:
            if candidate is None or candidate in attempted:
                continue
            attempted.append(candidate)
            try:
                return ctypes.CDLL(candidate)
            except OSError:
                continue
    raise NativeCudaRuntimeUnavailable(
        "Could not load required CUDA library; attempted: " + ", ".join(attempted)
    )


class _CudaDriver:
    def __init__(self) -> None:
        self.lib = _load_library("cuda", "libcuda.so.1")
        self._configure()

    def _configure(self) -> None:
        l = self.lib
        self.cuMemAlloc = _cuda_driver_symbol(l, "cuMemAlloc")
        self.cuMemFree = _cuda_driver_symbol(l, "cuMemFree")
        self.cuMemcpyHtoD = _cuda_driver_symbol(l, "cuMemcpyHtoD")
        self.cuMemcpyDtoH = _cuda_driver_symbol(l, "cuMemcpyDtoH")
        l.cuInit.argtypes = [ctypes.c_uint]
        l.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        l.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        l.cuDeviceComputeCapability.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        l.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        l.cuDevicePrimaryCtxRelease.argtypes = [ctypes.c_int]
        l.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        l.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        l.cuModuleUnload.argtypes = [ctypes.c_void_p]
        l.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        self.cuMemFree.argtypes = [ctypes.c_uint64]
        self.cuMemcpyHtoD.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        self.cuMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        l.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        l.cuCtxSynchronize.argtypes = []
        if hasattr(l, "cuGetErrorString"):
            l.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def check(self, result: int, statement: str) -> None:
        if result == 0:
            return
        error_text = ctypes.c_char_p()
        if hasattr(self.lib, "cuGetErrorString"):
            self.lib.cuGetErrorString(result, ctypes.byref(error_text))
        detail = error_text.value.decode("utf-8", errors="replace") if error_text.value else f"CUresult={result}"
        raise RuntimeError(f"{statement} failed: {detail}")


def _cuda_driver_symbol(driver_lib: object, name: str) -> object:
    versioned_name = f"{name}_v2"
    if hasattr(driver_lib, versioned_name):
        return getattr(driver_lib, versioned_name)
    try:
        return getattr(driver_lib, name)
    except AttributeError as exc:
        raise NativeCudaRuntimeUnavailable(
            f"CUDA driver library is missing required symbol {versioned_name} or {name}."
        ) from exc


def _check_cuda_preflight(driver: _CudaDriver, result: int, statement: str) -> None:
    try:
        driver.check(result, statement)
    except RuntimeError as exc:
        raise NativeCudaRuntimeUnavailable(str(exc)) from exc


class _Nvrtc:
    def __init__(self) -> None:
        self.lib = _load_library("nvrtc", "libnvrtc.so")
        self._configure()

    def _configure(self) -> None:
        l = self.lib
        l.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        l.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        l.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        l.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        l.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        l.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        l.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        l.nvrtcGetErrorString.argtypes = [ctypes.c_int]
        l.nvrtcGetErrorString.restype = ctypes.c_char_p

    def check(self, result: int, statement: str) -> None:
        if result == 0:
            return
        detail = self.lib.nvrtcGetErrorString(result).decode("utf-8", errors="replace")
        raise RuntimeError(f"{statement} failed: {detail}")


def _as_double_array(values: Sequence[float]) -> ctypes.Array[ctypes.c_double]:
    return (ctypes.c_double * len(values))(*(float(value) for value in values))


def _as_int_array(values: Sequence[int]) -> ctypes.Array[ctypes.c_int]:
    return (ctypes.c_int * len(values))(*(int(value) for value in values))


def _device_alloc_and_copy(driver: _CudaDriver, payload: ctypes.Array, size: int) -> ctypes.c_uint64:
    if size == 0:
        return ctypes.c_uint64(0)
    device = ctypes.c_uint64(0)
    driver.check(driver.cuMemAlloc(ctypes.byref(device), size), "cuMemAlloc")
    try:
        driver.check(driver.cuMemcpyHtoD(device.value, ctypes.cast(payload, ctypes.c_void_p), size), "cuMemcpyHtoD")
    except BaseException:
        driver.cuMemFree(device.value)
        raise
    return device


def _compile_kernel(nvrtc: _Nvrtc, kernel_source: str, *, major: int, minor: int) -> bytes:
    program = ctypes.c_void_p()
    nvrtc.check(
        nvrtc.lib.nvrtcCreateProgram(
            ctypes.byref(program),
            kernel_source.encode("utf-8"),
            b"psi_native_cuda.cu",
            0,
            None,
            None,
        ),
        "nvrtcCreateProgram",
    )
    try:
        options = (ctypes.c_char_p * 2)(
            f"--gpu-architecture=compute_{major}{minor}".encode("ascii"),
            b"--std=c++11",
        )
        compile_result = nvrtc.lib.nvrtcCompileProgram(program, len(options), options)
        if compile_result != 0:
            log_size = ctypes.c_size_t()
            nvrtc.check(nvrtc.lib.nvrtcGetProgramLogSize(program, ctypes.byref(log_size)), "nvrtcGetProgramLogSize")
            log = ctypes.create_string_buffer(log_size.value)
            nvrtc.check(nvrtc.lib.nvrtcGetProgramLog(program, log), "nvrtcGetProgramLog")
            raise RuntimeError("NVRTC failed to compile Psi NativeCuda kernel:\n" + log.value.decode("utf-8", errors="replace"))

        ptx_size = ctypes.c_size_t()
        nvrtc.check(nvrtc.lib.nvrtcGetPTXSize(program, ctypes.byref(ptx_size)), "nvrtcGetPTXSize")
        ptx = ctypes.create_string_buffer(ptx_size.value)
        nvrtc.check(nvrtc.lib.nvrtcGetPTX(program, ptx), "nvrtcGetPTX")
        return bytes(ptx.raw)
    finally:
        nvrtc.lib.nvrtcDestroyProgram(ctypes.byref(program))


def _prior_kind(prior: ConditionalPrior) -> int:
    if prior.kind == "normal":
        return 0
    if prior.kind == "uniform":
        return 1
    raise ValueError(f"Unsupported prior kind: {prior.kind!r}.")


def launch_psi_native_cuda_kernel(
    *,
    repo_root: Path,
    priors: Sequence[ConditionalPrior],
    subproblems: Sequence[SubProblemSamples],
    batch_parameters: Sequence[Sequence[float]],
    batch_log_priors: Sequence[float] | None = None,
) -> NativeCudaPsiRuntimeResult:
    if not batch_parameters:
        return NativeCudaPsiRuntimeResult(
            subproblem_log_likelihoods=tuple(() for _ in subproblems),
            batch_log_likelihoods=(),
            compute_capability=(0, 0),
        )

    driver = _CudaDriver()
    nvrtc = _Nvrtc()

    _check_cuda_preflight(driver, driver.lib.cuInit(0), "cuInit")
    device_count = ctypes.c_int()
    _check_cuda_preflight(driver, driver.lib.cuDeviceGetCount(ctypes.byref(device_count)), "cuDeviceGetCount")
    if device_count.value <= 0:
        raise NativeCudaRuntimeUnavailable("CUDA driver is available, but no CUDA devices were reported.")

    device = ctypes.c_int()
    _check_cuda_preflight(driver, driver.lib.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    context = ctypes.c_void_p()
    module = ctypes.c_void_p()
    allocations: list[int] = []

    try:
        _check_cuda_preflight(
            driver,
            driver.lib.cuDevicePrimaryCtxRetain(ctypes.byref(context), device.value),
            "cuDevicePrimaryCtxRetain",
        )
        _check_cuda_preflight(driver, driver.lib.cuCtxSetCurrent(context), "cuCtxSetCurrent")

        major = ctypes.c_int()
        minor = ctypes.c_int()
        _check_cuda_preflight(
            driver,
            driver.lib.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device.value),
            "cuDeviceComputeCapability",
        )
        kernel_source = read_vendored_psi_native_cuda_kernel_source(repo_root)
        ptx = _compile_kernel(nvrtc, kernel_source, major=major.value, minor=minor.value)

        ptx_buffer = ctypes.create_string_buffer(ptx)
        driver.check(driver.lib.cuModuleLoadData(ctypes.byref(module), ctypes.cast(ptx_buffer, ctypes.c_void_p)), "cuModuleLoadData")
        function = ctypes.c_void_p()
        driver.check(driver.lib.cuModuleGetFunction(ctypes.byref(function), module, KERNEL_ENTRYPOINT.encode("ascii")), "cuModuleGetFunction")

        flattened_parameters = [float(value) for row in batch_parameters for value in row]
        parameter_count = len(batch_parameters[0]) if batch_parameters else 0
        batch_size = len(batch_parameters)
        subproblem_count = len(subproblems)
        dynamic_prior_count = len(priors)

        batch_parameters_host = _as_double_array(flattened_parameters)
        batch_parameters_device = _device_alloc_and_copy(
            driver,
            batch_parameters_host,
            ctypes.sizeof(batch_parameters_host),
        )
        allocations.append(batch_parameters_device.value)

        prior_kinds_host = _as_int_array([_prior_kind(prior) for prior in priors])
        parameter_a_is_variable_host = _as_int_array([int(prior.parameter_a.position is not None) for prior in priors])
        parameter_a_positions_host = _as_int_array([prior.parameter_a.position or 0 for prior in priors])
        parameter_a_values_host = _as_double_array([prior.parameter_a.value or 0.0 for prior in priors])
        parameter_b_is_variable_host = _as_int_array([int(prior.parameter_b.position is not None) for prior in priors])
        parameter_b_positions_host = _as_int_array([prior.parameter_b.position or 0 for prior in priors])
        parameter_b_values_host = _as_double_array([prior.parameter_b.value or 0.0 for prior in priors])

        metadata_devices = [
            _device_alloc_and_copy(driver, prior_kinds_host, ctypes.sizeof(prior_kinds_host)),
            _device_alloc_and_copy(driver, parameter_a_is_variable_host, ctypes.sizeof(parameter_a_is_variable_host)),
            _device_alloc_and_copy(driver, parameter_a_positions_host, ctypes.sizeof(parameter_a_positions_host)),
            _device_alloc_and_copy(driver, parameter_a_values_host, ctypes.sizeof(parameter_a_values_host)),
            _device_alloc_and_copy(driver, parameter_b_is_variable_host, ctypes.sizeof(parameter_b_is_variable_host)),
            _device_alloc_and_copy(driver, parameter_b_positions_host, ctypes.sizeof(parameter_b_positions_host)),
            _device_alloc_and_copy(driver, parameter_b_values_host, ctypes.sizeof(parameter_b_values_host)),
        ]
        allocations.extend(device.value for device in metadata_devices)

        coordinate_devices: list[ctypes.c_uint64] = []
        base_weight_devices: list[ctypes.c_uint64] = []
        sample_counts: list[int] = []
        for subproblem in subproblems:
            sample_count = len(subproblem.base_log_weights)
            sample_counts.append(sample_count)
            dynamic_coordinates = [
                value
                for prior in priors
                for value in subproblem.coordinates_by_dimension[prior.sample_dimension]
            ]
            dynamic_coordinates_host = _as_double_array(dynamic_coordinates)
            base_weights_host = _as_double_array(subproblem.base_log_weights)
            coordinate_device = _device_alloc_and_copy(driver, dynamic_coordinates_host, ctypes.sizeof(dynamic_coordinates_host))
            base_weight_device = _device_alloc_and_copy(driver, base_weights_host, ctypes.sizeof(base_weights_host))
            coordinate_devices.append(coordinate_device)
            base_weight_devices.append(base_weight_device)
            allocations.extend((coordinate_device.value, base_weight_device.value))

        coordinate_pointer_host = (ctypes.c_uint64 * len(coordinate_devices))(*(device.value for device in coordinate_devices))
        base_weight_pointer_host = (ctypes.c_uint64 * len(base_weight_devices))(*(device.value for device in base_weight_devices))
        sample_counts_host = (ctypes.c_uint * len(sample_counts))(*sample_counts)
        coordinate_pointers_device = _device_alloc_and_copy(driver, coordinate_pointer_host, ctypes.sizeof(coordinate_pointer_host))
        base_weight_pointers_device = _device_alloc_and_copy(driver, base_weight_pointer_host, ctypes.sizeof(base_weight_pointer_host))
        sample_counts_device = _device_alloc_and_copy(driver, sample_counts_host, ctypes.sizeof(sample_counts_host))
        allocations.extend((coordinate_pointers_device.value, base_weight_pointers_device.value, sample_counts_device.value))

        output_size = batch_size * subproblem_count
        output_device = ctypes.c_uint64(0)
        driver.check(driver.cuMemAlloc(ctypes.byref(output_device), output_size * ctypes.sizeof(ctypes.c_double)), "cuMemAlloc output")
        allocations.append(output_device.value)

        parameter_count_arg = ctypes.c_uint(parameter_count)
        batch_size_arg = ctypes.c_uint(batch_size)
        dynamic_prior_count_arg = ctypes.c_uint(dynamic_prior_count)
        subproblem_count_arg = ctypes.c_uint(subproblem_count)

        args = [
            batch_parameters_device,
            parameter_count_arg,
            batch_size_arg,
            dynamic_prior_count_arg,
            metadata_devices[0],
            metadata_devices[1],
            metadata_devices[2],
            metadata_devices[3],
            metadata_devices[4],
            metadata_devices[5],
            metadata_devices[6],
            coordinate_pointers_device,
            base_weight_pointers_device,
            sample_counts_device,
            subproblem_count_arg,
            output_device,
        ]
        kernel_params = (ctypes.c_void_p * len(args))(
            *(ctypes.cast(ctypes.byref(arg), ctypes.c_void_p) for arg in args)
        )

        driver.check(
            driver.lib.cuLaunchKernel(
                function,
                batch_size,
                subproblem_count,
                1,
                256,
                1,
                1,
                0,
                None,
                kernel_params,
                None,
            ),
            "cuLaunchKernel",
        )
        driver.check(driver.lib.cuCtxSynchronize(), "cuCtxSynchronize")

        output_host = (ctypes.c_double * output_size)()
        driver.check(
            driver.cuMemcpyDtoH(ctypes.cast(output_host, ctypes.c_void_p), output_device.value, ctypes.sizeof(output_host)),
            "cuMemcpyDtoH output",
        )
        flattened_output = [float(value) for value in output_host]
        by_subproblem = tuple(
            tuple(flattened_output[subproblem_id * batch_size + batch_id] for batch_id in range(batch_size))
            for subproblem_id in range(subproblem_count)
        )

        reduced = []
        for batch_id, log_prior in enumerate(batch_log_priors or [0.0] * batch_size):
            if not math.isfinite(float(log_prior)):
                reduced.append(float("-inf"))
                continue
            partials = [by_subproblem[subproblem_id][batch_id] for subproblem_id in range(subproblem_count)]
            reduced.append(sum(partials) if all(math.isfinite(value) for value in partials) else float("-inf"))

        return NativeCudaPsiRuntimeResult(
            subproblem_log_likelihoods=by_subproblem,
            batch_log_likelihoods=tuple(reduced),
            compute_capability=(major.value, minor.value),
        )
    finally:
        for device_pointer in reversed(allocations):
            if device_pointer:
                driver.cuMemFree(device_pointer)
        if module:
            driver.lib.cuModuleUnload(module)
        if context:
            driver.lib.cuDevicePrimaryCtxRelease(device.value)


def expected_subproblem_log_likelihoods(
    *,
    priors: Sequence[ConditionalPrior],
    subproblems: Sequence[SubProblemSamples],
    batch_parameters: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            subproblem_log_likelihood(priors=priors, parameters=parameters, subproblem=subproblem)
            for parameters in batch_parameters
        )
        for subproblem in subproblems
    )


def expected_batch_log_likelihoods(
    *,
    priors: Sequence[ConditionalPrior],
    subproblems: Sequence[SubProblemSamples],
    batch_parameters: Sequence[Sequence[float]],
    batch_log_priors: Sequence[float] | None = None,
) -> tuple[float, ...]:
    return tuple(
        batch_log_likelihood(
            priors=priors,
            subproblems=subproblems,
            batch_parameters=batch_parameters,
            batch_log_priors=batch_log_priors,
        )
    )
