from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from tests import conftest
from tests.support.native_cuda_runtime import (
    NativeCudaRuntimeUnavailable,
    _check_cuda_preflight,
    _cuda_driver_symbol,
    _device_alloc_and_copy,
    launch_psi_native_cuda_kernel,
)


def test_cuda_driver_symbol_prefers_versioned_symbol_without_unversioned_alias() -> None:
    sentinel = object()
    driver_lib = SimpleNamespace(cuMemAlloc_v2=sentinel)

    assert _cuda_driver_symbol(driver_lib, "cuMemAlloc") is sentinel


def test_cuda_driver_symbol_falls_back_to_unversioned_symbol() -> None:
    sentinel = object()
    driver_lib = SimpleNamespace(cuMemAlloc=sentinel)

    assert _cuda_driver_symbol(driver_lib, "cuMemAlloc") is sentinel


def test_cuda_driver_symbol_reports_missing_required_symbol() -> None:
    with pytest.raises(NativeCudaRuntimeUnavailable, match="cuMemAlloc_v2 or cuMemAlloc"):
        _cuda_driver_symbol(SimpleNamespace(), "cuMemAlloc")


class _FailingDriver:
    def check(self, _result: int, statement: str) -> None:
        raise RuntimeError(f"{statement} failed: CUDA_ERROR_NO_DEVICE")


def test_cuda_preflight_failure_is_reported_as_runtime_unavailable() -> None:
    with pytest.raises(NativeCudaRuntimeUnavailable, match="cuInit failed: CUDA_ERROR_NO_DEVICE"):
        _check_cuda_preflight(_FailingDriver(), 100, "cuInit")


class _ZeroAllocationDriver:
    def cuMemAlloc(self, *_args: object) -> int:
        raise AssertionError("zero-byte copies must not allocate CUDA memory")

    def cuMemcpyHtoD(self, *_args: object) -> int:
        raise AssertionError("zero-byte copies must not transfer CUDA memory")


def test_zero_byte_device_copy_returns_null_pointer_without_cuda_allocation() -> None:
    payload = (ctypes.c_double * 0)()

    device = _device_alloc_and_copy(_ZeroAllocationDriver(), payload, 0)

    assert device.value == 0


def test_empty_batch_returns_without_loading_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_cuda_is_loaded() -> None:
        raise AssertionError("empty batch should not load the CUDA driver")

    monkeypatch.setattr("tests.support.native_cuda_runtime._CudaDriver", fail_if_cuda_is_loaded)

    result = launch_psi_native_cuda_kernel(
        repo_root=SimpleNamespace(),
        priors=(),
        subproblems=(SimpleNamespace(), SimpleNamespace()),
        batch_parameters=(),
    )

    assert result.subproblem_log_likelihoods == ((), ())
    assert result.batch_log_likelihoods == ()
    assert result.compute_capability == (0, 0)


@pytest.mark.parametrize(
    ("mark_expression", "expected"),
    [
        ("", False),
        ("korali", False),
        ("not cuda", False),
        ("not (cuda)", False),
        ("cuda", True),
        ("cuda and korali", True),
        ("gpu or cuda", True),
    ],
)
def test_cuda_marker_expression_requires_explicit_cuda_opt_in(mark_expression: str, expected: bool) -> None:
    assert conftest._mark_expression_requests_cuda(mark_expression) is expected
