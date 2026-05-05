from __future__ import annotations


class GVSamplingFailure(RuntimeError):
    """Base class for all GV sampling failures."""


class GVSamplingPlanError(GVSamplingFailure):
    """Raised when a sampling plan cannot be built."""


class GVSamplingExtractionError(GVSamplingFailure):
    """Raised when Mirheo outputs cannot be converted into finite channels."""


class GVCwdValidationError(GVSamplingFailure):
    """Raised when a configured command working directory is invalid."""


class GVCommandFailure(GVSamplingFailure):
    """Raised when a command exits with a non-zero status."""


class GVTimeoutFailure(GVSamplingFailure):
    """Raised when a command exceeds its timeout."""


class GVLogScanFailure(GVSamplingFailure):
    """Raised when NaN/Inf-like tokens are detected in runtime output."""


__all__ = [
    "GVSamplingFailure",
    "GVSamplingPlanError",
    "GVSamplingExtractionError",
    "GVCwdValidationError",
    "GVCommandFailure",
    "GVTimeoutFailure",
    "GVLogScanFailure",
]
