"""Platform policy contracts with dependency-light submodule imports."""

from __future__ import annotations

__all__ = [
    "PlatformPolicy",
    "PlatformPolicyLookupError",
    "PlatformPolicyRecord",
    "list_platform_policies",
    "lookup_platform_policy",
    "validate_platform_path_policy",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import policy

    return getattr(policy, name)
