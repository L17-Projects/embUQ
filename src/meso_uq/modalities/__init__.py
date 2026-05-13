"""Metadata-only modality descriptors for the package migration spine."""

from .registry import (
    MODALITY_REGISTRY,
    assert_modality_family,
    get_modality_descriptor,
    list_modality_descriptors,
)

__all__ = [
    "MODALITY_REGISTRY",
    "assert_modality_family",
    "get_modality_descriptor",
    "list_modality_descriptors",
]
