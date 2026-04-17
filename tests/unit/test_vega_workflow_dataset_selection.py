from __future__ import annotations

import pytest

from meso_uq.vega_workflows import select_workflow_datasets


def _datasets():
    return [(2.1, "compression_2.1um"), (2.8, "compression_2.8um")]


def test_select_workflow_datasets_returns_all_when_no_filter() -> None:
    selected = select_workflow_datasets(_datasets())
    assert selected == _datasets()


def test_select_workflow_datasets_filters_by_name() -> None:
    selected = select_workflow_datasets(_datasets(), dataset_name="compression_2.8um")
    assert selected == [(2.8, "compression_2.8um")]


def test_select_workflow_datasets_filters_by_diameter() -> None:
    selected = select_workflow_datasets(_datasets(), diameter=2.1)
    assert selected == [(2.1, "compression_2.1um")]


def test_select_workflow_datasets_rejects_conflicting_filters() -> None:
    with pytest.raises(ValueError, match="Use either dataset_name or diameter, not both."):
        select_workflow_datasets(_datasets(), dataset_name="compression_2.1um", diameter=2.1)
