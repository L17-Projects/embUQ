from __future__ import annotations

from pathlib import Path

import pytest

from meso_uq.scheduler_routing import (
    GpuPartitionPolicy,
    parse_slurm_time_limit,
    route_gpu_partition,
)


def test_parse_slurm_time_limit_hms() -> None:
    assert parse_slurm_time_limit("00:15:00") == 900
    assert parse_slurm_time_limit("00:30:00") == 1800
    assert parse_slurm_time_limit("01:00:05") == 3605


def test_parse_slurm_time_limit_day_format() -> None:
    assert parse_slurm_time_limit("1-00:00:00") == 86400
    assert parse_slurm_time_limit("2-01:02:03") == (2 * 86400 + 3600 + 120 + 3)


def test_parse_slurm_time_limit_invalid_raises() -> None:
    with pytest.raises(ValueError):
        parse_slurm_time_limit("foo")


def test_route_gpu_partition_strict_threshold() -> None:
    assert route_gpu_partition("00:29:59") == "dev"
    assert route_gpu_partition("00:30:00") == "gpu"
    assert route_gpu_partition("02:00:00") == "gpu"


def test_route_gpu_partition_custom_policy() -> None:
    policy = GpuPartitionPolicy(short_partition="gpu-short", long_partition="gpu-long")
    assert route_gpu_partition("00:10:00", policy=policy) == "gpu-short"
    assert route_gpu_partition("00:45:00", policy=policy) == "gpu-long"


def test_map_sbatch_templates_use_partition_router_script() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    map_template = repo_root / "scripts" / "vega" / "sbatch" / "workflow_map.sbatch"
    mirheo_template = repo_root / "scripts" / "vega" / "sbatch" / "workflow_map_mirheo.sbatch"

    map_text = map_template.read_text(encoding="utf-8")
    mirheo_text = mirheo_template.read_text(encoding="utf-8")
    assert "resolve_gpu_partition.py" in map_text
    assert "resolve_gpu_partition.py" in mirheo_text
    assert "strict GPU partition policy violation" in map_text
    assert "strict GPU partition policy violation" in mirheo_text


def test_map_mirheo_template_uses_supported_canary_floor() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    mirheo_template = repo_root / "scripts" / "vega" / "sbatch" / "workflow_map_mirheo.sbatch"
    mirheo_text = mirheo_template.read_text(encoding="utf-8")

    assert 'MAP_MIRHEO_NUMSTEPS="${MAP_MIRHEO_NUMSTEPS:-200}"' in mirheo_text
    assert 'MAP_MIRHEO_NUMSTEPS_EQ="${MAP_MIRHEO_NUMSTEPS_EQ:-200}"' in mirheo_text
