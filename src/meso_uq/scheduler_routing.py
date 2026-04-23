from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpuPartitionPolicy:
    short_partition: str = "dev"
    long_partition: str = "gpu"
    threshold_seconds: int = 30 * 60


def parse_slurm_time_limit(time_limit: str) -> int:
    """Parse SLURM style walltime strings into seconds.

    Supported formats:
    - MM
    - MM:SS
    - HH:MM:SS
    - D-HH
    - D-HH:MM
    - D-HH:MM:SS
    """
    text = str(time_limit).strip()
    if not text:
        raise ValueError("time_limit must be a non-empty string")

    days = 0
    clock = text
    if "-" in text:
        day_text, clock = text.split("-", 1)
        if not day_text.isdigit():
            raise ValueError(f"Invalid SLURM day component in time limit '{time_limit}'")
        days = int(day_text)

    fields = clock.split(":")
    if any(not field.isdigit() for field in fields):
        raise ValueError(f"Invalid SLURM time limit '{time_limit}'")

    if len(fields) == 1:
        hours, minutes, seconds = 0, int(fields[0]), 0
    elif len(fields) == 2:
        hours, minutes, seconds = 0, int(fields[0]), int(fields[1])
    elif len(fields) == 3:
        hours, minutes, seconds = int(fields[0]), int(fields[1]), int(fields[2])
    else:
        raise ValueError(f"Invalid SLURM time limit '{time_limit}'")

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def route_gpu_partition(
    time_limit: str,
    *,
    policy: GpuPartitionPolicy | None = None,
) -> str:
    effective_policy = policy or GpuPartitionPolicy()
    seconds = parse_slurm_time_limit(time_limit)
    if seconds < effective_policy.threshold_seconds:
        return effective_policy.short_partition
    return effective_policy.long_partition
