"""
HBV time utility helpers.

Centralizes timestep conversions to keep warmup handling consistent across
HBV components.
"""

from typing import Union


def timesteps_per_day(timestep_hours: Union[int, float]) -> float:
    """Return number of model timesteps per day."""
    if timestep_hours <= 0:
        raise ValueError(f"timestep_hours must be positive, got {timestep_hours}")
    return 24.0 / float(timestep_hours)


def warmup_timesteps(warmup_days: int, timestep_hours: Union[int, float]) -> int:
    """Convert warmup days to timesteps using rounding for non-divisor timesteps."""
    if warmup_days <= 0:
        return 0
    return int(round(warmup_days * timesteps_per_day(timestep_hours)))
