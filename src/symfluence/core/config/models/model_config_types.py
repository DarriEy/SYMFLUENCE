"""Shared type aliases for model-specific config models."""

from typing import Literal

# Spatial mode types for hydrological models
SpatialModeType = Literal['lumped', 'semi_distributed', 'distributed', 'auto']

__all__ = ['SpatialModeType']
