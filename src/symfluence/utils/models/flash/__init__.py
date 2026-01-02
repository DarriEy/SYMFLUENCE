"""
FLASH (Flow and Snow Hydrological LSTM) package.

This package contains components for running and managing FLASH neural network model simulations.
FLASH is an LSTM-based model for hydrological predictions, specifically for streamflow
and snow water equivalent (SWE).
"""

from .runner import FLASH
from .postprocessor import FLASHPostProcessor

__all__ = [
    'FLASH',
    'FLASHPostProcessor'
]
