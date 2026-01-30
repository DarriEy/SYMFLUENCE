"""VIC model calibration components.

This module provides:
- VICModelOptimizer: VIC-specific optimizer
- VICParameterManager: Handles VIC parameter bounds and updates
- VICWorker: Worker for VIC model calibration in optimization loops
"""

from .optimizer import VICModelOptimizer
from .parameter_manager import VICParameterManager
from .worker import VICWorker

__all__ = ["VICModelOptimizer", "VICParameterManager", "VICWorker"]
