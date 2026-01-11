"""
RHESSys Model Package

RHESSys (Regional Hydro-Ecologic Simulation System) is a distributed,
physically-based ecohydrological model that simulates water, carbon,
and nitrogen cycling.
"""
from .preprocessor import RHESSysPreprocessor
from .runner import RHESSysRunner
from .postprocessor import RHESSysPostProcessor

__all__ = ["RHESSysPreprocessor", "RHESSysRunner", "RHESSysPostProcessor"]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional
