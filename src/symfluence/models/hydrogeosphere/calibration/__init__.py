"""
HydroGeoSphere Calibration Module.

Provides calibration infrastructure for HydroGeoSphere (HGS).

Components:
    optimizer: HGSModelOptimizer — sets up parallel dirs
    parameter_manager: HGSParameterManager — manages HGS parameter bounds and file updates
    worker: HGSWorker — model execution and metric calculation
    targets: HGSStreamflowTarget — streamflow extraction for calibration
"""
from .targets import HGSStreamflowTarget  # noqa: F401 - triggers target registration
