"""
PIHM Calibration Module.

Provides calibration infrastructure for PIHM integrated hydrologic model.

Components:
    optimizer: PIHMModelOptimizer — sets up parallel dirs
    parameter_manager: PIHMParameterManager — manages PIHM parameter bounds and file updates
    worker: PIHMWorker — model execution and metric calculation
    targets: PIHMStreamflowTarget — streamflow extraction for calibration
"""
from .targets import PIHMStreamflowTarget  # noqa: F401 - triggers target registration
