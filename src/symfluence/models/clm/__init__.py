"""CLM (Community Land Model / CTSM 5.x) Integration.

This module implements CLM5 support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install clm`
- Preprocessing (domain, surface data, parameters, forcing)
- Model execution (standalone single-point)
- Result extraction
- Calibration support (26 parameters)

CLM5 is the land component of CESM (Community Earth System Model).
It is the most physics-heavy LSM in the SYMFLUENCE ensemble, covering
biogeophysics, biogeochemistry, hydrology, snow, and vegetation dynamics.

Key design: CIME is used only for the one-time build. At calibration
runtime, the compiled cesm.exe is invoked directly with modified
parameter NetCDF + namelists to avoid per-iteration rebuild overhead.

Model Architecture:
    CLM uses a single-point structure with:

    1. **Domain File**: NetCDF defining model grid (xc, yc, mask, frac, area)
    2. **Surface Data**: NetCDF with soil, PFT, topography properties
    3. **Parameter File**: clm5_params.nc with global CLM parameters
    4. **Forcing Files**: DATM stream format (one NetCDF/year)
    5. **Namelists**: user_nl_clm, drv_in, datm_in

Configuration Parameters:
    CLM_INSTALL_PATH: Path to CTSM installation
    CLM_EXE: Executable name (default: cesm.exe)
    CLM_PARAMS_TO_CALIBRATE: Calibration parameters
    CLM_TIMEOUT: Execution timeout in seconds

References:
    Lawrence, D. M., et al. (2019): The Community Land Model version 5:
    Description of new features, benchmarking, and impact of forcing
    uncertainty. JAMES, 11, 4245-4287.

    https://github.com/ESCOMP/CTSM
"""
from .preprocessor import CLMPreProcessor
from .runner import CLMRunner
from .extractor import CLMResultExtractor
from .postprocessor import CLMPostProcessor
from .config import CLMConfigAdapter

__all__ = [
    "CLMPreProcessor",
    "CLMRunner",
    "CLMResultExtractor",
    "CLMPostProcessor",
    "CLMConfigAdapter",
]

# Register build instructions (lightweight, no heavy deps)
try:
    from . import build_instructions  # noqa: F401
except ImportError:
    pass  # Build instructions optional

# Components are registered via decorators in their respective modules:
# - CLMPreProcessor: @ModelRegistry.register_preprocessor("CLM")
# - CLMRunner: @ModelRegistry.register_runner("CLM")
# - CLMResultExtractor: @ModelRegistry.register_result_extractor("CLM")
# - CLMPostProcessor: @ModelRegistry.register_postprocessor("CLM")
# CLMConfigAdapter needs explicit registration (no decorator)
from symfluence.models.registry import ModelRegistry
ModelRegistry.register_config_adapter('CLM')(CLMConfigAdapter)

# Register calibration components with OptimizerRegistry
try:
    from .calibration import CLMModelOptimizer  # noqa: F401
except ImportError:
    pass  # Calibration dependencies optional
