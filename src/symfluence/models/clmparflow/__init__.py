# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""ParFlow-CLM Tightly-Coupled Integrated Hydrologic Model Integration.

This module implements ParFlow-CLM support for SYMFLUENCE, including:
- Binary installation via `symfluence binary install clmparflow`
- Preprocessing (generates ParFlow .pfidb + CLM driver/vegetation/forcing files)
- Model execution (ParFlow compiled with -DPARFLOW_HAVE_CLM=ON)
- Result extraction (pressure head, saturation, overland flow from .pfb;
  ET, soil temperature from CLM .C.pfb output)
- Postprocessing (combined surface + subsurface flow; CLM handles ET/snow internally)

ParFlow-CLM is ParFlow compiled with the Common Land Model (CLM) embedded as
Fortran modules.  CLM adds land surface energy balance, evapotranspiration,
snow dynamics, and vegetation processes directly inside the ParFlow simulation,
eliminating the need for external coupling of a separate land surface model.

Configuration Parameters:
    CLMPARFLOW_INSTALL_PATH: Path to CLMParFlow installation
    CLMPARFLOW_EXE: Executable name (default: parflow)
    CLMPARFLOW_DIR: ParFlow install root (sets PARFLOW_DIR env var)
    CLMPARFLOW_K_SAT: Saturated hydraulic conductivity (m/hr)
    CLMPARFLOW_POROSITY: Porosity (dimensionless)
    CLMPARFLOW_VG_ALPHA: van Genuchten alpha (1/m)
    CLMPARFLOW_VG_N: van Genuchten n (dimensionless, > 1)
    CLMPARFLOW_TOP/BOT: Domain top/bottom elevation (m)
    CLMPARFLOW_MANNINGS_N: Manning's roughness for overland flow
    CLMPARFLOW_VEGM_FILE: CLM vegetation map file (drv_vegm.dat)
    CLMPARFLOW_VEGP_FILE: CLM vegetation parameter file (drv_vegp.dat)
    CLMPARFLOW_DRV_CLMIN_FILE: CLM driver input file (drv_clmin.dat)

References:
    Kollet, S.J. & Maxwell, R.M. (2008): Capturing the influence of
    groundwater dynamics on land surface processes using an integrated,
    distributed watershed model. Water Resources Research 44(2).

    Dai, Y. et al. (2003): The Common Land Model. Bull. Amer. Meteor. Soc.

    https://github.com/parflow/parflow
"""
from .config import CLMParFlowConfigAdapter
from .extractor import CLMParFlowResultExtractor
from .plotter import CLMParFlowPlotter
from .postprocessor import CLMParFlowPostProcessor
from .preprocessor import CLMParFlowPreProcessor
from .runner import CLMParFlowRunner

__all__ = [
    "CLMParFlowPreProcessor",
    "CLMParFlowRunner",
    "CLMParFlowResultExtractor",
    "CLMParFlowPostProcessor",
    "CLMParFlowConfigAdapter",
    "CLMParFlowPlotter",
]

# Register CLMParFlow config adapter via unified registry
# Note: preprocessor, runner, extractor, postprocessor are registered via
# decorators in their respective component modules.
from symfluence.core.registry import model_manifest

model_manifest(
    "CLMPARFLOW",
    config_adapter=CLMParFlowConfigAdapter,
    plotter=CLMParFlowPlotter,
    build_instructions_module="symfluence.models.clmparflow.build_instructions",
)
