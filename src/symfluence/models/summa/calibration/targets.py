#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SUMMA Calibration Targets

Provides calibration target classes specifically designed for SUMMA model output.
SUMMA is the default model for the base StreamflowEvaluator, so these targets
primarily provide explicit naming and registration for consistency with other models.
"""

import logging
from pathlib import Path
from typing import Dict, Any

from symfluence.evaluation.evaluators import StreamflowEvaluator, SnowEvaluator, ETEvaluator
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_calibration_target('SUMMA', 'streamflow')
class SUMMAStreamflowTarget(StreamflowEvaluator):
    """SUMMA-specific streamflow calibration target.

    SUMMA outputs runoff at HRU/GRU level which is then converted to basin-scale
    discharge. This target handles:
    1. Direct SUMMA output (averageRoutedRunoff, basin__TotalRunoff, etc.)
    2. mizuRoute routed output (when routing is enabled)

    The base StreamflowEvaluator was designed primarily for SUMMA, so this
    class provides explicit naming and registry registration for consistency
    with other model-specific calibration targets.

    Key Features:
        - Automatic detection of SUMMA vs mizuRoute output format
        - Unit conversion from mass flux (kg m⁻² s⁻¹) to volume flux (m³/s)
        - Area-weighted spatial aggregation for distributed models
        - Catchment area resolution from multiple sources

    Configuration:
        FIXED_CATCHMENT_AREA: Manual area override (m²)
        OBSERVATIONS_PATH: Override to observed streamflow file path
        ROUTING_DELINEATION: 'lumped' or 'river_network'
    """

    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        """Initialize SUMMA streamflow calibration target.

        Args:
            config: Configuration dictionary
            project_dir: Path to project directory
            logger: Logger instance
        """
        super().__init__(config, project_dir, logger)
        self.logger.debug("SUMMAStreamflowTarget initialized")


@OptimizerRegistry.register_calibration_target('SUMMA', 'snow')
class SUMMASnowTarget(SnowEvaluator):
    """SUMMA-specific snow calibration target.

    Handles snow water equivalent (SWE), snow cover area (SCA), and snow depth
    evaluation for SUMMA model outputs.

    SUMMA Snow Variables:
        - scalarSWE: Snow water equivalent (kg m⁻²)
        - scalarSnowDepth: Snow depth (m)
        - scalarSnowfall: Snowfall rate (kg m⁻² s⁻¹)

    Observation Sources:
        - Point observations (snow pillows, courses)
        - SNODAS gridded product
        - SNOTEL network data
    """

    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        """Initialize SUMMA snow calibration target.

        Args:
            config: Configuration dictionary
            project_dir: Path to project directory
            logger: Logger instance
        """
        super().__init__(config, project_dir, logger)
        self.logger.debug("SUMMASnowTarget initialized")


@OptimizerRegistry.register_calibration_target('SUMMA', 'et')
class SUMMAETTarget(ETEvaluator):
    """SUMMA-specific evapotranspiration calibration target.

    Handles ET evaluation for SUMMA model outputs using various observation
    sources including FLUXNET, MODIS, and point measurements.

    SUMMA ET Variables:
        - scalarLatHeatTotal: Latent heat flux (W m⁻²)
        - scalarSenHeatTotal: Sensible heat flux (W m⁻²)
        - scalarCanopyEvaporation: Canopy evaporation (kg m⁻² s⁻¹)
        - scalarGroundEvaporation: Ground evaporation (kg m⁻² s⁻¹)
    """

    def __init__(self, config: Dict[str, Any], project_dir: Path, logger: logging.Logger):
        """Initialize SUMMA ET calibration target.

        Args:
            config: Configuration dictionary
            project_dir: Path to project directory
            logger: Logger instance
        """
        super().__init__(config, project_dir, logger)
        self.logger.debug("SUMMAETTarget initialized")


__all__ = [
    'SUMMAStreamflowTarget',
    'SUMMASnowTarget',
    'SUMMAETTarget',
]
