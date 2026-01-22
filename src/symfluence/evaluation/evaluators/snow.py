#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Snow (SWE/SCA) Evaluator.

Evaluates snow water equivalent (SWE) and snow-covered area (SCA) from model outputs.
Supports multi-target calibration with automatic target selection.
"""

import logging
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import cast, Any, Dict, List, Optional, TYPE_CHECKING

from symfluence.evaluation.registry import EvaluationRegistry
from symfluence.evaluation.output_file_locator import OutputFileLocator
from .base import ModelEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@EvaluationRegistry.register('SCA')
@EvaluationRegistry.register('SWE')
@EvaluationRegistry.register('SNOW')
class SnowEvaluator(ModelEvaluator):
    """Snow evaluator for SWE and SCA calibration.

    Evaluates simulated snow using one of two metrics:
    1. SWE (Snow Water Equivalent): Mass of water in snowpack (kg/m²)
       - Directly comparable to observations (in-situ, satellite-derived)
       - Continuous variable (0 to ~1000 kg/m²)
    2. SCA (Snow-Covered Area): Fraction of basin covered by snow (%)
       - Derived from satellite observations (MODIS, Landsat)
       - Discontinuous (0 to 100%, or binary presence/absence)

    Multi-target Support:
        Can override target via kwargs (for multivariate calibration).
        Supports SWE, SCA, snow_depth targets.

    Target Resolution Priority:
        1. kwargs target override (for multivariate mode)
        2. config.optimization.target (typed config)
        3. CALIBRATION_VARIABLE in dict config (default: swe)
        4. Pattern matching: if 'swe'/'snow' in name → swe; if 'sca' → sca

    Output Variables:
        - scalarSWE: SUMMA SWE output (kg/m²)
        - scalarSCA: SUMMA fractional SCA (0-1)
        - spatial dimensions: HRU, GRU, point (averaged/selected)

    Attributes:
        optimization_target: 'swe', 'sca', or 'snow_depth'
        variable_name: Same as optimization_target
    """

    def __init__(self, config: 'SymfluenceConfig', project_dir: Path, logger: logging.Logger, **kwargs):
        """Initialize snow evaluator with target determination.

        Determines whether to evaluate SWE or SCA via multiple configuration sources.

        Args:
            config: Typed configuration object
            project_dir: Project root directory
            logger: Logger instance
            **kwargs: Optional target override (target='swe' or target='sca')
        """
        # Allow target override from kwargs (for multivariate calibration)
        self._target_override = kwargs.get('target')
        super().__init__(config, project_dir, logger)

        # Determine variable target: swe or sca
        self.optimization_target = self._target_override
        if self.optimization_target:
            self.optimization_target = self.optimization_target.lower()

        if not self.optimization_target:
            # Get from typed config, with fallback to flat config keys and CALIBRATION_VARIABLE
            opt_target = self._get_config_value(
                lambda: self.config.optimization.target,
                default=None
            )
            # Also check flat config keys (for worker processes with flattened config)
            if not opt_target:
                opt_target = self.config_dict.get('OPTIMIZATION_TARGET')

            calib_var = self.config_dict.get('CALIBRATION_VARIABLE', 'swe')
            self.logger.debug(f"[SNOW INIT] opt_target={opt_target}, calib_var={calib_var}")
            self.optimization_target = (opt_target or calib_var).lower()

        if self.optimization_target not in ['swe', 'sca', 'snow_depth']:
            # Check if OPTIMIZATION_TARGET contains swe/sca keywords
            opt_target = self.config_dict.get('OPTIMIZATION_TARGET', '').lower()
            if opt_target in ['swe', 'sca']:
                self.optimization_target = opt_target
            else:
                # Fall back to CALIBRATION_VARIABLE
                calibration_var = self.config_dict.get('CALIBRATION_VARIABLE', '').lower()
                if 'swe' in calibration_var or 'snow' in calibration_var:
                    self.optimization_target = 'swe'
                elif 'sca' in calibration_var:
                    self.optimization_target = 'sca'

        self.logger.debug(f"[SNOW INIT FINAL] optimization_target={self.optimization_target}")
        self.variable_name = self.optimization_target

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Locate snow output files containing SWE and/or SCA variables.

        Args:
            sim_dir: Directory containing simulation outputs

        Returns:
            List[Path]: Paths to snow output files (typically NetCDF)
        """
        locator = OutputFileLocator(self.logger)
        return locator.find_snow_files(sim_dir)

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract specified snow variable (SWE or SCA) from simulation output.

        Dispatches to _extract_swe_data() or _extract_sca_data() based on
        optimization_target determined during initialization.

        Args:
            sim_files: List of simulation output files
            **kwargs: Additional parameters (unused)

        Returns:
            pd.Series: Time series of selected snow variable

        Raises:
            Exception: If file cannot be read or variable not found
        """
        sim_file = sim_files[0]
        try:
            with xr.open_dataset(sim_file) as ds:
                if self.optimization_target == 'swe':
                    return self._extract_swe_data(ds)
                elif self.optimization_target == 'sca':
                    return self._extract_sca_data(ds)
                else:
                    return self._extract_swe_data(ds)
        except Exception as e:
            self.logger.error(f"Error extracting snow data from {sim_file}: {str(e)}")
            raise

    def _extract_swe_data(self, ds: xr.Dataset) -> pd.Series:
        """Extract Snow Water Equivalent (SWE) from SUMMA output.

        SWE is the mass of water contained in the snowpack (kg/m²).
        This method:
        1. Loads scalarSWE from NetCDF
        2. Collapses spatial dimensions (HRU/GRU) to basin scale
        3. Returns time series in kg/m²

        Spatial Aggregation:
            - Single HRU/GRU: selects that unit (isel)
            - Multiple units: averages across (mean)
            - Any other dimensions: selects first

        Args:
            ds: xarray Dataset with scalarSWE variable

        Returns:
            pd.Series: Time series of basin-scale SWE (kg/m²)

        Raises:
            ValueError: If scalarSWE not found in dataset
        """
        if 'scalarSWE' not in ds.variables:
            raise ValueError("scalarSWE variable not found")
        swe_var = ds['scalarSWE']

        # Collapse spatial dimensions
        sim_xr = swe_var
        for dim in ['hru', 'gru']:
            if dim in sim_xr.dims:
                if sim_xr.sizes[dim] == 1:
                    sim_xr = sim_xr.isel({dim: 0})
                else:
                    sim_xr = sim_xr.mean(dim=dim)

        # Handle any remaining non-time dimensions
        non_time_dims = [dim for dim in sim_xr.dims if dim != 'time']
        if non_time_dims:
            sim_xr = sim_xr.isel({d: 0 for d in non_time_dims})

        sim_data = cast(pd.Series, sim_xr.to_pandas())

        # DEBUG: Log extraction stats
        if len(sim_data) > 0:
            self.logger.debug(f"SWE extraction: min={sim_data.min():.3f}, max={sim_data.max():.3f}, mean={sim_data.mean():.3f} kg/m² (n={len(sim_data)})")

        return sim_data

    def _extract_sca_data(self, ds: xr.Dataset) -> pd.Series:
        sca_vars = ['scalarGroundSnowFraction', 'scalarSWE']
        for var_name in sca_vars:
            if var_name in ds.variables:
                sca_var = ds[var_name]

                # Collapse spatial dimensions
                sim_xr = sca_var
                for dim in ['hru', 'gru']:
                    if dim in sim_xr.dims:
                        if sim_xr.sizes[dim] == 1:
                            sim_xr = sim_xr.isel({dim: 0})
                        else:
                            sim_xr = sim_xr.mean(dim=dim)

                # Handle any remaining non-time dimensions
                non_time_dims = [dim for dim in sim_xr.dims if dim != 'time']
                if non_time_dims:
                    sim_xr = sim_xr.isel({d: 0 for d in non_time_dims})

                sim_data = cast(pd.Series, sim_xr.to_pandas())

                if var_name == 'scalarSWE':
                    swe_threshold = 1.0
                    sim_data = (sim_data > swe_threshold).astype(float)

                return sim_data
        raise ValueError("No suitable SCA variable found")

    def calculate_metrics(self, sim: Any, obs: Optional[pd.Series] = None,
                         mizuroute_dir: Optional[Path] = None,
                         calibration_only: bool = True, **kwargs) -> Optional[Dict[str, float]]:
        """
        Calculate performance metrics for simulated snow data.

        Args:
            sim: Either a Path to simulation directory or a pre-loaded pd.Series
            obs: Optional pre-loaded pd.Series of observations. If None, loads from file.
            mizuroute_dir: mizuRoute simulation directory (if needed and sim is Path)
            calibration_only: If True, only use calibration period
        """
        simulated_data = sim
        # Ensure we are using the correct target if provided in kwargs
        if 'target' in kwargs:
            self.optimization_target = kwargs['target'].lower()
            self.variable_name = self.optimization_target

        # Call base class with proper signature: sim, obs=None, mizuroute_dir=None, calibration_only=True
        return super().calculate_metrics(
            sim=simulated_data,
            obs=obs,
            mizuroute_dir=mizuroute_dir,
            calibration_only=calibration_only
        )

    def get_observed_data_path(self) -> Path:
        """Get path to preprocessed observed snow data."""
        if self.optimization_target == 'swe':
            # Check multiple possible locations for SWE observations
            paths = [
                # Primary location: snow/swe/processed/
                self.project_dir / "observations" / "snow" / "swe" / "processed" / f"{self.domain_name}_swe_processed.csv",
                # Fallback locations for backwards compatibility
                self.project_dir / "observations" / "snow" / "swe" / "preprocessed" / f"{self.domain_name}_swe_processed.csv",
                self.project_dir / "observations" / "snow" / "processed" / f"{self.domain_name}_snow_processed.csv",
                self.project_dir / "observations" / "snow" / "preprocessed" / f"{self.domain_name}_snow_processed.csv",
            ]
            for p in paths:
                if p.exists(): return p
            return paths[0]
        elif self.optimization_target == 'sca':
            # Check multiple possible locations for SCA observations
            paths = [
                self.project_dir / "observations" / "snow" / "sca" / "processed" / f"{self.domain_name}_sca_processed.csv",
                self.project_dir / "observations" / "snow" / "sca" / "preprocessed" / f"{self.domain_name}_sca_processed.csv",
                self.project_dir / "observations" / "snow" / "processed" / f"{self.domain_name}_modis_snow_processed.csv",
                self.project_dir / "observations" / "snow" / "preprocessed" / f"{self.domain_name}_modis_snow_processed.csv",
            ]
            for p in paths:
                if p.exists(): return p
            return paths[0]
        else:
            paths = [
                self.project_dir / "observations" / "snow" / "processed" / f"{self.domain_name}_snow_processed.csv",
                self.project_dir / "observations" / "snow" / "preprocessed" / f"{self.domain_name}_snow_processed.csv",
            ]
            for p in paths:
                if p.exists(): return p
            return paths[0]

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        if self.optimization_target == 'swe':
            # Check for exact match first
            for col in columns:
                if col.lower() == 'swe':
                    return col
            # Then check for patterns
            for col in columns:
                if any(term in col.lower() for term in ['swe', 'snow_water_equivalent', 'value', 'water_equiv']):
                    return col
        elif self.optimization_target == 'sca':
            for col in columns:
                if any(term in col.lower() for term in ['snow_cover_ratio', 'sca', 'snow_cover']):
                    return col
        return None

    def _load_observed_data(self) -> Optional[pd.Series]:
        try:
            obs_path = self.get_observed_data_path()
            if not obs_path.exists():
                self.logger.warning(f"Snow observation file not found: {obs_path}")
                return None

            obs_df = pd.read_csv(obs_path)

            self.logger.debug(f"[SNOW DEBUG] Target: {self.optimization_target}")
            self.logger.debug(f"[SNOW DEBUG] Columns: {list(obs_df.columns)}")

            date_col = self._find_date_column(obs_df.columns)
            data_col = self._get_observed_data_column(obs_df.columns)

            self.logger.debug(f"[SNOW DEBUG] Found Date: {date_col}, Data: {data_col}")

            if not date_col or not data_col:
                self.logger.warning(f"Could not find required columns in {obs_path}. Need Date and data column.")
                return None

            obs_df['DateTime'] = pd.to_datetime(obs_df[date_col], errors='coerce')
            obs_df = obs_df.dropna(subset=['DateTime'])
            obs_df.set_index('DateTime', inplace=True)

            obs_series = obs_df[data_col].copy()
            missing_indicators = ['', ' ', 'NA', 'na', 'N/A', 'n/a', 'NULL', 'null', '-', '--', '---', 'missing', 'Missing', 'MISSING']
            for indicator in missing_indicators:
                obs_series = obs_series.replace(indicator, np.nan)

            obs_series = pd.to_numeric(obs_series, errors='coerce')

            if self.optimization_target == 'swe':
                # Convert if data is likely in inches (common for NRCS)
                # If values are large, they are likely already in mm
                if obs_series.max() < 250: # 250 inches is a huge amount of SWE
                    obs_series = self._convert_swe_units(obs_series)
                obs_series = obs_series[obs_series >= 0]

            return obs_series.dropna()
        except Exception as e:
            self.logger.error(f"Error loading observed snow data: {str(e)}")
            return None

    def _convert_swe_units(self, obs_swe: pd.Series) -> pd.Series:
        """Convert SWE units from inches to kg/m² (mm water equivalent)"""
        # Assume inches if set up that way, otherwise just return
        return obs_swe * 25.4

    def needs_routing(self) -> bool:
        return False
