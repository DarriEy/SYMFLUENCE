"""
FUSE to mizuRoute Format Conversion Utilities

Handles conversion of FUSE model output to mizuRoute-compatible format.
Extracted from FUSEWorker for reusability and testability.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import warnings

# Suppress xarray FutureWarning about timedelta64 decoding
warnings.filterwarnings('ignore',
                       message='.*decode_timedelta.*',
                       category=FutureWarning,
                       module='xarray.*')

logger = logging.getLogger(__name__)


class FuseToMizurouteConverter:
    """
    Converts FUSE distributed output to mizuRoute-compatible format.

    Handles:
    - Variable name detection (q_routed, q_instnt, total_discharge, runoff)
    - Dimension reshaping (param_set, latitude, longitude → time, gru)
    - Unit conversion (mm/timestep → m/s, mm/s, mm/day, etc.)
    - Time rounding for mizuRoute precision
    """

    # FUSE variable names to try in order of preference
    FUSE_RUNOFF_VARS = ['q_routed', 'q_instnt', 'total_discharge', 'runoff']

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def convert(
        self,
        fuse_output_dir: Path,
        config: Dict[str, Any],
        proc_id: int = 0
    ) -> bool:
        """
        Convert FUSE distributed output to mizuRoute-compatible format.

        Args:
            fuse_output_dir: Directory containing FUSE output
            config: Configuration dictionary
            proc_id: Process ID for parallel calibration (used in filename)

        Returns:
            True if conversion successful
        """
        try:
            import xarray as xr

            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID')
            fuse_id = config.get('FUSE_FILE_ID', experiment_id)

            # Find FUSE output file (runs_def.nc for run_def mode)
            output_file = fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc"

            if not output_file.exists():
                self.logger.error(f"FUSE output file not found: {output_file}")
                return False

            # Open dataset
            ds = xr.open_dataset(output_file, decode_timedelta=True)

            # Find runoff variable
            q_fuse = self._find_runoff_variable(ds)
            if q_fuse is None:
                self.logger.error(f"FUSE output missing runoff variable. Available: {list(ds.variables)}")
                ds.close()
                return False

            # Get routing variable name
            routing_var = self._get_routing_var_name(config)

            # Reshape dimensions
            q_fuse = self._reshape_dimensions(q_fuse, config)

            # Convert units
            q_fuse_values, target_units = self._convert_units(q_fuse.values, config)

            # Prepare time values
            time_values = self._prepare_time_values(ds)

            # Create routing dataset
            ds_routing = self._create_routing_dataset(
                q_fuse_values, time_values, routing_var, target_units
            )

            # Close input dataset before writing
            ds.close()

            # Save converted output
            expected_filename = f"proc_{proc_id:02d}_{experiment_id}_timestep.nc"
            expected_file = fuse_output_dir / expected_filename

            ds_routing.to_netcdf(expected_file)
            self.logger.debug(f"Created mizuRoute input file: {expected_file}")

            ds_routing.close()
            return True

        except Exception as e:
            self.logger.error(f"Error converting FUSE output to mizuRoute format: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _find_runoff_variable(self, ds) -> Optional[Any]:
        """Find the runoff variable in FUSE output."""
        for var_name in self.FUSE_RUNOFF_VARS:
            if var_name in ds.variables:
                self.logger.debug(f"Using FUSE variable '{var_name}' for routing")
                return ds[var_name]
        return None

    def _get_routing_var_name(self, config: Dict[str, Any]) -> str:
        """Get the routing variable name from config."""
        routing_var = config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
        if routing_var in ('default', None, ''):
            routing_var = 'q_routed'
        return routing_var

    def _reshape_dimensions(self, q_fuse, config: Dict[str, Any]):
        """
        Reshape FUSE output dimensions for mizuRoute.

        FUSE: (time, param_set, latitude, longitude) → mizuRoute: (time, gru)
        """
        # Squeeze out param_set dimension if it exists
        if 'param_set' in q_fuse.dims:
            q_fuse = q_fuse.isel(param_set=0)

        # Handle spatial dimensions
        subcatchment_dim = config.get('FUSE_SUBCATCHMENT_DIM', 'longitude')

        if 'latitude' in q_fuse.dims and 'longitude' in q_fuse.dims:
            # Squeeze singleton dimensions
            if q_fuse.sizes.get('longitude', 1) == 1:
                q_fuse = q_fuse.squeeze('longitude', drop=True)
            if q_fuse.sizes.get('latitude', 1) == 1:
                q_fuse = q_fuse.squeeze('latitude', drop=True)

            # Rename the remaining spatial dimension to 'gru'
            if subcatchment_dim in q_fuse.dims:
                q_fuse = q_fuse.rename({subcatchment_dim: 'gru'})

        # If still no 'gru' dimension, create one (lumped case after squeeze)
        if 'gru' not in q_fuse.dims:
            q_fuse = q_fuse.expand_dims('gru')

        # Ensure dimensions are in (time, gru) order
        if 'time' in q_fuse.dims and 'gru' in q_fuse.dims:
            q_fuse = q_fuse.transpose('time', 'gru')

        return q_fuse

    def _convert_units(
        self,
        values: np.ndarray,
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, str]:
        """
        Convert FUSE output units (mm/timestep) to target units.

        Returns:
            Tuple of (converted values, target units string)
        """
        target_units = config.get('SETTINGS_MIZU_ROUTING_UNITS', 'm/s')
        if target_units in ('default', None, ''):
            target_units = 'm/s'

        timestep_seconds = int(config.get('FORCING_TIME_STEP_SIZE', 86400))

        self.logger.debug(f"Converting FUSE runoff to {target_units} (timestep={timestep_seconds}s)")

        if target_units == 'm/s':
            # mm/timestep -> m/s
            conversion_factor = 1.0 / (1000.0 * timestep_seconds)
            values = values * conversion_factor
            self.logger.debug(f"Applied conversion factor: {conversion_factor:.2e}")

        elif target_units == 'mm/s':
            # mm/timestep -> mm/s
            conversion_factor = 1.0 / timestep_seconds
            values = values * conversion_factor

        elif target_units in ['mm/d', 'mm/day']:
            # mm/timestep -> mm/day
            if timestep_seconds != 86400:
                conversion_factor = 86400.0 / timestep_seconds
                values = values * conversion_factor

        elif target_units in ['mm/h', 'mm/hr', 'mm/hour']:
            # mm/timestep -> mm/hour
            if timestep_seconds != 3600:
                conversion_factor = 3600.0 / timestep_seconds
                values = values * conversion_factor

        else:
            self.logger.warning(f"Unknown target units '{target_units}', passing through without conversion")

        return values, target_units

    def _prepare_time_values(self, ds):
        """Round time values to nearest hour for mizuRoute precision."""
        try:
            import pandas as pd
            time_values = pd.to_datetime(ds['time'].values).round('h')
            # Remove timezone if present
            if hasattr(time_values, 'tz') and time_values.tz is not None:
                time_values = time_values.tz_localize(None)
            return time_values
        except Exception as e:
            self.logger.warning(f"Could not round times for mizuRoute: {e}")
            return ds['time'].values

    def _create_routing_dataset(
        self,
        q_values: np.ndarray,
        time_values,
        routing_var: str,
        units: str
    ):
        """Create xarray Dataset for mizuRoute input."""
        import xarray as xr

        n_gru = q_values.shape[1] if q_values.ndim > 1 else 1

        ds_routing = xr.Dataset({
            routing_var: (('time', 'gru'), q_values)
        }, coords={
            'time': time_values,
            'gru': np.arange(n_gru)
        })

        # Set variable attributes
        ds_routing[routing_var].attrs['units'] = units
        ds_routing[routing_var].attrs['long_name'] = 'FUSE runoff for mizuRoute routing'

        # Add gruId variable (1-indexed as expected by mizuRoute)
        ds_routing['gruId'] = ('gru', np.arange(1, n_gru + 1))

        return ds_routing
