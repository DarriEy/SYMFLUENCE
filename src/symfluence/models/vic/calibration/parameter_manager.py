"""
VIC Parameter Manager

Handles VIC parameter bounds, normalization, and parameter file updates.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('VIC')
class VICParameterManager(BaseParameterManager):
    """Handles VIC parameter bounds, normalization, and file updates."""

    # VIC calibration parameters and their NetCDF variable names
    PARAM_VAR_MAP = {
        'infilt': 'infilt',
        'Ds': 'Ds',
        'Dsmax': 'Dsmax',
        'Ws': 'Ws',
        'c': 'c',
        'depth1': 'depth',  # Indexed by layer
        'depth2': 'depth',
        'depth3': 'depth',
        'Ksat': 'Ksat',           # Applied to all layers
        'expt': 'expt',           # Applied to all layers
        'Wcr_FRACT': 'Wcr_FRACT', # Applied to all layers
        'Wpwp_FRACT': 'Wpwp_FRACT', # Applied to all layers
        'snow_rough': 'snow_rough',
        'elev_offset': None,      # Special: shifts snow band elevations (not a direct NetCDF variable)
    }

    # Layer indices for depth parameters
    LAYER_PARAMS = {
        'depth1': 0,
        'depth2': 1,
        'depth3': 2,
    }

    def __init__(self, config: Dict, logger: logging.Logger, vic_settings_dir: Path):
        """
        Initialize VIC parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            vic_settings_dir: Path to VIC settings directory
        """
        super().__init__(config, logger, vic_settings_dir)

        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse VIC parameters to calibrate from config
        vic_params_str = None
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'vic'):
                vic_params_str = config.model.vic.params_to_calibrate
        except (AttributeError, TypeError):
            pass

        if vic_params_str is None:
            vic_params_str = config.get('VIC_PARAMS_TO_CALIBRATE')

        if vic_params_str is None:
            vic_params_str = 'infilt,Ds,Dsmax,Ws,c,depth1,depth2,depth3,expt,Ksat,Wcr_FRACT,Wpwp_FRACT,snow_rough'
            logger.warning(
                f"VIC_PARAMS_TO_CALIBRATE missing; using fallback: {vic_params_str}"
            )

        self.vic_params = [p.strip() for p in str(vic_params_str).split(',') if p.strip()]

        # Path to parameter file
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.params_dir = self.project_dir / 'VIC_input' / 'parameters'

        # Get parameter file name
        params_file = 'vic_params.nc'
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'vic'):
                params_file = config.model.vic.params_file or params_file
        except (AttributeError, TypeError):
            pass
        self.params_file = self.params_dir / params_file

    def _get_parameter_names(self) -> List[str]:
        """Return VIC parameter names from config."""
        return self.vic_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Return VIC parameter bounds.

        VIC parameters with typical calibration ranges:
        - infilt: [0.001, 0.9] - Variable infiltration curve parameter
        - Ds: [0.0, 1.0] - Fraction of Dsmax for nonlinear baseflow
        - Dsmax: [0.1, 30.0] mm/day - Maximum baseflow velocity
        - Ws: [0.1, 1.0] - Fraction of max soil moisture for baseflow
        - depth1/2/3: [0.05, 2.0] m - Soil layer depths
        - Ksat: [1.0, 5000.0] mm/day - Saturated hydraulic conductivity
        - expt: [4.0, 30.0] - Soil layer exponent
        """
        # VIC-specific bounds
        vic_bounds = {
            # --- Infiltration ---
            'infilt': {'min': 0.001, 'max': 0.9},
            # --- Baseflow ---
            'Ds': {'min': 0.0, 'max': 1.0},          # Fraction of Dsmax for nonlinear baseflow
            'Dsmax': {'min': 0.1, 'max': 30.0},       # Maximum baseflow velocity [mm/day]
            'Ws': {'min': 0.1, 'max': 1.0},           # Soil moisture fraction for nonlinear baseflow
            'c': {'min': 1.0, 'max': 6.0},            # ARNO baseflow exponent (VIC allows up to ~10)
            # --- Soil layers ---
            'depth1': {'min': 0.01, 'max': 0.5},      # Top layer depth [m]
            'depth2': {'min': 0.1, 'max': 2.0},       # Middle layer depth [m]
            'depth3': {'min': 0.1, 'max': 3.0},       # Bottom layer depth [m]
            'expt': {'min': 4.0, 'max': 30.0},        # Brooks-Corey exponent (all layers)
            'Ksat': {'min': 10.0, 'max': 5000.0},     # Saturated hydraulic conductivity [mm/day]
            # --- ET partitioning ---
            'Wcr_FRACT': {'min': 0.3, 'max': 0.9},    # Critical soil moisture fraction
            'Wpwp_FRACT': {'min': 0.05, 'max': 0.45},  # Wilting point fraction (must be < Wcr_FRACT)
            # --- Snow ---
            'snow_rough': {'min': 0.00001, 'max': 0.01},  # Snow surface roughness [m]
            # --- Temperature bias via snow band elevation offset ---
            # Positive offset raises band elevations → cooler bands → delayed snowmelt
            # Each +100m offset cools all bands by ~0.65°C via lapse rate
            # Range 0-1000m gives temperature bias of 0 to -6.5°C
            'elev_offset': {'min': 0.0, 'max': 1000.0},   # Snow band elevation offset [m]
        }

        # Check for config overrides
        config_bounds = self.config.get('VIC_PARAM_BOUNDS', {})
        if config_bounds:
            for param_name, bound_list in config_bounds.items():
                if isinstance(bound_list, (list, tuple)) and len(bound_list) == 2:
                    vic_bounds[param_name] = {
                        'min': float(bound_list[0]),
                        'max': float(bound_list[1])
                    }
                    self.logger.debug(
                        f"Using config bounds for {param_name}: [{bound_list[0]}, {bound_list[1]}]"
                    )

        # Log bounds for calibrated parameters
        for param_name in self.vic_params:
            if param_name in vic_bounds:
                b = vic_bounds[param_name]
                self.logger.info(f"VIC param {param_name}: bounds=[{b['min']:.4f}, {b['max']:.4f}]")

        return vic_bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update VIC parameter file with new parameter values."""
        return self.update_params_nc(params)

    def update_params_nc(self, params: Dict[str, float]) -> bool:
        """
        Update VIC parameter NetCDF file with new values.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if successful
        """
        try:
            if not self.params_file.exists():
                self.logger.error(f"VIC parameter file not found: {self.params_file}")
                return False

            # Open dataset for modification
            ds = xr.open_dataset(self.params_file)
            ds = ds.load()  # Load into memory for modification

            # Enforce constraint: Wpwp_FRACT must be < Wcr_FRACT
            if 'Wcr_FRACT' in params and 'Wpwp_FRACT' in params:
                if params['Wpwp_FRACT'] >= params['Wcr_FRACT']:
                    params['Wpwp_FRACT'] = params['Wcr_FRACT'] * 0.5

            # Handle elev_offset: shift snow band elevations to control snowmelt timing
            if 'elev_offset' in params and 'elevation' in ds:
                offset = params['elev_offset']
                for band in range(ds['elevation'].shape[0]):
                    mask = ~np.isnan(ds['elevation'].values[band])
                    ds['elevation'].values[band][mask] += offset
                self.logger.debug(f"Applied elev_offset = {offset:.0f}m to snow band elevations")

            for param_name, value in params.items():
                var_name = self.PARAM_VAR_MAP.get(param_name)
                if var_name is None:
                    continue  # Skip special params like elev_offset

                if var_name not in ds:
                    self.logger.warning(f"Variable {var_name} not in parameter file")
                    continue

                # Handle layer-specific parameters
                if param_name in self.LAYER_PARAMS:
                    layer_idx = self.LAYER_PARAMS[param_name]
                    if 'nlayer' in ds[var_name].dims:
                        # Update specific layer
                        mask = ~np.isnan(ds[var_name].values[layer_idx])
                        ds[var_name].values[layer_idx][mask] = value
                        self.logger.debug(f"Updated {param_name} (layer {layer_idx}) = {value}")
                    else:
                        self.logger.warning(
                            f"Variable {var_name} doesn't have nlayer dimension"
                        )
                else:
                    # Update all valid cells
                    if len(ds[var_name].dims) == 2:
                        # 2D variable (lat, lon)
                        mask = ~np.isnan(ds[var_name].values)
                        ds[var_name].values[mask] = value
                    elif len(ds[var_name].dims) == 3:
                        # 3D variable (nlayer, lat, lon) - update all layers
                        for layer in range(ds[var_name].shape[0]):
                            mask = ~np.isnan(ds[var_name].values[layer])
                            ds[var_name].values[layer][mask] = value
                    else:
                        ds[var_name].values = value

                    self.logger.debug(f"Updated {param_name} = {value}")

            # Save modified dataset
            # Write to temporary file then rename (atomic operation)
            temp_file = self.params_file.with_suffix('.nc.tmp')
            ds.to_netcdf(temp_file)
            ds.close()

            # Replace original
            temp_file.replace(self.params_file)

            return True

        except Exception as e:
            self.logger.error(f"Error updating VIC parameter file: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from parameter file or defaults."""
        try:
            if not self.params_file.exists():
                return self._get_default_initial_values()

            ds = xr.open_dataset(self.params_file)
            params = {}

            for param_name in self.vic_params:
                var_name = self.PARAM_VAR_MAP.get(param_name)
                if var_name and var_name in ds:
                    data = ds[var_name]

                    # Handle layer-specific parameters
                    if param_name in self.LAYER_PARAMS:
                        layer_idx = self.LAYER_PARAMS[param_name]
                        if 'nlayer' in data.dims:
                            values = data.values[layer_idx]
                            valid_values = values[~np.isnan(values)]
                            if len(valid_values) > 0:
                                params[param_name] = float(np.nanmean(valid_values))
                                continue

                    # Take mean of valid values
                    valid_values = data.values[~np.isnan(data.values)]
                    if len(valid_values) > 0:
                        params[param_name] = float(np.nanmean(valid_values))
                    else:
                        # Use midpoint of bounds
                        bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                        params[param_name] = (bounds['min'] + bounds['max']) / 2
                else:
                    # Use midpoint of bounds
                    bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                    params[param_name] = (bounds['min'] + bounds['max']) / 2

            ds.close()
            return params

        except Exception as e:
            self.logger.error(f"Error reading initial parameters: {e}")
            return self._get_default_initial_values()

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values (midpoint of bounds)."""
        params = {}
        for param_name in self.vic_params:
            bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
            params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params

    def copy_params_to_worker_dir(self, worker_params_dir: Path) -> bool:
        """
        Copy parameter file to a worker-specific directory for parallel calibration.

        Args:
            worker_params_dir: Target directory for worker's parameter files

        Returns:
            True if successful
        """
        import shutil

        try:
            worker_params_dir.mkdir(parents=True, exist_ok=True)

            # Copy parameter file
            if self.params_file.exists():
                shutil.copy2(self.params_file, worker_params_dir / self.params_file.name)

            # Also copy domain file if it exists
            domain_file = self.params_dir / 'vic_domain.nc'
            if domain_file.exists():
                shutil.copy2(domain_file, worker_params_dir / domain_file.name)

            return True

        except Exception as e:
            self.logger.error(f"Error copying params to {worker_params_dir}: {e}")
            return False
