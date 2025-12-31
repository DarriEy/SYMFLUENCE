import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

class ParameterManager:
    """Handles parameter bounds, normalization, file generation, and soil depth calculations"""
    
    def __init__(self, config: Dict, logger: logging.Logger, optimization_settings_dir: Path):
        self.config = config
        self.logger = logger
        self.optimization_settings_dir = optimization_settings_dir
        
        # Parse parameter lists
        self.local_params = [p.strip() for p in config.get('PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        self.basin_params = [p.strip() for p in config.get('BASIN_PARAMS_TO_CALIBRATE', '').split(',') if p.strip()]
        
        # Identify depth parameters
        self.depth_params = []
        if config.get('CALIBRATE_DEPTH', False):
            self.depth_params = ['total_mult', 'shape_factor']
        
        # Add special multiplier if in list
        if 'total_soil_depth_multiplier' in self.local_params:
            self.depth_params.append('total_soil_depth_multiplier')
            self.local_params.remove('total_soil_depth_multiplier')
            
        self.mizuroute_params = []
        
        if config.get('CALIBRATE_MIZUROUTE', False):
            mizuroute_params_str = config.get('MIZUROUTE_PARAMS_TO_CALIBRATE', 'velo,diff')
            self.mizuroute_params = [p.strip() for p in mizuroute_params_str.split(',') if p.strip()]
        
        # Load parameter bounds
        self.param_bounds = self._parse_all_bounds()
        
        # Load original soil depths if depth calibration enabled
        self.original_depths = None
        if self.depth_params:
            self.original_depths = self._load_original_depths()
        
        # Get attribute file path
        self.attr_file_path = self.optimization_settings_dir / config.get('SETTINGS_SUMMA_ATTRIBUTES', 'attributes.nc')
    
    @property
    def all_param_names(self) -> List[str]:
        """Get list of all parameter names"""
        return self.local_params + self.basin_params + self.depth_params + self.mizuroute_params
    
    def get_initial_parameters(self) -> Optional[Dict[str, np.ndarray]]:
        """Get initial parameter values from existing files or defaults"""
        # Try to load existing optimized parameters
        existing_params = self._load_existing_optimized_parameters()
        if existing_params:
            self.logger.info("Loaded existing optimized parameters")
            return existing_params
        
        # Extract parameters from model files
        #self.logger.info("Extracting initial parameters from default values")
        return self._extract_default_parameters()
    
    def normalize_parameters(self, params: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert parameter dictionary to normalized array [0,1]"""
        normalized = np.zeros(len(self.all_param_names))
        
        for i, param_name in enumerate(self.all_param_names):
            if param_name in params and param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                
                # Get parameter value
                param_values = params[param_name]
                if isinstance(param_values, np.ndarray) and len(param_values) > 1:
                    value = np.mean(param_values)  # Use mean for multi-value parameters
                else:
                    value = param_values[0] if isinstance(param_values, np.ndarray) else param_values
                
                # Normalize to [0,1]
                normalized[i] = (value - bounds['min']) / (bounds['max'] - bounds['min'])
        
        return np.clip(normalized, 0, 1)
    
    def denormalize_parameters(self, normalized_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Convert normalized array back to parameter dictionary"""
        params = {}
        
        for i, param_name in enumerate(self.all_param_names):
            if param_name in self.param_bounds:
                bounds = self.param_bounds[param_name]
                denorm_value = bounds['min'] + normalized_array[i] * (bounds['max'] - bounds['min'])
                
                # Validate bounds
                denorm_value = np.clip(denorm_value, bounds['min'], bounds['max'])
                
                # Format based on parameter type
                if param_name in self.depth_params:
                    params[param_name] = np.array([denorm_value])
                elif param_name in self.mizuroute_params:
                    params[param_name] = denorm_value
                elif param_name in self.basin_params:
                    params[param_name] = np.array([denorm_value])
                else:
                    # Local parameters - expand to HRU count
                    params[param_name] = self._expand_to_hru_count(denorm_value)
        
        return params
    
    
    def _parse_all_bounds(self) -> Dict[str, Dict[str, float]]:
        """Parse parameter bounds from all parameter info files"""
        bounds = {}
        
        # Parse local parameter bounds
        if self.local_params:
            local_param_file = Path(self.config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}" / 'settings' / 'SUMMA' / 'localParamInfo.txt'
            local_bounds = self._parse_param_info_file(local_param_file, self.local_params)
            bounds.update(local_bounds)
        
        # Parse basin parameter bounds
        if self.basin_params:
            basin_param_file = Path(self.config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{self.config.get('DOMAIN_NAME')}" / 'settings' / 'SUMMA' / 'basinParamInfo.txt'
            basin_bounds = self._parse_param_info_file(basin_param_file, self.basin_params)
            bounds.update(basin_bounds)
        
        # Add depth parameter bounds
        if self.depth_params:
            if 'total_mult' in self.depth_params or 'total_soil_depth_multiplier' in self.depth_params:
                bounds['total_mult'] = {'min': 0.1, 'max': 5.0}
                bounds['total_soil_depth_multiplier'] = {'min': 0.1, 'max': 5.0}
            if 'shape_factor' in self.depth_params:
                bounds['shape_factor'] = {'min': 0.1, 'max': 3.0}
        
        # Add mizuRoute parameter bounds
        if self.mizuroute_params:
            mizuroute_bounds = self._get_mizuroute_bounds()
            bounds.update(mizuroute_bounds)
        
        return bounds
    
    def _parse_param_info_file(self, file_path: Path, param_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Parse parameter bounds from a SUMMA parameter info file"""
        bounds = {}
        
        if not file_path.exists():
            self.logger.error(f"Parameter file not found: {file_path}")
            return bounds
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('!') or line.startswith("'"):
                        continue
                    
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) < 4:
                        continue
                    
                    param_name = parts[0]
                    if param_name in param_names:
                        try:
                            min_val = float(parts[2].replace('d','e').replace('D','e'))
                            max_val = float(parts[3].replace('d','e').replace('D','e'))
                            
                            if min_val > max_val:
                                min_val, max_val = max_val, min_val
                            
                            if min_val == max_val:
                                range_val = abs(min_val) * 0.1 if min_val != 0 else 0.1
                                min_val -= range_val
                                max_val += range_val
                            
                            bounds[param_name] = {'min': min_val, 'max': max_val}
                            
                        except ValueError as e:
                            self.logger.error(f"Could not parse bounds for {param_name}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error reading parameter file {file_path}: {str(e)}")
        
        return bounds
    
    def _get_mizuroute_bounds(self) -> Dict[str, Dict[str, float]]:
        """Get parameter bounds for mizuRoute parameters"""
        default_bounds = {
            'velo': {'min': 0.1, 'max': 5.0},
            'diff': {'min': 100.0, 'max': 5000.0},
            'mann_n': {'min': 0.01, 'max': 0.1},
            'wscale': {'min': 0.0001, 'max': 0.01},
            'fshape': {'min': 1.0, 'max': 5.0},
            'tscale': {'min': 3600, 'max': 172800}
        }
        
        bounds = {}
        for param in self.mizuroute_params:
            if param in default_bounds:
                bounds[param] = default_bounds[param]
            else:
                self.logger.warning(f"Unknown mizuRoute parameter: {param}")
        
        return bounds
    
    def _load_existing_optimized_parameters(self) -> Optional[Dict[str, np.ndarray]]:
        """Load existing optimized parameters from default settings"""
        trial_params_path = self.config.get('SYMFLUENCE_DATA_DIR')
        if trial_params_path == 'default':
            return None
        
        # Implementation would check for existing trialParams.nc file
        # For brevity, returning None - full implementation would load existing params
        return None
    
    def _extract_default_parameters(self) -> Dict[str, np.ndarray]:
        """Extract default parameter values from parameter info files"""
        defaults = {}
        
        # Parse local parameters
        if self.local_params:
            local_defaults = self._parse_defaults_from_file(
                self.optimization_settings_dir / 'localParamInfo.txt', 
                self.local_params
            )
            defaults.update(local_defaults)
        
        # Parse basin parameters
        if self.basin_params:
            basin_defaults = self._parse_defaults_from_file(
                self.optimization_settings_dir / 'basinParamInfo.txt',
                self.basin_params
            )
            defaults.update(basin_defaults)
        
        # Add depth parameters
        if self.depth_params:
            defaults['total_mult'] = np.array([1.0])
            defaults['shape_factor'] = np.array([1.0])
        
        # Add mizuRoute parameters
        if self.mizuroute_params:
            for param in self.mizuroute_params:
                defaults[param] = self._get_default_mizuroute_value(param)
        
        # Expand to HRU count
        return self._expand_defaults_to_hru_count(defaults)
    
    def _parse_defaults_from_file(self, file_path: Path, param_names: List[str]) -> Dict[str, np.ndarray]:
        """Parse default values from parameter info file"""
        defaults = {}
        
        if not file_path.exists():
            return defaults
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('!') or line.startswith("'"):
                        continue
                    
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 4:
                        param_name = parts[0]
                        if param_name in param_names:
                            try:
                                default_val = float(parts[1].replace('d','e').replace('D','e'))
                                defaults[param_name] = np.array([default_val])
                            except ValueError:
                                continue
        except Exception as e:
            self.logger.error(f"Error parsing defaults from {file_path}: {str(e)}")
        
        return defaults
    
    def _get_default_mizuroute_value(self, param_name: str) -> float:
        """Get default value for mizuRoute parameter"""
        defaults = {
            'velo': 1.0,
            'diff': 1000.0,
            'mann_n': 0.025,
            'wscale': 0.001,
            'fshape': 2.5,
            'tscale': 86400
        }
        return defaults.get(param_name, 1.0)
    
    def _expand_defaults_to_hru_count(self, defaults: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Expand parameter defaults to match HRU count"""
        try:
            # Get HRU count from attributes file
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)
            
            expanded_defaults = {}
            routing_params = ['routingGammaShape', 'routingGammaScale']
            
            for param_name, values in defaults.items():
                if param_name in self.basin_params or param_name in routing_params:
                    expanded_defaults[param_name] = values
                elif param_name in self.depth_params or param_name in self.mizuroute_params:
                    expanded_defaults[param_name] = values
                else:
                    expanded_defaults[param_name] = np.full(num_hrus, values[0])
            
            return expanded_defaults
            
        except Exception as e:
            self.logger.error(f"Error expanding defaults: {str(e)}")
            return defaults
    
    def _expand_to_hru_count(self, value: float) -> np.ndarray:
        """Expand single value to HRU count"""
        try:
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)
            return np.full(num_hrus, value)
        except:
            return np.array([value])
    
    def _load_original_depths(self) -> Optional[np.ndarray]:
        """Load original soil depths from coldState.nc"""
        try:
            coldstate_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
            
            if not coldstate_path.exists():
                return None

            with nc.Dataset(coldstate_path, 'r') as ds:
                if 'mLayerDepth' in ds.variables:
                    return ds.variables['mLayerDepth'][:, 0].copy()
            
        except Exception as e:
            self.logger.error(f"Error loading original depths: {str(e)}")
        
        return None
    
    def _update_soil_depths(self, params: Dict[str, np.ndarray]) -> bool:
        """Update soil depths in coldState.nc"""
        if self.original_depths is None:
            return False
        
        try:
            total_mult = params['total_mult'][0] if isinstance(params['total_mult'], np.ndarray) else params['total_mult']
            shape_factor = params['shape_factor'][0] if isinstance(params['shape_factor'], np.ndarray) else params['shape_factor']
            
            # Calculate new depths using shape method
            new_depths = self._calculate_new_depths(total_mult, shape_factor)
            if new_depths is None:
                return False
            
            # Calculate layer heights
            heights = np.zeros(len(new_depths) + 1)
            for i in range(len(new_depths)):
                heights[i + 1] = heights[i] + new_depths[i]
            
            # Update coldState.nc
            coldstate_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_COLDSTATE', 'coldState.nc')
            
            with nc.Dataset(coldstate_path, 'r+') as ds:
                if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                    return False
                
                num_hrus = ds.dimensions['hru'].size
                for h in range(num_hrus):
                    ds.variables['mLayerDepth'][:, h] = new_depths
                    ds.variables['iLayerHeight'][:, h] = heights
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating soil depths: {str(e)}")
            return False
    
    def _calculate_new_depths(self, total_mult: float, shape_factor: float) -> Optional[np.ndarray]:
        """Calculate new soil depths using shape method"""
        if self.original_depths is None:
            return None
        
        arr = self.original_depths.copy()
        n = len(arr)
        idx = np.arange(n)
        
        # Calculate shape weights
        if shape_factor > 1:
            w = np.exp(idx / (n - 1) * np.log(shape_factor))
        elif shape_factor < 1:
            w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
        else:
            w = np.ones(n)
        
        # Normalize weights
        w /= w.mean()
        
        # Apply multipliers
        new_depths = arr * w * total_mult
        
        return new_depths
    
    def _update_mizuroute_parameters(self, params: Dict) -> bool:
        """Update mizuRoute parameters in param.nml.default"""
        try:
            mizuroute_settings_dir = self.optimization_settings_dir.parent / "mizuRoute"
            param_file = mizuroute_settings_dir / "param.nml.default"
            
            if not param_file.exists():
                return True  # Skip if file doesn't exist
            
            # Read file
            with open(param_file, 'r') as f:
                content = f.read()
            
            # Update parameters
            updated_content = content
            for param_name in self.mizuroute_params:
                if param_name in params:
                    param_value = params[param_name]
                    pattern = rf'(\s+{param_name}\s*=\s*)[0-9.-]+'
                    
                    if param_name in ['tscale']:
                        replacement = rf'\g<1>{int(param_value)}'
                    else:
                        replacement = rf'\g<1>{param_value:.6f}'
                    
                    updated_content = re.sub(pattern, replacement, updated_content)
            
            # Write updated file
            with open(param_file, 'w') as f:
                f.write(updated_content)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating mizuRoute parameters: {str(e)}")
            return False
        
    def _generate_trial_params_file(self, params: Dict[str, np.ndarray]) -> bool:
        """Generate trialParams.nc file with proper dimensions"""
        try:
            trial_params_path = self.optimization_settings_dir / self.config.get('SETTINGS_SUMMA_TRIALPARAMS', 'trialParams.nc')
            
            # Get HRU and GRU counts from attributes
            with xr.open_dataset(self.attr_file_path) as ds:
                num_hrus = ds.sizes.get('hru', 1)
                num_grus = ds.sizes.get('gru', 1)
                
                # Get original hruId values
                if 'hruId' in ds.variables:
                    original_hru_ids = ds.variables['hruId'][:].copy()
                else:
                    original_hru_ids = np.arange(1, num_hrus + 1)
                    self.logger.warning(f"hruId not found in attributes.nc, using sequential IDs 1 to {num_hrus}")
                
                # Get original gruId values
                if 'gruId' in ds.variables:
                    original_gru_ids = ds.variables['gruId'][:].copy()
                else:
                    original_gru_ids = np.arange(1, num_grus + 1)
                    self.logger.warning(f"gruId not found in attributes.nc, using sequential IDs 1 to {num_grus}")
            
            # Define parameter levels
            routing_params = ['routingGammaShape', 'routingGammaScale']
            
            with nc.Dataset(trial_params_path, 'w', format='NETCDF4') as output_ds:
                # Create dimensions
                output_ds.createDimension('hru', num_hrus)
                output_ds.createDimension('gru', num_grus)
                
                # Create coordinate variables with ORIGINAL ID values
                hru_var = output_ds.createVariable('hruId', 'i4', ('hru',), fill_value=-9999)
                hru_var[:] = original_hru_ids  # ← USE ORIGINAL hruId VALUES
                
                gru_var = output_ds.createVariable('gruId', 'i4', ('gru',), fill_value=-9999)
                gru_var[:] = original_gru_ids  # ← USE ORIGINAL gruId VALUES
                
                # Add parameters
                for param_name, param_values in params.items():
                    param_values_array = np.asarray(param_values)
                    
                    if param_name in routing_params or param_name in self.basin_params:
                        # GRU-level parameters
                        param_var = output_ds.createVariable(param_name, 'f8', ('gru',), fill_value=np.nan)
                        param_var.long_name = f"Trial value for {param_name}"
                        
                        if len(param_values_array) == 1:
                            param_var[:] = param_values_array[0]
                        else:
                            param_var[:] = param_values_array[:num_grus]
                    else:
                        # HRU-level parameters
                        param_var = output_ds.createVariable(param_name, 'f8', ('hru',), fill_value=np.nan)
                        param_var.long_name = f"Trial value for {param_name}"
                        
                        if len(param_values_array) == num_hrus:
                            param_var[:] = param_values_array
                        elif len(param_values_array) == 1:
                            param_var[:] = param_values_array[0]
                        else:
                            param_var[:] = param_values_array[:num_hrus]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating trial params file: {str(e)}")
            return False
