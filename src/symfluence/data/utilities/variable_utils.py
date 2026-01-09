import pint_xarray # type: ignore
import pint # type: ignore
import xarray as xr # type: ignore
import yaml # type: ignore
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class VariableHandler:
    """
    Handles variable name mapping and unit conversion between different datasets and models.
    
    Attributes:
        variable_mappings (Dict): Dataset to model variable name mappings
        unit_registry (pint_xarray.UnitRegistry): Unit conversion registry
        logger (logging.Logger): SYMFLUENCE logger instance
    """
    
    # Dataset variable name mappings
    DATASET_MAPPINGS = {
        'ERA5': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'CARRA': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd_u': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'windspd_v': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'kg m-2 s-1'}
        },
        'RDRS': {
            'RDRS_v2.1_P_TT_1.5m': {'standard_name': 'air_temperature', 'units': 'K'},
            'RDRS_v2.1_P_P0_SFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'RDRS_v2.1_P_HU_1.5m': {'standard_name': 'specific_humidity', 'units': '1'},
            'RDRS_v2.1_P_UVC_10m': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'RDRS_v2.1_P_FI_SFC': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'RDRS_v2.1_P_FB_SFC': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'RDRS_v2.1_A_PR0_SFC': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'RDRS_v3.1': {
            'TT': {'standard_name': 'air_temperature', 'units': 'K'},
            'P0': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'HU': {'standard_name': 'specific_humidity', 'units': '1'},
            'UVC': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'FI': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'FB': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'PR0': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'CASR': {
            'CaSR_v3.1_A_TT_1.5m': {'standard_name': 'air_temperature', 'units': 'K'},
            'CaSR_v3.1_P_P0_SFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'CaSR_v3.1_P_HU_1.5m': {'standard_name': 'specific_humidity', 'units': '1'},
            'CaSR_v3.1_P_UVC_10m': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'CaSR_v3.1_P_FI_SFC': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'CaSR_v3.1_P_FB_SFC': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'CaSR_v3.1_P_PR0_SFC': {'standard_name': 'precipitation_flux', 'units': 'm'}
        },
        'DayMet': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'dayl': {'standard_name': 'day_length', 'units': 's/day'},
            'prcp': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'srad': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'swe': {'standard_name': 'snow_water_equivalent', 'units': 'kg/m^2'},
            'tmax': {'standard_name': 'air_temperature_max', 'units': 'degC'},
            'tmin': {'standard_name': 'air_temperature_min', 'units': 'degC'},
            'vp': {'standard_name': 'water_vapor_pressure', 'units': 'Pa'}
        },
        'NEX-GDDP': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'tas': {'standard_name': 'air_temperature', 'units': 'K'},
            'tasmax': {'standard_name': 'air_temperature_max', 'units': 'K'},
            'tasmin': {'standard_name': 'air_temperature_min', 'units': 'K'},
            'hurs': {'standard_name': 'relative_humidity', 'units': '%'},
            'huss': {'standard_name': 'specific_humidity', 'units': '1'},
            'rlds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'sfcWind': {'standard_name': 'wind_speed', 'units': 'm/s'}
        },
        'GWF-I': {
            'PSFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'Q2': {'standard_name': 'specific_humidity', 'units': '1'},
            'T2': {'standard_name': 'air_temperature', 'units': 'K'},
            'U10': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'V10': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'PREC_ACC_NC': {'standard_name': 'precipitation_flux', 'units': 'mm/hr'},  
            'SWDOWN': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'GLW': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'GWF-II': {
            'PSFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'Q2': {'standard_name': 'specific_humidity', 'units': '1'},
            'T2': {'standard_name': 'air_temperature', 'units': 'K'},
            'U10': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'V10': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'PREC_ACC_NC': {'standard_name': 'precipitation_flux', 'units': 'mm/hr'},  
            'SWDOWN': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'GLW': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'CCRN-CanRCM4': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'ta': {'standard_name': 'air_temperature', 'units': 'K'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'hus': {'standard_name': 'specific_humidity', 'units': '1'},
            'wind': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'lsds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'CCRN-WFDEI': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'ta': {'standard_name': 'air_temperature', 'units': 'K'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'hus': {'standard_name': 'specific_humidity', 'units': '1'},
            'wind': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'lsds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'}
        },
        'Ouranos-ESPO': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'},
            'tasmax': {'standard_name': 'air_temperature_max', 'units': 'K'},
            'tasmin': {'standard_name': 'air_temperature_min', 'units': 'K'}
        },
        'Ouranos-MRCC5': {
            'tas': {'standard_name': 'air_temperature', 'units': 'K'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'huss': {'standard_name': 'specific_humidity', 'units': '1'},
            'uas': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'vas': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'rlds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'AGCD': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'},
            'tmax': {'standard_name': 'air_temperature_max', 'units': 'degC'},
            'tmin': {'standard_name': 'air_temperature_min', 'units': 'degC'}
        },
        'CERRA': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'relhum': {'standard_name': 'relative_humidity', 'units': '%'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd_u': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'windspd_v': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'kg m-2 s-1'}
        },
        'AORC': {
            'APCP_surface': {'standard_name': 'precipitation_flux', 'units': 'kg m-2 s-1'},
            'TMP_2maboveground': {'standard_name': 'air_temperature', 'units': 'K'},
            'SPFH_2maboveground': {'standard_name': 'specific_humidity', 'units': '1'},
            'PRES_surface': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'DLWRF_surface': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'DSWRF_surface': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'UGRD_10maboveground': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'VGRD_10maboveground': {'standard_name': 'northward_wind', 'units': 'm/s'}
        },
        'CONUS404': {
            'T2': {'standard_name': 'air_temperature', 'units': 'K'},
            'Q2': {'standard_name': 'specific_humidity', 'units': '1'},
            'PSFC': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'U10': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'V10': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'GLW': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWDOWN': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'RAINRATE': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'HRRR': {
            'TMP': {'standard_name': 'air_temperature', 'units': 'K'},
            'SPFH': {'standard_name': 'specific_humidity', 'units': '1'},
            'PRES': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'DLWRF': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'DSWRF': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'UGRD': {'standard_name': 'eastward_wind', 'units': 'm/s'},
            'VGRD': {'standard_name': 'northward_wind', 'units': 'm/s'},
            'APCP': {'standard_name': 'precipitation_flux', 'units': 'kg m-2 s-1'}
        },
        'NEX-GDDP-CMIP6': {
            'pr': {'standard_name': 'precipitation_flux', 'units': 'kg m-2 s-1'},
            'tas': {'standard_name': 'air_temperature', 'units': 'K'},
            'huss': {'standard_name': 'specific_humidity', 'units': '1'},
            'ps': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'rlds': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'rsds': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'sfcWind': {'standard_name': 'wind_speed', 'units': 'm/s'}
        }

    }

    # Model variable requirements
    MODEL_REQUIREMENTS = {
        'SUMMA': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'FUSE': {
            'temp': {'standard_name': 'air_temperature', 'units': 'degC'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'}
        },
        'GR': {
            'temp': {'standard_name': 'air_temperature', 'units': 'degC'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'}
        }, 
        'HYPE': {
            'temp': {'standard_name': 'air_temperature', 'units': 'degC'},
            'pr': {'standard_name': 'precipitation_flux', 'units': 'mm/day'}
        },
        'MESH': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'NGEN': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'LSTM': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'K'},
            'airpres': {'standard_name': 'surface_air_pressure', 'units': 'Pa'},
            'spechum': {'standard_name': 'specific_humidity', 'units': '1'},
            'windspd': {'standard_name': 'wind_speed', 'units': 'm/s'},
            'LWRadAtm': {'standard_name': 'surface_downwelling_longwave_flux', 'units': 'W/m^2'},
            'SWRadAtm': {'standard_name': 'surface_downwelling_shortwave_flux', 'units': 'W/m^2'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/s'}
        },
        'RHESSys': {
            'airtemp': {'standard_name': 'air_temperature', 'units': 'degC'},
            'pptrate': {'standard_name': 'precipitation_flux', 'units': 'mm/day'}
        }
    }

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, dataset: str, model: str):
        """
        Initialize VariableHandler with configuration settings.
        
        Args:
            config: SYMFLUENCE configuration dictionary
            logger: SYMFLUENCE logger instance
        """
        self.config = config
        self.logger = logger
        self.dataset = dataset if dataset is not None else config.get('FORCING_DATASET')
        self.model = model if model is not None else config.get('HYDROLOGICAL_MODEL')
        
        # Initialize pint for unit handling
        self.ureg = pint.UnitRegistry()
        pint_xarray.setup_registry(self.ureg)
        
        # Validate dataset and model are supported
        if self.dataset not in self.DATASET_MAPPINGS:
            self.logger.error(f"Unsupported dataset: {self.dataset}")
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.model not in self.MODEL_REQUIREMENTS:
            self.logger.error(f"Unsupported model: {self.model}")
            raise ValueError(f"Unsupported model: {self.model}")

    def get_dataset_variables(self, dataset: Optional[str] = None) -> str:
        """
        Get the forcing variable keys for a specified dataset as a comma-separated string.
        
        Args:
            dataset (Optional[str]): Name of the dataset. If None, uses the instance's dataset.
            
        Returns:
            str: Comma-separated string of variable keys for the specified dataset
            
        Raises:
            ValueError: If the specified dataset is not supported
        """
        # Use instance dataset if none provided
        dataset_name = dataset if dataset is not None else self.dataset
        
        # Check if dataset exists in mappings
        if dataset_name not in self.DATASET_MAPPINGS:
            self.logger.error(f"Unsupported dataset: {dataset_name}")
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        return ','.join(self.DATASET_MAPPINGS[dataset_name].keys())
    
    def process_forcing_data(self, data: xr.Dataset) -> xr.Dataset:
        """Process forcing data by mapping variable names and converting units."""
        self.logger.debug("Starting forcing data unit processing")
        
        processed_data = data.copy()
        
        # Get dataset and model mappings
        dataset_map = self.DATASET_MAPPINGS[self.dataset]
        model_map = self.MODEL_REQUIREMENTS[self.model]
        
        # Process each required model variable
        for model_var, model_req in model_map.items():
            # Find corresponding dataset variable
            dataset_var = self._find_matching_variable(model_req['standard_name'], dataset_map)
            
            if dataset_var is None:
                self.logger.error(f"Required variable {model_var} not found in dataset {self.dataset}")
                raise ValueError(f"Required variable {model_var} not found in dataset {self.dataset}")
            
            # Rename variable
            if dataset_var in processed_data:
                self.logger.debug(f"Processing {dataset_var} -> {model_var}")
                
                # Get units: prioritize metadata from the DataArray over hardcoded mapping
                data_units = str(processed_data[dataset_var].attrs.get('units', '')).lower()
                source_units = dataset_map[dataset_var]['units']
                
                # If metadata exists and looks different from our mapping, trust metadata
                # BUT perform a range check for temperature to handle inconsistent files
                if data_units and data_units != source_units.lower():
                    # Handle minor string variations (e.g. 'degc' vs 'degC')
                    if data_units in ['degc', 'celsius', 'c']:
                        actual_source_units = 'degC'
                    elif data_units in ['k', 'kelvin']:
                        actual_source_units = 'K'
                    else:
                        actual_source_units = data_units
                    
                    if actual_source_units != source_units:
                        # Perform range check for temperature
                        if model_req['standard_name'] == 'air_temperature':
                            temp_mean = float(processed_data[dataset_var].mean())
                            if temp_mean > 100 and actual_source_units == 'degC':
                                self.logger.warning(f"Metadata for {dataset_var} says 'degC' but mean value is {temp_mean:.2f}. Assuming 'K'.")
                                actual_source_units = 'K'
                            elif temp_mean < 100 and actual_source_units == 'K':
                                self.logger.warning(f"Metadata for {dataset_var} says 'K' but mean value is {temp_mean:.2f}. Assuming 'degC'.")
                                actual_source_units = 'degC'
                        
                        if actual_source_units != source_units:
                            self.logger.info(f"Using metadata units '{actual_source_units}' instead of mapping '{source_units}' for {dataset_var}")
                            source_units = actual_source_units

                target_units = model_req['units']
                
                # Convert units if needed
                if source_units != target_units:
                    self.logger.debug(f"Converting units for {dataset_var}: {source_units} -> {target_units}")
                    try:
                        processed_data[dataset_var] = self._convert_units(
                            processed_data[dataset_var], 
                            source_units, 
                            target_units
                        )
                    except Exception as e:
                        self.logger.error(f"Unit conversion failed for {dataset_var}: {str(e)}")
                        raise
                
                # Rename after conversion
                processed_data = processed_data.rename({dataset_var: model_var})
        
        self.logger.debug("Forcing data unit processing completed")
        return processed_data

    def _find_matching_variable(self, standard_name: str, dataset_map: Dict) -> Optional[str]:
        """Find dataset variable matching the required standard_name."""
        for var, attrs in dataset_map.items():
            if attrs['standard_name'] == standard_name:
                return var
        self.logger.warning(f"No matching variable found for standard_name: {standard_name}")
        return None

    def _normalize_unit_string(self, unit_str: str) -> str:
        """
        Normalize unit strings to formats that Pint handles reliably.
        Example: 'mm hour-1' -> 'mm / hour'
        Example: 'kg m-2 s-1' -> 'kg m^-2 s^-1'
        """
        if not unit_str:
            return unit_str
            
        import re
        norm = unit_str.strip()
        
        # Handle 'X-N' format (e.g. m-2, s-1) -> 'X^-N'
        # This prevents Pint from interpreting '-' as subtraction
        norm = re.sub(r'([a-zA-Z_]\w*)-(\d+)', r'\1^-\2', norm)
        
        # Handle 'X**-1' or 'X^-1' -> '/ X' (legacy support if needed, but ^-N covers it)
        # We can keep specific cleanups if they help
        
        # Standardize spaces around operators
        norm = norm.replace('/', ' / ')
        norm = norm.replace('*', ' * ')
        
        # Final cleanup of any potential double slashes or extra spaces
        norm = ' '.join(norm.split())
        norm = norm.replace('/ /', '/')
        
        if norm != unit_str:
            self.logger.debug(f"Normalized units: '{unit_str}' -> '{norm}'")
            
        return norm

    def _convert_units(self, data: xr.DataArray, from_units: str, to_units: str) -> xr.DataArray:
        """
        Convert variable units using pint-xarray.
        
        Args:
            data: DataArray to convert
            from_units: Source units
            to_units: Target units
            
        Returns:
            DataArray with converted units
        """
        # Normalize unit strings for pint
        orig_from = from_units
        from_units = self._normalize_unit_string(from_units)
        to_units = self._normalize_unit_string(to_units)
        
        try:
            # Special case for precipitation flux conversions (very common source of errors)
            # Check for various formats including normalized 'kg m^-2 s^-1'
            # Note: normalized strings have spaces around operators like ' / '
            is_mass_flux = ('kg/m2/s' in from_units or 
                           'kilogram / meter ** 2 / second' in from_units or 
                           'kg m^-2 s^-1' in from_units)
            
            is_mm_day = ('mm/day' in to_units or 'mm / day' in to_units)
            
            if is_mass_flux and is_mm_day:
                # 1 kg/m² = 1 mm of water
                # Convert kg/m²/s to mm/s, then to mm/day
                converted = data * 86400  # multiply by seconds per day
                return converted

            # Regular unit conversion
            try:
                data = data.pint.quantify(from_units)
                converted = data.pint.to(to_units)
                return converted.pint.dequantify()
            except Exception as pe:
                self.logger.warning(f"Pint conversion failed for {orig_from} -> {to_units}: {pe}. Trying manual fallback.")
                # Manual fallbacks for common meteorological variables
                f_low = from_units.lower()
                t_low = to_units.lower()
                
                # Temperature: Kelvin to Celsius
                if 'k' in f_low and 'c' in t_low and 'deg' in t_low:
                    return data - 273.15
                # Temperature: Celsius to Kelvin
                if 'c' in f_low and 'deg' in f_low and 'k' in t_low:
                    return data + 273.15
                # Precipitation: mm/h to mm/day
                if 'mm' in f_low and 'hour' in f_low and 'mm' in t_low and 'day' in t_low:
                    return data * 24.0
                # Precipitation: mm/s to mm/day
                if 'mm' in f_low and 's' in f_low and 'mm' in t_low and 'day' in t_low:
                    return data * 86400.0
                # Precipitation: kg m-2 s-1 to mm/day (catch-all for mass flux)
                if ('kg' in f_low and 'm' in f_low and 's' in f_low) and ('mm' in t_low and 'day' in t_low):
                     return data * 86400.0
                
                raise pe
        except Exception as e:
            self.logger.error(f"Unit conversion failed: {orig_from} -> {to_units}: {str(e)}")
            raise

    def save_mappings(self, filepath: Path):
        """Save current mappings to YAML file."""
        self.logger.info(f"Saving variable mappings to: {filepath}")
        mappings = {
            'dataset_mappings': self.DATASET_MAPPINGS,
            'model_requirements': self.MODEL_REQUIREMENTS
        }
        
        try:
            with open(filepath, 'w') as f:
                yaml.dump(mappings, f)
            self.logger.info("Variable mappings saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save mappings: {str(e)}")
            raise

    @classmethod
    def load_mappings(cls, filepath: Path, logger: logging.Logger):
        """Load mappings from YAML file."""
        logger.info(f"Loading variable mappings from: {filepath}")
        try:
            with open(filepath, 'r') as f:
                mappings = yaml.safe_load(f)
                
            cls.DATASET_MAPPINGS = mappings['dataset_mappings']
            cls.MODEL_REQUIREMENTS = mappings['model_requirements']
            logger.info("Variable mappings loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load mappings: {str(e)}")
            raise