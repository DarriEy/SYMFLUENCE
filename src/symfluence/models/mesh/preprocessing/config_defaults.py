"""
MESH Configuration Defaults

Default variable mappings, units, and parameter values for MESH model.
"""

from typing import Dict, Any


class MESHConfigDefaults:
    """
    Provides default configuration values for MESH preprocessing.

    Contains mappings for forcing variables, landcover classes,
    drainage database variables, and GRU parameters.
    """

    # meshflow expects: standard_name -> actual_file_variable_name
    FORCING_VARS: Dict[str, str] = {
        "air_pressure": "airpres",
        "specific_humidity": "spechum",
        "air_temperature": "airtemp",
        "wind_speed": "windspd",
        "precipitation": "pptrate",
        "shortwave_radiation": "SWRadAtm",
        "longwave_radiation": "LWRadAtm",
    }

    # Units from source data
    FORCING_UNITS: Dict[str, str] = {
        "air_pressure": 'Pa',
        "specific_humidity": 'kg/kg',
        "air_temperature": 'K',
        "wind_speed": 'm/s',
        "precipitation": 'm/s',
        "shortwave_radiation": 'W/m^2',
        "longwave_radiation": 'W/m^2',
    }

    # Target units for MESH
    FORCING_TO_UNITS: Dict[str, str] = {
        "air_pressure": 'Pa',
        "specific_humidity": 'kg/kg',
        "air_temperature": 'K',
        "wind_speed": 'm/s',
        "precipitation": 'mm/s',
        "shortwave_radiation": 'W/m^2',
        "longwave_radiation": 'W/m^2',
    }

    # NALCMS 2020 landcover classes (integer keys for meshflow compatibility)
    LANDCOVER_CLASSES: Dict[int, str] = {
        1: 'Temperate or sub-polar needleleaf forest',
        2: 'Sub-polar taiga needleleaf forest',
        3: 'Tropical or sub-tropical broadleaf evergreen forest',
        4: 'Tropical or sub-tropical broadleaf deciduous forest',
        5: 'Temperate or sub-polar broadleaf deciduous forest',
        6: 'Mixed forest',
        7: 'Tropical or sub-tropical shrubland',
        8: 'Temperate or sub-polar shrubland',
        9: 'Tropical or sub-tropical grassland',
        10: 'Temperate or sub-polar grassland',
        11: 'Sub-polar or polar shrubland-lichen-moss',
        12: 'Sub-polar or polar grassland-lichen-moss',
        13: 'Sub-polar or polar barren-lichen-moss',
        14: 'Wetland',
        15: 'Cropland',
        16: 'Barren lands',
        17: 'Urban',
        18: 'Water',
        19: 'Snow and Ice',
    }

    # ddb_vars maps standard names -> input shapefile column names
    DDB_VARS: Dict[str, str] = {
        'river_slope': 'Slope',
        'river_length': 'Length',
        'river_class': 'strmOrder',
    }

    DDB_UNITS: Dict[str, str] = {
        'river_slope': 'm/m',
        'river_length': 'm',
        'rank': 'dimensionless',
        'next': 'dimensionless',
        'gru': 'dimensionless',
        'subbasin_area': 'm^2',
    }

    DDB_MIN_VALUES: Dict[str, float] = {
        'river_slope': 1e-6,
        'river_length': 1e-3,
        'subbasin_area': 1e-3,
    }

    # Full NALCMS to CLASS parameter type mapping
    FULL_GRU_MAPPING: Dict[int, str] = {
        0: 'needleleaf',  # Unknown -> default to needleleaf
        1: 'needleleaf',  # Temperate or sub-polar needleleaf forest
        2: 'needleleaf',  # Sub-polar taiga needleleaf forest
        3: 'broadleaf',   # Tropical or sub-tropical broadleaf evergreen forest
        4: 'broadleaf',   # Tropical or sub-tropical broadleaf deciduous forest
        5: 'broadleaf',   # Temperate or sub-polar broadleaf deciduous forest
        6: 'broadleaf',   # Mixed forest
        7: 'grass',       # Tropical or sub-tropical shrubland
        8: 'grass',       # Temperate or sub-polar shrubland
        9: 'grass',       # Tropical or sub-tropical grassland
        10: 'grass',      # Temperate or sub-polar grassland
        11: 'grass',      # Sub-polar or polar shrubland-lichen-moss
        12: 'grass',      # Sub-polar or polar grassland-lichen-moss
        13: 'barrenland', # Sub-polar or polar barren-lichen-moss
        14: 'water',      # Wetland
        15: 'crops',      # Cropland
        16: 'barrenland', # Barren lands
        17: 'urban',      # Urban
        18: 'water',      # Water
        19: 'water',      # Snow and Ice
    }

    # MESH variable name mappings
    MESH_VAR_NAMES: Dict[str, str] = {
        'air_pressure': 'PRES',
        'specific_humidity': 'QA',
        'air_temperature': 'TA',
        'wind_speed': 'UV',
        'precipitation': 'PRE',
        'shortwave_radiation': 'FSIN',
        'longwave_radiation': 'FLIN',
    }

    # MESH 1.5 variable mapping to file names
    VAR_TO_FILE: Dict[str, str] = {
        'FSIN': 'basin_shortwave.nc',
        'FLIN': 'basin_longwave.nc',
        'PRES': 'basin_pres.nc',
        'TA': 'basin_temperature.nc',
        'QA': 'basin_humidity.nc',
        'UV': 'basin_wind.nc',
        'PRE': 'basin_rain.nc',
    }

    @classmethod
    def get_var_long_name(cls, var: str) -> str:
        """Get long name for MESH variable."""
        names = {
            'FSIN': 'downward shortwave radiation',
            'FLIN': 'downward longwave radiation',
            'PRES': 'air pressure',
            'TA': 'air temperature',
            'QA': 'specific humidity',
            'UV': 'wind speed',
            'PRE': 'precipitation rate',
        }
        return names.get(var, var)

    @classmethod
    def get_var_units(cls, var: str) -> str:
        """Get units for MESH variable."""
        units = {
            'FSIN': 'W m-2',
            'FLIN': 'W m-2',
            'PRES': 'Pa',
            'TA': 'K',
            'QA': 'kg kg-1',
            'UV': 'm s-1',
            'PRE': 'kg m-2 s-1',
        }
        return units.get(var, '1')

    @classmethod
    def get_gru_mapping_for_classes(cls, detected_classes: list) -> Dict[int, str]:
        """
        Get GRU mapping filtered to only include detected classes.

        Args:
            detected_classes: List of GRU class numbers present in data

        Returns:
            Filtered GRU mapping dictionary
        """
        if detected_classes:
            return {k: cls.FULL_GRU_MAPPING.get(k, 'needleleaf')
                    for k in detected_classes}
        return cls.FULL_GRU_MAPPING

    @classmethod
    def get_default_settings(
        cls,
        forcing_start_date: str,
        sim_start_date: str,
        sim_end_date: str,
        gru_mapping: Dict[int, str]
    ) -> Dict[str, Any]:
        """
        Build default meshflow settings dictionary.

        Args:
            forcing_start_date: Start date for forcing data
            sim_start_date: Simulation start date
            sim_end_date: Simulation end date
            gru_mapping: GRU class to type mapping

        Returns:
            Settings dictionary for meshflow
        """
        return {
            'core': {
                'forcing_files': 'single',
                'forcing_start_date': forcing_start_date,
                'simulation_start_date': sim_start_date,
                'simulation_end_date': sim_end_date,
                'forcing_time_zone': 'UTC',
                'output_path': 'results',
            },
            'class_params': {
                'measurement_heights': {
                    'wind_speed': 10.0,
                    'specific_humidity': 2.0,
                    'air_temperature': 2.0,
                    'roughness_length': 50.0,
                },
                'copyright': {
                    'author': 'University of Calgary',
                    'location': 'SYMFLUENCE',
                },
                'grus': gru_mapping,
            },
            'hydrology_params': {
                'routing': [
                    {
                        'r2n': 0.4,
                        'r1n': 0.02,
                        'pwr': 2.37,
                        'flz': 0.001,
                    },
                ],
                'hydrology': {},
            },
            'run_options': {
                'flags': {
                    'etc': {
                        'RUNMODE': 'noroute',
                    },
                },
            },
        }
