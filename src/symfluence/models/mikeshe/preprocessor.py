"""
MIKE-SHE Model Preprocessor

Handles preparation of MIKE-SHE model inputs including:
- ERA5 forcing data conversion to MIKE-SHE compatible format (CSV)
- Setup file generation (.she XML)
- Directory structure creation
"""
import logging
from typing import Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base.base_preprocessor import BaseModelPreProcessor
from symfluence.models.registry import ModelRegistry
from symfluence.models.mixins import ObservationLoaderMixin

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("MIKESHE")
class MIKESHEPreProcessor(BaseModelPreProcessor, ObservationLoaderMixin):  # type: ignore[misc]
    """
    Prepares inputs for a MIKE-SHE model run.

    MIKE-SHE requires:
    - .she setup file (XML) defining model configuration
    - Forcing data in CSV or dfs0-compatible text format
    - Spatial data (topography, soil, land use)
    """

    def __init__(self, config, logger):
        """
        Initialize the MIKE-SHE preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        # Setup MIKE-SHE-specific directories
        self.mikeshe_input_dir = self.project_dir / "MIKESHE_input"
        self.settings_dir = self.mikeshe_input_dir / "settings"
        self.forcing_dir = self.mikeshe_input_dir / "forcing"
        self.params_dir = self.mikeshe_input_dir / "parameters"

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "MIKESHE"

    def run_preprocessing(self) -> bool:
        """
        Run the complete MIKE-SHE preprocessing workflow.

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            logger.info("Starting MIKE-SHE preprocessing...")

            # Create directory structure
            self._create_directory_structure()

            # Generate forcing files from ERA5
            self._generate_forcing_files()

            # Generate .she setup file
            self._generate_setup_file()

            logger.info("MIKE-SHE preprocessing complete.")
            return True

        except Exception as e:
            logger.error(f"MIKE-SHE preprocessing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _create_directory_structure(self) -> None:
        """Create MIKE-SHE input directory structure."""
        dirs = [
            self.mikeshe_input_dir,
            self.settings_dir,
            self.forcing_dir,
            self.params_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created MIKE-SHE input directories at {self.mikeshe_input_dir}")

    def _get_simulation_dates(self) -> Tuple[datetime, datetime]:
        """Get simulation start and end dates from configuration."""
        start_str = self._get_config_value(lambda: self.config.domain.time_start)
        end_str = self._get_config_value(lambda: self.config.domain.time_end)

        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        return start_date.to_pydatetime(), end_date.to_pydatetime()

    def _get_catchment_properties(self) -> Dict:
        """
        Get catchment properties from shapefile.

        Returns:
            Dict with centroid lat/lon, area, and elevation
        """
        try:
            import geopandas as gpd

            catchment_path = self.get_catchment_path()
            if catchment_path.exists():
                gdf = gpd.read_file(catchment_path)

                # Get centroid
                centroid = gdf.geometry.centroid.iloc[0]
                lon, lat = centroid.x, centroid.y

                # Project to UTM for accurate area
                utm_zone = int((lon + 180) / 6) + 1
                hemisphere = 'north' if lat >= 0 else 'south'
                utm_crs = (
                    f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
                )
                gdf_proj = gdf.to_crs(utm_crs)
                area_m2 = gdf_proj.geometry.area.sum()

                # Get elevation if available
                elev = (
                    float(gdf.get('elev_mean', [1000])[0])
                    if 'elev_mean' in gdf.columns
                    else 1000.0
                )

                return {
                    'lat': lat,
                    'lon': lon,
                    'area_m2': area_m2,
                    'elev': elev
                }
        except Exception as e:
            logger.warning(f"Could not read catchment properties: {e}")

        # Defaults
        return {
            'lat': 51.0,
            'lon': -115.0,
            'area_m2': 1e8,
            'elev': 1000.0
        }

    def _generate_forcing_files(self) -> None:
        """
        Generate MIKE-SHE forcing files from ERA5 data.

        Converts ERA5 NetCDF forcing data to CSV format compatible with
        MIKE-SHE. Output CSV contains columns:
        - datetime: ISO format timestamp
        - precipitation_mm: Precipitation [mm/timestep]
        - temperature_C: Air temperature [deg C]
        - pet_mm: Potential evapotranspiration [mm/timestep]
        """
        logger.info("Generating MIKE-SHE forcing files...")

        start_date, end_date = self._get_simulation_dates()

        # Try to load forcing data
        try:
            forcing_ds = self._load_forcing_data()
            self._write_forcing_csv(forcing_ds, start_date, end_date)
        except Exception as e:
            logger.warning(f"Could not load forcing data: {e}, using synthetic")
            self._generate_synthetic_forcing(start_date, end_date)

    def _load_forcing_data(self) -> xr.Dataset:
        """Load basin-averaged forcing data."""
        forcing_files = list(self.forcing_basin_path.glob("*.nc"))

        if not forcing_files:
            merged_path = self.project_dir / 'forcing' / 'merged_path'
            if merged_path.exists():
                forcing_files = list(merged_path.glob("*.nc"))

        if not forcing_files:
            raise FileNotFoundError(
                f"No forcing data found in {self.forcing_basin_path}"
            )

        logger.info(f"Loading forcing from {len(forcing_files)} files")

        try:
            ds = xr.open_mfdataset(forcing_files, combine='by_coords')
        except ValueError:
            try:
                ds = xr.open_mfdataset(
                    forcing_files, combine='nested', concat_dim='time'
                )
            except Exception:
                datasets = [xr.open_dataset(f) for f in forcing_files]
                ds = xr.merge(datasets)

        # Subset to simulation period
        ds = self.subset_to_simulation_time(ds, "Forcing")
        return ds

    def _write_forcing_csv(
        self,
        forcing_ds: xr.Dataset,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Write forcing data in MIKE-SHE compatible CSV format."""
        # Variable name candidates for each forcing type
        precip_vars = ['pptrate', 'precipitation', 'pr', 'precip', 'tp', 'PREC']
        temp_vars = ['airtemp', 'temperature', 'tas', 'temp', 't2m', 'AIR_TEMP']
        pet_vars = ['pet', 'PET', 'evspsbl', 'potential_evapotranspiration']

        times = pd.to_datetime(forcing_ds['time'].values)

        # Determine timestep for rate conversions
        if len(times) > 1:
            dt_seconds = float(
                (pd.Timestamp(times[1]) - pd.Timestamp(times[0])).total_seconds()
            )
        else:
            dt_seconds = 86400.0  # default daily

        # Extract precipitation
        precip = None
        for var in precip_vars:
            if var in forcing_ds:
                data = forcing_ds[var].values
                src_units = forcing_ds[var].attrs.get('units', '')

                # Flatten spatial dims if present
                while data.ndim > 1:
                    data = np.nanmean(data, axis=-1)

                # Unit conversion to mm/timestep
                if 'mm/s' in src_units or var == 'pptrate':
                    data = data * dt_seconds
                elif src_units == 'm' or var == 'tp':
                    data = data * 1000.0
                elif 'kg' in src_units and 'm-2' in src_units and 's-1' in src_units:
                    data = data * dt_seconds

                precip = data[:len(times)]
                logger.info(f"Loaded precipitation from '{var}'")
                break

        if precip is None:
            logger.warning("No precipitation found, using zeros")
            precip = np.zeros(len(times))

        # Extract temperature
        temperature = None
        for var in temp_vars:
            if var in forcing_ds:
                data = forcing_ds[var].values
                src_units = forcing_ds[var].attrs.get('units', '')

                while data.ndim > 1:
                    data = np.nanmean(data, axis=-1)

                # Convert K to C if needed
                if src_units == 'K' or np.nanmean(data) > 100:
                    data = data - 273.15

                temperature = data[:len(times)]
                logger.info(f"Loaded temperature from '{var}'")
                break

        if temperature is None:
            logger.warning("No temperature found, using synthetic seasonal cycle")
            day_frac = np.arange(len(times)) / (24.0 if dt_seconds < 86400 else 1.0)
            temperature = 5.0 + 10.0 * np.sin(2 * np.pi * day_frac / 365)

        # Extract or estimate PET
        pet = None
        for var in pet_vars:
            if var in forcing_ds:
                data = forcing_ds[var].values
                while data.ndim > 1:
                    data = np.nanmean(data, axis=-1)
                pet = data[:len(times)]
                logger.info(f"Loaded PET from '{var}'")
                break

        if pet is None:
            # Estimate PET using Hamon method (simple temperature-based)
            logger.warning("No PET found, estimating via Hamon method")
            temp_c = np.maximum(temperature, 0.0)
            # Saturated vapor density (g/m3)
            svd = 4.95 * np.exp(0.062 * temp_c) / 100.0
            # Approximate daylight hours (assume 12h average)
            pet = 0.55 * 12.0 * svd  # mm/day
            if dt_seconds < 86400:
                pet = pet * dt_seconds / 86400.0

        # Write CSV
        df = pd.DataFrame({
            'datetime': times,
            'precipitation_mm': precip,
            'temperature_C': temperature,
            'pet_mm': pet,
        })

        forcing_csv = self.forcing_dir / f'{self.domain_name}_forcing.csv'
        df.to_csv(forcing_csv, index=False, float_format='%.6f')
        logger.info(f"Forcing CSV written: {forcing_csv} ({len(df)} records)")

    def _generate_synthetic_forcing(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """Generate synthetic forcing data for testing."""
        self._get_catchment_properties()
        dates = pd.date_range(start_date, end_date, freq='D')
        n = len(dates)

        day_frac = np.arange(n)
        precip = np.random.exponential(2.0, n)
        temperature = 5.0 + 10.0 * np.sin(2 * np.pi * day_frac / 365)
        temp_c = np.maximum(temperature, 0.0)
        svd = 4.95 * np.exp(0.062 * temp_c) / 100.0
        pet = 0.55 * 12.0 * svd

        df = pd.DataFrame({
            'datetime': dates,
            'precipitation_mm': precip,
            'temperature_C': temperature,
            'pet_mm': pet,
        })

        forcing_csv = self.forcing_dir / f'{self.domain_name}_forcing.csv'
        df.to_csv(forcing_csv, index=False, float_format='%.6f')
        logger.info(f"Synthetic forcing CSV written: {forcing_csv}")

    def _generate_setup_file(self) -> None:
        """
        Generate the MIKE-SHE .she setup file (XML format).

        Creates a minimal but valid .she XML structure with references
        to forcing data, default parameters, and simulation settings.
        """
        import xml.etree.ElementTree as ET  # nosec B405

        logger.info("Generating MIKE-SHE .she setup file...")

        setup_file_name = self._get_config_value(
            lambda: self.config.model.mikeshe.setup_file,
            default='model.she'
        )
        setup_path = self.settings_dir / setup_file_name

        start_date, end_date = self._get_simulation_dates()
        props = self._get_catchment_properties()

        from .parameters import DEFAULT_PARAMS

        # Build XML structure
        root = ET.Element('MikeSheSetup')
        root.set('xmlns', 'http://www.dhigroup.com/mikeshe')

        # Simulation period
        sim = ET.SubElement(root, 'SimulationPeriod')
        ET.SubElement(sim, 'StartDate').text = start_date.strftime('%Y-%m-%d %H:%M:%S')
        ET.SubElement(sim, 'EndDate').text = end_date.strftime('%Y-%m-%d %H:%M:%S')

        # Domain
        domain = ET.SubElement(root, 'Domain')
        ET.SubElement(domain, 'Latitude').text = str(props['lat'])
        ET.SubElement(domain, 'Longitude').text = str(props['lon'])
        ET.SubElement(domain, 'Area_m2').text = str(props['area_m2'])
        ET.SubElement(domain, 'Elevation').text = str(props['elev'])

        # Forcing reference
        forcing = ET.SubElement(root, 'Forcing')
        forcing_csv = self.forcing_dir / f'{self.domain_name}_forcing.csv'
        ET.SubElement(forcing, 'ForcingFile').text = str(forcing_csv)

        # Overland Flow parameters
        of = ET.SubElement(root, 'OverlandFlow')
        ET.SubElement(of, 'ManningM').text = str(DEFAULT_PARAMS['manning_m'])
        ET.SubElement(of, 'DetentionStorage').text = str(
            DEFAULT_PARAMS['detention_storage']
        )

        # Unsaturated Zone parameters
        uz = ET.SubElement(root, 'UnsaturatedFlow')
        ET.SubElement(uz, 'HydraulicConductivity').text = str(
            DEFAULT_PARAMS['Ks_uz']
        )
        ET.SubElement(uz, 'SaturatedMoistureContent').text = str(
            DEFAULT_PARAMS['theta_sat']
        )
        ET.SubElement(uz, 'FieldCapacity').text = str(DEFAULT_PARAMS['theta_fc'])
        ET.SubElement(uz, 'WiltingPoint').text = str(DEFAULT_PARAMS['theta_wp'])

        # Saturated Zone parameters
        sz = ET.SubElement(root, 'SaturatedFlow')
        ET.SubElement(sz, 'HorizontalConductivity').text = str(
            DEFAULT_PARAMS['Ks_sz_h']
        )
        ET.SubElement(sz, 'SpecificYield').text = str(
            DEFAULT_PARAMS['specific_yield']
        )

        # Snow parameters
        snow = ET.SubElement(root, 'SnowMelt')
        ET.SubElement(snow, 'DegreeDayFactor').text = str(DEFAULT_PARAMS['ddf'])
        ET.SubElement(snow, 'ThresholdTemperature').text = str(
            DEFAULT_PARAMS['snow_threshold']
        )

        # Vegetation parameters
        veg = ET.SubElement(root, 'Vegetation')
        ET.SubElement(veg, 'MaxCanopyStorage').text = str(
            DEFAULT_PARAMS['max_canopy_storage']
        )

        # Output settings
        output = ET.SubElement(root, 'Output')
        ET.SubElement(output, 'OutputDirectory').text = str(self.project_dir / 'simulations')
        ET.SubElement(output, 'OutputFormat').text = 'csv'

        # Write XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(setup_path, encoding='utf-8', xml_declaration=True)

        logger.info(f"MIKE-SHE setup file written: {setup_path}")

    def preprocess(self, **kwargs):
        """Alternative entry point for preprocessing."""
        return self.run_preprocessing()
