"""Acquisition Service

Unified facade for all data acquisition workflows in SYMFLUENCE. Coordinates
downloading and processing of geospatial attributes, forcing data, and
observations from diverse sources (cloud, HPC, local). Acts as high-level
orchestrator delegating to specialized acquisition handlers and cloud
downloaders.

Architecture:
    AcquisitionService provides two parallel acquisition paths:

    1. CLOUD Mode (CloudForcingDownloader):
       - Cloud-based data providers with direct HTTP/S3 access
       - DEM sources: Copernicus GLO-30, FABDEM, NASADEM local tiles
       - Soil class: SoilGrids via WCS subsetting
       - Land cover: MODIS Landcover (multi-year mode), USGS NLCD
       - Forcing: ERA5 (CDS), CARRA/CERRA (CDS), AORC (AWS/GCS), NEX-GDDP (Zenodo)
       - Observations: USGS, WSC, SMHI, SNOTEL, GRACE, MODIS snow/ET

    2. MAF Mode (gistoolRunner, datatoolRunner):
       - HPC-based data access via external MAF tools on supercomputers
       - gistool: MERIT-Hydro elevation, MODIS landcover, SoilGrids soil class
       - datatool: ERA5, RDRS, CASR forcing data with Slurm job monitoring
       - Configuration: Generates MAF JSON configs and executes MAF scheduler
       - Output: Same directory structure as CLOUD mode

Data Acquisition Workflows:
    1. Attribute Acquisition (acquire_attributes)
       - DEM/elevation: Multiple sources with fallback logic
       - Soil classification: SoilGrids primary, gistool fallback
       - Land cover: MODIS or USGS depending on availability
       - Output: GeoTIFF rasters at project_dir/attributes/{type}/

    2. Forcing Data Download (acquire_forcings)
       - Datasets: ERA5, CARRA, CERRA, AORC, NEX-GDDP
       - Mode selection: CLOUD vs MAF based on config.domain.data_access
       - Caching: RawForcingCache with automatic TTL/checksum validation
       - Unit conversion: Via VariableHandler for dataset-specific mappings
       - Output: NetCDF at project_dir/forcing/{dataset}_raw/

    3. Observation Data Retrieval (acquire_observations)
       - Streamflow: USGS (NWIS), WSC (Canada), SMHI (Nordic)
       - Gridded: GRACE, MODIS Snow, MODIS ET, FLUXNET
       - Point sensors: SNOTEL (NOAA snow/precip/temp)
       - Output: CSV at project_dir/observations/{type}/processed/

    4. EM-Earth Supplementary Data (acquire_em_earth_forcings)
       - Gridded ERA5 re-analysis supplementing point/coarse data
       - Subsetting: Via bounding box
       - Averaging: Spatial mean over domain
       - Output: NetCDF at project_dir/forcing/em_earth_supplementary/

Configuration Parameters:
    Data Source Selection:
        domain.data_access: 'CLOUD' or 'MAF' (default: 'MAF')
        domain.dem_source: 'merit_hydro', 'copernicus', 'fabdem', 'nasadem'
        domain.land_class_source: 'modis', 'usgs_nlcd' (cloud only)
        domain.bounding_box_coords: 'lat_min/lon_min/lat_max/lon_max'

    Download Flags:
        domain.download_dem: Enable DEM acquisition (default: True)
        domain.download_soil: Enable soil class acquisition (default: True)
        domain.download_landcover: Enable land cover acquisition (default: True)

    Observation Sources:
        optimization.observation_variables: List of variables to download
        evaluation.targets: Evaluation targets (e.g., 'streamflow')

    MAF Configuration:
        domain.hpc_account: HPC account for job submission
        domain.hpc_cache_dir: HPC cache directory
        domain.hpc_job_timeout: Max seconds to wait for jobs

Caching and Error Handling:
    Raw Forcing Cache:
    - RawForcingCache manages downloaded forcing files
    - TTL: Files cached for configurable duration (default: 30 days)
    - Validation: Checksum-based integrity checking
    - Fallback: Automatic re-download if cache corrupted

    Error Recovery:
    - Network failures: Retry with exponential backoff
    - Partial downloads: Cleanup and retry
    - Missing data: Warn and continue with available sources
    - Configuration errors: Validate early and report clearly

Examples:
    >>> # Create service and run all acquisitions
    >>> from symfluence.data.acquisition.acquisition_service import AcquisitionService
    >>> acq = AcquisitionService(config, logger, reporting_manager=reporter)
    >>> acq.acquire_attributes()
    >>> acq.acquire_forcings()
    >>> acq.acquire_observations()
    >>> acq.acquire_em_earth_forcings()

    >>> # Cloud-only mode (faster for small domains)
    >>> # Set config.domain.data_access = 'CLOUD'
    >>> acq.acquire_attributes()

    >>> # MAF mode (for large domains on HPC)
    >>> # Set config.domain.data_access = 'MAF'
    >>> acq.acquire_attributes()

References:
    - MERIT-Hydro: Yamazaki et al. (2019) Global Hydrology, Earth System Science
    - Copernicus DEM: https://copernicus-dem-30m.s3.amazonaws.com/
    - FABDEM: Hawker et al. (2022) Scientific Data
    - SoilGrids: Poggio et al. (2021) Scientific Data
    - MODIS: Justice et al. (2002) Remote Sensing Reviews
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING
from datetime import datetime
import xarray as xr
import pandas as pd
from symfluence.data.acquisition.cloud_downloader import CloudForcingDownloader, check_cloud_access_availability
from symfluence.data.acquisition.maf_pipeline import gistoolRunner, datatoolRunner
from symfluence.data.utils.variable_utils import VariableHandler
from symfluence.geospatial.raster_utils import calculate_landcover_mode
from symfluence.data.cache import RawForcingCache
from symfluence.core.mixins import ConfigurableMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class AcquisitionService(ConfigurableMixin):
    """Unified data acquisition service for all SYMFLUENCE data needs.

    High-level facade orchestrating geospatial attributes, forcing data, and
    observation data acquisition from multiple sources (cloud, HPC, local).
    Provides flexible acquisition modes (CLOUD vs MAF) and handles caching,
    error recovery, and visualization.

    Acquisition Modes:
        CLOUD Mode:
        - Direct HTTP/S3 access to cloud providers
        - Faster for small domains, requires internet access
        - DEM sources: Copernicus GLO-30, FABDEM, NASADEM local
        - Forcing: ERA5 (CDS), CARRA/CERRA, AORC, NEX-GDDP
        - Suitable for research, testing, small basins

        MAF Mode:
        - HPC-based via external MAF tools (gistool, datatool)
        - Better for large domains, requires HPC access
        - Same output format as CLOUD mode
        - Handles job queuing and monitoring via Slurm
        - Suitable for operational, large-scale applications

    Data Acquisition Methods:
        acquire_attributes(): Geospatial attributes (DEM, soil, landcover)
        acquire_forcings(): Meteorological forcing data (ERA5, CARRA, etc.)
        acquire_observations(): Validation data (streamflow, GRACE, SNOTEL, etc.)
        acquire_em_earth_forcings(): Supplementary forcing from EM-Earth

    Key Features:
        - Multi-source geospatial data with automatic fallbacks
        - Caching with TTL and checksum-based validation
        - Parallel downloading where supported
        - Progress visualization via reporting_manager
        - Comprehensive error handling and logging
        - Configuration-driven mode selection

    Attributes:
        config: Typed SymfluenceConfig instance
        logger: Logger for acquisition progress tracking
        data_dir: Root data directory (from config.system.data_dir)
        domain_name: Domain identifier (from config.domain.name)
        project_dir: Project-specific directory (data_dir/domain_{domain_name})
        reporting_manager: Optional visualization manager
        variable_handler: VariableHandler for dataset-specific unit conversion

    Configuration:
        domain.data_access: 'CLOUD' or 'MAF' (default: 'MAF')
        domain.dem_source: DEM provider ('merit_hydro', 'copernicus', 'fabdem', 'nasadem')
        domain.land_class_source: Land cover provider ('modis', 'usgs_nlcd')
        domain.download_dem: Enable DEM acquisition (default: True)
        domain.download_soil: Enable soil class (default: True)
        domain.download_landcover: Enable land cover (default: True)

    Examples:
        >>> # Create service with config and logger
        >>> acq = AcquisitionService(config, logger, reporting_manager=reporter)

        >>> # Run complete acquisition workflow
        >>> acq.acquire_attributes()   # DEM, soil, landcover
        >>> acq.acquire_forcings()     # ERA5, CARRA, etc.
        >>> acq.acquire_observations() # Streamflow, GRACE, etc.
        >>> acq.acquire_em_earth_forcings()  # Supplementary data

        >>> # Cloud-only mode (small domain)
        >>> config.domain.data_access = 'CLOUD'
        >>> acq.acquire_attributes()

        >>> # MAF mode (large domain on HPC)
        >>> config.domain.data_access = 'MAF'
        >>> acq.acquire_forcings()

    See Also:
        CloudForcingDownloader: Cloud-based data source handlers
        gistoolRunner: HPC geospatial data extraction
        datatoolRunner: HPC forcing data extraction
        RawForcingCache: Forcing data caching system
    """

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        reporting_manager: Any = None
    ):
        # Set up typed config via ConfigurableMixin
        from symfluence.core.config.models import SymfluenceConfig
        if isinstance(config, dict):
            self._config = SymfluenceConfig(**config)
        else:
            self._config = config
        # Backward compatibility alias
        self.config = self._config

        self.logger = logger
        self.reporting_manager = reporting_manager
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir))
        self.domain_name = self._get_config_value(lambda: self.config.domain.name)
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.variable_handler = VariableHandler(self.config, self.logger, 'ERA5', 'SUMMA')

    def acquire_attributes(self):
        """Acquire geospatial attributes including DEM, soil, and land cover data."""
        self.logger.info("Starting attribute acquisition")

        data_access = self._get_config_value(lambda: self.config.domain.data_access, default='MAF').upper()
        dem_source = self._get_config_value(lambda: self.config.domain.dem_source, default='merit_hydro').lower()

        dem_dir = self.project_dir / 'attributes' / 'elevation' / 'dem'
        soilclass_dir = self.project_dir / 'attributes' / 'soilclass'
        landclass_dir = self.project_dir / 'attributes' / 'landclass'

        for dir_path in [dem_dir, soilclass_dir, landclass_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        if data_access == 'CLOUD':
            self.logger.info(f"Cloud data access enabled for attributes (DEM_SOURCE: {dem_source})")

            try:
                downloader = CloudForcingDownloader(self.config, self.logger)

                if self._get_config_value(lambda: self.config.domain.download_dem, default=True):
                    if dem_source == 'copernicus':
                        self.logger.info("Acquiring Copernicus DEM GLO-30 (30m) from AWS")
                        elev_file = downloader.download_copernicus_dem()
                        self.logger.info(f"✓ Copernicus DEM acquired: {elev_file}")

                    elif dem_source == 'fabdem':
                        self.logger.info("Acquiring FABDEM (30m, vegetation/building removed)")
                        elev_file = downloader.download_fabdem()
                        self.logger.info(f"✓ FABDEM acquired: {elev_file}")

                    elif dem_source == 'nasadem':
                        if self.config.get('NASADEM_LOCAL_DIR'):
                            self.logger.info("Acquiring NASADEM (30m) from local tiles")
                            elev_file = downloader.download_nasadem_local()
                            self.logger.info(f"✓ NASADEM acquired: {elev_file}")
                        else:
                            raise ValueError("DEM_SOURCE set to 'nasadem' but NASADEM_LOCAL_DIR not configured.")

                    elif dem_source == 'merit_hydro':
                        self.logger.info("DEM_SOURCE is merit_hydro - using MAF/gistool for elevation")
                        gr = gistoolRunner(self.config, self.logger)
                        bbox = self._get_config_value(lambda: self.config.domain.bounding_box_coords).split('/')
                        latlims = f"{bbox[0]},{bbox[2]}"
                        lonlims = f"{bbox[1]},{bbox[3]}"
                        self._acquire_elevation_data(gr, dem_dir, latlims, lonlims)
                        self.logger.info("✓ MERIT-Hydro elevation acquired via MAF")
                        elev_file = dem_dir / f"domain_{self.domain_name}_elv.tif" # Assuming standard name

                    else:
                        raise ValueError(f"Unsupported DEM_SOURCE: '{dem_source}'.")

                    if self.reporting_manager and 'elev_file' in locals() and elev_file and elev_file.exists():
                        self.reporting_manager.visualize_spatial_coverage(elev_file, 'elevation', 'acquisition')

                else:
                    self.logger.info("Skipping DEM acquisition (DOWNLOAD_DEM is False)")

                if self._get_config_value(lambda: self.config.domain.download_soil, default=True):
                    self.logger.info("Acquiring soil class data from SoilGrids")
                    soil_file = downloader.download_global_soilclasses()
                    self.logger.info(f"✓ SoilGrids data acquired: {soil_file}")

                    if self.reporting_manager and soil_file and soil_file.exists():
                        self.reporting_manager.visualize_spatial_coverage(soil_file, 'soil_class', 'acquisition')
                else:
                    self.logger.info("Skipping soil class acquisition (DOWNLOAD_SOIL is False)")

                if self._get_config_value(lambda: self.config.domain.download_landcover, default=True):
                    land_source = self._get_config_value(lambda: self.config.domain.land_class_source, default='modis').lower()
                    self.logger.info(f"Acquiring land cover data (cloud mode, source: {land_source})")

                    try:
                        if land_source == 'modis':
                            lc_file = downloader.download_modis_landcover()
                        elif land_source == 'usgs_nlcd':
                            lc_file = downloader.download_usgs_landcover()
                        else:
                            raise ValueError(f"Unsupported LAND_CLASS_SOURCE: '{land_source}'. Supported: 'modis', 'usgs_nlcd'.")

                        self.logger.info(f"✓ Land cover data acquired: {lc_file}")

                        if self.reporting_manager and lc_file and lc_file.exists():
                            self.reporting_manager.visualize_spatial_coverage(lc_file, 'land_class', 'acquisition')
                    except Exception as e_lc:
                        self.logger.error(f"Land cover acquisition failed: {e_lc}")
                        raise
                else:
                    self.logger.info("Skipping land cover acquisition (DOWNLOAD_LAND_COVER is False)")

                # Glacier data acquisition (optional)
                if self._get_config_value(lambda: self.config.data.download_glacier_data, default=False):
                    self.logger.info("Acquiring glacier data from RGI")
                    try:
                        glacier_file = downloader.download_glacier_data()
                        self.logger.info(f"✓ Glacier data acquired: {glacier_file}")
                    except Exception as e_glacier:
                        self.logger.warning(f"Glacier data acquisition failed: {e_glacier}")
                        # Don't raise - glacier data is optional

            except Exception as e:
                self.logger.error(f"Error during cloud attribute acquisition: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        else:
            self.logger.info("Using traditional MAF attribute acquisition workflow")
            gr = gistoolRunner(self.config, self.logger)
            bbox = self._get_config_value(lambda: self.config.domain.bounding_box_coords).split('/')
            latlims = f"{bbox[0]},{bbox[2]}"
            lonlims = f"{bbox[1]},{bbox[3]}"

            try:
                self._acquire_elevation_data(gr, dem_dir, latlims, lonlims)
                self._acquire_landcover_data(gr, landclass_dir, latlims, lonlims)
                self._acquire_soilclass_data(gr, soilclass_dir, latlims, lonlims)
                self.logger.info("Attribute acquisition completed successfully")

                if self.reporting_manager:
                    # Attempt to visualize acquired files
                    try:
                        dem_file = dem_dir / f"domain_{self.domain_name}_elv.tif"
                        if dem_file.exists():
                            self.reporting_manager.visualize_spatial_coverage(dem_file, 'elevation', 'acquisition')

                        land_file = landclass_dir / f"domain_{self.domain_name}_land_classes.tif"
                        if land_file.exists():
                            self.reporting_manager.visualize_spatial_coverage(land_file, 'land_class', 'acquisition')

                        soil_file = soilclass_dir / f"domain_{self.domain_name}_soil_classes.tif"
                        if soil_file.exists():
                            self.reporting_manager.visualize_spatial_coverage(soil_file, 'soil_class', 'acquisition')
                    except Exception as e_viz:
                        self.logger.warning(f"Failed to visualize MAF attributes: {e_viz}")

            except Exception as e:
                self.logger.error(f"Error during attribute acquisition: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

    def _acquire_elevation_data(self, gistool_runner, output_dir: Path, lat_lims: str, lon_lims: str):
        self.logger.info("Acquiring elevation data")
        gistool_command = gistool_runner.create_gistool_command(
            dataset='MERIT-Hydro',
            output_dir=output_dir,
            lat_lims=lat_lims,
            lon_lims=lon_lims,
            variables='elv'
        )
        gistool_runner.execute_gistool_command(gistool_command)

    def _acquire_landcover_data(self, gistool_runner, output_dir: Path, lat_lims: str, lon_lims: str):
        self.logger.info("Acquiring land cover data")
        start_year = 2001
        end_year = 2020
        modis_var = "MCD12Q1.006"

        gistool_command = gistool_runner.create_gistool_command(
            dataset='MODIS',
            output_dir=output_dir,
            lat_lims=lat_lims,
            lon_lims=lon_lims,
            variables=modis_var,
            start_date=f"{start_year}-01-01",
            end_date=f"{end_year}-01-01"
        )
        gistool_runner.execute_gistool_command(gistool_command)

        land_name = self._get_config_value(lambda: self.config.domain.land_class_name, default='default')
        if land_name == 'default':
            land_name = f"domain_{self.domain_name}_land_classes.tif"

        if start_year != end_year:
            input_dir = output_dir / modis_var
            output_file = output_dir / land_name
            self.logger.info("Calculating land cover mode across years")
            calculate_landcover_mode(input_dir, output_file, start_year, end_year, self.domain_name)

    def _acquire_soilclass_data(self, gistool_runner, output_dir: Path, lat_lims: str, lon_lims: str):
        self.logger.info("Acquiring soil class data")
        gistool_command = gistool_runner.create_gistool_command(
            dataset='soil_class',
            output_dir=output_dir,
            lat_lims=lat_lims,
            lon_lims=lon_lims,
            variables='soil_classes'
        )
        gistool_runner.execute_gistool_command(gistool_command)

    def _expected_forcing_times(self, dataset: str) -> Optional[pd.DatetimeIndex]:
        resolution_hours = {
            "CARRA": 1,
            "CERRA": 3,
        }
        dataset_key = dataset.upper()
        if dataset_key not in resolution_hours:
            return None

        start = pd.to_datetime(self.config.get("EXPERIMENT_TIME_START"))
        end = pd.to_datetime(self.config.get("EXPERIMENT_TIME_END"))
        if pd.isna(start) or pd.isna(end) or end < start:
            return None

        freq = f"{resolution_hours[dataset_key]}h"
        return pd.date_range(start, end, freq=freq)

    def _cached_forcing_has_expected_times(
        self, cached_file: Path, expected_times: pd.DatetimeIndex
    ) -> bool:
        try:
            with xr.open_dataset(cached_file) as ds:
                if "time" not in ds:
                    return False
                actual_times = pd.to_datetime(ds["time"].values)
        except Exception as exc:
            self.logger.warning(f"Failed to validate cached forcing file {cached_file}: {exc}")
            return False

        if len(actual_times) < len(expected_times):
            return False

        return actual_times[0] <= expected_times[0] and actual_times[-1] >= expected_times[-1]

    def acquire_forcings(self):
        """Acquire forcing data for the model simulation."""
        self.logger.info("Starting forcing data acquisition")

        data_access = self._get_config_value(lambda: self.config.domain.data_access, default='MAF').upper()
        forcing_dataset = self._get_config_value(lambda: self.config.forcing.dataset, default='').upper()
        if forcing_dataset in {"CARRA", "CERRA"} and not self.config.get("FORCING_TIME_STEP_SIZE"):
            self.config["FORCING_TIME_STEP_SIZE"] = 10800
            self.logger.info(
                f"Defaulting FORCING_TIME_STEP_SIZE to 10800s for {forcing_dataset}"
            )

        if data_access == 'CLOUD':
            self.logger.info(f"Cloud data access enabled for {forcing_dataset}")

            if not check_cloud_access_availability(forcing_dataset, self.logger):
                raise ValueError(f"Dataset '{forcing_dataset}' does not support DATA_ACCESS: cloud.")

            raw_data_dir = self.project_dir / 'forcing' / 'raw_data'
            raw_data_dir.mkdir(parents=True, exist_ok=True)

            # Initialize cache
            cache_root = self.data_dir / 'cache' / 'raw_forcing'
            cache = RawForcingCache(
                cache_root=cache_root,
                max_size_gb=self.config.get('FORCING_CACHE_SIZE_GB', 3.0),
                ttl_days=self.config.get('FORCING_CACHE_TTL_DAYS', 30),
                enable_checksum=self.config.get('FORCING_CACHE_CHECKSUM', True)
            )

            # Generate cache key
            bbox = self._get_config_value(lambda: self.config.domain.bounding_box_coords)
            time_start = self._get_config_value(lambda: self.config.domain.time_start)
            time_end = self._get_config_value(lambda: self.config.domain.time_end)
            variables = self.config.get('FORCING_VARIABLES')

            cache_key = cache.generate_cache_key(
                dataset=forcing_dataset,
                bbox=bbox,
                time_start=time_start,
                time_end=time_end,
                variables=variables if isinstance(variables, list) else None
            )

            # Check cache first
            cached_file = cache.get(cache_key)
            if cached_file and not self.config.get('FORCE_DOWNLOAD', False):
                expected_times = self._expected_forcing_times(forcing_dataset)
                if expected_times is not None and not self._cached_forcing_has_expected_times(
                    cached_file, expected_times
                ):
                    self.logger.warning(
                        f"Cached forcing data {cached_file} does not cover the requested time range; "
                        "re-downloading from source."
                    )
                    cached_file = None

            if cached_file and not self.config.get('FORCE_DOWNLOAD', False):
                self.logger.info(f"✓ Using cached forcing data: {cache_key}")
                # Copy from cache to project directory
                import shutil
                output_file = raw_data_dir / cached_file.name
                shutil.copy(cached_file, output_file)
                self.logger.info(f"✓ Copied cached file to: {output_file}")
            else:
                # Cache miss - download from source
                if cached_file:
                    self.logger.info("FORCE_DOWNLOAD enabled - skipping cache")
                else:
                    self.logger.info("Cache miss - downloading from source")

                try:
                    downloader = CloudForcingDownloader(self.config, self.logger)
                    output_file = downloader.download_forcing_data(raw_data_dir)
                    self.logger.info(f"✓ Cloud forcing data acquisition completed: {output_file}")

                    # Handle case where output is a directory (e.g. non-aggregated files)
                    if output_file.is_dir():
                        self.logger.info("Output is a directory - skipping single-file caching and visualization")

                        # Find a sample file for visualization
                        sample_files = list(output_file.glob("*.nc"))
                        if sample_files:
                            sample_file = sample_files[0]
                            if self.reporting_manager:
                                self.reporting_manager.visualize_spatial_coverage(sample_file, 'forcing_sample', 'acquisition')

                        self.logger.warning("Caching is not currently supported for non-aggregated forcing files. Skipping cache.")
                    else:
                        if self.reporting_manager and output_file and output_file.exists():
                            self.reporting_manager.visualize_spatial_coverage(output_file, 'forcing_sample', 'acquisition')

                        # Store in cache
                        try:
                            cache.put(
                                cache_key=cache_key,
                                file_path=output_file,
                                metadata={
                                    'dataset': forcing_dataset,
                                    'bbox': bbox,
                                    'time_range': f"{time_start} to {time_end}",
                                    'variables': variables if isinstance(variables, list) else str(variables),
                                    'domain_name': self.domain_name
                                }
                            )
                        except Exception as cache_error:
                            self.logger.warning(f"Failed to cache downloaded file: {cache_error}")
                            # Don't fail the acquisition if caching fails

                except Exception as e:
                    self.logger.error(f"Error during cloud data acquisition: {str(e)}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    raise

        else:
            self.logger.info("Using traditional MAF data acquisition workflow")

            if forcing_dataset == 'AORC':
                raise ValueError("AORC is not supported with DATA_ACCESS: MAF.")

            dr = datatoolRunner(self.config, self.logger)
            raw_data_dir = self.project_dir / 'forcing' / 'raw_data'
            raw_data_dir.mkdir(parents=True, exist_ok=True)

            bbox = self._get_config_value(lambda: self.config.domain.bounding_box_coords).split('/')
            latlims = f"{bbox[2]},{bbox[0]}"
            lonlims = f"{bbox[1]},{bbox[3]}"

            variables = self._get_config_value(lambda: self.config.forcing.variables, default='default')
            if variables == 'default':
                variables = self.variable_handler.get_dataset_variables(
                    dataset=self._get_config_value(lambda: self.config.forcing.dataset)
                )

            try:
                datatool_command = dr.create_datatool_command(
                    dataset=self._get_config_value(lambda: self.config.forcing.dataset),
                    output_dir=raw_data_dir,
                    lat_lims=latlims,
                    lon_lims=lonlims,
                    variables=variables,
                    start_date=self._get_config_value(lambda: self.config.domain.time_start),
                    end_date=self._get_config_value(lambda: self.config.domain.time_end)
                )
                dr.execute_datatool_command(datatool_command)
                self.logger.info("Primary forcing data acquisition completed successfully")

                if self.reporting_manager:
                    # Find a sample forcing file
                    sample_files = list(raw_data_dir.glob("*.nc"))
                    if sample_files:
                        self.reporting_manager.visualize_spatial_coverage(sample_files[0], 'forcing_sample', 'acquisition')

            except Exception as e:
                self.logger.error(f"Error during forcing data acquisition: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise

        if self._get_config_value(lambda: self.config.forcing.supplement, default=False):
            self.logger.info("SUPPLEMENT_FORCING enabled - acquiring EM-Earth data")
            self.acquire_em_earth_forcings()

    def acquire_observations(self):
        """
        Acquire additional observations based on configuration.
        This handles registry-based observations (GRACE, MODIS, etc.)
        that require an 'acquire' step before processing.
        """
        from symfluence.data.observation.registry import ObservationRegistry

        additional_obs = self._get_config_value(lambda: self.config.data.additional_observations) or []
        if isinstance(additional_obs, str):
            additional_obs = [o.strip() for o in additional_obs.split(',')]
        elif additional_obs is None:
            additional_obs = []

        # Auto-detect observation types based on config flags (matching process_observed_data logic)
        streamflow_provider = (self._get_config_value(lambda: self.config.data.streamflow_data_provider) or '').upper()
        if streamflow_provider == 'USGS' and 'USGS_STREAMFLOW' not in additional_obs:
            additional_obs.append('USGS_STREAMFLOW')
        elif streamflow_provider == 'WSC' and 'WSC_STREAMFLOW' not in additional_obs:
            additional_obs.append('WSC_STREAMFLOW')
        elif streamflow_provider == 'SMHI' and 'SMHI_STREAMFLOW' not in additional_obs:
            additional_obs.append('SMHI_STREAMFLOW')
        elif streamflow_provider == 'LAMAH_ICE' and 'LAMAH_ICE_STREAMFLOW' not in additional_obs:
            additional_obs.append('LAMAH_ICE_STREAMFLOW')

        # Check for USGS Groundwater download
        download_usgs_gw = self.config.get('DOWNLOAD_USGS_GW', False)
        if isinstance(download_usgs_gw, str):
            download_usgs_gw = download_usgs_gw.lower() == 'true'
        if download_usgs_gw and 'USGS_GW' not in additional_obs:
            additional_obs.append('USGS_GW')

        # Check for MODIS Snow
        if self.config.get('DOWNLOAD_MODIS_SNOW', False) and 'MODIS_SNOW' not in additional_obs:
            additional_obs.append('MODIS_SNOW')

        # Check for SNOTEL
        download_snotel = self.config.get('DOWNLOAD_SNOTEL', False)
        if isinstance(download_snotel, str):
            download_snotel = download_snotel.lower() == 'true'
        if download_snotel and 'SNOTEL' not in additional_obs:
            additional_obs.append('SNOTEL')

        # Check for GRACE
        if self.config.get('DOWNLOAD_GRACE', False) and 'GRACE' not in additional_obs:
            additional_obs.append('GRACE')

        # Check for MOD16 ET (based on ET_OBS_SOURCE or OPTIMIZATION_TARGET)
        et_obs_source = str(self.config.get('ET_OBS_SOURCE', '')).lower()
        optimization_target = str(self.config.get('OPTIMIZATION_TARGET', '')).lower()
        if et_obs_source in ('mod16', 'modis', 'modis_et', 'mod16a2'):
            if 'MODIS_ET' not in additional_obs and 'MOD16' not in additional_obs:
                additional_obs.append('MODIS_ET')
        elif optimization_target == 'et' and not et_obs_source:
            # Default to MOD16 if ET calibration without explicit source
            if 'MODIS_ET' not in additional_obs:
                additional_obs.append('MODIS_ET')

        # Check for FLUXNET data (based on config flags or ET_OBS_SOURCE)
        if self.config.get('DOWNLOAD_FLUXNET', False) or et_obs_source == 'fluxnet':
            if 'FLUXNET' not in additional_obs and 'FLUXNET_ET' not in additional_obs:
                additional_obs.append('FLUXNET_ET')

        # Check for multi-source ET (both FLUXNET and MOD16)
        if self.config.get('MULTI_SOURCE_ET', False):
            if 'FLUXNET_ET' not in additional_obs and 'FLUXNET' not in additional_obs:
                additional_obs.append('FLUXNET_ET')
            if 'MODIS_ET' not in additional_obs and 'MOD16' not in additional_obs:
                additional_obs.append('MODIS_ET')

        if not additional_obs:
            return

        self.logger.info(f"Acquiring additional observations: {additional_obs}")

        for obs_type in additional_obs:
            try:
                if ObservationRegistry.is_registered(obs_type):
                    self.logger.info(f"Acquiring registry-based observation: {obs_type}")
                    handler = ObservationRegistry.get_handler(obs_type, self.config, self.logger)
                    handler.acquire()
                else:
                    self.logger.debug(f"Skipping acquisition for {obs_type}: no registry handler")
            except Exception as e:
                self.logger.warning(f"Failed to acquire additional observation {obs_type}: {e}")

    def acquire_em_earth_forcings(self):
        """Acquire EM-Earth precipitation and temperature data."""
        self.logger.info("Starting EM-Earth forcing data acquisition")

        try:
            em_earth_dir = self.project_dir / 'forcing' / 'raw_data_em_earth'
            em_earth_dir.mkdir(parents=True, exist_ok=True)

            em_region = self.config.get('EM_EARTH_REGION', 'NorthAmerica')
            em_earth_prcp_dir = self.config.get('EM_EARTH_PRCP_DIR',
                f"/anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly/prcp/{em_region}")
            em_earth_tmean_dir = self.config.get('EM_EARTH_TMEAN_DIR',
                f"/anvil/datasets/meteorological/EM-Earth/EM_Earth_v1/deterministic_hourly/tmean/{em_region}")

            if not Path(em_earth_prcp_dir).exists():
                raise FileNotFoundError(f"EM-Earth precipitation directory not found: {em_earth_prcp_dir}")
            if not Path(em_earth_tmean_dir).exists():
                raise FileNotFoundError(f"EM-Earth temperature directory not found: {em_earth_tmean_dir}")

            bbox = self._get_config_value(lambda: self.config.domain.bounding_box_coords)
            bbox_parts = bbox.split('/')
            lat_max, lon_min, lat_min, lon_max = map(float, bbox_parts)
            lat_range = lat_max - lat_min
            lon_range = lon_max - lon_min

            self.logger.info(f"Watershed bounding box: {bbox}")
            self.logger.info(f"Watershed size: {lat_range:.4f}° x {lon_range:.4f}°")

            min_bbox_size = self.config.get('EM_EARTH_MIN_BBOX_SIZE', 0.1)
            if lat_range < min_bbox_size or lon_range < min_bbox_size:
                self.logger.warning("Very small watershed detected. EM-Earth processing will use spatial averaging.")

            try:
                start_date = datetime.strptime(self._get_config_value(lambda: self.config.domain.time_start), '%Y-%m-%d %H:%M')
                end_date = datetime.strptime(self._get_config_value(lambda: self.config.domain.time_end), '%Y-%m-%d %H:%M')
            except ValueError as e:
                raise ValueError(f"Invalid date format in configuration: {str(e)}")

            self.logger.info(f"Processing EM-Earth data for period: {start_date} to {end_date}")

            year_months = self._generate_year_month_list(start_date, end_date)

            if not year_months:
                raise ValueError("No valid year-month combinations found for the specified time period")

            processed_files = []
            failed_months = []

            for i, year_month in enumerate(year_months, 1):
                try:
                    self.logger.info(f"Processing month {i}/{len(year_months)}: {year_month}")
                    processed_file = self._process_em_earth_month(
                        year_month, em_earth_prcp_dir, em_earth_tmean_dir, em_earth_dir, bbox
                    )
                    if processed_file:
                        processed_files.append(processed_file)
                        self.logger.info(f"✓ Successfully processed EM-Earth data for {year_month}")
                    else:
                        failed_months.append(year_month)
                        self.logger.warning(f"✗ Failed to process EM-Earth data for {year_month}")

                except Exception as e:
                    failed_months.append(year_month)
                    self.logger.warning(f"✗ Failed to process EM-Earth data for {year_month}: {str(e)}")
                    continue

            if not processed_files:
                raise ValueError("No EM-Earth data files were successfully processed")

            success_rate = len(processed_files) / len(year_months) * 100
            self.logger.info(f"EM-Earth forcing data acquisition completed. Success rate: {success_rate:.1f}%")

            if failed_months and success_rate < 50:
                raise ValueError(f"EM-Earth processing success rate too low ({success_rate:.1f}%).")

            if self.reporting_manager and processed_files:
                # Visualize one sample file
                self.reporting_manager.visualize_spatial_coverage(processed_files[0], 'em_earth_sample', 'acquisition')

        except Exception as e:
            self.logger.error(f"Error during EM-Earth forcing data acquisition: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _generate_year_month_list(self, start_date: datetime, end_date: datetime) -> List[str]:
        year_months = []
        current_date = start_date.replace(day=1)

        while current_date <= end_date:
            year_month = current_date.strftime('%Y%m')
            year_months.append(year_month)
            if current_date.month == 12:
                current_date = current_date.replace(year=current_date.year + 1, month=1)
            else:
                current_date = current_date.replace(month=current_date.month + 1)
        return year_months

    def _process_em_earth_month(self, year_month: str, prcp_dir: str, tmean_dir: str,
                               output_dir: Path, bbox: str) -> Optional[Path]:
        em_region = self.config.get('EM_EARTH_REGION', 'NorthAmerica')

        prcp_pattern = f"EM_Earth_deterministic_hourly_{em_region}_{year_month}.nc"
        tmean_pattern = f"EM_Earth_deterministic_hourly_{em_region}_{year_month}.nc"

        prcp_file = Path(prcp_dir) / prcp_pattern
        tmean_file = Path(tmean_dir) / tmean_pattern

        if not prcp_file.exists():
            self.logger.warning(f"EM-Earth precipitation file not found: {prcp_file}")
            return None
        if not tmean_file.exists():
            self.logger.warning(f"EM-Earth temperature file not found: {tmean_file}")
            return None

        output_file = output_dir / f"watershed_subset_{year_month}.nc"

        if output_file.exists() and not self.config.get('FORCE_RUN_ALL_STEPS', False):
            self.logger.info(f"EM-Earth file already exists, skipping: {output_file}")
            return output_file

        try:
            self._process_em_earth_data(str(prcp_file), str(tmean_file), str(output_file), bbox)
            return output_file
        except Exception as e:
            self.logger.error(f"Error processing EM-Earth data for {year_month}: {str(e)}")
            return None

    def _process_em_earth_data(self, prcp_file: str, tmean_file: str, output_file: str, bbox: str):
        """Process EM-Earth precipitation and temperature data for a specific bounding box."""
        import xarray as xr

        bbox_parts = bbox.split('/')
        if len(bbox_parts) != 4:
            raise ValueError(f"Invalid bounding box format: {bbox}. Expected lat_max/lon_min/lat_min/lon_max")

        lat_max, lon_min, lat_min, lon_max = map(float, bbox_parts)
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min

        min_bbox_size = 0.1
        original_bbox = (lat_min, lat_max, lon_min, lon_max)

        if lat_range < min_bbox_size or lon_range < min_bbox_size:
            self.logger.warning(f"Very small watershed detected (lat: {lat_range:.4f}°, lon: {lon_range:.4f}°)")

            lat_center = (lat_min + lat_max) / 2
            lon_center = (lon_min + lon_max) / 2

            lat_min_extract = lat_center - min_bbox_size/2
            lat_max_extract = lat_center + min_bbox_size/2
            lon_min_extract = lon_center - min_bbox_size/2
            lon_max_extract = lon_center + min_bbox_size/2
        else:
            lat_min_extract, lat_max_extract = lat_min, lat_max
            lon_min_extract, lon_max_extract = lon_min, lon_max

        try:
            prcp_ds = xr.open_dataset(prcp_file)
            tmean_ds = xr.open_dataset(tmean_file)
        except Exception as e:
            raise ValueError(f"Error opening EM-Earth files: {str(e)}")

        try:
            if lon_min_extract > lon_max_extract:
                prcp_subset = prcp_ds.where(
                    (prcp_ds.lat >= lat_min_extract) & (prcp_ds.lat <= lat_max_extract) &
                    ((prcp_ds.lon >= lon_min_extract) | (prcp_ds.lon <= lon_max_extract)), drop=True
                )
                tmean_subset = tmean_ds.where(
                    (tmean_ds.lat >= lat_min_extract) & (tmean_ds.lat <= lat_max_extract) &
                    ((tmean_ds.lon >= lon_min_extract) | (tmean_ds.lon <= lon_max_extract)), drop=True
                )
            else:
                prcp_subset = prcp_ds.where(
                    (prcp_ds.lat >= lat_min_extract) & (prcp_ds.lat <= lat_max_extract) &
                    (prcp_ds.lon >= lon_min_extract) & (prcp_ds.lon <= lon_max_extract), drop=True
                )
                tmean_subset = tmean_ds.where(
                    (tmean_ds.lat >= lat_min_extract) & (tmean_ds.lat <= lat_max_extract) &
                    (tmean_ds.lon >= lon_min_extract) & (tmean_ds.lon <= lon_max_extract), drop=True
                )

            if prcp_subset.sizes.get('lat', 0) == 0 or prcp_subset.sizes.get('lon', 0) == 0:
                self.logger.warning("No precipitation data found with initial expansion, trying larger expansion")
                larger_expand = 0.2
                lat_center = (original_bbox[0] + original_bbox[1]) / 2
                lon_center = (original_bbox[2] + original_bbox[3]) / 2

                lat_min_large = lat_center - larger_expand
                lat_max_large = lat_center + larger_expand
                lon_min_large = lon_center - larger_expand
                lon_max_large = lon_center + larger_expand

                prcp_subset = prcp_ds.where(
                    (prcp_ds.lat >= lat_min_large) & (prcp_ds.lat <= lat_max_large) &
                    (prcp_ds.lon >= lon_min_large) & (prcp_ds.lon <= lon_max_large), drop=True
                )
                tmean_subset = tmean_ds.where(
                    (tmean_ds.lat >= lat_min_large) & (tmean_ds.lat <= lat_max_large) &
                    (tmean_ds.lon >= lon_min_large) & (tmean_ds.lon <= lon_max_large), drop=True
                )

            if prcp_subset.sizes.get('lat', 0) == 0 or prcp_subset.sizes.get('lon', 0) == 0:
                raise ValueError("No precipitation data found within the expanded bounding box.")
            if tmean_subset.sizes.get('lat', 0) == 0 or tmean_subset.sizes.get('lon', 0) == 0:
                raise ValueError("No temperature data found within the expanded bounding box.")

        except Exception as e:
            raise ValueError(f"Error subsetting EM-Earth data: {str(e)}")

        if (lat_min_extract, lat_max_extract, lon_min_extract, lon_max_extract) != original_bbox:
            self.logger.info("Computing spatial average over expanded area to represent the small watershed")
            prcp_subset = prcp_subset.mean(dim=['lat', 'lon'], keep_attrs=True)
            tmean_subset = tmean_subset.mean(dim=['lat', 'lon'], keep_attrs=True)

            prcp_subset = prcp_subset.expand_dims({'lat': [original_bbox[0] + (original_bbox[1] - original_bbox[0])/2]})
            prcp_subset = prcp_subset.expand_dims({'lon': [original_bbox[2] + (original_bbox[3] - original_bbox[2])/2]})
            tmean_subset = tmean_subset.expand_dims({'lat': [original_bbox[0] + (original_bbox[1] - original_bbox[0])/2]})
            tmean_subset = tmean_subset.expand_dims({'lon': [original_bbox[2] + (original_bbox[3] - original_bbox[2])/2]})

        try:
            merged_ds = xr.Dataset()
            merged_ds = merged_ds.assign_coords({
                'lat': prcp_subset.lat,
                'lon': prcp_subset.lon,
                'time': prcp_subset.time
            })

            for var in prcp_subset.data_vars:
                if 'prcp' in var:
                    merged_ds[var] = prcp_subset[var]

            for var in tmean_subset.data_vars:
                if 'tmean' in var or 'temp' in var:
                    temp_interp = tmean_subset[var].interp(
                        lat=prcp_subset.lat,
                        lon=prcp_subset.lon,
                        method='linear'
                    )
                    merged_ds[var] = temp_interp

            is_small_watershed = lat_range < min_bbox_size or lon_range < min_bbox_size
            is_spatially_averaged = (lat_min_extract, lat_max_extract, lon_min_extract, lon_max_extract) != original_bbox

            merged_ds.attrs.update({
                'small_watershed_processing': int(is_small_watershed),
                'spatial_averaging_applied': int(is_spatially_averaged),
                'subset_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })

            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            merged_ds.to_netcdf(output_file)

        except Exception as e:
            raise ValueError(f"Error merging EM-Earth datasets: {str(e)}")

        finally:
            prcp_ds.close()
            tmean_ds.close()
