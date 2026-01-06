"""
Forcing data processing utilities for SUMMA model.

This module contains the SummaForcingProcessor class which handles all forcing data
processing operations including lapse rate corrections, time coordinate fixes,
NaN value handling, and data validation for SUMMA model compatibility.
"""

# Standard library imports
import gc
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Third-party imports
import geopandas as gpd  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import psutil  # type: ignore
import xarray as xr  # type: ignore
from symfluence.core.constants import PhysicalConstants
from ..utilities import BaseForcingProcessor


class SummaForcingProcessor(BaseForcingProcessor):
    """
    Processor for SUMMA forcing data with comprehensive quality control and corrections.

    This class handles:
    - Temperature lapse rate corrections
    - Time coordinate standardization for SUMMA compatibility
    - NaN value interpolation and filling
    - Data range validation and clipping
    - Batch processing for memory efficiency
    - Forcing file list generation
    - HRU ID filtering

    Attributes:
        config: Configuration dictionary containing processing parameters
        logger: Logger instance for recording operations
        forcing_basin_path: Path to basin-averaged forcing data
        forcing_summa_path: Path to output SUMMA-compatible forcing data
        intersect_path: Path to catchment intersection shapefiles
        catchment_path: Path to catchment shapefiles
        project_dir: Root project directory
        setup_dir: Path to SUMMA setup/settings directory
        domain_name: Name of the domain being processed
        forcing_dataset: Name of the forcing dataset (e.g., 'era5', 'rdrs')
        data_step: Time step size in seconds
        gruId: Name of GRU ID field in configuration
        hruId: Name of HRU ID field in configuration
        catchment_name: Name of the catchment shapefile
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        forcing_basin_path: Path,
        forcing_summa_path: Path,
        intersect_path: Path,
        catchment_path: Path,
        project_dir: Path,
        setup_dir: Path,
        domain_name: str,
        forcing_dataset: str,
        data_step: int,
        gruId: str,
        hruId: str,
        catchment_name: str
    ):
        """
        Initialize the SUMMA forcing processor.

        Args:
            config: Configuration dictionary with processing parameters
            logger: Logger instance
            forcing_basin_path: Path to input basin-averaged forcing data
            forcing_summa_path: Path to output SUMMA-compatible forcing data
            intersect_path: Path to catchment intersection shapefiles
            catchment_path: Path to catchment shapefiles
            project_dir: Root project directory
            setup_dir: SUMMA settings directory
            domain_name: Domain name for file naming
            forcing_dataset: Forcing dataset identifier
            data_step: Time step size in seconds
            gruId: GRU ID field name
            hruId: HRU ID field name
            catchment_name: Catchment shapefile filename
        """
        super().__init__(
            config=config,
            logger=logger,
            input_path=forcing_basin_path,
            output_path=forcing_summa_path,
            intersect_path=intersect_path,
            catchment_path=catchment_path,
            project_dir=project_dir,
            setup_dir=setup_dir
        )
        # Keep original attribute names for backward compatibility
        self.forcing_basin_path = self.input_path
        self.forcing_summa_path = self.output_path
        self.domain_name = domain_name
        self.forcing_dataset = forcing_dataset
        self.data_step = data_step
        self.gruId = gruId
        self.hruId = hruId
        self.catchment_name = catchment_name

    @property
    def model_name(self) -> str:
        """Return model name for logging."""
        return "SUMMA"

    def apply_datastep_and_lapse_rate(self):
        """
        Apply temperature lapse rate corrections to the forcing data with improved memory efficiency.

        This optimized version:
        - Processes files in batches to control memory usage
        - Uses explicit garbage collection
        - Minimizes intermediate object creation
        - Provides progress monitoring and memory usage tracking
        """
        self.logger.info("Starting memory-efficient temperature lapse rate and data step application")

        # Find intersection file
        intersect_base = f"{self.domain_name}_{self.config.get('FORCING_DATASET')}_intersected_shapefile"
        intersect_csv = self.intersect_path / f"{intersect_base}.csv"
        intersect_shp = self.intersect_path / f"{intersect_base}.shp"

        # Fallback for legacy naming in data bundle
        if not intersect_csv.exists() and not intersect_shp.exists() and self.domain_name == 'bow_banff_minimal':
            legacy_base = f"Bow_at_Banff_lumped_{self.config.get('FORCING_DATASET')}_intersected_shapefile"
            if (self.intersect_path / f"{legacy_base}.csv").exists():
                intersect_csv = self.intersect_path / f"{legacy_base}.csv"
                self.logger.info(f"Using legacy intersection CSV: {intersect_csv.name}")
            elif (self.intersect_path / f"{legacy_base}.shp").exists():
                intersect_shp = self.intersect_path / f"{legacy_base}.shp"
                self.logger.info(f"Using legacy intersection SHP: {intersect_shp.name}")

        # Handle shapefile to CSV conversion if needed
        if not intersect_csv.exists() and intersect_shp.exists():
            self.logger.info(f"Converting {intersect_shp} to CSV format")
            try:
                shp_df = gpd.read_file(intersect_shp)
                shp_df['weight'] = shp_df['AP1']
                shp_df.to_csv(intersect_csv, index=False)
                self.logger.info(f"Successfully created {intersect_csv}")
                del shp_df  # Explicit cleanup
                gc.collect()
            except Exception as e:
                self.logger.error(f"Failed to convert shapefile to CSV: {str(e)}")
                raise
        elif not intersect_csv.exists() and not intersect_shp.exists():
            # Fallback: check for remapping weights file which often contains the same info
            hru_id_field = self.config.get('CATCHMENT_SHP_HRUID')
            case_name = f"{self.domain_name}_{self.config.get('FORCING_DATASET')}"
            remap_file = self.intersect_path / f"{case_name}_{hru_id_field}_remapping.csv"
            
            if remap_file.exists():
                self.logger.info(f"Intersected shapefile missing, falling back to remapping weights: {remap_file.name}")
                intersect_csv = remap_file
            else:
                self.logger.error(f"Missing both intersected shapefile and remapping weights in {self.intersect_path}")
                self.logger.error(f"Expected intersect base: {intersect_base}")
                self.logger.error(f"Expected remap file: {remap_file.name}")
                raise FileNotFoundError(f"Neither {intersect_csv} nor {intersect_shp} exist")

        # Load topology data efficiently
        self.logger.info("Loading topology data...")
        try:
            # First read column names to check for truncation
            # Shapefiles truncate field names to 10 characters
            sample_df = pd.read_csv(intersect_csv, nrows=0)

            # Build dtype dict with potentially truncated column names
            dtype_dict = {}

            # Handle GRU ID (may be truncated)
            gru_col = f'S_1_{self.gruId}'
            if gru_col not in sample_df.columns:
                # Try to find truncated version (10 char limit)
                gru_col_truncated = f'S_1_{self.gruId}'[:10]
                if gru_col_truncated in sample_df.columns:
                    dtype_dict[gru_col_truncated] = 'int32'
                else:
                    self.logger.warning(f"Column {gru_col} not found in CSV, will try to load without dtype")
            else:
                dtype_dict[gru_col] = 'int32'

            # Handle HRU ID (may be truncated)
            hru_col = f'S_1_{self.hruId}'
            if hru_col not in sample_df.columns:
                # Try to find truncated version (10 char limit)
                hru_col_truncated = f'S_1_{self.hruId}'[:10]
                if hru_col_truncated in sample_df.columns:
                    dtype_dict[hru_col_truncated] = 'int32'
                    self.logger.info(f"Using truncated column name: {hru_col_truncated} (original: {hru_col})")
                else:
                    self.logger.warning(f"Column {hru_col} not found in CSV, will try to load without dtype")
            else:
                dtype_dict[hru_col] = 'int32'

            # Add other standard columns
            dtype_dict.update({
                'S_2_ID': 'Int32',  # Nullable integer to handle NA values
                'S_2_elev_m': 'float32',
                'weight': 'float32'
            })

            # Handle elevation column (may be S_1_elev_m or S_1_elev_mean)
            if 'S_1_elev_m' in sample_df.columns:
                dtype_dict['S_1_elev_m'] = 'float32'
            elif 'S_1_elev_mean' in sample_df.columns:
                dtype_dict['S_1_elev_mean'] = 'float32'

            # Use chunked reading for very large CSV files
            topo_data = pd.read_csv(intersect_csv, dtype=dtype_dict)

            # Rename truncated columns back to full names for consistency
            rename_dict = {}
            if f'S_1_{self.hruId}' not in topo_data.columns:
                hru_col_truncated = f'S_1_{self.hruId}'[:10]
                if hru_col_truncated in topo_data.columns:
                    rename_dict[hru_col_truncated] = f'S_1_{self.hruId}'

            if f'S_1_{self.gruId}' not in topo_data.columns:
                gru_col_truncated = f'S_1_{self.gruId}'[:10]
                if gru_col_truncated in topo_data.columns:
                    rename_dict[gru_col_truncated] = f'S_1_{self.gruId}'

            if rename_dict:
                self.logger.info(f"Renaming truncated columns: {rename_dict}")
                topo_data.rename(columns=rename_dict, inplace=True)

            self.logger.info(f"Loaded topology data: {len(topo_data)} rows, {topo_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            self.logger.info(f"Columns after rename: {topo_data.columns.tolist()[:15]}")
            self.logger.info(f"Sample HRU IDs: {topo_data[f'S_1_{self.hruId}'].head(5).tolist()}")
        except Exception as e:
            self.logger.error(f"Error loading topology data: {str(e)}")
            raise

        # Get forcing files and log memory info
        forcing_files = [f for f in os.listdir(self.forcing_basin_path)
                        if f.startswith(f"{self.domain_name}") and f.endswith('.nc')]
        forcing_files.sort()

        total_files = len(forcing_files)
        self.logger.info(f"Found {total_files} forcing files to process")

        if total_files == 0:
            raise FileNotFoundError(f"No forcing files found in {self.forcing_basin_path}")

        # Prepare output directory
        self.forcing_summa_path.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.domain_name}_{self.forcing_dataset}".lower()
        for existing_file in self.forcing_summa_path.glob("*.nc"):
            if not existing_file.name.lower().startswith(prefix):
                continue
            try:
                existing_file.unlink()
                self.logger.info(f"Removed stale SUMMA forcing file {existing_file}")
            except OSError as exc:
                self.logger.warning(f"Failed to remove stale SUMMA forcing file {existing_file}: {exc}")

        # Define column names and lapse rate
        gru_id = f'S_1_{self.gruId}'
        hru_id = f'S_1_{self.hruId}'
        forcing_id = 'S_2_ID'
        catchment_elev = 'S_1_elev_m'
        forcing_elev = 'S_2_elev_m'
        weights = 'weight'
        lapse_rate = float(self.config.get('LAPSE_RATE'))  # [K m-1]

        if catchment_elev not in topo_data.columns and 'S_1_elev_mean' in topo_data.columns:
            catchment_elev = 'S_1_elev_mean'

        # Pre-calculate lapse values efficiently
        self.logger.info("Pre-calculating lapse rate corrections...")
        topo_data['lapse_values'] = topo_data[weights] * lapse_rate * (topo_data[forcing_elev] - topo_data[catchment_elev])

        # Calculate weighted lapse values for each HRU
        if gru_id == hru_id:
            lapse_values = topo_data.groupby([hru_id])['lapse_values'].sum().reset_index()
        else:
            lapse_values = topo_data.groupby([gru_id, hru_id])['lapse_values'].sum().reset_index()

        # Sort and set hruID as index
        lapse_values = lapse_values.sort_values(hru_id).set_index(hru_id)

        # Clean up topology data to free memory
        del topo_data
        gc.collect()
        self.logger.info(f"Prepared lapse corrections for {len(lapse_values)} HRUs")
        self.logger.info(f"Lapse values HRU IDs: {lapse_values.index.tolist()}")

        # Determine batch size based on available memory and file count
        batch_size = self._determine_batch_size(total_files)
        self.logger.info(f"Processing files in batches of {batch_size}")

        # Process files in batches
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = forcing_files[batch_start:batch_end]

            # Log memory usage before batch
            memory_before = psutil.Process().memory_info().rss / 1024**2
            self.logger.debug(f"Memory usage before batch: {memory_before:.1f} MB")

            # Process each file in the batch
            for i, file in enumerate(batch_files):
                try:
                    self._process_single_file(file, lapse_values, lapse_rate)

                    # Log progress every 10 files or for small batches
                    if (i + 1) % 10 == 0 or batch_size <= 10:
                        files_processed = batch_start + i + 1

                except Exception as e:
                    self.logger.error(f"Error processing file {file}: {str(e)}")
                    raise

            # Force garbage collection after each batch
            gc.collect()

            # Log memory usage after batch
            memory_after = psutil.Process().memory_info().rss / 1024**2
            self.logger.debug(f"Memory usage after batch: {memory_after:.1f} MB "
                            f"(delta: {memory_after - memory_before:+.1f} MB)")

        # Final cleanup
        del lapse_values
        gc.collect()

        self.logger.info(f"Completed processing of {total_files} {self.forcing_dataset.upper()} forcing files with temperature lapsing")

    def _process_single_file(self, file: str, lapse_values: pd.DataFrame, lapse_rate: float):
        """
        Process a single forcing file with comprehensive fixes for SUMMA compatibility.

        Fixes:
        1. Time coordinate format (convert to seconds since reference)
        2. NaN values in forcing data (interpolation)
        3. Data validation and quality checks

        Args:
            file: Filename to process
            lapse_values: Pre-calculated lapse values
            lapse_rate: Lapse rate value
        """
        input_path = self.forcing_basin_path / file
        output_path = self.forcing_summa_path / file

        self.logger.debug(f"Processing file: {file}")

        # Use context manager and process efficiently
        with xr.open_dataset(input_path) as dat:
            # Create a copy to avoid modifying the original
            dat = dat.copy()

            # 1. FIX TIME COORDINATE FIRST
            dat = self._fix_time_coordinate_comprehensive(dat, file)

            # Find which HRU IDs exist in the forcing data but not in the lapse values
            valid_hru_mask = np.isin(dat['hruId'].values, lapse_values.index)

            # Log and filter invalid HRUs
            if not np.all(valid_hru_mask):
                missing_hrus = dat['hruId'].values[~valid_hru_mask]
                if len(missing_hrus) <= 10:
                    self.logger.warning(f"File {file}: Removing {len(missing_hrus)} HRU IDs without lapse values: {missing_hrus}")
                else:
                    self.logger.warning(f"File {file}: Removing {len(missing_hrus)} HRU IDs without lapse values")

                # Filter the dataset
                dat = dat.sel(hru=valid_hru_mask)

                if len(dat.hru) == 0:
                    raise ValueError(f"File {file}: No valid HRUs found after filtering")

            # 2. FIX NaN VALUES IN FORCING DATA
            dat = self._fix_nan_values(dat, file)

            # 2b. ENSURE REQUIRED VARIABLES EXIST
            dat = self._ensure_required_forcing_variables(dat, file)

            # 3. VALIDATE DATA RANGES
            dat = self._validate_and_fix_data_ranges(dat, file)

            # Apply data step (memory efficient - in-place operation)
            dat['data_step'] = self.data_step
            dat.data_step.attrs.update({
                'long_name': 'data step length in seconds',
                'units': 's'
            })

            # Update precipitation units if present
            if 'pptrate' in dat:
                dat.pptrate.attrs.update({
                    'units': 'mm/s',
                    'long_name': 'Mean total precipitation rate'
                })

                # Apply lapse rate correction efficiently if enabled
            if self.config.get('APPLY_LAPSE_RATE') == True:
                # Get lapse values for the HRUs (vectorized operation)
                hru_lapse_values = lapse_values.loc[dat['hruId'].values, 'lapse_values'].values

                # Create correction array more efficiently, handling both (time, hru) and (hru, time)
                if dat['airtemp'].dims == ('time', 'hru'):
                    lapse_correction = np.broadcast_to(hru_lapse_values[np.newaxis, :], dat['airtemp'].shape)
                elif dat['airtemp'].dims == ('hru', 'time'):
                    lapse_correction = np.broadcast_to(hru_lapse_values[:, np.newaxis], dat['airtemp'].shape)
                else:
                    self.logger.warning(f"Unexpected airtemp dimensions {dat['airtemp'].dims}, skipping lapse correction")
                    lapse_correction = 0

                # Store original attributes
                tmp_units = dat['airtemp'].attrs.get('units', 'K')

                # Apply correction (in-place operation)
                if not isinstance(lapse_correction, int):
                    dat['airtemp'].values += lapse_correction
                dat.airtemp.attrs['units'] = tmp_units

                # Clean up temporary arrays
                del hru_lapse_values, lapse_correction

            # 4. FINAL VALIDATION BEFORE SAVING
            self._final_validation(dat, file)

            # Prepare encoding with time coordinate fix
            encoding = {
                var: {'zlib': True, 'complevel': 1, 'shuffle': True}
                for var in dat.data_vars
            }

            # Ensure time coordinate is properly encoded for SUMMA
            encoding['time'] = {
                'dtype': 'float64',
                'zlib': True,
                'complevel': 1,
                '_FillValue': None
            }

            dat.to_netcdf(output_path, encoding=encoding)

            # Explicit cleanup
            dat.close()
            del dat

    def _fix_time_coordinate_comprehensive(self, dataset: xr.Dataset, filename: str) -> xr.Dataset:
        """
        Fix time coordinate to ensure SUMMA compatibility using only the data's time coordinate.
        No filename parsing - just uses the actual time data which is always authoritative.

        Args:
            dataset: Input dataset
            filename: Filename for logging

        Returns:
            Dataset with corrected time coordinate
        """
        try:
            # Check if time exists in the dataset
            if 'time' not in dataset.dims and 'time' not in dataset.coords:
                raise ValueError(f"Dataset has no 'time' dimension or coordinate")

            # Use bracket notation to access time safely
            time_coord = dataset['time']

            self.logger.debug(f"File {filename}: Original time dtype: {time_coord.dtype}")

            # Convert any time format to pandas datetime first
            if time_coord.dtype.kind == 'M':  # datetime64
                pd_times = pd.to_datetime(time_coord.values)
            elif np.issubdtype(time_coord.dtype, np.number):
                if 'units' in time_coord.attrs and 'since' in time_coord.attrs['units']:
                    # Parse existing time units to understand the reference
                    units_str = time_coord.attrs['units']
                    if 'since' in units_str:
                        reference_str = units_str.split('since ')[1]
                        pd_times = pd.to_datetime(time_coord.values, unit='s', origin=reference_str)
                    else:
                        # Assume seconds since unix epoch if no reference given
                        pd_times = pd.to_datetime(time_coord.values, unit='s')
                else:
                    # Try to interpret as seconds since unix epoch
                    pd_times = pd.to_datetime(time_coord.values, unit='s')
            else:
                # Try direct conversion
                pd_times = pd.to_datetime(time_coord.values)

            self.logger.debug(f"File {filename}: Time range from data: {pd_times[0]} to {pd_times[-1]}")

            # Get time step from config
            time_step_seconds = int(self.config.get('FORCING_TIME_STEP_SIZE', 3600))
            num_steps = len(pd_times)

            # Convert to SUMMA's expected format: seconds since 1990-01-01 00:00:00
            reference_date = pd.Timestamp('1990-01-01 00:00:00')
            seconds_since_ref = (pd_times - reference_date).total_seconds().values

            # Ensure perfect integer seconds to avoid floating point precision issues
            seconds_since_ref = np.round(seconds_since_ref).astype(np.int64).astype(np.float64)

            # Replace the time coordinate
            dataset = dataset.assign_coords(time=seconds_since_ref)
            
            # Ensure time is monotonic
            dataset = dataset.sortby('time')

            # Set proper attributes for SUMMA
            dataset.time.attrs = {
                'units': 'seconds since 1990-01-01 00:00:00',
                'calendar': 'standard',
                'long_name': 'time',
                'axis': 'T'
            }

            self.logger.debug(f"File {filename}: Final time range: {seconds_since_ref[0]:.0f} to {seconds_since_ref[-1]:.0f} seconds")

            # Validate the conversion
            if len(seconds_since_ref) == 0:
                raise ValueError(f"Empty time coordinate after conversion")

            if np.any(np.isnan(seconds_since_ref)):
                raise ValueError(f"NaN values in converted time coordinate")

            # Check time step consistency (but don't force it - preserve actual data timing)
            if len(seconds_since_ref) > 1:
                time_diffs = np.diff(seconds_since_ref)
                expected_step = time_step_seconds

                # Check if most time steps match expected (allowing for some variability)
                step_matches = np.abs(time_diffs - expected_step) < (expected_step * 0.01)  # 1% tolerance
                match_percentage = np.sum(step_matches) / len(step_matches) * 100

                if match_percentage < 90:
                    self.logger.warning(f"File {filename}: Only {match_percentage:.1f}% of time steps match expected step size")
                    actual_median_step = int(np.median(time_diffs))
                    self.logger.warning(
                        f"File {filename}: Expected step: {expected_step}s, Actual median: {actual_median_step:.0f}s"
                    )
                    if actual_median_step > 0 and abs(actual_median_step - expected_step) > expected_step * 0.01:
                        self.logger.info(
                            f"File {filename}: Updating data_step from {self.data_step}s to {actual_median_step}s "
                            f"based on actual forcing timestep"
                        )
                        self.data_step = actual_median_step
                else:
                    self.logger.debug(f"File {filename}: Time steps are consistent ({match_percentage:.1f}% match)")

            return dataset

        except Exception as e:
            self.logger.error(f"File {filename}: Error fixing time coordinate: {str(e)}")
            raise ValueError(f"Cannot fix time coordinate in file {filename}: {str(e)}")

    def _fix_nan_values(self, dataset: xr.Dataset, filename: str) -> xr.Dataset:
        """
        Fix NaN values in forcing data through interpolation and filling.
        Handles CASR data pattern where only every 3rd temperature value is valid.

        Args:
            dataset: Input dataset
            filename: Filename for logging

        Returns:
            Dataset with NaN values filled
        """
        forcing_vars = ['airtemp', 'airpres', 'spechum', 'windspd', 'pptrate', 'LWRadAtm', 'SWRadAtm']

        for var in forcing_vars:
            if var not in dataset:
                continue

            var_data = dataset[var]

            # Count NaN values
            nan_count = np.isnan(var_data.values).sum()
            total_count = var_data.size

            if nan_count > 0:
                nan_percentage = (nan_count / total_count) * 100

                # Apply interpolation strategy based on variable type
                if var == 'pptrate':
                    # For precipitation, fill NaN with 0 (no precipitation)
                    filled_data = var_data.fillna(0.0)
                    self.logger.debug(f"File {filename}: Filled {var} NaN values with 0")

                elif var in ['SWRadAtm']:
                    # For solar radiation, interpolate during day, zero at night
                    filled_data = var_data.interpolate_na(dim='time', method='linear')
                    filled_data = filled_data.ffill(dim='time').bfill(dim='time')
                    filled_data = filled_data.fillna(0.0)
                    self.logger.debug(f"File {filename}: Interpolated {var} NaN values")

                elif var == 'airtemp' and nan_percentage > 50:
                    # Special handling for CASR temperature pattern (high NaN percentage)

                    # Use scipy cubic interpolation for better results with sparse temperature data
                    try:
                        from scipy import interpolate  # type: ignore
                        filled_data = var_data.copy()

                        # Process each HRU separately
                        for hru_idx in range(var_data.shape[-1] if len(var_data.shape) == 2 else 1):
                            if len(var_data.shape) == 2:
                                temp_values = var_data.values[:, hru_idx]
                            else:
                                temp_values = var_data.values

                            # Find valid (non-NaN) indices
                            valid_mask = ~np.isnan(temp_values)
                            valid_indices = np.where(valid_mask)[0]
                            valid_values = temp_values[valid_mask]

                            if len(valid_values) >= 2:
                                # Use cubic for smooth interpolation if enough points, otherwise linear
                                # Avoid extrapolation with cubic as it can produce wild values at boundaries
                                kind = 'cubic' if len(valid_values) >= 4 else 'linear'

                                f = interpolate.interp1d(
                                    valid_indices,
                                    valid_values,
                                    kind=kind,
                                    bounds_error=False,
                                    fill_value=(valid_values[0], valid_values[-1])
                                )

                                # Interpolate all time steps
                                all_indices = np.arange(len(temp_values))
                                interpolated_values = f(all_indices)

                                # Update the data
                                if len(var_data.shape) == 2:
                                    filled_data.values[:, hru_idx] = interpolated_values
                                else:
                                    filled_data.values[:] = interpolated_values
                            else:
                                # Not enough valid values, use default
                                if len(var_data.shape) == 2:
                                    filled_data.values[:, hru_idx] = PhysicalConstants.KELVIN_OFFSET  # 0°C
                                else:
                                    filled_data.values[:] = PhysicalConstants.KELVIN_OFFSET

                        # Clip to reasonable temperature range
                        filled_data = filled_data.clip(min=200.0, max=350.0)

                    except ImportError:
                        self.logger.warning(f"File {filename}: scipy not available, using xarray interpolation")
                        filled_data = var_data.interpolate_na(dim='time', method='linear')
                        filled_data = filled_data.ffill(dim='time').bfill(dim='time')
                        filled_data = filled_data.fillna(PhysicalConstants.KELVIN_OFFSET)
                        filled_data = filled_data.clip(min=200.0, max=350.0)

                    self.logger.debug(f"File {filename}: Applied CASR temperature interpolation")

                elif nan_percentage > 80:  # Only reject if >80% NaN for non-temperature variables
                    self.logger.error(f"File {filename}: Too many NaN values in {var} ({nan_percentage:.1f}%)")
                    raise ValueError(f"Variable {var} has too many NaN values to interpolate reliably")

                else:
                    # Standard interpolation for other variables
                    filled_data = var_data.interpolate_na(dim='time', method='linear')
                    filled_data = filled_data.ffill(dim='time').bfill(dim='time')

                    # If still NaN, use reasonable defaults
                    if np.any(np.isnan(filled_data.values)):
                        if var == 'airtemp':
                            default_val = PhysicalConstants.KELVIN_OFFSET  # 0°C in Kelvin
                        elif var == 'airpres':
                            default_val = 101325.0  # Standard pressure in Pa
                        elif var == 'spechum':
                            default_val = 0.005  # Reasonable specific humidity
                        elif var == 'windspd':
                            default_val = 2.0  # Light wind in m/s
                        elif var == 'LWRadAtm':
                            default_val = 300.0  # Reasonable longwave radiation
                        else:
                            default_val = 0.0

                        filled_data = filled_data.fillna(default_val)
                        self.logger.warning(f"File {filename}: Used default value {default_val} for remaining {var} NaN values")

                    self.logger.debug(f"File {filename}: Interpolated {var} NaN values")

                # Replace the variable in dataset
                dataset[var] = filled_data

                # Verify no NaN values remain
                remaining_nans = np.isnan(dataset[var].values).sum()
                if remaining_nans > 0:
                    self.logger.error(f"File {filename}: Still have {remaining_nans} NaN values in {var} after fixing")
                    raise ValueError(f"Failed to remove all NaN values from {var}")

        return dataset

    def _ensure_required_forcing_variables(self, dataset: xr.Dataset, filename: str) -> xr.Dataset:
        """
        Ensure all required forcing variables exist in the dataset.

        Some forcing products (e.g., CARRA) can omit variables like LWRadAtm.
        SUMMA expects a full set of forcing variables, so add missing variables
        with reasonable defaults and log a warning.
        """
        required_vars = ['airtemp', 'airpres', 'spechum', 'windspd', 'pptrate', 'LWRadAtm', 'SWRadAtm']
        defaults = {
            'airtemp': PhysicalConstants.KELVIN_OFFSET,   # 0°C in Kelvin
            'airpres': 101325.0, # Standard pressure in Pa
            'spechum': 0.005,    # Reasonable specific humidity
            'windspd': 2.0,      # Light wind in m/s
            'pptrate': 0.0,      # No precipitation
            'LWRadAtm': 300.0,   # Reasonable longwave radiation W/m^2
            'SWRadAtm': 0.0      # Default shortwave radiation W/m^2
        }
        units = {
            'airtemp': 'K',
            'airpres': 'Pa',
            'spechum': 'kg/kg',
            'windspd': 'm/s',
            'pptrate': 'mm/s',
            'LWRadAtm': 'W/m2',
            'SWRadAtm': 'W/m2'
        }

        time_len = len(dataset.time)
        hru_len = len(dataset.hru)

        for var in required_vars:
            if var in dataset:
                continue
            default_val = defaults[var]
            self.logger.warning(
                f"File {filename}: Missing {var}; filling with default {default_val}"
            )
            data = xr.DataArray(
                np.full((time_len, hru_len), default_val, dtype=np.float32),
                dims=('time', 'hru')
            )
            data.attrs.update({'units': units.get(var, '')})
            dataset[var] = data

        return dataset

    def _validate_and_fix_data_ranges(self, dataset: xr.Dataset, filename: str) -> xr.Dataset:
        """
        Validate and fix unrealistic data ranges that could cause SUMMA to fail.

        Args:
            dataset: Input dataset
            filename: Filename for logging

        Returns:
            Dataset with validated data ranges
        """
        # Define reasonable ranges for variables
        valid_ranges = {
            'airtemp': (200.0, 350.0),      # -73°C to 77°C
            'airpres': (50000.0, 110000.0), # 50-110 kPa
            'spechum': (0.0, 0.1),          # 0-100 g/kg
            'windspd': (0.0, 100.0),        # 0-100 m/s
            'pptrate': (0.0, 0.1),          # 0-360 mm/hr in mm/s
            'LWRadAtm': (50.0, 600.0),      # Longwave radiation W/m²
            'SWRadAtm': (0.0, 1500.0)       # Shortwave radiation W/m²
        }

        for var, (min_val, max_val) in valid_ranges.items():
            if var not in dataset:
                continue

            var_data = dataset[var]

            # Check for out-of-range values
            below_min = (var_data < min_val).sum()
            above_max = (var_data > max_val).sum()

            if below_min > 0 or above_max > 0:
                # Clip to valid range
                clipped_data = var_data.clip(min=min_val, max=max_val)
                dataset[var] = clipped_data

                self.logger.debug(f"File {filename}: Clipped {var} to range [{min_val}, {max_val}]")

        return dataset

    def _final_validation(self, dataset: xr.Dataset, filename: str):
        """
        Final validation to ensure dataset is ready for SUMMA.

        Args:
            dataset: Dataset to validate
            filename: Filename for logging
        """
        # Check time coordinate
        time_coord = dataset.time

        if not np.issubdtype(time_coord.dtype, np.number):
            raise ValueError(f"File {filename}: Time coordinate is not numeric after fixing")

        if 'units' not in time_coord.attrs or 'since' not in time_coord.attrs['units']:
            raise ValueError(f"File {filename}: Time coordinate missing proper units")

        # Check for any remaining NaN values in critical variables
        critical_vars = ['airtemp', 'airpres', 'spechum', 'windspd']
        for var in critical_vars:
            if var in dataset:
                nan_count = np.isnan(dataset[var].values).sum()
                if nan_count > 0:
                    raise ValueError(f"File {filename}: Variable {var} still has {nan_count} NaN values")

        # Check that all arrays have consistent shapes
        expected_shape = (len(dataset.time), len(dataset.hru))
        for var in dataset.data_vars:
            if var not in ['data_step', 'latitude', 'longitude', 'hruId'] and hasattr(dataset[var], 'shape'):
                if dataset[var].shape != expected_shape:
                    self.logger.warning(f"File {filename}: Variable {var} has unexpected shape {dataset[var].shape}, "
                                    f"expected {expected_shape}")

        self.logger.debug(f"File {filename}: Passed final validation for SUMMA compatibility")

    def _infer_forcing_step_from_filenames(self, forcing_files: List[str]) -> int | None:
        forcing_times = []
        for forcing_file in forcing_files:
            stem = Path(forcing_file).stem
            time_token = stem.split("_")[-1]
            try:
                forcing_times.append(datetime.strptime(time_token, "%Y-%m-%d-%H-%M-%S"))
            except ValueError:
                continue

        if len(forcing_times) < 2:
            return None

        forcing_times.sort()
        diffs = [
            (forcing_times[idx] - forcing_times[idx - 1]).total_seconds()
            for idx in range(1, len(forcing_times))
            if forcing_times[idx] > forcing_times[idx - 1]
        ]
        if not diffs:
            return None

        return int(np.median(diffs))

    def _determine_batch_size(self, total_files: int) -> int:
        """
        Determine optimal batch size based on available memory and file count.

        Args:
            total_files: Total number of files to process

        Returns:
            Optimal batch size
        """
        try:
            # Get available memory in MB
            available_memory = psutil.virtual_memory().available / 1024**2

            # Conservative estimate: assume each file uses ~50MB during processing
            # (this includes temporary arrays, xarray overhead, etc.)
            estimated_memory_per_file = 50

            # Use at most 70% of available memory for batch processing
            max_memory_for_batch = available_memory * 0.7

            # Calculate batch size based on memory constraint
            memory_based_batch_size = max(1, int(max_memory_for_batch / estimated_memory_per_file))

            # Set reasonable bounds
            min_batch_size = 1
            max_batch_size = min(100, total_files)  # Don't exceed 100 files per batch

            # Choose the most conservative estimate
            batch_size = max(min_batch_size, min(memory_based_batch_size, max_batch_size))

            self.logger.debug(f"Batch size calculation: available_memory={available_memory:.1f}MB, "
                            f"memory_based_size={memory_based_batch_size}, "
                            f"chosen_size={batch_size}")

            return batch_size

        except Exception as e:
            self.logger.warning(f"Could not determine optimal batch size: {str(e)}. Using default.")
            return min(10, total_files)  # Conservative fallback

    def create_forcing_file_list(self):
        """
        Create a list of forcing files for SUMMA.

        This method performs the following steps:
        1. Determine the forcing dataset from the configuration
        2. Find all relevant forcing files in the SUMMA input directory
        3. Sort the files to ensure chronological order
        4. Write the sorted file list to a text file

        The resulting file list is used by SUMMA to locate and read the forcing data.

        Raises:
            FileNotFoundError: If no forcing files are found.
            IOError: If there are issues writing the file list.
        """
        self.logger.info("Creating forcing file list")

        forcing_dataset = self.config.get("FORCING_DATASET")
        domain_name = self.config.get("DOMAIN_NAME")
        forcing_path = self.project_dir / "forcing" / "SUMMA_input"
        file_list_path = (
            self.setup_dir / self.config.get("SETTINGS_SUMMA_FORCING_LIST")
        )

        forcing_dataset_upper = forcing_dataset.upper()

        # All datasets we *know* about and expect to behave like the others
        supported_datasets = {
            "CARRA",
            "ERA5",
            "RDRS",
            "CASR",
            "AORC",
            "CONUS404",
            "NEX-GDDP-CMIP6",
            "HRRR",
        }

        if forcing_dataset_upper in supported_datasets:
            prefix = f"{domain_name}_{forcing_dataset}"
        else:
            # Fall back to a generic prefix so future datasets still work,
            # but emit a warning so we notice.
            self.logger.warning(
                "Forcing dataset %s is not in the supported list %s; "
                "using generic prefix '%s_' for SUMMA forcing files.",
                forcing_dataset,
                supported_datasets,
                domain_name,
            )
            prefix = f"{domain_name}_"

        self.logger.info(
            "Looking for SUMMA forcing files in %s with prefix '%s' and extension '.nc'",
            forcing_path,
            prefix,
        )

        if not forcing_path.exists():
            self.logger.error("Forcing SUMMA_input directory does not exist: %s", forcing_path)
            raise FileNotFoundError(f"SUMMA forcing directory not found: {forcing_path}")

        forcing_files = [
            f for f in os.listdir(forcing_path)
            if f.startswith(prefix) and f.endswith(".nc")
        ]

        if not forcing_files:
            self.logger.error(
                "No forcing files found for dataset %s in %s (prefix '%s')",
                forcing_dataset,
                forcing_path,
                prefix,
            )
            raise FileNotFoundError(
                f"No {forcing_dataset} forcing files found in {forcing_path}"
            )

        # Robust chronological sorting by extracting dates from filenames
        def extract_date(filename):
            import re
            # Pattern 1: YYYY-MM-DD-HH-MM-SS
            match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", filename)
            if match:
                try:
                    return datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")
                except ValueError:
                    pass
            # Pattern 2: YYYYMM at the end (before .nc)
            match = re.search(r"(\d{6})\.nc$", filename)
            if match:
                try:
                    return datetime.strptime(match.group(1), "%Y%m")
                except ValueError:
                    pass
            # Pattern 3: YYYYMMDD at the end
            match = re.search(r"(\d{8})\.nc$", filename)
            if match:
                try:
                    return datetime.strptime(match.group(1), "%Y%m%d")
                except ValueError:
                    pass
            # Fallback to a very old date for non-matching files
            return datetime(1900, 1, 1)

        # Sort and deduplicate (prefer files with longer names which usually contain full timestamps)
        forcing_files.sort(key=lambda x: (extract_date(x), -len(x)))
        
        unique_files = []
        seen_dates = set()
        for f in forcing_files:
            date = extract_date(f)
            if date not in seen_dates:
                unique_files.append(f)
                seen_dates.add(date)
            else:
                self.logger.warning(f"Skipping duplicate forcing file for date {date}: {f}")

        forcing_files = unique_files
        
        self.logger.info(
            "Found %d unique %s forcing files for SUMMA",
            len(forcing_files),
            forcing_dataset,
        )

        with open(file_list_path, "w") as fobj:
            for fname in forcing_files:
                fobj.write(f"{fname}\n")

        self.logger.info(
            "Forcing file list created at %s with %d files",
            file_list_path,
            len(forcing_files),
        )

    def _filter_forcing_hru_ids(self, forcing_hru_ids):
        """
        Filter forcing HRU IDs against catchment shapefile to ensure consistency.

        Args:
            forcing_hru_ids: List or array of HRU IDs from forcing data

        Returns:
            Filtered list of HRU IDs that exist in catchment shapefile
        """
        forcing_hru_ids = list(forcing_hru_ids)
        try:
            shp = gpd.read_file(self.catchment_path / self.catchment_name)
            shp = shp.set_index(self.config.get('CATCHMENT_SHP_HRUID'))
            shp.index = shp.index.astype(int)
            available_hru_ids = set(shp.index.astype(int))
        except Exception as exc:
            self.logger.warning(
                "Unable to filter forcing HRU IDs against catchment shapefile: %s",
                exc,
            )
            return forcing_hru_ids

        missing_hru_ids = [hru_id for hru_id in forcing_hru_ids if hru_id not in available_hru_ids]
        if missing_hru_ids:
            self.logger.warning(
                "Forcing HRU IDs not found in catchment shapefile; filtering missing IDs. "
                "Missing count: %s (showing first 10): %s",
                len(missing_hru_ids),
                missing_hru_ids[:10],
            )
            forcing_hru_ids = [hru_id for hru_id in forcing_hru_ids if hru_id in available_hru_ids]
        if len(forcing_hru_ids) == 0:
            raise ValueError("No forcing HRU IDs match catchment shapefile HRU IDs.")
        return forcing_hru_ids
