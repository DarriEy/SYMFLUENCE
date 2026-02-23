"""
MizuRoute Model Runner.

Manages the execution of the mizuRoute routing model.
Refactored to use the Unified Model Execution Framework.
"""

import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.models.base import BaseModelRunner
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_runner('MIZUROUTE', method_name='run_mizuroute')
class MizuRouteRunner(BaseModelRunner):  # type: ignore[misc]
    """
    A class to run the mizuRoute model.

    This class handles the execution of the mizuRoute model, including setting up paths,
    running the model, and managing log files.

    Uses the Unified Model Execution Framework for subprocess execution.

    Attributes:

        config (Dict[str, Any]): Configuration settings for the model run.
        logger (Any): Logger object for recording run information.
        root_path (Path): Root path for the project.
        domain_name (str): Name of the domain being processed.
        project_dir (Path): Directory for the current project.
    """

    MODEL_NAME = "MizuRoute"
    def __init__(self, config: Dict[str, Any], logger: logging.Logger, reporting_manager: Optional[Any] = None):
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # MizuRoute uses 'root_path' alias for backwards compatibility
        self.setup_path_aliases({'root_path': 'data_dir'})

    def _should_create_output_dir(self) -> bool:
        """MizuRoute creates directories on-demand."""
        return False

    def _get_time_rounding_freq(self) -> Optional[str]:
        """Get time rounding frequency from config.

        Returns:
            Rounding frequency string (e.g., 'h', 'min', 's') or None if disabled.
        """
        freq = self._get_config_value(
            lambda: self.config.model.mizuroute.time_rounding_freq, default='h'
        )
        if freq and freq.lower() == 'none':
            return None
        return freq

    def fix_time_precision(self) -> Optional[Path]:
        """
        Fix model output time precision by rounding to configurable frequency.
        This fixes compatibility issues with mizuRoute time matching.
        Now supports both SUMMA and FUSE outputs with proper time format detection.
        Rounding frequency can be configured via MIZUROUTE_TIME_ROUNDING_FREQ
        (default: 'h' for hour, use 'none' to disable rounding).
        Returns the path to the runoff file if resolved, None otherwise.
        """
        # Determine which model's output to process
        models_raw = self.hydrological_model or ''
        mizu_from = self._get_config_value(
            lambda: self.config.model.mizuroute.from_model, default=''
        )

        # Combine models and filter out 'DEFAULT' or empty strings
        all_models = f"{models_raw},{mizu_from}".split(',')
        active_models = sorted(list(set([
            m.strip().upper() for m in all_models
            if m.strip() and m.strip().upper() != 'DEFAULT'
        ])))

        self.logger.debug(f"Detected active models for time precision fix: {active_models}")

        # For FUSE, check if it has already converted its output
        if 'FUSE' in active_models:
            self.logger.info("Fixing FUSE time precision for mizuRoute compatibility")
            experiment_output_fuse = self._get_config_value(
                lambda: self.config.model.fuse.experiment_output, default='default'
            )
            if experiment_output_fuse == 'default' or not experiment_output_fuse:
                experiment_output_dir = self.project_dir / f"simulations/{self.experiment_id}" / 'FUSE'
            else:
                experiment_output_dir = Path(experiment_output_fuse)
            fuse_file_id = self._get_config_value(
                lambda: self.config.model.fuse.file_id, default=None
            )
            if not fuse_file_id:
                fuse_file_id = self.experiment_id or 'fuse'
                # Replicate FUSE preprocessor's 6-char truncation for Fortran compatibility
                if len(fuse_file_id) > 6:
                    import hashlib
                    fuse_file_id = hashlib.md5(fuse_file_id.encode(), usedforsecurity=False).hexdigest()[:6]
            runoff_filename = f"{self.domain_name}_{fuse_file_id}_runs_def.nc"
        elif 'GR' in active_models:
            self.logger.info("Fixing GR time precision for mizuRoute compatibility")
            experiment_output_gr = self._get_config_value(lambda: None, default='default', dict_key='EXPERIMENT_OUTPUT_GR')
            if experiment_output_gr == 'default' or not experiment_output_gr:
                experiment_output_dir = self.project_dir / f"simulations/{self.experiment_id}" / 'GR'
            else:
                experiment_output_dir = Path(experiment_output_gr)
            runoff_filename = f"{self.domain_name}_{self.experiment_id}_runs_def.nc"
        elif 'HYPE' in active_models:
            self.logger.info("Fixing HYPE time precision for mizuRoute compatibility")
            experiment_output_hype = self._get_config_value(lambda: None, default='default', dict_key='EXPERIMENT_OUTPUT_HYPE')
            if experiment_output_hype == 'default' or not experiment_output_hype:
                experiment_output_dir = self.project_dir / f"simulations/{self.experiment_id}" / 'HYPE'
            else:
                experiment_output_dir = Path(experiment_output_hype)
            runoff_filename = f"{self.experiment_id}_timestep.nc"
        else:
            self.logger.info(f"Fixing SUMMA time precision for mizuRoute compatibility (Active models: {active_models})")
            experiment_output_summa = self._get_config_value(
                lambda: self.config.model.summa.experiment_output, default='default'
            )
            if experiment_output_summa == 'default' or not experiment_output_summa:
                experiment_output_dir = self.project_dir / f"simulations/{self.experiment_id}" / 'SUMMA'
            else:
                experiment_output_dir = Path(experiment_output_summa)
            runoff_filename = f"{self.experiment_id}_timestep.nc"

        runoff_filepath = experiment_output_dir / runoff_filename
        self.logger.info(f"Resolved runoff filepath: {runoff_filepath} (Exists: {runoff_filepath.exists()})")

        if not runoff_filepath.exists():
            self.logger.warning(f"Model output file not found: {runoff_filepath}. Checking if any other output files exist in {experiment_output_dir}...")
            if experiment_output_dir.exists():
                nc_files = [f for f in experiment_output_dir.glob("*.nc") if '_para_' not in f.name]
                if nc_files:
                    runoff_filepath = nc_files[0]
                    self.logger.info(f"Using fallback output file: {runoff_filepath}")
                else:
                    self.logger.error(f"No NetCDF output files found in {experiment_output_dir}")
                    return None
            else:
                self.logger.error(f"Output directory does not exist: {experiment_output_dir}")
                return None

        try:
            import os

            import xarray as xr

            self.logger.debug(f"Processing {runoff_filepath}")

            # Open dataset and examine time format
            try:
                ds = xr.open_dataset(runoff_filepath, decode_times=False)
            except (OSError, RuntimeError, ValueError) as nc_err:
                self.logger.error(f"Failed to open model output file: {runoff_filepath}")
                self.logger.error("The file appears to be corrupt or incomplete.")
                self.logger.error("This usually happens when the upstream hydrological model (e.g., SUMMA) fails or times out before finishing.")
                self.logger.error(f"Underlying error: {nc_err}")
                raise

            # Detect the time format by examining attributes and values
            time_attrs = ds.time.attrs
            time_values = ds.time.values

            self.logger.debug(f"Time units: {time_attrs.get('units', 'No units specified')}")

            # Check if time_values is empty
            if len(time_values) == 0:
                self.logger.error(f"Time dimension in {runoff_filepath} is empty (no time values found)")
                self.logger.error("This indicates the upstream model output is incomplete or corrupted")
                self.logger.error("Please verify the model run completed successfully and produced valid output")
                ds.close()
                raise ValueError(f"Empty time dimension in model output: {runoff_filepath}")

            self.logger.debug(f"Time range: {time_values.min()} to {time_values.max()}")

            # Check if time precision fix is needed and determine format
            needs_fix = False
            time_format_detected = None

            if 'units' in time_attrs:
                units_str = time_attrs['units'].lower()

                if 'since 1990-01-01' in units_str:
                    # SUMMA-style format: seconds since 1990-01-01
                    time_format_detected = 'summa_seconds_1990'
                    first_time = pd.to_datetime(time_values[0], unit='s', origin='1990-01-01')
                    rounded_time = first_time.round('h')
                    needs_fix = (first_time != rounded_time)

                elif 'since' in units_str:
                    # Other reference time format - extract the reference
                    import re
                    since_match = re.search(r'since\s+([0-9-]+(?:\s+[0-9:]+)?)', units_str)
                    if since_match:
                        ref_time_str = since_match.group(1).strip()
                        time_format_detected = f'generic_since_{ref_time_str}'

                        # Determine the unit (seconds, hours, days)
                        if 'second' in units_str:
                            first_time = pd.to_datetime(time_values[0], unit='s', origin=ref_time_str)
                            time_unit = 's'
                        elif 'hour' in units_str:
                            first_time = pd.Timestamp(ref_time_str) + pd.Timedelta(hours=float(time_values[0]))
                            time_unit = 'h'
                        elif 'day' in units_str:
                            first_time = pd.to_datetime(time_values[0], unit='D', origin=ref_time_str)
                            time_unit = 'D'
                        else:
                            # Default to seconds
                            first_time = pd.to_datetime(time_values[0], unit='s', origin=ref_time_str)
                            time_unit = 's'

                        rounded_time = first_time.round('h')
                        needs_fix = (first_time != rounded_time)

                else:
                    # No 'since' found, might be already in datetime format
                    time_format_detected = 'unknown'
                    try:
                        # Try to interpret as datetime directly
                        ds_decoded = xr.open_dataset(runoff_filepath, decode_times=True)
                        first_time = pd.Timestamp(ds_decoded.time.values[0])
                        rounded_time = first_time.round('h')
                        needs_fix = (first_time != rounded_time)
                        ds_decoded.close()
                        time_format_detected = 'datetime64'
                    except (ValueError, TypeError, KeyError) as e:
                        self.logger.warning(f"Could not determine time format - skipping time precision fix: {e}")
                        ds.close()
                        return runoff_filepath
            else:
                # No units attribute - try to decode directly
                try:
                    ds_decoded = xr.open_dataset(runoff_filepath, decode_times=True)
                    first_time = pd.Timestamp(ds_decoded.time.values[0])
                    rounded_time = first_time.round('h')
                    needs_fix = (first_time != rounded_time)
                    ds_decoded.close()
                    time_format_detected = 'datetime64'
                except (ValueError, TypeError, KeyError) as e:
                    self.logger.warning(f"No time units and cannot decode times - skipping time precision fix: {e}")
                    ds.close()
                    return runoff_filepath

            self.logger.debug(f"Detected time format: {time_format_detected}")
            self.logger.debug(f"Needs time precision fix: {needs_fix}")

            if not needs_fix:
                self.logger.debug("Time precision is already correct")
                ds.close()
                return runoff_filepath

            # Apply the appropriate fix based on detected format
            if time_format_detected == 'summa_seconds_1990':
                # Original SUMMA logic
                rounding_freq = self._get_time_rounding_freq()
                self.logger.info(f"Applying SUMMA-style time precision fix (rounding to '{rounding_freq}')")
                time_stamps = pd.to_datetime(time_values, unit='s', origin='1990-01-01')
                if rounding_freq:
                    rounded_stamps = time_stamps.round(rounding_freq)
                    max_shift = np.abs(rounded_stamps - time_stamps).max()
                    if max_shift > pd.Timedelta(minutes=5):
                        self.logger.info(f"Time rounding applied, max shift: {max_shift}")
                else:
                    rounded_stamps = time_stamps  # No rounding
                reference = pd.Timestamp('1990-01-01')
                rounded_seconds = (rounded_stamps - reference).total_seconds().values

                ds = ds.assign_coords(time=rounded_seconds)
                ds.time.attrs.clear()
                ds.time.attrs['units'] = 'seconds since 1990-01-01 00:00:00'
                ds.time.attrs['calendar'] = 'standard'
                ds.time.attrs['long_name'] = 'time'

            elif time_format_detected.startswith('generic_since_'):
                # Generic 'since' format
                rounding_freq = self._get_time_rounding_freq()
                self.logger.info(f"Applying generic time precision fix for format: {time_format_detected} (rounding to '{rounding_freq}')")
                ref_time_str = time_format_detected.split('generic_since_')[1]

                reference = pd.Timestamp(ref_time_str)
                # Convert time values based on unit
                if time_unit == 's':
                    time_stamps = reference + pd.to_timedelta(time_values, unit='s')
                elif time_unit == 'h':
                    time_stamps = reference + pd.to_timedelta(time_values, unit='h')
                elif time_unit == 'D':
                    time_stamps = reference + pd.to_timedelta(time_values, unit='D')
                else:
                    time_stamps = reference + pd.to_timedelta(time_values, unit='s')
                if rounding_freq:
                    rounded_stamps = time_stamps.round(rounding_freq)
                    max_shift = np.abs(rounded_stamps - time_stamps).max()
                    if max_shift > pd.Timedelta(minutes=5):
                        self.logger.info(f"Time rounding applied, max shift: {max_shift}")
                else:
                    rounded_stamps = time_stamps  # No rounding

                if time_unit == 's':
                    rounded_values = (rounded_stamps - reference).total_seconds().values
                elif time_unit == 'h':
                    rounded_values = (rounded_stamps - reference) / pd.Timedelta(hours=1)
                elif time_unit == 'D':
                    rounded_values = (rounded_stamps - reference) / pd.Timedelta(days=1)

                ds = ds.assign_coords(time=rounded_values)
                ds.time.attrs.clear()
                # Normalize unit name and reference time for CF/mizuRoute compliance
                _unit_names = {'s': 'seconds', 'h': 'hours', 'D': 'days'}
                full_unit_name = _unit_names.get(time_unit, time_unit)
                normalized_ref = reference.strftime('%Y-%m-%d %H:%M:%S')
                ds.time.attrs['units'] = f"{full_unit_name} since {normalized_ref}"
                ds.time.attrs['calendar'] = 'standard'
                ds.time.attrs['long_name'] = 'time'

            elif time_format_detected == 'datetime64':
                # Already in datetime format, just round
                rounding_freq = self._get_time_rounding_freq()
                self.logger.info(f"Applying datetime64 time precision fix (rounding to '{rounding_freq}')")
                ds_decoded = xr.open_dataset(runoff_filepath, decode_times=True)
                time_stamps = pd.to_datetime(ds_decoded.time.values)
                if rounding_freq:
                    rounded_stamps = time_stamps.round(rounding_freq)
                    max_shift = np.abs(rounded_stamps - time_stamps).max()
                    if max_shift > pd.Timedelta(minutes=5):
                        self.logger.info(f"Time rounding applied, max shift: {max_shift}")
                else:
                    rounded_stamps = time_stamps  # No rounding

                # Keep original format but with rounded times
                ds = ds_decoded.assign_coords(time=rounded_stamps)
                ds_decoded.close()

            # Save the corrected file safely using a temp file
            ds.load()
            temp_filepath = runoff_filepath.with_suffix('.tmp.nc')

            # Ensure permissions are set on temp file after creation
            ds.to_netcdf(temp_filepath, format='NETCDF4')
            ds.close()

            if os.name != 'nt':
                os.chmod(temp_filepath, 0o664)  # nosec B103 - Group-writable for HPC shared access
            temp_filepath.replace(runoff_filepath)

            self.logger.info("Time precision fixed successfully")
            return runoff_filepath

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error fixing time precision: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def sync_control_file_dimensions(self, control_path: Path, netcdf_path: Path):
        """
        Ensure mizuRoute control file dimension/variable names match the NetCDF input.
        This prevents hangs/crashes when preprocessor assumes 'gru' but SUMMA outputs 'hru'.
        """
        try:
            import xarray as xr
            self.logger.debug(f"Syncing control file dimensions for {netcdf_path}")

            with xr.open_dataset(netcdf_path, decode_times=False) as ds:
                dname = None
                # Detect dimension name
                if 'gru' in ds.dims:
                    dname = 'gru'
                elif 'hru' in ds.dims:
                    dname = 'hru'
                else:
                    self.logger.warning(f"Could not find 'gru' or 'hru' dimension in {netcdf_path}. Available: {list(ds.dims)}")

                # Detect ID variable
                vname = None
                if 'gruId' in ds.variables:
                    vname = 'gruId'
                elif 'hruId' in ds.variables:
                    vname = 'hruId'
                # fallback checks
                elif 'gru_id' in ds.variables:
                    vname = 'gru_id'
                elif 'hru_id' in ds.variables:
                    vname = 'hru_id'

                if dname and not vname:
                     self.logger.warning(f"Could not find ID variable in {netcdf_path}")
                     # Try to find integer variable with same name as dim?
                     if dname in ds.variables:
                         vname = dname

            if dname and vname:
                self.logger.debug(f"Detected in NetCDF: dimension='{dname}', variable='{vname}'")

                # Read control file
                with open(control_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                new_lines = []
                modified = False
                for line in lines:
                    if '<dname_hruid>' in line:
                        # Check if update is needed
                        if dname not in line:
                            parts = line.split('!')
                            comment = '!' + parts[1] if len(parts) > 1 else ''
                            new_lines.append(f"<dname_hruid>           {dname}    {comment}")
                            modified = True
                        else:
                            new_lines.append(line)
                    elif '<vname_hruid>' in line:
                         # Check if update is needed
                        if vname not in line:
                            parts = line.split('!')
                            comment = '!' + parts[1] if len(parts) > 1 else ''
                            new_lines.append(f"<vname_hruid>           {vname}    {comment}")
                            modified = True
                        else:
                            new_lines.append(line)
                    else:
                        new_lines.append(line)

                # Write back if modified
                if modified:
                    self.logger.info(f"Updating control file to use dimension '{dname}' and variable '{vname}'")
                    with open(control_path, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                else:
                    self.logger.debug("Control file already matches NetCDF dimensions.")
            else:
                self.logger.warning("Could not determine dimensions to sync.")

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error syncing control file dimensions: {e}")

    def run_mizuroute(self):
        """
        Run the mizuRoute model.

        This method sets up the necessary paths, executes the mizuRoute model,
        and handles any errors that occur during the run.
        """
        self.logger.debug("Starting mizuRoute run")

        with symfluence_error_handler(
            "mizuRoute model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            runoff_path = self.fix_time_precision()

            # Set up paths and filenames
            # Legacy keys (INSTALL_PATH_MIZUROUTE, EXE_NAME_MIZUROUTE) are handled
            # by DEPRECATED_KEYS in transformers.py and mapped to the standard keys.
            self.mizu_exe = self.get_model_executable(
                install_path_key='MIZUROUTE_INSTALL_PATH',
                default_install_subpath='installs/mizuRoute/route/bin',
                exe_name_key='MIZUROUTE_EXE',
                default_exe_name='mizuRoute.exe',
                must_exist=True
            )
            settings_path = self.get_config_path('SETTINGS_MIZU_PATH', 'settings/mizuRoute/')
            control_file = self._get_config_value(
                lambda: self.config.model.mizuroute.control_file, default='mizuroute.control'
            )

            # Sane defaults for control file if not specified
            if not control_file or control_file == 'default':
                mizu_from = self._get_config_value(
                    lambda: self.config.model.mizuroute.from_model, default=''
                ).upper()
                if mizu_from == 'GR':
                    control_file = 'mizuRoute_control_GR.txt'
                elif mizu_from == 'FUSE':
                    control_file = 'mizuRoute_control_FUSE.txt'
                else:
                    control_file = 'mizuroute.control'
                self.logger.debug(f"Using default mizuRoute control file: {control_file}")

            # Sync control file dimensions with actual runoff file
            if runoff_path and runoff_path.exists():
                control_path = settings_path / control_file
                if control_path.exists():
                    self.sync_control_file_dimensions(control_path, runoff_path)
                else:
                    self.logger.warning(f"Control file not found at {control_path}, skipping dimension sync")

            mizu_log_path = self.get_config_path('EXPERIMENT_LOG_MIZUROUTE', f"simulations/{self.experiment_id}/mizuRoute/mizuRoute_logs/")
            mizu_log_name = "mizuRoute_log.txt"

            mizu_out_path = self.get_config_path('EXPERIMENT_OUTPUT_MIZUROUTE', f"simulations/{self.experiment_id}/mizuRoute/")

            # Backup settings if required
            backup = self._get_config_value(
                lambda: self.config.model.summa.backup_settings, default='no',
                dict_key='EXPERIMENT_BACKUP_SETTINGS'
            )
            if backup == 'yes':
                self.backup_settings(settings_path, backup_subdir="run_settings")

            # Run mizuRoute
            mizu_log_path.mkdir(parents=True, exist_ok=True)
            mizu_command = [str(self.mizu_exe), str(settings_path / control_file)]
            self.logger.debug(f'Running mizuRoute with command: {" ".join(mizu_command)}')

            self.execute_subprocess(
                mizu_command,
                mizu_log_path / mizu_log_name,
                success_message="mizuRoute run completed successfully"
            )

            return mizu_out_path
