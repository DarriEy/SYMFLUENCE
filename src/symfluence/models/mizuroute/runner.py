"""
MizuRoute Model Runner.

Manages the execution of the mizuRoute routing model.
Refactored to use the Unified Model Execution Framework.
"""

import os
import sys
import pandas as pd
import subprocess
import traceback
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional, List

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelRunner
from symfluence.models.execution import ModelExecutor


@ModelRegistry.register_runner('MIZUROUTE', method_name='run_mizuroute')
class MizuRouteRunner(BaseModelRunner, ModelExecutor):
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
    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # MizuRoute uses 'root_path' alias for backwards compatibility
        self.setup_path_aliases({'root_path': 'data_dir'})

    def _get_model_name(self) -> str:
        """Return model name for MizuRoute."""
        return "MizuRoute"

    def _should_create_output_dir(self) -> bool:
        """MizuRoute creates directories on-demand."""
        return False

    def fix_time_precision(self):
        """
        Fix model output time precision by rounding to nearest hour.
        This fixes compatibility issues with mizuRoute time matching.
        Now supports both SUMMA and FUSE outputs with proper time format detection.
        """
        # Determine which model's output to process
        models_raw = self.config_dict.get('HYDROLOGICAL_MODEL', '')
        mizu_from = self.config_dict.get('MIZU_FROM_MODEL', '')
        
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
            experiment_output_fuse = self.config_dict.get('EXPERIMENT_OUTPUT_FUSE', 'default')
            if experiment_output_fuse == 'default' or not experiment_output_fuse:
                # Try relative to current experiment first
                experiment_id = self.config_dict.get('EXPERIMENT_ID')
                experiment_output_dir = self.project_dir / f"simulations/{experiment_id}" / 'FUSE'
            else:
                experiment_output_dir = Path(experiment_output_fuse)
            runoff_filename = f"{self.config_dict.get('DOMAIN_NAME')}_{self.config_dict.get('EXPERIMENT_ID')}_runs_def.nc"
        elif 'GR' in active_models:
            self.logger.info("Fixing GR time precision for mizuRoute compatibility")
            experiment_output_gr = self.config_dict.get('EXPERIMENT_OUTPUT_GR', 'default')
            if experiment_output_gr == 'default' or not experiment_output_gr:
                # Try relative to current experiment first
                experiment_id = self.config_dict.get('EXPERIMENT_ID')
                experiment_output_dir = self.project_dir / f"simulations/{experiment_id}" / 'GR'
            else:
                experiment_output_dir = Path(experiment_output_gr)
            runoff_filename = f"{self.config_dict.get('DOMAIN_NAME')}_{self.config_dict.get('EXPERIMENT_ID')}_runs_def.nc"
        elif 'HYPE' in active_models:
            self.logger.info("Fixing HYPE time precision for mizuRoute compatibility")
            experiment_output_hype = self.config_dict.get('EXPERIMENT_OUTPUT_HYPE', 'default')
            if experiment_output_hype == 'default' or not experiment_output_hype:
                # Try relative to current experiment first
                experiment_id = self.config_dict.get('EXPERIMENT_ID')
                experiment_output_dir = self.project_dir / f"simulations/{experiment_id}" / 'HYPE'
            else:
                experiment_output_dir = Path(experiment_output_hype)
            runoff_filename = f"{self.config_dict.get('EXPERIMENT_ID')}_timestep.nc"
        else:
            self.logger.info(f"Fixing SUMMA time precision for mizuRoute compatibility (Active models: {active_models})")
            experiment_output_summa = self.config_dict.get('EXPERIMENT_OUTPUT_SUMMA', 'default')
            if experiment_output_summa == 'default' or not experiment_output_summa:
                # Try relative to current experiment first
                experiment_id = self.config_dict.get('EXPERIMENT_ID')
                experiment_output_dir = self.project_dir / f"simulations/{experiment_id}" / 'SUMMA'
            else:
                experiment_output_dir = Path(experiment_output_summa)
            runoff_filename = f"{self.config_dict.get('EXPERIMENT_ID')}_timestep.nc"
        
        runoff_filepath = experiment_output_dir / runoff_filename
        self.logger.info(f"Resolved runoff filepath: {runoff_filepath} (Exists: {runoff_filepath.exists()})")
        
        if not runoff_filepath.exists():
            self.logger.warning(f"Model output file not found: {runoff_filepath}. Checking if any other output files exist in {experiment_output_dir}...")
            if experiment_output_dir.exists():
                nc_files = list(experiment_output_dir.glob("*.nc"))
                if nc_files:
                    runoff_filepath = nc_files[0]
                    self.logger.info(f"Using fallback output file: {runoff_filepath}")
                else:
                    self.logger.error(f"No NetCDF output files found in {experiment_output_dir}")
                    return
            else:
                self.logger.error(f"Output directory does not exist: {experiment_output_dir}")
                return
        
        try:
            import xarray as xr
            import os
            
            self.logger.debug(f"Processing {runoff_filepath}")
            
            # Open dataset and examine time format
            ds = xr.open_dataset(runoff_filepath, decode_times=False)
            
            # Detect the time format by examining attributes and values
            time_attrs = ds.time.attrs
            time_values = ds.time.values
            
            self.logger.debug(f"Time units: {time_attrs.get('units', 'No units specified')}")
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
                            first_time = pd.to_datetime(time_values[0], unit='h', origin=ref_time_str)
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
                    except:
                        self.logger.warning("Could not determine time format - skipping time precision fix")
                        ds.close()
                        return
            else:
                # No units attribute - try to decode directly
                try:
                    ds_decoded = xr.open_dataset(runoff_filepath, decode_times=True)
                    first_time = pd.Timestamp(ds_decoded.time.values[0])
                    rounded_time = first_time.round('h')
                    needs_fix = (first_time != rounded_time)
                    ds_decoded.close()
                    time_format_detected = 'datetime64'
                except:
                    self.logger.warning("No time units and cannot decode times - skipping time precision fix")
                    ds.close()
                    return
            
            self.logger.debug(f"Detected time format: {time_format_detected}")
            self.logger.debug(f"Needs time precision fix: {needs_fix}")
            
            if not needs_fix:
                self.logger.debug("Time precision is already correct")
                ds.close()
                return
            
            # Apply the appropriate fix based on detected format
            if time_format_detected == 'summa_seconds_1990':
                # Original SUMMA logic
                self.logger.info("Applying SUMMA-style time precision fix")
                time_stamps = pd.to_datetime(time_values, unit='s', origin='1990-01-01')
                rounded_stamps = time_stamps.round('h')
                reference = pd.Timestamp('1990-01-01')
                rounded_seconds = (rounded_stamps - reference).total_seconds().values
                
                ds = ds.assign_coords(time=rounded_seconds)
                ds.time.attrs.clear()
                ds.time.attrs['units'] = 'seconds since 1990-01-01 00:00:00'
                ds.time.attrs['calendar'] = 'standard'
                ds.time.attrs['long_name'] = 'time'
                
            elif time_format_detected.startswith('generic_since_'):
                # Generic 'since' format
                self.logger.info(f"Applying generic time precision fix for format: {time_format_detected}")
                ref_time_str = time_format_detected.split('generic_since_')[1]
                
                time_stamps = pd.to_datetime(time_values, unit=time_unit, origin=ref_time_str)
                rounded_stamps = time_stamps.round('h')
                reference = pd.Timestamp(ref_time_str)
                
                if time_unit == 's':
                    rounded_values = (rounded_stamps - reference).total_seconds().values
                elif time_unit == 'h':
                    rounded_values = (rounded_stamps - reference) / pd.Timedelta(hours=1)
                elif time_unit == 'D':
                    rounded_values = (rounded_stamps - reference) / pd.Timedelta(days=1)
                
                ds = ds.assign_coords(time=rounded_values)
                ds.time.attrs.clear()
                ds.time.attrs['units'] = f"{time_unit} since {ref_time_str}"
                ds.time.attrs['calendar'] = 'standard'
                ds.time.attrs['long_name'] = 'time'
                
            elif time_format_detected == 'datetime64':
                # Already in datetime format, just round
                self.logger.info("Applying datetime64 time precision fix")
                ds_decoded = xr.open_dataset(runoff_filepath, decode_times=True)
                time_stamps = pd.to_datetime(ds_decoded.time.values)
                rounded_stamps = time_stamps.round('h')
                
                # Keep original format but with rounded times
                ds = ds_decoded.assign_coords(time=rounded_stamps)
                ds_decoded.close()
            
            # Save the corrected file
            ds.load()
            
            os.chmod(runoff_filepath, 0o664)
            ds.to_netcdf(runoff_filepath, format='NETCDF4')
            ds.close()
            self.logger.info("Time precision fixed successfully")
            
        except Exception as e:
            self.logger.error(f"Error fixing time precision: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def run_mizuroute(self):
        """
        Run the mizuRoute model.

        This method sets up the necessary paths, executes the mizuRoute model,
        and handles any errors that occur during the run.
        """
        self.logger.debug("Starting mizuRoute run")
        self.fix_time_precision()

        # Set up paths and filenames
        self.mizu_exe = self.get_model_executable(
            install_path_key='INSTALL_PATH_MIZUROUTE',
            default_install_subpath='installs/mizuRoute/route/bin',
            exe_name_key='EXE_NAME_MIZUROUTE',
            default_exe_name='mizuroute.exe',
            must_exist=True
        )
        settings_path = self.get_config_path('SETTINGS_MIZU_PATH', 'settings/mizuRoute/')
        control_file = self.config_dict.get('SETTINGS_MIZU_CONTROL_FILE')
        
        # Sane defaults for control file if not specified
        if not control_file or control_file == 'default':
            mizu_from = self.config_dict.get('MIZU_FROM_MODEL', '').upper()
            if mizu_from == 'GR':
                control_file = 'mizuRoute_control_GR.txt'
            elif mizu_from == 'FUSE':
                control_file = 'mizuRoute_control_FUSE.txt'
            else:
                control_file = 'mizuroute.control'
            self.logger.debug(f"Using default mizuRoute control file: {control_file}")

        experiment_id = self.config_dict.get('EXPERIMENT_ID')
        mizu_log_path = self.get_config_path('EXPERIMENT_LOG_MIZUROUTE', f"simulations/{experiment_id}/mizuRoute/mizuRoute_logs/")
        mizu_log_name = "mizuRoute_log.txt"

        mizu_out_path = self.get_config_path('EXPERIMENT_OUTPUT_MIZUROUTE', f"simulations/{experiment_id}/mizuRoute/")

        # Backup settings if required
        if self.config_dict.get('EXPERIMENT_BACKUP_SETTINGS') == 'yes':
            self.backup_settings(settings_path, backup_subdir="run_settings")

        # Run mizuRoute
        mizu_log_path.mkdir(parents=True, exist_ok=True)
        mizu_command = f"{self.mizu_exe} {settings_path / control_file}"
        self.logger.debug(f'Running mizuRoute with command: {mizu_command}')

        self.execute_model_subprocess(
            mizu_command,
            mizu_log_path / mizu_log_name,
            shell=True,
            success_message="mizuRoute run completed successfully"
        )

        return mizu_out_path

