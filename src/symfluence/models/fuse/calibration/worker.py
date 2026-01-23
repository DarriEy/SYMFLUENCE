"""
FUSE Worker

Worker implementation for FUSE model optimization.
Delegates to existing worker functions while providing BaseWorker interface.
"""

import logging
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.core.constants import UnitConversion
from symfluence.models.utilities.routing_decider import RoutingDecider
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.models.fuse.utilities import FuseToMizurouteConverter

# Suppress xarray FutureWarning about timedelta64 decoding
warnings.filterwarnings('ignore',
                       message='.*decode_timedelta.*',
                       category=FutureWarning,
                       module='xarray.*')

logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('FUSE')
class FUSEWorker(BaseWorker):
    """
    Worker for FUSE model calibration.

    Handles parameter application to NetCDF files, FUSE execution,
    and metric calculation for streamflow calibration.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize FUSE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    # Shared utilities
    _routing_decider = RoutingDecider()
    _streamflow_metrics = StreamflowMetrics()
    _format_converter = FuseToMizurouteConverter()

    def needs_routing(self, config: Dict[str, Any], settings_dir: Optional[Path] = None) -> bool:
        """
        Determine if routing (mizuRoute) is needed for FUSE.

        Delegates to shared RoutingDecider utility.

        Args:
            config: Configuration dictionary
            settings_dir: Optional settings directory to check for mizuRoute control files

        Returns:
            True if routing is needed
        """
        return self._routing_decider.needs_routing(config, 'FUSE', settings_dir)

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to FUSE para_def.nc file directly.

        FUSE's run_def mode reads parameters from para_def.nc, so we must
        update this file directly for calibration to work. We also update
        the constraints file for consistency.

        Args:
            params: Parameter values to apply
            settings_dir: FUSE settings directory
            **kwargs: Must include 'config' and 'sim_dir' for path resolution

        Returns:
            True if successful
        """

        try:
            config = kwargs.get('config', self.config)
            _sim_dir = kwargs.get('sim_dir')  # noqa: F841

            # Log parameters being applied at DEBUG level to reduce spam
            self.logger.debug(f"APPLY_PARAMS: Applying {len(params)} parameters to {settings_dir}")
            for p, v in list(params.items())[:5]:  # Log first 5 params
                self.logger.debug(f"  PARAM: {p} = {v:.4f}")

            # =====================================================================
            # CRITICAL: Update para_def.nc directly for run_def mode
            # FUSE reads parameters from para_def.nc, not from constraints file
            # =====================================================================

            # Determine para_def.nc location
            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID', 'run_1')
            fuse_id = config.get('FUSE_FILE_ID', experiment_id)

            # para_def.nc is in the SETTINGS directory, not the simulation output directory
            # Check for FUSE subdirectory (common in parallel setup)
            if (settings_dir / 'FUSE').exists():
                fuse_settings_for_para = settings_dir / 'FUSE'
            else:
                fuse_settings_for_para = settings_dir

            para_def_path = fuse_settings_for_para / f"{domain_name}_{fuse_id}_para_def.nc"

            # Fallback to project-level path if not found in settings
            if not para_def_path.exists():
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                project_dir = data_dir / f"domain_{domain_name}"
                para_def_path = project_dir / 'simulations' / experiment_id / 'FUSE' / f"{domain_name}_{fuse_id}_para_def.nc"

            if para_def_path.exists():
                # Update para_def.nc directly
                params_updated_nc = self._update_para_def_nc(para_def_path, params)
                if params_updated_nc:
                    self.logger.debug(f"APPLY_PARAMS: Updated {len(params_updated_nc)} params in {para_def_path.name}")
                else:
                    self.logger.warning(f"APPLY_PARAMS: No params updated in {para_def_path.name}")
            else:
                self.logger.warning(f"APPLY_PARAMS: para_def.nc not found at {para_def_path}, will try constraints file")

            # =====================================================================
            # Also update constraints file for consistency (and for FUSE modes that
            # regenerate para_def.nc from constraints)
            # =====================================================================

            # Check for FUSE subdirectory (common in parallel setup)
            if (settings_dir / 'FUSE').exists():
                fuse_settings_dir = settings_dir / 'FUSE'
            else:
                fuse_settings_dir = settings_dir

            # Find the constraints file in settings_dir
            constraints_file = fuse_settings_dir / 'fuse_zConstraints_snow.txt'

            if constraints_file.exists():
                params_updated_txt = self._update_constraints_file(constraints_file, params)
                if params_updated_txt:
                    self.logger.debug(f"APPLY_PARAMS: Updated {len(params_updated_txt)} params in constraints file")

            return True

        except (FileNotFoundError, OSError) as e:
            self.logger.error(f"File error applying FUSE parameters: {e}")
            return False
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data error applying FUSE parameters: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _update_para_def_nc(self, para_def_path: Path, params: Dict[str, float]) -> set:
        """
        Update FUSE para_def.nc file with new parameter values.

        Args:
            para_def_path: Path to para_def.nc file
            params: Parameter values to apply

        Returns:
            Set of parameter names that were updated
        """
        import netCDF4 as nc

        params_updated = set()

        try:
            with nc.Dataset(para_def_path, 'r+') as ds:
                # Verify the file structure
                if 'par' not in ds.dimensions:
                    self.logger.error(f"Missing 'par' dimension in {para_def_path}")
                    return params_updated

                par_size = ds.dimensions['par'].size
                if par_size == 0:
                    self.logger.error(f"Empty 'par' dimension in {para_def_path}")
                    return params_updated

                for param_name, value in params.items():
                    if param_name in ds.variables:
                        try:
                            # Always use index 0 for single parameter set
                            before = float(ds.variables[param_name][0])
                            ds.variables[param_name][0] = float(value)
                            after = float(ds.variables[param_name][0])
                            self.logger.debug(f"  NC: {param_name}: {before:.4f} -> {after:.4f}")
                            params_updated.add(param_name)
                        except (IndexError, ValueError, TypeError) as e:
                            self.logger.warning(f"Error updating {param_name} in NetCDF: {e}")
                    else:
                        self.logger.debug(f"  NC: {param_name} not in file (may be structure param)")

                # Force sync to disk
                ds.sync()

        except (OSError, IOError) as e:
            self.logger.error(f"I/O error updating {para_def_path}: {e}")
        except (KeyError, ValueError) as e:
            self.logger.error(f"Data error updating {para_def_path}: {e}")

        return params_updated

    def _update_constraints_file(self, constraints_file: Path, params: Dict[str, float]) -> set:
        """
        Update FUSE constraints file with new parameter default values.

        FUSE uses Fortran fixed-width format: (L1,1X,I1,1X,3(F9.3,1X),...)
        The default value column starts at position 4 and is exactly 9 characters.

        Args:
            constraints_file: Path to constraints file
            params: Parameter values to apply

        Returns:
            Set of parameter names that were updated
        """
        params_updated = set()

        try:
            # Read the constraints file
            with open(constraints_file, 'r') as f:
                lines = f.readlines()

            # Fortran format: (L1,1X,I1,1X,3(F9.3,1X),...)
            # Default value column: position 4-12 (9 chars, F9.3 format)
            DEFAULT_VALUE_START = 4
            DEFAULT_VALUE_WIDTH = 9

            updated_lines = []

            for line in lines:
                # Skip header line (starts with '(') and comment lines
                stripped = line.strip()
                if stripped.startswith('(') or stripped.startswith('*') or stripped.startswith('!'):
                    updated_lines.append(line)
                    continue

                # Check if this line contains any of our parameters
                updated = False
                for param_name, value in params.items():
                    # Match exact parameter name (avoid partial matches)
                    parts = line.split()
                    if len(parts) >= 14 and param_name in parts:
                        # Parameter name is at index 13 in parts
                        if parts[13] == param_name:
                            # Format value to exactly 9 characters (F9.3 format)
                            new_value = f"{value:9.3f}"

                            # Replace the fixed-width column in the line
                            # Position 4-12 is the default value (9 characters)
                            if len(line) > DEFAULT_VALUE_START + DEFAULT_VALUE_WIDTH:
                                new_line = (
                                    line[:DEFAULT_VALUE_START] +
                                    new_value +
                                    line[DEFAULT_VALUE_START + DEFAULT_VALUE_WIDTH:]
                                )
                                updated_lines.append(new_line)
                                params_updated.add(param_name)
                                updated = True
                                break

                if not updated:
                    updated_lines.append(line)

            # Write updated constraints file
            with open(constraints_file, 'w') as f:
                f.writelines(updated_lines)

        except (OSError, IOError) as e:
            self.logger.warning(f"I/O error updating constraints file: {e}")
        except (IndexError, ValueError) as e:
            self.logger.warning(f"Format error updating constraints file: {e}")

        return params_updated

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run FUSE model.

        Args:
            config: Configuration dictionary
            settings_dir: FUSE settings directory
            output_dir: Output directory (not used directly by FUSE)
            **kwargs: Additional arguments including 'mode'

        Returns:
            True if model ran successfully
        """
        try:
            import subprocess

            # Optimization modifies para_def.nc and runs with run_def mode
            # run_def reads parameters from para_def.nc automatically
            mode = kwargs.get('mode', 'run_def')

            # Get FUSE executable path
            fuse_install = config.get('FUSE_INSTALL_PATH', 'default')
            if fuse_install == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                fuse_exe = data_dir / 'installs' / 'fuse' / 'bin' / 'fuse.exe'
            else:
                fuse_exe = Path(fuse_install) / 'fuse.exe'

            # Get file manager path using settings_dir
            # Check for FUSE subdirectory (common in parallel setup)
            if (settings_dir / 'FUSE').exists():
                filemanager_path = settings_dir / 'FUSE' / 'fm_catch.txt'
                # Update settings_dir to point to the FUSE subdir for execution context
                execution_cwd = settings_dir / 'FUSE'
            else:
                filemanager_path = settings_dir / 'fm_catch.txt'
                execution_cwd = settings_dir

            if not fuse_exe.exists():
                self.logger.error(f"FUSE executable not found: {fuse_exe}")
                return False

            if not filemanager_path.exists():
                self.logger.error(f"FUSE file manager not found: {filemanager_path}")
                return False

            # Use sim_dir for FUSE output (consistent with SUMMA structure)
            # sim_dir = process_X/simulations/run_1/FUSE
            fuse_output_dir = kwargs.get('sim_dir', output_dir)
            if fuse_output_dir:
                Path(fuse_output_dir).mkdir(parents=True, exist_ok=True)

            # Update file manager with isolated paths, experiment_id, and FMODEL_ID
            # We use a short alias 'sim' for the domain ID to avoid Fortran string length limits
            # and create symlinks for the input files in the execution directory
            fuse_run_id = 'sim'

            # Create symlinks for input files
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            domain_name = config.get('DOMAIN_NAME')
            project_dir = data_dir / f"domain_{domain_name}"
            fuse_input_dir = project_dir / 'forcing' / 'FUSE_input'
            experiment_id = config.get('EXPERIMENT_ID', 'run_1')
            fuse_id = config.get('FUSE_FILE_ID', experiment_id)

            # Define input files to symlink
            input_files = [
                (fuse_input_dir / f"{domain_name}_input.nc", f"{fuse_run_id}_input.nc"),
                (fuse_input_dir / f"{domain_name}_elev_bands.nc", f"{fuse_run_id}_elev_bands.nc")
            ]

            # Also symlink the parameter file (para_def.nc) to match the short alias
            # The optimizer generates {domain_name}_{fuse_id}_para_def.nc in execution_cwd
            # FUSE with 'sim' alias will look for sim_{fuse_id}_para_def.nc
            param_file_src = execution_cwd / f"{domain_name}_{fuse_id}_para_def.nc"
            param_file_dst = f"{fuse_run_id}_{fuse_id}_para_def.nc"
            input_files.append((param_file_src, param_file_dst))

            # Ensure configuration files are present (input_info.txt, fuse_zNumerix.txt, etc.)
            # These should have been copied by the optimizer, but if missing, symlink from main settings
            project_settings_dir = project_dir / 'settings' / 'FUSE'
            self.logger.debug(f"Checking for config files in: {project_settings_dir}")

            config_files = ['input_info.txt', 'fuse_zNumerix.txt']

            # Add decisions file to the list
            # Try to find the specific decisions file for this experiment
            actual_decisions_file = f"fuse_zDecisions_{experiment_id}.txt"
            if (project_settings_dir / actual_decisions_file).exists():
                config_files.append(actual_decisions_file)
            else:
                self.logger.warning(f"Decisions file {actual_decisions_file} not found in {project_settings_dir}")
                # Fallback: find any decisions file
                try:
                    decisions = list(project_settings_dir.glob("fuse_zDecisions_*.txt"))
                    if decisions:
                        actual_decisions_file = decisions[0].name
                        config_files.append(actual_decisions_file)
                        self.logger.warning(f"Using fallback decisions file: {actual_decisions_file}")
                except Exception as e:
                    self.logger.warning(f"Error searching for decisions files: {e}")

            for cfg_file in config_files:
                target_path = execution_cwd / cfg_file
                if not target_path.exists():
                    src_path = project_settings_dir / cfg_file
                    if src_path.exists():
                        input_files.append((src_path, cfg_file))
                        self.logger.warning(f"Restoring missing config file: {cfg_file}")
                    else:
                        self.logger.error(f"Source config file not found: {src_path}")

            for src, link_name in input_files:
                if src.exists():
                    link_path = execution_cwd / link_name
                    # Remove existing link/file if it exists
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    try:
                        link_path.symlink_to(src)
                        self.logger.debug(f"Created symlink: {link_path} -> {src}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create symlink {link_path}: {e}")

            # Pass use_local_input=True and actual decisions file to _update_file_manager
            if not self._update_file_manager(filemanager_path, execution_cwd, fuse_output_dir,
                                          config=config, use_local_input=True,
                                          decisions_file=actual_decisions_file):
                return False

            # List files in execution directory at DEBUG level to reduce spam
            self.logger.debug(f"Files in execution CWD ({execution_cwd}):")
            try:
                for f in execution_cwd.iterdir():
                    if f.is_symlink():
                        self.logger.debug(f"  {f.name} -> {f.resolve()}")
                    else:
                        self.logger.debug(f"  {f.name}")
            except Exception as e:
                self.logger.debug(f"Could not list directory: {e}")

            # Execute FUSE using the short alias
            # cmd = [str(fuse_exe), str(filemanager_path.name), domain_name, mode]
            cmd = [str(fuse_exe), str(filemanager_path.name), fuse_run_id, mode]

            # For run_pre mode, we need to pass the parameter file as argument
            # For run_def mode, FUSE reads parameters from para_def.nc automatically
            if mode == 'run_pre':
                experiment_id = config.get('EXPERIMENT_ID')
                fuse_id = config.get('FUSE_FILE_ID', experiment_id)
                # Note: para_def.nc is usually generated by the optimizer in the execution dir
                # If it uses domain_name, we might need to symlink it too or rely on FUSE behavior
                param_file = execution_cwd / f"{domain_name}_{fuse_id}_para_def.nc"
                if param_file.exists():
                    cmd.append(str(param_file.name))
                else:
                    self.logger.error(f"Parameter file not found for run_pre: {param_file}")
                    return False

            # Use execution_cwd as cwd
            self.logger.debug(f"Executing FUSE: {' '.join(cmd)} in {execution_cwd}")
            result = subprocess.run(
                cmd,
                cwd=str(execution_cwd),
                capture_output=True,
                text=True,
                timeout=config.get('FUSE_TIMEOUT', 300)
            )

            if result.returncode != 0:
                self.logger.error(f"FUSE failed with return code {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False

            # Log FUSE output only at DEBUG level to reduce spam
            if result.stdout:
                self.logger.debug(f"FUSE stdout (last 500 chars): {result.stdout[-500:]}")
            if result.stderr:
                self.logger.debug(f"FUSE stderr: {result.stderr}")

            # Validate that FUSE actually produced output (FUSE can return 0 but fail silently)
            # Output will now use the short alias 'sim' and the FMODEL_ID from file manager
            fuse_id = config.get('FUSE_FILE_ID', config.get('EXPERIMENT_ID'))

            # The output filename format is {domain_id}_{fmodel_id}_{suffix}
            # FUSE writes to execution_cwd because we set OUTPUT_PATH to ./
            local_output_filename = f"{fuse_run_id}_{fuse_id}_runs_def.nc"
            local_output_path = execution_cwd / local_output_filename

            # Final destination
            final_output_path = fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc"

            if local_output_path.exists():
                try:
                    # Move to final destination and rename
                    if final_output_path.exists():
                        final_output_path.unlink()
                    shutil.move(str(local_output_path), str(final_output_path))
                    self.logger.debug(f"Moved output from {local_output_path} to {final_output_path}")
                except Exception as e:
                    self.logger.error(f"Failed to move output file: {e}")
                    return False
            else:
                self.logger.error(f"FUSE returned success but local output file not created: {local_output_path}")
                # Only log first and last 1000 chars of stdout to avoid massive log spam
                if result.stdout:
                    stdout_lines = result.stdout.split('\n')
                    if len(stdout_lines) > 20:
                        self.logger.error(f"FUSE stdout (first 10 lines): {chr(10).join(stdout_lines[:10])}")
                        self.logger.error(f"FUSE stdout (last 10 lines): {chr(10).join(stdout_lines[-10:])}")
                    else:
                        self.logger.error(f"FUSE stdout: {result.stdout}")
                return False

            self.logger.debug(f"FUSE completed successfully, output: {final_output_path}")

            # Run routing if needed
            # Pass settings_dir to check for mizuRoute control files
            needs_routing_check = self.needs_routing(config, settings_dir=settings_dir)
            self.logger.debug(f"Routing check: needs_routing={needs_routing_check}, settings_dir={settings_dir}")

            if needs_routing_check:
                self.logger.debug("Running mizuRoute for FUSE output")

                # Get proc_id for parallel calibration (used for unique filenames)
                proc_id = kwargs.get('proc_id', 0)

                # Determine output directories
                sim_dir = kwargs.get('sim_dir')
                if sim_dir:
                    mizuroute_dir = Path(sim_dir).parent / 'mizuRoute'
                else:
                    mizuroute_dir = Path(fuse_output_dir).parent / 'mizuRoute'

                mizuroute_dir.mkdir(parents=True, exist_ok=True)

                # Convert FUSE output to mizuRoute format
                # Pass proc_id for correct filename generation in parallel calibration
                if not self._convert_fuse_to_mizuroute_format(
                    fuse_output_dir, config, execution_cwd, proc_id=proc_id
                ):
                    self.logger.error("Failed to convert FUSE output to mizuRoute format")
                    return False

                # Run mizuRoute
                # Pass settings_dir explicitly since it's a positional arg, not in kwargs
                # Remove keys that are passed explicitly to avoid duplicate argument errors
                keys_to_remove = {'proc_id', 'mizuroute_dir', 'settings_dir'}
                kwargs_filtered = {k: v for k, v in kwargs.items() if k not in keys_to_remove}
                if not self._run_mizuroute_for_fuse(
                    config, fuse_output_dir, mizuroute_dir,
                    settings_dir=settings_dir, proc_id=proc_id, **kwargs_filtered
                ):
                    self.logger.warning("Routing failed, but FUSE succeeded")
                    # Continue - routing failure may be acceptable for some workflows

            return True

        except subprocess.TimeoutExpired:
            self.logger.error("FUSE execution timed out")
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Required file not found for FUSE: {e}")
            return False
        except (OSError, IOError) as e:
            self.logger.error(f"I/O error running FUSE: {e}")
            return False
        except (subprocess.SubprocessError, RuntimeError) as e:
            # Catch subprocess errors and runtime issues during model execution
            self.logger.error(f"Error running FUSE: {e}")
            return False

    def _update_file_manager(self, filemanager_path: Path, settings_dir: Path, output_dir: Path,
                              experiment_id: str = None, config: Dict[str, Any] = None,
                              use_local_input: bool = False, decisions_file: str = None) -> bool:
        """
        Update FUSE file manager with isolated paths for parallel execution.

        Args:
            filemanager_path: Path to fm_catch.txt
            settings_dir: Isolated settings directory (where input files are)
            output_dir: Isolated output directory
            experiment_id: Experiment ID to use for FMODEL_ID and decisions file
            config: Configuration dictionary
            use_local_input: If True, set INPUT_PATH to ./ and expect files to be symlinked
            decisions_file: Actual decisions filename to use (if known from pre-check)

        Returns:
            True if successful
        """
        try:
            with open(filemanager_path, 'r') as f:
                lines = f.readlines()

            # Get experiment_id from config if not provided
            if experiment_id is None and config:
                experiment_id = config.get('EXPERIMENT_ID', 'run_1')
            elif experiment_id is None:
                experiment_id = 'run_1'

            # Get fuse_id for output files
            fuse_id = experiment_id
            if config:
                fuse_id = config.get('FUSE_FILE_ID', experiment_id)

            updated_lines = []

            # Use relative paths where possible to avoid Fortran string length limits (often 128 chars)
            # FUSE execution CWD is settings_dir (or settings_dir/FUSE)
            execution_cwd = filemanager_path.parent

            try:
                # SETNGS_PATH is the execution directory itself
                settings_path_str = "./"

                # Use local output path to avoid FUSE path length/symlink issues
                output_path_str = "./"

                self.logger.debug(f"Using paths - Settings: {settings_path_str}, Output: {output_path_str}")
            except Exception as e:
                self.logger.warning(f"Error setting paths: {e}")
                settings_path_str = "./"
                output_path_str = "./"
                self.logger.warning(f"Could not calculate relative paths: {e}. Falling back to absolute.")
                settings_path_str = str(settings_dir)
                if not settings_path_str.endswith('/'):
                    settings_path_str += '/'
                output_path_str = str(output_dir)
                if not output_path_str.endswith('/'):
                    output_path_str += '/'

            # Get input path from config (forcing directory)
            if use_local_input:
                input_path_str = "./"
            else:
                input_path_str = None
                if config:
                    data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                    domain_name = config.get('DOMAIN_NAME', '')
                    project_dir = data_dir / f"domain_{domain_name}"
                    fuse_input_dir = project_dir / 'forcing' / 'FUSE_input'
                    if fuse_input_dir.exists():
                        input_path_str = str(fuse_input_dir)
                        if not input_path_str.endswith('/'):
                            input_path_str += '/'

            # Get simulation dates - prefer actual forcing file dates over config
            sim_start = None
            sim_end = None
            eval_start = None
            eval_end = None

            if config:
                # First, try to read actual dates from forcing file
                # This handles the case where daily resampling shifts the start date
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                domain_name = config.get('DOMAIN_NAME', '')
                project_dir = data_dir / f"domain_{domain_name}"
                forcing_file = project_dir / 'forcing' / 'FUSE_input' / f"{domain_name}_input.nc"

                if forcing_file.exists():
                    try:
                        import xarray as xr
                        import pandas as pd
                        import numpy as np
                        with xr.open_dataset(forcing_file) as ds:
                            time_vals = ds['time'].values
                            if len(time_vals) > 0:
                                # Check type of time values
                                first_val = time_vals[0]
                                last_val = time_vals[-1]

                                # Check if datetime64 type
                                if np.issubdtype(type(first_val), np.datetime64):
                                    # Already datetime64
                                    forcing_start = pd.Timestamp(first_val)
                                    forcing_end = pd.Timestamp(last_val)
                                elif isinstance(first_val, (int, float, np.integer, np.floating)):
                                    # Numeric - assume 'days since 1970-01-01'
                                    forcing_start = pd.Timestamp('1970-01-01') + pd.Timedelta(days=float(first_val))
                                    forcing_end = pd.Timestamp('1970-01-01') + pd.Timedelta(days=float(last_val))
                                else:
                                    # Try direct conversion
                                    forcing_start = pd.Timestamp(first_val)
                                    forcing_end = pd.Timestamp(last_val)

                                sim_start = forcing_start.strftime('%Y-%m-%d')
                                sim_end = forcing_end.strftime('%Y-%m-%d')
                                self.logger.debug(f"Using forcing file dates: {sim_start} to {sim_end}")
                    except Exception as e:
                        self.logger.warning(f"Could not read forcing file dates: {e}")

                # Fallback to config dates if forcing file not available
                if sim_start is None:
                    exp_start = config.get('EXPERIMENT_TIME_START', '')
                    if exp_start:
                        sim_start = str(exp_start).split()[0]
                if sim_end is None:
                    exp_end = config.get('EXPERIMENT_TIME_END', '')
                    if exp_end:
                        sim_end = str(exp_end).split()[0]

                # Calibration period from config
                calib_period = config.get('CALIBRATION_PERIOD', '')
                if calib_period and ',' in str(calib_period):
                    parts = str(calib_period).split(',')
                    eval_start = parts[0].strip()
                    eval_end = parts[1].strip()

            for line in lines:
                stripped = line.strip()
                # Only match actual path lines (start with quote), not comment lines
                if stripped.startswith("'") and 'SETNGS_PATH' in line:
                    # Replace path inside single quotes
                    updated_lines.append(f"'{settings_path_str}'     ! SETNGS_PATH\n")
                elif stripped.startswith("'") and 'INPUT_PATH' in line:
                    if input_path_str:
                        updated_lines.append(f"'{input_path_str}'        ! INPUT_PATH\n")
                    else:
                        updated_lines.append(line)  # Keep original if not found
                elif stripped.startswith("'") and 'OUTPUT_PATH' in line:
                    updated_lines.append(f"'{output_path_str}'       ! OUTPUT_PATH\n")
                elif stripped.startswith("'") and 'M_DECISIONS' in line:
                    # Use the passed decisions_file if provided, otherwise try to find one
                    actual_decisions = decisions_file
                    if not actual_decisions:
                        actual_decisions = f"fuse_zDecisions_{experiment_id}.txt"
                        # Check if file exists, if not use what's available
                        if not (execution_cwd / actual_decisions).exists():
                            # Find any decisions file
                            found_files = list(execution_cwd.glob('fuse_zDecisions_*.txt'))
                            if found_files:
                                actual_decisions = found_files[0].name
                                self.logger.debug(f"Using available decisions file: {actual_decisions}")
                    updated_lines.append(f"'{actual_decisions}'        ! M_DECISIONS        = definition of model decisions\n")
                elif stripped.startswith("'") and 'FMODEL_ID' in line:
                    # Update FMODEL_ID to match fuse_id (used in output filename)
                    updated_lines.append(f"'{fuse_id}'                            ! FMODEL_ID          = string defining FUSE model, only used to name output files\n")
                elif stripped.startswith("'") and 'FORCING INFO' in line:
                    # Ensure input_info.txt doesn't have trailing spaces
                    updated_lines.append("'input_info.txt'                 ! FORCING INFO       = definition of the forcing file\n")
                elif stripped.startswith("'") and 'date_start_sim' in line and sim_start:
                    updated_lines.append(f"'{sim_start}'                     ! date_start_sim     = date start simulation\n")
                elif stripped.startswith("'") and 'date_end_sim' in line and sim_end:
                    updated_lines.append(f"'{sim_end}'                     ! date_end_sim       = date end simulation\n")
                elif stripped.startswith("'") and 'date_start_eval' in line and eval_start:
                    updated_lines.append(f"'{eval_start}'                     ! date_start_eval    = date start evaluation period\n")
                elif stripped.startswith("'") and 'date_end_eval' in line and eval_end:
                    updated_lines.append(f"'{eval_end}'                     ! date_end_eval      = date end evaluation period\n")
                else:
                    updated_lines.append(line)

            with open(filemanager_path, 'w') as f:
                f.writelines(updated_lines)

            self.logger.debug(f"Updated file manager: decisions={experiment_id}, fmodel_id={fuse_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update FUSE file manager: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from FUSE output.

        Args:
            output_dir: Directory containing model outputs (not used directly)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            import xarray as xr
            import pandas as pd

            # Get paths
            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"

            # Read observed streamflow
            obs_file_path = config.get('OBSERVATIONS_PATH', 'default')
            if obs_file_path == 'default':
                obs_file_path = (project_dir / 'observations' / 'streamflow' / 'preprocessed' /
                                f"{domain_name}_streamflow_processed.csv")
            else:
                obs_file_path = Path(obs_file_path)

            if not obs_file_path.exists():
                self.logger.error(f"Observation file not found: {obs_file_path}")
                return {'kge': self.penalty_score}

            # Read observations
            df_obs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True, dayfirst=True)

            # Ensure DatetimeIndex for resampling (fallback if parse_dates failed)
            if not isinstance(df_obs.index, pd.DatetimeIndex):
                try:
                    df_obs.index = pd.to_datetime(df_obs.index)
                    self.logger.debug("Converted observation index to DatetimeIndex")
                except Exception as e:
                    self.logger.error(f"Failed to convert observation time index to DatetimeIndex: {e}")
                    return {'kge': self.penalty_score}

            observed_streamflow = df_obs['discharge_cms'].resample('D').mean()

            # Check if routing was used - prioritize routed output over direct FUSE output
            mizuroute_dir = kwargs.get('mizuroute_dir')
            proc_id = kwargs.get('proc_id', 0)
            use_routed_output = False

            if mizuroute_dir and Path(mizuroute_dir).exists():
                mizuroute_dir = Path(mizuroute_dir)
                # Look for mizuRoute output - mizuRoute uses {case_name}.h.{start_time}.nc pattern
                # For parallel calibration, case_name = proc_{proc_id:02d}_{experiment_id}
                case_name = f"proc_{proc_id:02d}_{experiment_id}"

                # Try to find the output file using glob pattern
                mizuroute_dir / f"{case_name}.h.*.nc"
                mizu_output_files = list(mizuroute_dir.glob(f"{case_name}.h.*.nc"))

                if mizu_output_files:
                    # Sort by file size (largest first) to get file with actual data
                    # Empty/incomplete files will be smaller
                    mizu_output_files.sort(key=lambda f: f.stat().st_size, reverse=True)
                    sim_file_path = mizu_output_files[0]
                    use_routed_output = True
                    self.logger.debug(f"Using mizuRoute output for metrics calculation: {sim_file_path} (size: {sim_file_path.stat().st_size} bytes)")
                else:
                    # Fallback to non-prefixed pattern (for backward compatibility / default runs)
                    mizu_output_files_fallback = list(mizuroute_dir.glob(f"{experiment_id}.h.*.nc"))
                    if mizu_output_files_fallback:
                        # Sort by file size to get file with actual data
                        mizu_output_files_fallback.sort(key=lambda f: f.stat().st_size, reverse=True)
                        sim_file_path = mizu_output_files_fallback[0]
                        use_routed_output = True
                        self.logger.info(f"Using mizuRoute output for metrics calculation: {sim_file_path} (size: {sim_file_path.stat().st_size} bytes)")
                    else:
                        # Also try the older timestep naming convention
                        old_pattern = mizuroute_dir / f"{experiment_id}_timestep.nc"
                        if old_pattern.exists():
                            sim_file_path = old_pattern
                            use_routed_output = True
                            self.logger.info(f"Using mizuRoute output for metrics calculation: {sim_file_path}")

            # If no routed output, use FUSE output
            if not use_routed_output:
                # Read FUSE simulation output from sim_dir (or fallback to output_dir)
                # sim_dir = process_X/simulations/run_1/FUSE (consistent with SUMMA structure)
                fuse_id = config.get('FUSE_FILE_ID', experiment_id)
                fuse_output_dir = kwargs.get('sim_dir', output_dir)
                if fuse_output_dir:
                    fuse_output_dir = Path(fuse_output_dir)
                else:
                    fuse_output_dir = output_dir

                # FUSE runs in 'run_def' mode which reads from para_def.nc and produces runs_def.nc
                # Try runs_def first, then other possible output files
                sim_file_path = None
                candidates = [
                    fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc",   # run_def mode (default)
                    fuse_output_dir / f"{domain_name}_{fuse_id}_runs_best.nc",  # run_best mode
                    fuse_output_dir / f"{domain_name}_{fuse_id}_runs_pre.nc",   # run_pre mode (legacy)
                    fuse_output_dir.parent / f"{domain_name}_{fuse_id}_runs_def.nc",
                    output_dir.parent / f"{domain_name}_{fuse_id}_runs_pre.nc",
                ]
                for cand in candidates:
                    if cand.exists():
                        sim_file_path = cand
                        break

                if sim_file_path is None or not sim_file_path.exists():
                    self.logger.error(f"Simulation file not found. Searched: {[str(c) for c in candidates]}")
                    return {'kge': self.penalty_score}

                self.logger.debug("Using FUSE output for metrics calculation")

            # Read simulations
            # Explicitly decode times to ensure proper DatetimeIndex conversion
            with xr.open_dataset(sim_file_path, decode_times=True, decode_timedelta=True) as ds:
                if use_routed_output:
                    # mizuRoute output is already in m³/s
                    if 'IRFroutedRunoff' in ds.variables:
                        simulated = ds['IRFroutedRunoff'].isel(seg=0)
                    elif 'dlayRunoff' in ds.variables:
                        simulated = ds['dlayRunoff'].isel(seg=0)
                    else:
                        self.logger.error(f"No routed runoff variable in mizuRoute output. Variables: {list(ds.variables.keys())}")
                        return {'kge': self.penalty_score}

                    simulated_streamflow = simulated.to_pandas()

                    # Ensure DatetimeIndex for resampling (fallback if xarray decoding failed)
                    if not isinstance(simulated_streamflow.index, pd.DatetimeIndex):
                        simulated_streamflow.index = pd.to_datetime(simulated_streamflow.index)

                    # mizuRoute output is already in m³/s, no conversion needed
                    # Just resample to daily if needed
                    simulated_streamflow = simulated_streamflow.resample('D').mean()

                else:
                    # FUSE dimensions: (time, param_set, latitude, longitude)
                    # In distributed mode, latitude contains multiple subcatchments
                    spatial_mode = config.get('FUSE_SPATIAL_MODE', 'lumped')

                    if 'q_routed' in ds.variables:
                        runoff_var = ds['q_routed']
                    elif 'q_instnt' in ds.variables:
                        runoff_var = ds['q_instnt']
                    else:
                        self.logger.error(f"No runoff variable found in FUSE output. Variables: {list(ds.variables.keys())}")
                        return {'kge': self.penalty_score}

                    # Handle distributed mode: sum across all subcatchments
                    # FUSE output in mm/day represents depth over each subcatchment
                    if spatial_mode == 'distributed' and 'latitude' in runoff_var.dims and runoff_var.sizes.get('latitude', 1) > 1:
                        # For distributed mode without routing, we need to aggregate subcatchments
                        # Sum the volumetric runoff (convert each subcatchment from mm/day to m3/s, then sum)
                        # Or take area-weighted mean if areas are equal
                        n_subcatchments = runoff_var.sizes['latitude']
                        self.logger.info(f"Distributed mode: aggregating {n_subcatchments} subcatchments")

                        # Get individual subcatchment areas if available, otherwise assume equal distribution
                        total_area_km2 = self._get_catchment_area(config, project_dir)

                        # Select param_set and longitude, keep latitude for aggregation
                        runoff_selected = runoff_var.isel(param_set=0, longitude=0)

                        # For equal-area subcatchments: total flow = sum of individual flows
                        # Each subcatchment's mm/day * (total_area/n_subcatchments) / 86.4 gives m3/s
                        # Sum gives total m3/s
                        subcatchment_area = total_area_km2 / n_subcatchments

                        # Convert each subcatchment to m3/s and sum
                        simulated_cms = (runoff_selected * subcatchment_area / UnitConversion.MM_DAY_TO_CMS).sum(dim='latitude')
                        simulated_streamflow = simulated_cms.to_pandas()
                        self.logger.info(f"Aggregated distributed output: mean flow = {simulated_streamflow.mean():.2f} m³/s")
                    else:
                        # Lumped mode or single subcatchment
                        simulated = runoff_var.isel(param_set=0, latitude=0, longitude=0)
                        simulated_streamflow = simulated.to_pandas()

                        # Get catchment area for unit conversion
                        area_km2 = self._get_catchment_area(config, project_dir)

                        # Convert FUSE output from mm/day to cms
                        # Q(cms) = Q(mm/day) * Area(km2) / 86.4
                        simulated_streamflow = simulated_streamflow * area_km2 / UnitConversion.MM_DAY_TO_CMS

            # Ensure simulated_streamflow has a DatetimeIndex (fallback if xarray decoding failed)
            if not isinstance(simulated_streamflow.index, pd.DatetimeIndex):
                try:
                    simulated_streamflow.index = pd.to_datetime(simulated_streamflow.index)
                    self.logger.debug("Converted simulated streamflow index to DatetimeIndex")
                except Exception as e:
                    self.logger.error(f"Failed to convert time index to DatetimeIndex: {e}")
                    return {'kge': self.penalty_score}

            # Align time series
            common_index = observed_streamflow.index.intersection(simulated_streamflow.index)
            if len(common_index) == 0:
                self.logger.error("No overlapping time period")
                return {'kge': self.penalty_score}

            obs_aligned = observed_streamflow.loc[common_index].dropna()
            sim_aligned = simulated_streamflow.loc[common_index].dropna()

            # Filter to calibration period if specified
            calib_period = config.get('CALIBRATION_PERIOD', '')
            if calib_period and ',' in str(calib_period):
                try:
                    calib_start, calib_end = [s.strip() for s in str(calib_period).split(',')]
                    calib_start = pd.Timestamp(calib_start)
                    calib_end = pd.Timestamp(calib_end)

                    # Filter to calibration period
                    mask_obs = (obs_aligned.index >= calib_start) & (obs_aligned.index <= calib_end)
                    mask_sim = (sim_aligned.index >= calib_start) & (sim_aligned.index <= calib_end)

                    obs_aligned = obs_aligned[mask_obs]
                    sim_aligned = sim_aligned[mask_sim]

                    self.logger.debug(f"Filtered to calibration period {calib_start} to {calib_end}: {len(obs_aligned)} points")
                except Exception as e:
                    self.logger.warning(f"Could not parse calibration period '{calib_period}': {e}")

            common_index = obs_aligned.index.intersection(sim_aligned.index)
            obs_values = obs_aligned.loc[common_index].values
            sim_values = sim_aligned.loc[common_index].values

            if len(obs_values) == 0:
                self.logger.error("No valid data points")
                return {'kge': self.penalty_score}

            # Calculate metrics using shared utility
            metrics = self._streamflow_metrics.calculate_metrics(
                obs_values, sim_values, metrics=['kge', 'nse', 'rmse', 'mae']
            )

            # Debug: Log computed metrics to trace score flow
            self.logger.debug(f"FUSE metrics computed: KGE={metrics['kge']:.4f}, NSE={metrics['nse']:.4f}, n_points={len(obs_values)}")

            return metrics

        except FileNotFoundError as e:
            self.logger.error(f"Output or observation file not found: {e}")
            return {'kge': self.penalty_score}
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data error calculating FUSE metrics: {e}")
            return {'kge': self.penalty_score}
        except (ImportError, OSError) as e:
            # Catch xarray/pandas import issues or I/O errors
            self.logger.error(f"Error calculating FUSE metrics: {e}")
            return {'kge': self.penalty_score}

    def _get_catchment_area(self, config: Dict[str, Any], project_dir: Path) -> float:
        """
        Get catchment area for FUSE unit conversion. Delegates to shared utility.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path

        Returns:
            Catchment area in km2
        """
        domain_name = config.get('DOMAIN_NAME')
        return self._streamflow_metrics.get_catchment_area(config, project_dir, domain_name)

    def _convert_fuse_to_mizuroute_format(
        self,
        fuse_output_dir: Path,
        config: Dict[str, Any],
        settings_dir: Path,
        proc_id: int = 0
    ) -> bool:
        """
        Convert FUSE distributed output to mizuRoute-compatible format.

        Delegates to FuseToMizurouteConverter for the actual conversion.

        Args:
            fuse_output_dir: Directory containing FUSE output
            config: Configuration dictionary
            settings_dir: Settings directory (unused, kept for API compatibility)
            proc_id: Process ID for parallel calibration (used in filename)

        Returns:
            True if conversion successful
        """
        # Use instance logger for the converter
        converter = FuseToMizurouteConverter(logger=self.logger)
        return converter.convert(fuse_output_dir, config, proc_id)

    def _run_mizuroute_for_fuse(
        self,
        config: Dict[str, Any],
        fuse_output_dir: Path,
        mizuroute_dir: Path,
        **kwargs
    ) -> bool:
        """
        Execute mizuRoute for FUSE output.

        Args:
            config: Configuration dictionary
            fuse_output_dir: Directory containing FUSE output
            mizuroute_dir: Output directory for mizuRoute
            **kwargs: Additional arguments (settings_dir, sim_dir, etc.)

        Returns:
            True if mizuRoute ran successfully
        """
        try:
            import subprocess

            # Get mizuRoute executable
            mizuroute_install = config.get('MIZUROUTE_INSTALL_PATH', 'default')
            if mizuroute_install == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                mizuroute_exe = data_dir / 'installs' / 'mizuRoute' / 'route' / 'bin' / 'mizuRoute.exe'
            else:
                mizuroute_exe = Path(mizuroute_install) / 'mizuRoute.exe'

            if not mizuroute_exe.exists():
                self.logger.error(f"mizuRoute executable not found: {mizuroute_exe}")
                return False

            # Get process-specific control file
            # The optimizer should have copied and configured mizuRoute settings
            # to the process-specific settings directory
            # settings_dir structure: .../process_N/settings/FUSE/
            # mizuRoute settings are at: .../process_N/settings/mizuRoute/

            # Try to get from kwargs first (set by BaseModelOptimizer)
            mizuroute_settings_dir = kwargs.get('mizuroute_settings_dir')
            if mizuroute_settings_dir:
                control_file = Path(mizuroute_settings_dir) / 'mizuroute.control'
            else:
                settings_dir_path = Path(kwargs.get('settings_dir', Path('.')))
                # Check both in settings_dir and settings_dir.parent to handle FUSE subdirectory
                control_file = settings_dir_path / 'mizuRoute' / 'mizuroute.control'
                if not control_file.exists() and settings_dir_path.name == 'FUSE':
                    control_file = settings_dir_path.parent / 'mizuRoute' / 'mizuroute.control'

            # Fallback to main control file (default runs)
            if not control_file or not control_file.exists():
                domain_name = config.get('DOMAIN_NAME')
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                project_dir = data_dir / f"domain_{domain_name}"
                control_file = project_dir / 'settings' / 'mizuRoute' / 'mizuroute.control'

            if not control_file.exists():
                self.logger.error(f"mizuRoute control file not found: {control_file}")
                return False

            self.logger.debug(f"Using control file: {control_file}")

            # Execute mizuRoute
            cmd = [str(mizuroute_exe), str(control_file)]

            self.logger.debug(f"Executing mizuRoute: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.get('MIZUROUTE_TIMEOUT', 600)
            )

            if result.returncode != 0:
                self.logger.error(f"mizuRoute failed with return code {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False

            self.logger.debug("mizuRoute completed successfully")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error("mizuRoute execution timed out")
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Required file not found for mizuRoute: {e}")
            return False
        except (OSError, subprocess.SubprocessError) as e:
            self.logger.error(f"Error running mizuRoute: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_fuse_parameters_worker(task_data)


def _evaluate_fuse_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    worker = FUSEWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
