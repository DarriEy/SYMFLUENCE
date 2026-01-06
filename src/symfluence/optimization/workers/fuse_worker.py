"""
FUSE Worker

Worker implementation for FUSE model optimization.
Delegates to existing worker functions while providing BaseWorker interface.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from ..registry import OptimizerRegistry
from symfluence.core.constants import UnitConversion


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

    def needs_routing(self, config: Dict[str, Any], settings_dir: Optional[Path] = None) -> bool:
        """
        Determine if routing (mizuRoute) is needed for FUSE.

        Args:
            config: Configuration dictionary
            settings_dir: Optional settings directory to check for mizuRoute control files

        Returns:
            True if routing is needed
        """
        calibration_var = config.get('CALIBRATION_VARIABLE', 'streamflow')

        if calibration_var != 'streamflow':
            return False

        # Check if FUSE routing integration is enabled
        routing_integration = config.get('FUSE_ROUTING_INTEGRATION', 'none')

        # If 'default', inherit from ROUTING_MODEL
        if routing_integration == 'default':
            routing_model = config.get('ROUTING_MODEL', 'none')
            if routing_model == 'mizuRoute':
                routing_integration = 'mizuRoute'

        if routing_integration != 'mizuRoute':
            return False

        # Check spatial mode and routing delineation
        spatial_mode = config.get('FUSE_SPATIAL_MODE', 'lumped')
        routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')

        # Distributed modes need routing
        if spatial_mode in ['semi_distributed', 'distributed']:
            return True

        # Lumped with river network routing needs routing
        if spatial_mode == 'lumped' and routing_delineation == 'river_network':
            return True

        # Also check if mizuRoute control files exist in settings_dir
        # This handles cases where routing was set up by the optimizer
        # but config flags don't explicitly indicate it
        if settings_dir:
            settings_dir = Path(settings_dir)
            # Check for process-specific mizuRoute settings (parallel calibration)
            mizu_control = settings_dir.parent / 'mizuRoute' / 'mizuroute.control'
            if mizu_control.exists():
                self.logger.debug(f"Found mizuRoute control file at {mizu_control}, enabling routing")
                return True

        return False

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to FUSE constraints file.

        FUSE's run_def mode regenerates para_def.nc from the constraints file,
        so we must modify the constraints file to change parameter values.

        FUSE uses Fortran fixed-width format: (L1,1X,I1,1X,3(F9.3,1X),...)
        The default value column starts at position 4 and is exactly 9 characters.

        Args:
            params: Parameter values to apply
            settings_dir: FUSE settings directory
            **kwargs: Must include 'config' for path resolution

        Returns:
            True if successful
        """
        try:
            config = kwargs.get('config', self.config)

            # Find the constraints file in settings_dir
            constraints_file = settings_dir / 'fuse_zConstraints_snow.txt'

            if not constraints_file.exists():
                self.logger.error(f"FUSE constraints file not found: {constraints_file}")
                return False

            # Read the constraints file
            with open(constraints_file, 'r') as f:
                lines = f.readlines()

            # Fortran format: (L1,1X,I1,1X,3(F9.3,1X),...)
            # Default value column: position 4-12 (9 chars, F9.3 format)
            DEFAULT_VALUE_START = 4
            DEFAULT_VALUE_WIDTH = 9

            updated_lines = []
            params_updated = set()

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
                    if len(parts) >= 13 and param_name in parts:
                        # Verify this is the parameter we want (check parameter name column)
                        param_col_idx = 13  # Parameter name is typically at index 13 in parts
                        if parts[param_col_idx] == param_name or param_name in parts:
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

            # Log which parameters were updated
            if params_updated:
                self.logger.debug(f"Updated FUSE constraints: {params_updated}")

            missing = set(params.keys()) - params_updated
            if missing:
                self.logger.warning(f"Parameters not found in constraints file: {missing}")

            return True

        except Exception as e:
            self.logger.error(f"Error applying FUSE parameters: {e}")
            return False

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
            if not self._update_file_manager(filemanager_path, execution_cwd, fuse_output_dir, config=config):
                return False

            # Execute FUSE
            domain_name = config.get('DOMAIN_NAME')
            cmd = [str(fuse_exe), str(filemanager_path.name), domain_name, mode]

            # For run_pre mode, we need to pass the parameter file as argument
            # For run_def mode, FUSE reads parameters from para_def.nc automatically
            if mode == 'run_pre':
                experiment_id = config.get('EXPERIMENT_ID')
                fuse_id = config.get('FUSE_FILE_ID', experiment_id)
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

            # Log FUSE output for debugging (even on success)
            if result.stdout:
                self.logger.debug(f"FUSE stdout: {result.stdout[-500:]}")  # Last 500 chars
            if result.stderr:
                self.logger.warning(f"FUSE stderr: {result.stderr}")

            # Validate that FUSE actually produced output (FUSE can return 0 but fail silently)
            domain_name = config.get('DOMAIN_NAME')
            fuse_id = config.get('FUSE_FILE_ID', config.get('EXPERIMENT_ID'))
            expected_output = fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc"

            if not expected_output.exists():
                self.logger.error(f"FUSE returned success but output file not created: {expected_output}")
                self.logger.error(f"FUSE stdout: {result.stdout}")
                self.logger.error(f"Check forcing file format matches FUSE_SPATIAL_MODE setting")
                return False

            self.logger.debug(f"FUSE completed successfully, output: {expected_output}")

            # Run routing if needed
            # Pass settings_dir to check for mizuRoute control files
            needs_routing_check = self.needs_routing(config, settings_dir=settings_dir)
            self.logger.info(f"Routing check: needs_routing={needs_routing_check}, settings_dir={settings_dir}")

            if needs_routing_check:
                self.logger.info("Running mizuRoute for FUSE output")

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
        except Exception as e:
            self.logger.error(f"Error running FUSE: {e}")
            return False

    def _update_file_manager(self, filemanager_path: Path, settings_dir: Path, output_dir: Path,
                              experiment_id: str = None, config: Dict[str, Any] = None) -> bool:
        """
        Update FUSE file manager with isolated paths for parallel execution.

        Args:
            filemanager_path: Path to fm_catch.txt
            settings_dir: Isolated settings directory (where input files are)
            output_dir: Isolated output directory
            experiment_id: Experiment ID to use for FMODEL_ID and decisions file
            config: Configuration dictionary

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
            settings_path_str = str(settings_dir)
            if not settings_path_str.endswith('/'):
                settings_path_str += '/'

            output_path_str = str(output_dir)
            if not output_path_str.endswith('/'):
                output_path_str += '/'

            # Get input path from config (forcing directory)
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

            # Get simulation dates from config
            sim_start = None
            sim_end = None
            eval_start = None
            eval_end = None
            if config:
                # Parse dates from config
                exp_start = config.get('EXPERIMENT_TIME_START', '')
                exp_end = config.get('EXPERIMENT_TIME_END', '')
                calib_period = config.get('CALIBRATION_PERIOD', '')

                # Extract date part (without time)
                if exp_start:
                    sim_start = str(exp_start).split()[0]
                if exp_end:
                    sim_end = str(exp_end).split()[0]
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
                    # Update decisions file to match experiment_id
                    decisions_file = f"fuse_zDecisions_{experiment_id}.txt"
                    # Check if file exists, if not use what's available
                    if not (settings_dir / decisions_file).exists():
                        # Find any decisions file
                        import glob
                        decisions_files = list(settings_dir.glob('fuse_zDecisions_*.txt'))
                        if decisions_files:
                            decisions_file = decisions_files[0].name
                            self.logger.debug(f"Using available decisions file: {decisions_file}")
                    updated_lines.append(f"'{decisions_file}'        ! M_DECISIONS        = definition of model decisions\n")
                elif stripped.startswith("'") and 'FMODEL_ID' in line:
                    # Update FMODEL_ID to match fuse_id (used in output filename)
                    updated_lines.append(f"'{fuse_id}'                            ! FMODEL_ID          = string defining FUSE model, only used to name output files\n")
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
    ) -> Dict[str, float]:
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
            from symfluence.evaluation.metrics import kge, nse, rmse, mae

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
            df_obs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True)
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
                mizu_output_pattern = mizuroute_dir / f"{case_name}.h.*.nc"
                mizu_output_files = list(mizuroute_dir.glob(f"{case_name}.h.*.nc"))

                if mizu_output_files:
                    # Use the first (or most recent) output file
                    sim_file_path = mizu_output_files[0]
                    use_routed_output = True
                    self.logger.info(f"Using mizuRoute output for metrics calculation: {sim_file_path}")
                else:
                    # Fallback to non-prefixed pattern (for backward compatibility / default runs)
                    mizu_output_files_fallback = list(mizuroute_dir.glob(f"{experiment_id}.h.*.nc"))
                    if mizu_output_files_fallback:
                        sim_file_path = mizu_output_files_fallback[0]
                        use_routed_output = True
                        self.logger.info(f"Using mizuRoute output for metrics calculation: {sim_file_path}")
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

                self.logger.info("Using FUSE output for metrics calculation")

            # Read simulations
            with xr.open_dataset(sim_file_path) as ds:
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

            # Align time series
            common_index = observed_streamflow.index.intersection(simulated_streamflow.index)
            if len(common_index) == 0:
                self.logger.error("No overlapping time period")
                return {'kge': self.penalty_score}

            obs_aligned = observed_streamflow.loc[common_index].dropna()
            sim_aligned = simulated_streamflow.loc[common_index].dropna()

            common_index = obs_aligned.index.intersection(sim_aligned.index)
            obs_values = obs_aligned.loc[common_index].values
            sim_values = sim_aligned.loc[common_index].values

            if len(obs_values) == 0:
                self.logger.error("No valid data points")
                return {'kge': self.penalty_score}

            # Calculate metrics
            metrics = {
                'kge': float(kge(obs_values, sim_values, transfo=1)),
                'nse': float(nse(obs_values, sim_values, transfo=1)),
                'rmse': float(rmse(obs_values, sim_values, transfo=1)),
                'mae': float(mae(obs_values, sim_values, transfo=1)),
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating FUSE metrics: {e}")
            return {'kge': self.penalty_score}

    def _get_catchment_area(self, config: Dict[str, Any], project_dir: Path) -> float:
        """
        Get catchment area for FUSE unit conversion.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path

        Returns:
            Catchment area in km2
        """
        try:
            import geopandas as gpd

            # Get catchment shapefile path
            catchment_path = config.get('CATCHMENT_PATH', 'default')
            catchment_name = config.get('CATCHMENT_SHP_NAME', 'default')

            if catchment_path == 'default':
                catchment_path = project_dir / 'shapefiles' / 'catchment'
            else:
                catchment_path = Path(catchment_path)

            if catchment_name == 'default':
                domain_name = config.get('DOMAIN_NAME')
                discretization = config.get('DOMAIN_DISCRETIZATION', 'elevation')
                catchment_name = f"{domain_name}_HRUs_{discretization}.shp"

            catchment_file = catchment_path / catchment_name

            if not catchment_file.exists():
                self.logger.warning(f"Catchment file not found: {catchment_file}")
                return 1000.0  # Default value

            # Read catchment and calculate area
            gdf = gpd.read_file(catchment_file)

            if 'GRU_area' in gdf.columns:
                total_area_m2 = gdf['GRU_area'].sum()
                return total_area_m2 / 1e6

            # Calculate from geometry
            if gdf.crs and not gdf.crs.is_geographic:
                total_area_m2 = gdf.geometry.area.sum()
            else:
                gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
                total_area_m2 = gdf_utm.geometry.area.sum()

            return total_area_m2 / 1e6

        except Exception as e:
            self.logger.warning(f"Error calculating catchment area: {e}")
            return 1000.0

    def _convert_fuse_to_mizuroute_format(
        self,
        fuse_output_dir: Path,
        config: Dict[str, Any],
        settings_dir: Path,
        proc_id: int = 0
    ) -> bool:
        """
        Convert FUSE distributed output to mizuRoute-compatible format.

        Args:
            fuse_output_dir: Directory containing FUSE output
            config: Configuration dictionary
            settings_dir: Settings directory
            proc_id: Process ID for parallel calibration (used in filename)

        Returns:
            True if conversion successful
        """
        try:
            import xarray as xr
            import numpy as np

            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID')
            fuse_id = config.get('FUSE_FILE_ID', experiment_id)

            # Find FUSE output file (runs_def.nc for run_def mode)
            output_file = fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc"

            if not output_file.exists():
                self.logger.error(f"FUSE output file not found: {output_file}")
                return False

            # Open dataset
            ds = xr.open_dataset(output_file)

            # FUSE outputs q_routed (routed runoff) which we need for mizuRoute
            # Try multiple variable names that FUSE might use
            fuse_runoff_vars = ['q_routed', 'q_instnt', 'total_discharge', 'runoff']
            q_fuse = None
            for var_name in fuse_runoff_vars:
                if var_name in ds.variables:
                    q_fuse = ds[var_name]
                    self.logger.debug(f"Using FUSE variable '{var_name}' for routing")
                    break

            if q_fuse is None:
                self.logger.error(f"FUSE output missing runoff variable. Available: {list(ds.variables)}")
                ds.close()
                return False

            # mizuRoute control file expects 'q_routed' as variable name (see vname_qsim in control)
            # Handle the case where config has 'default' as value - use model-specific default
            routing_var_config = config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
            if routing_var_config in ('default', None, ''):
                routing_var = 'q_routed'  # FUSE default for routing
            else:
                routing_var = routing_var_config

            # Convert FUSE output to routing format
            # FUSE: (time, param_set, latitude, longitude) -> mizuRoute: (time, gru)

            # Squeeze out param_set dimension if it exists
            if 'param_set' in q_fuse.dims:
                q_fuse = q_fuse.isel(param_set=0)

            # Handle spatial dimensions to create 'gru' dimension for mizuRoute
            # For distributed mode: latitude encodes subcatchments (N values), longitude is singleton (1)
            # For lumped mode: both are singleton (1)
            subcatchment_dim = config.get('FUSE_SUBCATCHMENT_DIM', 'longitude')

            if 'latitude' in q_fuse.dims and 'longitude' in q_fuse.dims:
                # Squeeze singleton dimensions
                if q_fuse.sizes.get('longitude', 1) == 1:
                    q_fuse = q_fuse.squeeze('longitude', drop=True)
                if q_fuse.sizes.get('latitude', 1) == 1:
                    q_fuse = q_fuse.squeeze('latitude', drop=True)

                # Rename the remaining spatial dimension to 'gru'
                if subcatchment_dim in q_fuse.dims:
                    q_fuse = q_fuse.rename({subcatchment_dim: 'gru'})

            # If still no 'gru' dimension, create one (lumped case after squeeze)
            if 'gru' not in q_fuse.dims:
                q_fuse = q_fuse.expand_dims('gru')

            # Determine number of GRUs
            n_gru = q_fuse.sizes['gru']

            # CRITICAL: Convert units from FUSE output (mm/timestep) to what mizuRoute expects
            target_units = config.get('SETTINGS_MIZU_ROUTING_UNITS', 'm/s')
            # Handle 'default' value - resolve to 'm/s'
            if target_units in ('default', None, ''):
                target_units = 'm/s'
            timestep_seconds = int(config.get('FORCING_TIME_STEP_SIZE', 86400))

            self.logger.info(f"Converting FUSE runoff to {target_units} (timestep={timestep_seconds}s)")

            # FUSE outputs in 'mm timestep-1' - convert to target units
            q_fuse_values = q_fuse.values

            if target_units == 'm/s':
                # Convert mm/timestep -> m/s
                # mm/timestep * (1m / 1000mm) / timestep_seconds = m/s
                conversion_factor = 1.0 / (1000.0 * timestep_seconds)
                q_fuse_values = q_fuse_values * conversion_factor
                self.logger.info(f"Applied conversion factor: {conversion_factor:.2e}")
            elif target_units == 'mm/s':
                # Convert mm/timestep -> mm/s
                conversion_factor = 1.0 / timestep_seconds
                q_fuse_values = q_fuse_values * conversion_factor
            elif target_units in ['mm/d', 'mm/day']:
                # Convert mm/timestep -> mm/day
                if timestep_seconds != 86400:
                    conversion_factor = 86400.0 / timestep_seconds
                    q_fuse_values = q_fuse_values * conversion_factor
                # else: Already in mm/day, no conversion needed
            elif target_units in ['mm/h', 'mm/hr', 'mm/hour']:
                # Convert mm/timestep -> mm/hour
                if timestep_seconds != 3600:
                    conversion_factor = 3600.0 / timestep_seconds
                    q_fuse_values = q_fuse_values * conversion_factor
            else:
                self.logger.warning(f"Unknown target units '{target_units}', passing through without conversion")

            # Create new dataset with routing variable and simple integer gru coordinate
            ds_routing = xr.Dataset({
                routing_var: (('time', 'gru'), q_fuse_values)
            }, coords={
                'time': ds['time'].values,
                'gru': np.arange(n_gru)
            })

            # Set units attribute
            ds_routing[routing_var].attrs['units'] = target_units
            ds_routing[routing_var].attrs['long_name'] = 'FUSE runoff for mizuRoute routing'

            # Add gruId variable (1-indexed as expected by mizuRoute)
            ds_routing['gruId'] = ('gru', np.arange(1, n_gru + 1))

            # Add time attributes
            ds_routing['time'].attrs = ds['time'].attrs

            # CRITICAL: Close input dataset BEFORE writing to avoid permission errors
            ds.close()

            # Create the file with the name mizuRoute expects
            # For parallel calibration, use proc_XX prefix to match mizuRoute control file
            # The control file is configured with fname_qsim = proc_{proc_id:02d}_{experiment_id}_timestep.nc
            expected_filename = f"proc_{proc_id:02d}_{experiment_id}_timestep.nc"
            expected_file = fuse_output_dir / expected_filename

            # Save converted output directly to expected file (don't overwrite FUSE output)
            ds_routing.to_netcdf(expected_file)
            self.logger.info(f"Created mizuRoute input file: {expected_file}")

            ds_routing.close()

            self.logger.info(f"Converted FUSE output to mizuRoute format: {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error converting FUSE output to mizuRoute format: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

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
            # settings_dir structure: .../process_N/settings/FUSE
            # mizuRoute settings are at: .../process_N/settings/mizuRoute
            settings_dir_path = kwargs.get('settings_dir', Path('.'))

            # Check if this is a process-specific directory (parallel calibration)
            if 'process_' in str(settings_dir_path):
                # Use process-specific control file
                control_file = settings_dir_path.parent / 'mizuRoute' / 'mizuroute.control'
            else:
                # Fallback to main control file (default runs)
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

            self.logger.info("mizuRoute completed successfully")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error("mizuRoute execution timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running mizuRoute: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
