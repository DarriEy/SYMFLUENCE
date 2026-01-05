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

            # Update file manager with isolated paths
            if not self._update_file_manager(filemanager_path, execution_cwd, fuse_output_dir):
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

            self.logger.debug(f"FUSE completed successfully")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error("FUSE execution timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running FUSE: {e}")
            return False

    def _update_file_manager(self, filemanager_path: Path, settings_dir: Path, output_dir: Path) -> bool:
        """
        Update FUSE file manager with isolated paths for parallel execution.

        Args:
            filemanager_path: Path to fm_catch.txt
            settings_dir: Isolated settings directory (where input files are)
            output_dir: Isolated output directory

        Returns:
            True if successful
        """
        try:
            with open(filemanager_path, 'r') as f:
                lines = f.readlines()

            updated_lines = []
            settings_path_str = str(settings_dir)
            if not settings_path_str.endswith('/'):
                settings_path_str += '/'
            
            output_path_str = str(output_dir)
            if not output_path_str.endswith('/'):
                output_path_str += '/'

            for line in lines:
                stripped = line.strip()
                # Only match actual path lines (start with quote), not comment lines
                if stripped.startswith("'") and 'SETNGS_PATH' in line:
                    # Replace path inside single quotes
                    updated_lines.append(f"'{settings_path_str}'     ! SETNGS_PATH\n")
                elif stripped.startswith("'") and 'OUTPUT_PATH' in line:
                    updated_lines.append(f"'{output_path_str}'       ! OUTPUT_PATH\n")
                else:
                    updated_lines.append(line)

            with open(filemanager_path, 'w') as f:
                f.writelines(updated_lines)
            
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

            # Read simulations
            with xr.open_dataset(sim_file_path) as ds:
                # FUSE dimensions: (time, param_set, latitude, longitude)
                if 'q_routed' in ds.variables:
                    simulated = ds['q_routed'].isel(param_set=0, latitude=0, longitude=0)
                elif 'q_instnt' in ds.variables:
                    simulated = ds['q_instnt'].isel(param_set=0, latitude=0, longitude=0)
                else:
                    self.logger.error(f"No runoff variable found in FUSE output. Variables: {list(ds.variables.keys())}")
                    return {'kge': self.penalty_score}

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
