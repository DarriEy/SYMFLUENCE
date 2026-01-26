"""
RHESSys Worker

Worker implementation for RHESSys model optimization.
"""

import logging
import os
import subprocess
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.core.constants import ModelDefaults


@OptimizerRegistry.register_worker('RHESSys')
class RHESSysWorker(BaseWorker):
    """
    Worker for RHESSys model calibration.

    Handles parameter application, RHESSys execution, and metric calculation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize RHESSys worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    # Shared utilities
    _streamflow_metrics = StreamflowMetrics()

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to RHESSys definition files.

        Args:
            params: Parameter values to apply
            settings_dir: RHESSys settings directory (contains defs/ subdirectory)
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        try:
            self.logger.debug(f"Applying RHESSys parameters: {params}")

            # The settings_dir should contain a 'defs' subdirectory
            defs_dir = settings_dir / 'defs'
            if not defs_dir.exists():
                self.logger.error(
                    f"RHESSys defs directory not found: {defs_dir}. "
                    "This indicates that RHESSys input files were not correctly copied to the worker directory. "
                    "Check that 'RHESSys_input/defs' exists in the domain directory."
                )
                return False

            # Update definition files with new parameters
            return self._update_def_files(defs_dir, params)

        except Exception as e:
            self.logger.error(f"Error applying RHESSys parameters: {e}")
            return False

    def _update_def_files(self, defs_dir: Path, params: Dict[str, float]) -> bool:
        """
        Update RHESSys definition files with new parameter values.

        Args:
            defs_dir: Path to defs directory
            params: Parameters to update

        Returns:
            True if successful
        """
        import re

        # Mapping from parameter names to definition files
        PARAM_FILE_MAP = {
            'sat_to_gw_coeff': 'basin.def',
            'gw_loss_coeff': 'basin.def',
            'n_routing_power': 'basin.def',
            'psi_air_entry': 'basin.def',
            'pore_size_index': 'basin.def',
            'porosity_0': 'soil.def',
            'porosity_decay': 'soil.def',
            'Ksat_0': 'soil.def',
            'Ksat_0_v': 'soil.def',
            'm': 'soil.def',
            'm_z': 'soil.def',
            'soil_depth': 'soil.def',
            'active_zone_z': 'soil.def',
            'snow_melt_Tcoef': 'soil.def',
            'maximum_snow_energy_deficit': 'soil.def',
            'max_snow_temp': 'zone.def',
            'min_rain_temp': 'zone.def',
            'epc.max_lai': 'stratum.def',
            'epc.gl_smax': 'stratum.def',
            'epc.gl_c': 'stratum.def',
            'epc.vpd_open': 'stratum.def',
            'epc.vpd_close': 'stratum.def',
        }

        # Group parameters by file
        params_by_file: Dict[str, Dict[str, float]] = {}
        for param_name, value in params.items():
            def_file_name = PARAM_FILE_MAP.get(param_name)
            if def_file_name:
                if def_file_name not in params_by_file:
                    params_by_file[def_file_name] = {}
                params_by_file[def_file_name][param_name] = value

        # Update each file
        for def_file_name, file_params in params_by_file.items():
            def_file = defs_dir / def_file_name
            if not def_file.exists():
                self.logger.warning(f"Definition file not found: {def_file}")
                continue

            with open(def_file, 'r') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                updated = False
                for param_name, value in file_params.items():
                    # Match: value<whitespace>param_name (allow trailing comments)
                    # Pattern matches: optional start ^, float group(1), whitespace group(2), param_name group(3), remaining group(4)
                    pattern = rf'^([\d\.\-\+eE]+)(\s+)({re.escape(param_name)})(\s.*|)$'
                    match = re.match(pattern, line)
                    if match:
                        new_line = f"{value:.6f}{match.group(2)}{match.group(3)}{match.group(4)}\n"
                        # Strip double newlines if they happen
                        new_line = new_line.replace('\n\n', '\n')
                        updated_lines.append(new_line)
                        updated = True
                        break
                if not updated:
                    updated_lines.append(line)

            with open(def_file, 'w') as f:
                f.writelines(updated_lines)
            self.logger.debug(f"Updated {len(file_params)} params in {def_file_name}")

        return True

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run RHESSys model.

        Args:
            config: Configuration dictionary
            settings_dir: RHESSys settings directory
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        try:
            # Get paths
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'
            rhessys_input_dir = project_dir / 'RHESSys_input'

            # Use sim_dir for output if provided
            rhessys_output_dir = Path(kwargs.get('sim_dir', output_dir))
            rhessys_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean up stale output files to ensure fresh model run
            self._cleanup_stale_output(rhessys_output_dir)

            # Log cleanup status
            output_file = rhessys_output_dir / 'rhessys_basin.daily'
            self.logger.debug(f"After cleanup, output file exists: {output_file.exists()}")

            # Get executable
            rhessys_exe = self._get_rhessys_executable(config, data_dir)
            if not rhessys_exe.exists():
                self.logger.error(f"RHESSys executable not found: {rhessys_exe}")
                return False

            # Build command
            cmd = self._build_command(
                rhessys_exe,
                config,
                rhessys_input_dir,
                settings_dir,
                rhessys_output_dir
            )

            # Set library path for WMFire
            env = os.environ.copy()
            lib_paths = []
            rhessys_bin_dir = rhessys_exe.parent
            wmfire_lib_dir = data_dir / "installs" / "wmfire" / "lib"

            for lib_dir in [rhessys_bin_dir, wmfire_lib_dir]:
                if lib_dir.exists():
                    lib_paths.append(str(lib_dir))

            if lib_paths:
                lib_path_str = ":".join(lib_paths)
                if sys.platform == "darwin":
                    env["DYLD_LIBRARY_PATH"] = f"{lib_path_str}:{env.get('DYLD_LIBRARY_PATH', '')}"
                else:
                    env["LD_LIBRARY_PATH"] = f"{lib_path_str}:{env.get('LD_LIBRARY_PATH', '')}"

            # Run RHESSys
            import time as time_module
            run_start = time_module.time()
            result = subprocess.run(
                cmd,
                cwd=str(rhessys_output_dir),
                env=env,
                capture_output=True,
                text=True,
                timeout=config.get('RHESSYS_TIMEOUT', 3600)
            )
            run_time = time_module.time() - run_start
            self.logger.info(f"RHESSys completed in {run_time:.2f}s with return code {result.returncode}")

            if result.returncode != 0:
                self._last_error = f"RHESSys failed: {result.stderr[-500:]}"
                self.logger.error(self._last_error)
                return False

            # Verify output exists and log details
            output_file = rhessys_output_dir / 'rhessys_basin.daily'
            if not output_file.exists():
                self._last_error = "No basin output file produced"
                self.logger.error(f"Expected output at {output_file}")
                return False

            # Log output file size to verify it was actually written
            file_size = output_file.stat().st_size
            self.logger.debug(f"Output file {output_file} size: {file_size} bytes")

            return True

        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error running RHESSys: {e}")
            return False

    def _get_rhessys_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get RHESSys executable path."""
        install_path = config.get('RHESSYS_INSTALL_PATH', 'default')
        exe_name = config.get('RHESSYS_EXE', 'rhessys')
        if install_path == 'default':
            return data_dir / "installs" / "rhessys" / "bin" / exe_name
        # If install_path is a directory, append exe_name
        install_path = Path(install_path)
        if install_path.is_dir():
            return install_path / exe_name
        # If it's already a full path to executable
        return install_path

    def _build_command(
        self,
        exe: Path,
        config: Dict[str, Any],
        rhessys_input_dir: Path,
        settings_dir: Path,
        output_dir: Path
    ) -> list:
        """Build RHESSys command line."""
        domain_name = config.get('DOMAIN_NAME')

        # Use worker-specific defs directory if available
        worker_defs_dir = settings_dir / 'defs'

        # Strategy: Copy world file to settings_dir and ensure header matches.
        # This ensures RHESSys finds the modified header file (which points to modified defs)
        # by looking for <world_file>.hdr in the same directory, avoiding unsupported flags.
        original_world = rhessys_input_dir / 'worldfiles' / f'{domain_name}.world'
        worker_world = settings_dir / f'{domain_name}.world'
        original_hdr = rhessys_input_dir / 'worldfiles' / f'{domain_name}.world.hdr'
        worker_hdr = settings_dir / f'{domain_name}.world.hdr'

        if worker_defs_dir.exists():
            # Copy world file if it doesn't exist in worker dir
            if original_world.exists() and not worker_world.exists():
                import shutil
                shutil.copy2(original_world, worker_world)

            # Create/Update modified header in worker dir if it doesn't exist
            if original_hdr.exists() and not worker_hdr.exists():
                # Copy and modify header to point to worker defs
                self._create_worker_header(original_hdr, worker_hdr, worker_defs_dir, rhessys_input_dir)

        # Parse dates
        start_str = config.get('EXPERIMENT_TIME_START', '2004-01-01 01:00')
        end_str = config.get('EXPERIMENT_TIME_END', '2004-12-31 23:00')
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)

        # Use worker world if it was created successfully, otherwise fallback to original
        world_to_use = worker_world if worker_world.exists() else original_world

        tecfile = rhessys_input_dir / 'tecfiles' / f'{domain_name}.tec'
        routing = rhessys_input_dir / 'routing' / f'{domain_name}.routing'

        # Get scaling parameters from config if available (default to None to match Runner)
        s1 = config.get('RHESSYS_S1')
        s2 = config.get('RHESSYS_S2')
        s3 = config.get('RHESSYS_S3')
        gw1 = config.get('RHESSYS_GW1')
        gw2 = config.get('RHESSYS_GW2')

        cmd = [
            str(exe),
            '-w', str(world_to_use),
            '-t', str(tecfile),
            '-r', str(routing),
            '-st', str(start_date.year), str(start_date.month), str(start_date.day), '1',
            '-ed', str(end_date.year), str(end_date.month), str(end_date.day), '1',
            '-pre', 'rhessys',
            '-b',  # Basin output
        ]

        if gw1 is not None and gw2 is not None:
            cmd.extend(['-gw', str(gw1), str(gw2)])

        if s1 is not None and s2 is not None and s3 is not None:
            cmd.extend(['-s', str(s1), str(s2), str(s3)])

        cmd.extend([
            '-sv', '1.0', '1.0',
            '-svalt', '1.0', '1.0',
        ])

        # Fire spread if WMFire is enabled
        wmfire_enabled = config.get('RHESSYS_USE_WMFIRE', False)
        if wmfire_enabled:
            fire_dir = rhessys_input_dir / "fire"
            patch_grid = fire_dir / "patch_grid.txt"
            dem_grid = fire_dir / "dem_grid.txt"
            if patch_grid.exists() and dem_grid.exists():
                resolution = config.get('WMFIRE_GRID_RESOLUTION', 30)
                cmd.extend(["-firespread", str(resolution), str(patch_grid), str(dem_grid)])
                self.logger.debug(f"WMFire fire spread enabled: {resolution}m resolution")
            else:
                self.logger.warning(
                    f"WMFire is enabled but fire grid files not found at {fire_dir}. "
                    "Fire spread will be disabled for calibration."
                )

        return cmd

    def _create_worker_header(
        self,
        original_hdr: Path,
        worker_hdr: Path,
        worker_defs_dir: Path,
        rhessys_input_dir: Path
    ):
        """Create a worker-specific header file pointing to worker defs."""
        with open(original_hdr, 'r') as f:
            content = f.read()

        # Replace def file paths with worker-specific paths.
        # Use normalized strings to ensure replacement works despite path variations.
        original_defs_str = os.path.normpath(str(rhessys_input_dir / 'defs'))
        worker_defs_str = os.path.normpath(str(worker_defs_dir))

        content = content.replace(original_defs_str, worker_defs_str)

        with open(worker_hdr, 'w') as f:
            f.write(content)

    def _cleanup_stale_output(self, output_dir: Path, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Remove stale RHESSys output files before a new model run.

        This ensures that each calibration iteration produces fresh output
        and prevents reusing stale results from previous runs.

        Args:
            output_dir: Directory containing RHESSys output files
            config: Optional config dict to get experiment_id for nested path cleanup
        """
        # RHESSys output file patterns to clean up
        output_patterns = [
            'rhessys_basin.daily',
            'rhessys_basin.hourly',
            'rhessys_basin.monthly',
            'rhessys_basin.yearly',
        ]

        files_removed = 0

        # Clean at the direct output directory level
        for pattern in output_patterns:
            direct_file = output_dir / pattern
            if direct_file.exists():
                try:
                    direct_file.unlink()
                    files_removed += 1
                    self.logger.debug(f"Removed stale output: {direct_file}")
                except (OSError, IOError) as e:
                    self.logger.warning(f"Could not remove stale file {direct_file}: {e}")

        # Clean the RHESSys-specific output file with wildcard
        for file_path in output_dir.glob('rhessys_*.daily'):
            try:
                file_path.unlink()
                files_removed += 1
            except (OSError, IOError) as e:
                self.logger.warning(f"Could not remove stale file {file_path}: {e}")

        for file_path in output_dir.glob('rhessys_*.hourly'):
            try:
                file_path.unlink()
                files_removed += 1
            except (OSError, IOError) as e:
                self.logger.warning(f"Could not remove stale file {file_path}: {e}")

        if files_removed > 0:
            self.logger.info(f"Cleaned up {files_removed} stale RHESSys output files from {output_dir}")

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from RHESSys output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            # Get simulation directory
            sim_dir = Path(kwargs.get('sim_dir', output_dir))

            # Read RHESSys output - robustly locate the file
            # RHESSys produces output in a nested structure: [sim_dir]/simulations/[experiment_id]/RHESSys/
            experiment_id = config.get('EXPERIMENT_ID', 'run_1')
            possible_paths = [
                sim_dir / 'rhessys_basin.daily',
                sim_dir / 'simulations' / experiment_id / 'RHESSys' / 'rhessys_basin.daily',
            ]

            sim_file = None
            for path in possible_paths:
                if path.exists():
                    sim_file = path
                    break

            if not sim_file:
                # Last resort: try recursive glob
                found = list(sim_dir.glob('**/rhessys_basin.daily'))
                if found:
                    sim_file = found[0]

            if not sim_file:
                self.logger.error(f"rhessys_basin.daily not found in {sim_dir}")
                return {'kge': self.penalty_score, 'error': 'rhessys_basin.daily not found'}

            self.logger.info(f"Calculating RHESSys metrics from: {sim_file}")
            sim_df = pd.read_csv(sim_file, sep=r'\s+', header=0)
            self.logger.info(f"Read {len(sim_df)} rows from simulation output")

            # Get streamflow in mm/day
            streamflow_mm = sim_df['streamflow'].values

            # Convert to m³/s using catchment area from shared utility
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'

            area_km2 = self._streamflow_metrics.get_catchment_area(
                config, project_dir, domain_name, source='shapefile'
            )
            area_m2 = area_km2 * 1e6  # Convert km² to m²
            self.logger.info(f"Using catchment area: {area_km2:.2f} km² ({area_m2:.2e} m²)")

            # Q (m³/s) = Q (mm/day) * area (m²) / 86400 / 1000
            streamflow_m3s = streamflow_mm * area_m2 / 86400 / 1000

            # Check for NaN values in simulation
            nan_count = pd.isna(streamflow_m3s).sum()
            if nan_count > 0:
                self.logger.warning(
                    f"RHESSys output contains {nan_count} NaN values out of {len(streamflow_m3s)} timesteps"
                )

            # Check for zero discharge (model didn't produce runoff)
            if streamflow_m3s.sum() == 0:
                self.logger.warning(
                    "RHESSys simulation produced zero streamflow - check model parameters"
                )
                return {'kge': self.penalty_score, 'error': 'Zero streamflow from model'}

            # Create dates
            sim_dates = pd.to_datetime(
                sim_df.apply(
                    lambda r: f"{int(r['year'])}-{int(r['month']):02d}-{int(r['day']):02d}",
                    axis=1
                )
            )
            sim_series = pd.Series(streamflow_m3s, index=sim_dates)

            # Load observations (domain_name and project_dir already set above)
            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq='D'
            )
            if obs_values is None:
                self.logger.error("Observations not found for metric calculation")
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_series = pd.Series(obs_values, index=obs_index)
            self.logger.info(f"Loaded {len(obs_series)} observations")

            # Align and calculate metrics
            try:
                # Parse calibration period if specified
                calib_period_tuple = None
                calib_period_str = config.get('CALIBRATION_PERIOD', '')
                if calib_period_str:
                    try:
                        start_str, end_str = calib_period_str.split(',')
                        calib_period_tuple = (start_str.strip(), end_str.strip())
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Could not parse calibration period '{calib_period_str}': {e}")

                # Let StreamflowMetrics handle alignment and period filtering
                obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                    sim_series, obs_series, calibration_period=calib_period_tuple
                )
                self.logger.info(f"Aligned timeseries length: {len(obs_aligned)}")

                results = self._streamflow_metrics.calculate_metrics(
                    obs_aligned, sim_aligned, metrics=['kge', 'nse']
                )
                self.logger.info(f"Calculated metrics: {results}")
                return results

            except ValueError as e:
                self.logger.error(f"Alignment error: {e}")
                return {'kge': self.penalty_score, 'error': str(e)}

        except Exception as e:
            self.logger.error(f"Error calculating RHESSys metrics: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_rhessys_parameters_worker(task_data)


def _evaluate_rhessys_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    import os
    import sys
    import signal
    import random
    import time
    import traceback

    # Set up signal handler for clean termination
    def signal_handler(signum, frame):
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass

    # Force single-threaded execution
    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NETCDF_DISABLE_LOCKING': '1',
        'HDF5_USE_FILE_LOCKING': 'FALSE',
        'HDF5_DISABLE_VERSION_CHECK': '1',
    })

    # Add small random delay
    initial_delay = random.uniform(0.1, 0.8)
    time.sleep(initial_delay)

    try:
        worker = RHESSysWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'Critical RHESSys worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
