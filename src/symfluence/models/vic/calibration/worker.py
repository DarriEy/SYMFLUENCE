"""
VIC Worker

Worker implementation for VIC model optimization.
"""

import logging
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.core.constants import ModelDefaults


@OptimizerRegistry.register_worker('VIC')
class VICWorker(BaseWorker):
    """
    Worker for VIC model calibration.

    Handles parameter application, VIC execution, and metric calculation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize VIC worker.

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
        Apply parameters to VIC parameter file.

        Args:
            params: Parameter values to apply
            settings_dir: VIC settings directory (contains parameters/ subdirectory)
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        import shutil

        try:
            self.logger.debug(f"Applying VIC parameters to {settings_dir}")

            # The settings_dir should contain a 'parameters' subdirectory
            params_dir = settings_dir / 'parameters'

            # Always copy fresh from the original VIC_input location
            # to ensure dimensions match the current domain file
            config = kwargs.get('config', self.config) or {}
            domain_name = config.get('DOMAIN_NAME', '')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            original_params_dir = data_dir / f'domain_{domain_name}' / 'VIC_input' / 'parameters'

            if original_params_dir.exists():
                params_dir.mkdir(parents=True, exist_ok=True)
                for f in original_params_dir.glob('*.nc'):
                    shutil.copy2(f, params_dir / f.name)
                self.logger.debug(f"Copied VIC parameters from {original_params_dir} to {params_dir}")
            elif not params_dir.exists():
                self.logger.error(f"VIC parameters directory not found: {params_dir} "
                                  f"(original also missing: {original_params_dir})")
                return False

            # Find parameter file
            params_file = params_dir / 'vic_params.nc'
            if not params_file.exists():
                # Try to find any .nc file
                nc_files = list(params_dir.glob('*params*.nc'))
                if nc_files:
                    params_file = nc_files[0]
                else:
                    self.logger.error(f"VIC parameter file not found in {params_dir}")
                    return False

            # Update parameter file
            return self._update_params_file(params_file, params)

        except Exception as e:
            self.logger.error(f"Error applying VIC parameters: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_params_file(self, params_file: Path, params: Dict[str, float]) -> bool:
        """
        Update VIC parameter NetCDF file with new values.

        Args:
            params_file: Path to parameter NetCDF file
            params: Parameters to update

        Returns:
            True if successful
        """
        # Parameter to variable mapping
        PARAM_VAR_MAP = {
            'infilt': 'infilt',
            'Ds': 'Ds',
            'Dsmax': 'Dsmax',
            'Ws': 'Ws',
            'c': 'c',
            'depth1': 'depth',
            'depth2': 'depth',
            'depth3': 'depth',
            'Ksat': 'Ksat',
            'expt': 'expt',
            'Wcr_FRACT': 'Wcr_FRACT',
            'Wpwp_FRACT': 'Wpwp_FRACT',
            'snow_rough': 'snow_rough',
        }

        # Special parameters handled outside the standard loop
        SPECIAL_PARAMS = {'elev_offset'}

        LAYER_PARAMS = {
            'depth1': 0,
            'depth2': 1,
            'depth3': 2,
        }

        try:
            ds = xr.open_dataset(params_file)
            ds = ds.load()

            # Enforce constraint: Wpwp_FRACT must be < Wcr_FRACT
            if 'Wcr_FRACT' in params and 'Wpwp_FRACT' in params:
                if params['Wpwp_FRACT'] >= params['Wcr_FRACT']:
                    params['Wpwp_FRACT'] = params['Wcr_FRACT'] * 0.5

            # Handle elev_offset: shift snow band elevations to control snowmelt timing
            # Positive offset raises band elevations → cooler bands → delayed snowmelt
            # Each +100m offset cools all bands by ~0.65°C via lapse rate
            if 'elev_offset' in params and 'elevation' in ds:
                offset = params['elev_offset']
                for band in range(ds['elevation'].shape[0]):
                    mask = ~np.isnan(ds['elevation'].values[band])
                    ds['elevation'].values[band][mask] += offset
                self.logger.debug(f"Applied elev_offset = {offset:.0f}m to snow band elevations")

            for param_name, value in params.items():
                if param_name in SPECIAL_PARAMS:
                    continue

                var_name = PARAM_VAR_MAP.get(param_name)
                if var_name is None or var_name not in ds:
                    continue

                if param_name in LAYER_PARAMS:
                    layer_idx = LAYER_PARAMS[param_name]
                    if 'nlayer' in ds[var_name].dims:
                        mask = ~np.isnan(ds[var_name].values[layer_idx])
                        ds[var_name].values[layer_idx][mask] = value
                else:
                    if len(ds[var_name].dims) == 2:
                        mask = ~np.isnan(ds[var_name].values)
                        ds[var_name].values[mask] = value
                    elif len(ds[var_name].dims) == 3:
                        for layer in range(ds[var_name].shape[0]):
                            mask = ~np.isnan(ds[var_name].values[layer])
                            ds[var_name].values[layer][mask] = value
                    else:
                        ds[var_name].values = value

                self.logger.debug(f"Updated {param_name} = {value}")

            # Save
            temp_file = params_file.with_suffix('.nc.tmp')
            ds.to_netcdf(temp_file)
            ds.close()
            temp_file.replace(params_file)

            return True

        except Exception as e:
            self.logger.error(f"Error updating VIC parameters: {e}")
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run VIC model.

        Args:
            config: Configuration dictionary
            settings_dir: VIC settings directory
            output_dir: Output directory
            **kwargs: Additional arguments (sim_dir, proc_id)

        Returns:
            True if model ran successfully
        """
        try:
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'
            vic_input_dir = project_dir / 'VIC_input'

            # Use sim_dir for output if provided
            vic_output_dir = Path(kwargs.get('sim_dir', output_dir))
            vic_output_dir.mkdir(parents=True, exist_ok=True)

            # Clean up stale output files
            self._cleanup_stale_output(vic_output_dir)

            # Get executable
            vic_exe = self._get_vic_executable(config, data_dir)
            if not vic_exe.exists():
                self.logger.error(f"VIC executable not found: {vic_exe}")
                return False

            # Get or create global parameter file
            global_param_file = self._get_or_create_global_file(
                config, vic_input_dir, settings_dir, vic_output_dir
            )

            if not global_param_file.exists():
                self.logger.error(f"Global parameter file not found: {global_param_file}")
                return False

            # Build command
            cmd = [str(vic_exe), '-g', str(global_param_file)]

            # Set environment
            env = os.environ.copy()

            # Run with timeout
            timeout = config.get('VIC_TIMEOUT', 300)

            stdout_file = vic_output_dir / 'vic_stdout.log'
            stderr_file = vic_output_dir / 'vic_stderr.log'

            try:
                with open(stdout_file, 'w') as stdout_f, open(stderr_file, 'w') as stderr_f:
                    result = subprocess.run(
                        cmd,
                        cwd=str(vic_output_dir),
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout_f,
                        stderr=stderr_f,
                        timeout=timeout
                    )
            except subprocess.TimeoutExpired:
                self.logger.warning(f"VIC timed out after {timeout}s")
                return False

            if result.returncode != 0:
                self._last_error = f"VIC failed with return code {result.returncode}"
                self.logger.error(self._last_error)
                return False

            # Verify output
            output_files = list(vic_output_dir.glob('*.nc'))
            if not output_files:
                self._last_error = "No output files produced"
                self.logger.error(self._last_error)
                return False

            return True

        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error running VIC: {e}")
            return False

    def _get_vic_executable(self, config: Dict[str, Any], data_dir: Path) -> Path:
        """Get VIC executable path."""
        install_path = config.get('VIC_INSTALL_PATH', 'default')
        exe_name = config.get('VIC_EXE', 'vic_image.exe')

        if install_path == 'default':
            return data_dir / "installs" / "vic" / "bin" / exe_name

        install_path = Path(install_path)
        if install_path.is_dir():
            return install_path / exe_name
        return install_path

    def _get_or_create_global_file(
        self,
        config: Dict[str, Any],
        vic_input_dir: Path,
        settings_dir: Path,
        output_dir: Path
    ) -> Path:
        """Get or create a worker-specific global parameter file."""
        # First try to find existing global file
        global_file_name = config.get('VIC_GLOBAL_PARAM_FILE', 'vic_global.txt')
        original_global = vic_input_dir / 'settings' / global_file_name

        if not original_global.exists():
            self.logger.warning(f"Original global file not found: {original_global}")
            return original_global

        # Create worker-specific version with updated output path
        worker_global = settings_dir / global_file_name

        with open(original_global, 'r') as f:
            content = f.read()

        # Update RESULT_DIR and PARAMETERS to point to worker directories
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if line.strip().startswith('RESULT_DIR'):
                new_lines.append(f'RESULT_DIR             {output_dir}')
            elif line.strip().startswith('PARAMETERS'):
                # Always point to worker-specific params if available
                worker_params = settings_dir / 'parameters' / 'vic_params.nc'
                if worker_params.exists():
                    new_lines.append(f'PARAMETERS             {worker_params}')
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        with open(worker_global, 'w') as f:
            f.write('\n'.join(new_lines))

        return worker_global

    def _cleanup_stale_output(self, output_dir: Path) -> None:
        """Remove stale VIC output files."""
        for pattern in ['*.nc', '*.log']:
            for file_path in output_dir.glob(pattern):
                try:
                    file_path.unlink()
                except (OSError, IOError):
                    pass

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from VIC output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            sim_dir = Path(kwargs.get('sim_dir', output_dir))

            # Find output file
            output_files = list(sim_dir.glob('*fluxes*.nc'))
            if not output_files:
                output_files = list(sim_dir.glob('vic_output*.nc'))
            if not output_files:
                output_files = list(sim_dir.glob('*.nc'))

            if not output_files:
                self.logger.error(f"No VIC output files found in {sim_dir}")
                return {'kge': self.penalty_score, 'error': 'No output files'}

            sim_file = output_files[0]

            # Extract streamflow
            ds = xr.open_dataset(sim_file)

            # Get runoff + baseflow
            runoff = None
            baseflow = None

            for var in ['OUT_RUNOFF', 'RUNOFF']:
                if var in ds:
                    runoff = ds[var]
                    break

            for var in ['OUT_BASEFLOW', 'BASEFLOW']:
                if var in ds:
                    baseflow = ds[var]
                    break

            if runoff is None:
                ds.close()
                return {'kge': self.penalty_score, 'error': 'No runoff variable'}

            # Aggregate spatially — mean across grid cells (each cell is mm/day)
            spatial_dims = [d for d in runoff.dims if d not in ['time']]
            total_runoff = runoff.mean(dim=spatial_dims).values
            if baseflow is not None:
                total_runoff += baseflow.mean(dim=spatial_dims).values

            times = pd.to_datetime(ds['time'].values)
            ds.close()

            # Convert to m³/s using catchment area
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f'domain_{domain_name}'

            area_km2 = self._streamflow_metrics.get_catchment_area(
                config, project_dir, domain_name, source='shapefile'
            )
            area_m2 = area_km2 * 1e6

            # mm/day -> m³/s
            streamflow_m3s = total_runoff * area_m2 / 1000 / 86400
            sim_series = pd.Series(streamflow_m3s, index=times)

            # Load observations
            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq='D'
            )
            if obs_values is None:
                return {'kge': self.penalty_score, 'error': 'No observations'}

            obs_series = pd.Series(obs_values, index=obs_index)

            # Align and calculate metrics
            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series
            )

            results = self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=['kge', 'nse']
            )
            return results

        except Exception as e:
            self.logger.error(f"Error calculating VIC metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score, 'error': str(e)}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_vic_parameters_worker(task_data)


def _evaluate_vic_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    import os
    import signal
    import random
    import time
    import traceback

    # Set up signal handler
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
    })

    # Small random delay
    time.sleep(random.uniform(0.1, 0.5))

    try:
        worker = VICWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'VIC worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
