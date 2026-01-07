"""
HYPE Worker

Worker implementation for HYPE model optimization.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from ..registry import OptimizerRegistry
from symfluence.evaluation.metrics import kge, nse
from symfluence.models.hype.preprocessor import HYPEPreProcessor
from symfluence.models.hype.runner import HYPERunner


@OptimizerRegistry.register_worker('HYPE')
class HYPEWorker(BaseWorker):
    """
    Worker for HYPE model calibration.

    Handles parameter application, HYPE execution, and metric calculation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize HYPE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    def needs_routing(self, config: Dict[str, Any], settings_dir: Optional[Path] = None) -> bool:
        """
        Determine if routing (mizuRoute) is needed for HYPE.

        Args:
            config: Configuration dictionary
            settings_dir: Optional settings directory to check for mizuRoute control files

        Returns:
            True if routing is needed
        """
        calibration_var = config.get('CALIBRATION_VARIABLE', 'streamflow')

        if calibration_var != 'streamflow':
            return False

        # Check routing model
        routing_model = config.get('ROUTING_MODEL', 'none')
        if routing_model in ['mizuRoute', 'default']:
            return True

        # Check spatial mode and routing delineation
        spatial_mode = config.get('HYPE_SPATIAL_MODE', 'lumped')
        routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')
        domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')

        # Distributed modes or non-lumped domains need routing
        if spatial_mode in ['semi_distributed', 'distributed']:
            return True
        
        if domain_method not in ['point', 'lumped']:
            return True

        # Lumped with river network routing needs routing
        if routing_delineation == 'river_network':
            return True

        # Also check if mizuRoute control files exist in settings_dir
        if settings_dir:
            mizu_control = settings_dir / 'mizuRoute' / 'mizuroute.control'
            if mizu_control.exists():
                return True

        return False

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to HYPE configuration files.

        Args:
            params: Parameter values to apply
            settings_dir: HYPE settings directory
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        try:
            from symfluence.models.hype.hypeFlow import write_hype_par_file
            
            # Ensure settings directory exists
            settings_dir.mkdir(parents=True, exist_ok=True)
            
            # Update par.txt in the isolated settings directory
            # write_hype_par_file will read GeoClass.txt from settings_dir 
            # to determine class counts and land uses
            write_hype_par_file(settings_dir, params=params)
            
            return True

        except Exception as e:
            self.logger.error(f"Error applying HYPE parameters: {e}")
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run HYPE model.

        Args:
            config: Configuration dictionary
            settings_dir: HYPE settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if model ran successfully
        """
        try:
            # Diagnostic check: Ensure all required files exist in the worker settings dir
            required_files = ['info.txt', 'par.txt', 'GeoData.txt', 'GeoClass.txt', 'ForcKey.txt', 'Pobs.txt']
            missing_files = [f for f in required_files if not (settings_dir / f).exists()]
            if missing_files:
                self.logger.error(f"Worker {kwargs.get('proc_id', '?')} missing required files in {settings_dir}: {missing_files}")
                return False

            # Use sim_dir for HYPE output if provided (consistent with SUMMA/FUSE structure)
            hype_output_dir = kwargs.get('sim_dir', output_dir)
            hype_output_dir = Path(hype_output_dir).absolute()
            hype_output_dir.mkdir(parents=True, exist_ok=True)

            # Ensure the info.txt file points to the correct isolated output directory
            from symfluence.models.hype.hypeFlow import write_hype_info_filedir_files
            
            spinup_days = config.get('HYPE_SPINUP_DAYS', 0)
            experiment_start = config.get('EXPERIMENT_TIME_START')
            experiment_end = config.get('EXPERIMENT_TIME_END')
            
            # HYPE results dir MUST have a trailing slash
            results_dir_str = str(hype_output_dir).rstrip('/') + '/'
            
            # Update info.txt in settings_dir to point to this worker's output dir
            write_hype_info_filedir_files(
                settings_dir.absolute(),
                spinup_days,
                results_dir_str,
                experiment_start=experiment_start,
                experiment_end=experiment_end
            )

            # Create a modified config for the runner that disables routing
            # The worker will handle routing with process-specific filenames
            # IMPORTANT: Read original value BEFORE copying/modifying
            original_routing_model = config.get('ROUTING_MODEL', 'none')
            runner_config = config.copy()
            runner_config['ROUTING_MODEL'] = 'none'  # Disable routing in runner

            # Initialize HYPE runner with modified config
            runner = HYPERunner(runner_config, self.logger)

            # Override paths for the worker
            runner.setup_dir = settings_dir.absolute()
            runner.output_dir = hype_output_dir
            runner.output_path = hype_output_dir

            # Run HYPE (without routing - we'll handle routing separately with correct filenames)
            result_path = runner.run_hype()

            # Restore original routing model for needs_routing check
            config['ROUTING_MODEL'] = original_routing_model
            
            if result_path is None:
                return False

            # NOTE: For HYPE calibration, we use direct HYPE output (timeCOUT.txt) rather
            # than routing through mizuRoute. This is because HYPE's timeCOUT.txt contains
            # accumulated discharge at each subbasin outlet, not local runoff. Converting
            # this to mizuRoute input would give incorrect results.
            #
            # mizuRoute routing is only appropriate when we have local runoff per HRU
            # (e.g., from SUMMA or FUSE). For HYPE, the outlet discharge is already
            # correctly routed internally by HYPE itself.

            return True

        except Exception as e:
            self._last_error = str(e)
            self.logger.error(f"Error running HYPE: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate metrics from HYPE or mizuRoute output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            import xarray as xr
            
            # For HYPE, we use direct output (timeCOUT.txt) rather than mizuRoute
            # because HYPE's output is already accumulated/routed discharge, not local runoff.
            use_routed_output = False

            if use_routed_output:
                # Read from mizuRoute NetCDF
                with xr.open_dataset(sim_file_path) as ds:
                    # Find routing variable (try multiple common names)
                    routing_vars = ['IRFroutedRunoff', 'KWTroutedRunoff', 'averageRoutedRunoff']
                    routing_var = None
                    for v in routing_vars:
                        if v in ds.variables:
                            routing_var = v
                            break
                    
                    if routing_var is None:
                        return {'kge': self.penalty_score, 'error': f'No routing variable found in {sim_file_path}'}
                    
                    var = ds[routing_var]
                    
                    # Select outlet segment (highest mean runoff)
                    if 'seg' in var.dims:
                        seg_means = var.mean(dim='time').values
                        outlet_idx = np.argmax(seg_means)
                        simulated = var.isel(seg=outlet_idx)
                        self.logger.debug(f"Selected mizuRoute outlet segment {outlet_idx}")
                    elif 'reachID' in var.dims:
                        reach_means = var.mean(dim='time').values
                        outlet_idx = np.argmax(reach_means)
                        simulated = var.isel(reachID=outlet_idx)
                        self.logger.debug(f"Selected mizuRoute outlet reach {outlet_idx}")
                    else:
                        # Fallback to first if dimension unknown
                        simulated = var.isel({var.dims[1]: 0})
                    
                    sim_series = simulated.to_pandas()
                    # Resample to daily if needed (mizuRoute might be hourly)
                    sim_series = sim_series.resample('D').mean()
                    sim = sim_series.values
                    sim_dates = sim_series.index
            else:
                # Fallback to direct HYPE output (lumped)
                sim_file = output_dir / 'timeCOUT.txt'
                if not sim_file.exists():
                    return {'kge': self.penalty_score, 'error': 'timeCOUT.txt not found'}

                sim_df = pd.read_csv(sim_file, sep='\s+', skiprows=1)
                if 'DATE' in sim_df.columns:
                    sim_df = sim_df.set_index('DATE')
                elif 'time' in sim_df.columns:
                    sim_df = sim_df.set_index('time')

                # Select outlet subbasin (highest mean flow) instead of blindly using first column
                # This ensures we evaluate against the actual watershed outlet
                subbasin_cols = [col for col in sim_df.columns if col not in ['DATE', 'time']]
                if len(subbasin_cols) > 1:
                    # Calculate mean flow for each subbasin and select the outlet (highest flow)
                    subbasin_means = sim_df[subbasin_cols].mean()
                    outlet_col = subbasin_means.idxmax()
                    self.logger.info(f"Auto-selected outlet subbasin {outlet_col} with mean flow {subbasin_means[outlet_col]:.2f} m3/s")
                    self.logger.info(f"All subbasin means: {subbasin_means.to_dict()}")
                    sim = pd.to_numeric(sim_df[outlet_col], errors='coerce').values
                else:
                    # Single subbasin case - use first (and only) column
                    sim = pd.to_numeric(sim_df.iloc[:, 0], errors='coerce').values

                sim_dates = pd.to_datetime(sim_df.index, format='%Y-%m-%d', errors='coerce')

            # Load observations
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            obs_file = (data_dir / f'domain_{domain_name}' / 'observations' /
                       'streamflow' / 'preprocessed' / f'{domain_name}_streamflow_processed.csv')

            if not obs_file.exists():
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
            
            # Resample observations to daily means to match HYPE's daily output
            obs_daily = obs_df.resample('D').mean()
            
            # Align simulation and observations
            sim_series = pd.Series(sim, index=sim_dates).dropna()
            
            # Align by index
            common_idx = sim_series.index.intersection(obs_daily.index)
            if len(common_idx) == 0:
                return {'kge': self.penalty_score, 'error': 'No common dates between sim and obs'}
                
            obs_aligned = obs_daily.loc[common_idx].iloc[:, 0].values
            sim_aligned = sim_series.loc[common_idx].values

            # Remove any remaining NaNs in aligned data
            mask = ~np.isnan(obs_aligned) & ~np.isnan(sim_aligned)
            if np.sum(mask) < 2:
                return {'kge': self.penalty_score, 'error': 'Too few valid data points after alignment'}
                
            obs_aligned = obs_aligned[mask]
            sim_aligned = sim_aligned[mask]

            # Calculate metrics
            # If all sim values are the same (e.g. all 0 during spinup), KGE/NSE will be NaN
            if np.std(sim_aligned) == 0:
                kge_val = -1.0 # Poor but not penalty
                nse_val = -1.0
            else:
                kge_val = kge(obs_aligned, sim_aligned, transfo=1)
                nse_val = nse(obs_aligned, sim_aligned, transfo=1)

            # Handle NaN values from metric functions
            if pd.isna(kge_val) or np.isinf(kge_val):
                kge_val = self.penalty_score
            if pd.isna(nse_val) or np.isinf(nse_val):
                nse_val = self.penalty_score

            return {'kge': float(kge_val), 'nse': float(nse_val)}

        except Exception as e:
            self.logger.error(f"Error calculating HYPE metrics: {e}")
            return {'kge': self.penalty_score}

    def _convert_hype_to_mizuroute_format(
        self,
        hype_output_dir: Path,
        config: Dict[str, Any],
        settings_dir: Path,
        proc_id: int = 0
    ) -> bool:
        """
        Convert HYPE subbasin output to mizuRoute-compatible format.

        Args:
            hype_output_dir: Directory containing HYPE output
            config: Configuration dictionary
            settings_dir: Settings directory
            proc_id: Process ID for parallel calibration

        Returns:
            True if conversion successful
        """
        try:
            import xarray as xr
            import numpy as np

            experiment_id = config.get('EXPERIMENT_ID', 'run_1')
            
            # HYPE subbasin output for computed discharge
            sim_file = hype_output_dir / 'timeCOUT.txt'
            if not sim_file.exists():
                self.logger.error(f"HYPE output file not found: {sim_file}")
                return False

            # Read simulation
            sim_df = pd.read_csv(sim_file, sep='\s+', skiprows=1)
            
            # Extract time index
            if 'DATE' in sim_df.columns:
                time_col = 'DATE'
            elif 'time' in sim_df.columns:
                time_col = 'time'
            else:
                self.logger.error("No time column found in HYPE output")
                return False
                
            times = pd.to_datetime(sim_df[time_col], format='%Y-%m-%d')
            sim_df = sim_df.drop(columns=[time_col])
            
            # Columns are subbasin IDs in order of GeoData.txt
            # We need to map them to 'gru' dimension (0 to N-1)
            # mizuRoute expects gruId variable to map correctly
            subids = [int(c) for c in sim_df.columns]
            n_gru = len(subids)
            
            # mizuRoute expects 'q_routed' or similar variable
            routing_var = config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
            if routing_var in ('default', None, ''):
                routing_var = 'q_routed'

            # Convert units: HYPE cout is usually m3/s at the subbasin outlet
            # mizuRoute usually expects unit runoff (m/s or mm/s)
            # BUT if we use mizuRoute as a simple routing tool, we might pass m3/s
            # and set mizuRoute to not apply area conversion.
            # Standard SYMFLUENCE approach: convert to m/s
            
            # We need subbasin areas to convert m3/s to m/s
            # Read GeoData.txt from settings_dir
            geodata_file = settings_dir / 'GeoData.txt'
            if not geodata_file.exists():
                self.logger.error(f"GeoData.txt not found in {settings_dir}")
                return False
                
            geodata = pd.read_csv(geodata_file, sep='\t')
            # Map subid to area
            area_map = dict(zip(geodata['subid'], geodata['area']))
            
            q_values = sim_df.values # (time, subbasin)
            
            # Convert m3/s to m/s: Q(m3/s) / Area(m2) = V(m/s)
            v_values = np.zeros_like(q_values, dtype=float)
            for i, subid in enumerate(subids):
                area = area_map.get(subid)
                if area and area > 0:
                    v_values[:, i] = q_values[:, i] / area
                else:
                    v_values[:, i] = 0.0

            # Create NetCDF with hru dimension (matches mizuRoute control file expectations)
            ds_routing = xr.Dataset({
                routing_var: (('time', 'hru'), v_values)
            }, coords={
                'time': times.values,
                'hru': np.arange(n_gru)
            })

            ds_routing['hruId'] = ('hru', np.array(subids, dtype=int))
            ds_routing[routing_var].attrs['units'] = 'm/s'
            
            # Save to expected file
            expected_filename = f"proc_{proc_id:02d}_{experiment_id}_timestep.nc"
            expected_file = hype_output_dir / expected_filename
            
            ds_routing.to_netcdf(expected_file)
            ds_routing.close()
            
            return True

        except Exception as e:
            self.logger.error(f"Error converting HYPE output for mizuRoute: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _run_mizuroute_for_hype(
        self,
        config: Dict[str, Any],
        hype_output_dir: Path,
        mizuroute_dir: Path,
        **kwargs
    ) -> bool:
        """
        Execute mizuRoute for HYPE output.
        """
        try:
            import subprocess

            # Get mizuRoute executable
            mizuroute_install = config.get('MIZUROUTE_INSTALL_PATH', 'default')
            if mizuroute_install == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                # Standard SYMFLUENCE install location: installs/mizuRoute/route/bin/mizuRoute.exe
                mizuroute_exe = data_dir / 'installs' / 'mizuRoute' / 'route' / 'bin' / 'mizuRoute.exe'
                # Fallback to alternate locations if not found
                if not mizuroute_exe.exists():
                    alt_paths = [
                        data_dir / 'installs' / 'mizuroute' / 'bin' / 'mizuroute.exe',
                        data_dir / 'installs' / 'mizuRoute' / 'bin' / 'mizuRoute.exe',
                    ]
                    for alt_path in alt_paths:
                        if alt_path.exists():
                            mizuroute_exe = alt_path
                            break
            else:
                mizuroute_exe = Path(mizuroute_install)
                if mizuroute_exe.is_dir():
                    mizuroute_exe = mizuroute_exe / 'mizuRoute.exe'

            if not mizuroute_exe.exists():
                self.logger.error(f"mizuRoute executable not found: {mizuroute_exe}")
                return False

            # settings_dir is HYPE settings (.../settings/HYPE/)
            # mizuRoute settings are in sibling directory (.../settings/mizuRoute/)
            settings_dir_path = Path(kwargs.get('settings_dir', Path('.')))
            control_file = settings_dir_path.parent / 'mizuRoute' / 'mizuroute.control'

            if not control_file.exists():
                self.logger.error(f"mizuRoute control file not found: {control_file}")
                self.logger.debug(f"  settings_dir_path={settings_dir_path}, parent={settings_dir_path.parent}")
                return False

            # Execute mizuRoute
            cmd = [str(mizuroute_exe), str(control_file)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.get('MIZUROUTE_TIMEOUT', 600)
            )

            return result.returncode == 0

        except Exception as e:
            self.logger.error(f"Error running mizuRoute for HYPE: {e}")
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
        return _evaluate_hype_parameters_worker(task_data)


def _evaluate_hype_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
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
        # Signal only works in main thread of the process
        pass

    # Force single-threaded execution for libraries and disable file locking
    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NETCDF_DISABLE_LOCKING': '1',
        'HDF5_USE_FILE_LOCKING': 'FALSE',
        'HDF5_DISABLE_VERSION_CHECK': '1',
    })

    # Add small random delay to stagger file system access in parallel
    initial_delay = random.uniform(0.1, 0.8)
    time.sleep(initial_delay)

    try:
        worker = HYPEWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': -999.0,
            'error': f'Critical HYPE worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
