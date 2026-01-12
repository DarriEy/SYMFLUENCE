"""
HYPE Worker

Worker implementation for HYPE model optimization.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
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
            config = kwargs.get('config', self.config)
            
            # Use HYPEPreProcessor to regenerate configs with new params
            # We only need to regenerate the par.txt file, but calling
            # preprocess_models with params is the cleanest way.
            preprocessor = HYPEPreProcessor(config, self.logger, params=params)

            # Set model-specific paths to point to the worker's settings dir
            # Ensure settings_dir is a Path object
            settings_dir = Path(settings_dir) if not isinstance(settings_dir, Path) else settings_dir

            preprocessor.output_path = settings_dir
            preprocessor.hype_setup_dir = settings_dir
            # IMPORTANT: forcing_data_dir must point to where forcing files are located
            # Forcing files are copied to worker's settings dir by copy_base_settings
            preprocessor.forcing_data_dir = settings_dir

            # CRITICAL: Also update the manager output paths so par.txt and config
            # files are written to the worker's isolated directory, not the shared default
            preprocessor.config_manager.output_path = settings_dir
            preprocessor.geodata_manager.output_path = settings_dir

            # Use isolated output directory for the worker
            output_dir = kwargs.get('proc_output_dir') or kwargs.get('output_dir')
            if output_dir:
                preprocessor.hype_results_dir = Path(output_dir)
                preprocessor.hype_results_dir_str = str(Path(output_dir)).rstrip('/') + '/'

            # Only regenerate par.txt and info.txt/filedir.txt
            # GeoData and forcing files should already be copied to worker directory
            # and don't change during calibration
            self.logger.debug("Regenerating par.txt and info.txt for calibration")
            self.logger.debug(f"  output_path: {preprocessor.output_path}")
            self.logger.debug(f"  forcing_data_dir: {preprocessor.forcing_data_dir}")

            # Get land uses from existing GeoClass.txt (should already be copied)
            geoclass_file = settings_dir / 'GeoClass.txt'
            if not geoclass_file.exists():
                self.logger.error(f"GeoClass.txt not found at {geoclass_file}")
                self.logger.error("GeoData files must be copied to worker directory before calibration")
                return False

            # Read land uses from GeoClass.txt
            import pandas as pd
            try:
                geoclass_df = pd.read_csv(geoclass_file, sep='\t', skiprows=1, header=None)
                land_uses = geoclass_df.iloc[:, 1].unique()
                self.logger.debug(f"Read {len(land_uses)} land use types from GeoClass.txt")
            except Exception as e:
                self.logger.error(f"Failed to read GeoClass.txt: {e}")
                return False

            # Write parameter file with calibration parameters
            preprocessor.config_manager.write_par_file(
                params=params,
                land_uses=land_uses
            )

            # Get experiment dates from config
            experiment_start = preprocessor.config_dict.get('EXPERIMENT_TIME_START')
            experiment_end = preprocessor.config_dict.get('EXPERIMENT_TIME_END')

            # Write info and file directory files
            preprocessor.config_manager.write_info_filedir(
                spinup_days=preprocessor.spinup_days,
                results_dir=preprocessor.hype_results_dir_str,
                experiment_start=experiment_start,
                experiment_end=experiment_end,
                forcing_data_dir=preprocessor.forcing_data_dir
            )

            self.logger.debug("par.txt and info.txt regeneration completed")

            # Debug: Check if forcing files exist
            forcing_files = ['Pobs.txt', 'TMAXobs.txt', 'TMINobs.txt', 'Tobs.txt']
            for f in forcing_files:
                fpath = settings_dir / f
                if not fpath.exists():
                    self.logger.error(f"Missing forcing file during calibration: {fpath}")
                else:
                    self.logger.debug(f"Forcing file exists: {fpath}")

            # Debug: Check if filedir.txt points to correct location
            filedir_path = settings_dir / 'filedir.txt'
            if filedir_path.exists():
                with open(filedir_path, 'r') as f:
                    forcing_path_in_filedir = f.read().strip()
                self.logger.debug(f"filedir.txt points to: {forcing_path_in_filedir}")
                if not Path(forcing_path_in_filedir).exists():
                    self.logger.error(f"Forcing path in filedir.txt does not exist: {forcing_path_in_filedir}")
            else:
                self.logger.error(f"filedir.txt not found at {filedir_path}")

            # Debug: Check if par.txt was updated with calibration parameters
            par_file = settings_dir / 'par.txt'
            if par_file.exists() and params:
                with open(par_file, 'r') as f:
                    par_content = f.read()
                for param_name in list(params.keys())[:3]:  # Check first 3 params
                    if param_name in par_content:
                        self.logger.debug(f"Parameter {param_name} found in par.txt")
                    else:
                        self.logger.warning(f"Parameter {param_name} NOT found in par.txt")

            return True

        except Exception as e:
            self.logger.error(f"Error applying HYPE parameters: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
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
            # Ensure paths are Path objects
            settings_dir = Path(settings_dir) if not isinstance(settings_dir, Path) else settings_dir
            output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir

            self.logger.debug(f"HYPE Worker run_model called with settings_dir={settings_dir}, output_dir={output_dir}")

            # Initialize HYPE runner
            runner = HYPERunner(config, self.logger)

            self.logger.debug(f"HYPE Runner initialized with setup_dir={runner.setup_dir}")

            # Override paths for the worker
            runner.setup_dir = settings_dir
            runner.output_dir = output_dir
            runner.output_path = output_dir

            self.logger.debug(f"HYPE Runner paths overridden: setup_dir={runner.setup_dir}, output_dir={runner.output_dir}")

            # Debug: Check if required input files exist before running
            required_files = ['par.txt', 'info.txt', 'filedir.txt', 'GeoData.txt', 'GeoClass.txt', 'ForcKey.txt']
            for f in required_files:
                fpath = settings_dir / f
                if not fpath.exists():
                    self.logger.error(f"Required HYPE input file missing before run: {fpath}")

            # Run HYPE
            result_path = runner.run_hype()

            if result_path is None:
                self.logger.error("HYPE run_hype returned None - model may have failed or outputs not found")
                self.logger.error(f"Expected outputs in: {runner.output_dir}")
                # Debug: Check if any output files were created
                if output_dir.exists():
                    output_files = list(output_dir.glob('*.txt'))
                    if output_files:
                        self.logger.error(f"Found {len(output_files)} .txt files in output_dir: {[f.name for f in output_files]}")
                    else:
                        self.logger.error(f"No .txt files found in output_dir: {output_dir}")

            return result_path is not None

        except Exception as e:
            self.logger.error(f"Error running HYPE: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate metrics from HYPE output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        output_dir = Path(output_dir)

        try:
            # HYPE output file for computed discharge
            sim_file = output_dir / 'timeCOUT.txt'
            if not sim_file.exists():
                self.logger.error(f"timeCOUT.txt not found at {sim_file}")
                return {'kge': self.penalty_score, 'error': 'timeCOUT.txt not found'}

            # Read simulation (HYPE output is tab-separated, first row is comment)
            sim_df = pd.read_csv(sim_file, sep='\t', skiprows=1)

            # Parse DATE column
            if 'DATE' in sim_df.columns:
                sim_df['DATE'] = pd.to_datetime(sim_df['DATE'])
                sim_df = sim_df.set_index('DATE')
            elif 'time' in sim_df.columns:
                sim_df['time'] = pd.to_datetime(sim_df['time'])
                sim_df = sim_df.set_index('time')

            # Get subbasin columns (exclude date columns)
            subbasin_cols = [col for col in sim_df.columns if col not in ['DATE', 'time']]
            if len(subbasin_cols) == 0:
                return {'kge': self.penalty_score, 'error': 'No subbasin columns in output'}

            # For lumped domains or auto-select outlet (highest mean flow)
            if len(subbasin_cols) > 1:
                # Convert to numeric first
                for col in subbasin_cols:
                    sim_df[col] = pd.to_numeric(sim_df[col], errors='coerce')
                subbasin_means = sim_df[subbasin_cols].mean()
                outlet_col = subbasin_means.idxmax()
                sim_series = sim_df[outlet_col]
            else:
                outlet_col = subbasin_cols[0]
                sim_series = pd.to_numeric(sim_df[outlet_col], errors='coerce')

            # Load observations
            # Handle both flat config dict and nested Pydantic model config
            if hasattr(config, 'domain') and hasattr(config.domain, 'name'):
                # Pydantic model
                domain_name = config.domain.name
                data_dir = Path(config.system.data_dir) if hasattr(config, 'system') else Path('.')
            elif isinstance(config, dict):
                # Dict format - check for nested keys first, then flat uppercase keys
                domain = config.get('domain', {})
                system = config.get('system', {})
                # Try nested format first
                domain_name = domain.get('name') if isinstance(domain, dict) and domain else None
                data_dir_val = system.get('data_dir') if isinstance(system, dict) and system else None
                # Fall back to flat uppercase keys
                if not domain_name:
                    domain_name = config.get('DOMAIN_NAME')
                if not data_dir_val:
                    data_dir_val = config.get('SYMFLUENCE_DATA_DIR')
                data_dir = Path(data_dir_val) if data_dir_val else Path('.')
            else:
                domain_name = None
                data_dir = Path('.')

            obs_file = (data_dir / f'domain_{domain_name}' / 'observations' /
                       'streamflow' / 'preprocessed' / f'{domain_name}_streamflow_processed.csv')

            if not obs_file.exists():
                self.logger.error(f"Observations file not found at {obs_file} (domain_name={domain_name})")
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

            # HYPE outputs daily data, observations may be hourly
            # Resample observations to daily mean if they are sub-daily
            obs_freq = pd.infer_freq(obs_df.index[:10])
            if obs_freq and obs_freq in ['H', 'h', 'T', 'min', 'S', 's']:
                # Hourly or sub-hourly observations - resample to daily mean
                obs_daily = obs_df.resample('D').mean()
            else:
                obs_daily = obs_df

            # Normalize both indices to date-only for alignment
            sim_series.index = sim_series.index.normalize()
            obs_daily.index = obs_daily.index.normalize()

            # Apply calibration period if configured
            calib_period = config.get('CALIBRATION_PERIOD')
            if calib_period:
                try:
                    start_str, end_str = [s.strip() for s in calib_period.split(',')]
                    calib_start = pd.to_datetime(start_str)
                    calib_end = pd.to_datetime(end_str)
                    sim_series = sim_series[(sim_series.index >= calib_start) & (sim_series.index <= calib_end)]
                    obs_daily = obs_daily[(obs_daily.index >= calib_start) & (obs_daily.index <= calib_end)]
                except Exception as e:
                    self.logger.warning(f"Could not apply calibration period: {e}")

            # Find common dates
            common_idx = sim_series.index.intersection(obs_daily.index)
            if len(common_idx) == 0:
                return {'kge': self.penalty_score, 'error': 'No common dates between sim and obs'}

            # Get the discharge column from observations
            obs_col = obs_daily.columns[0] if len(obs_daily.columns) > 0 else 'discharge_cms'
            obs_aligned = obs_daily.loc[common_idx, obs_col].values.flatten() if obs_col in obs_daily.columns else obs_daily.loc[common_idx].values.flatten()
            sim_aligned = sim_series.loc[common_idx].values.flatten()

            # Check for all-zero simulations (model didn't produce discharge)
            if sim_aligned.sum() == 0:
                self.logger.debug("HYPE simulation produced zero discharge")
                return {'kge': self.penalty_score, 'error': 'Zero discharge from model'}

            kge_val = kge(obs_aligned, sim_aligned, transfo=1)
            nse_val = nse(obs_aligned, sim_aligned, transfo=1)

            # Handle NaN values
            if pd.isna(kge_val):
                kge_val = self.penalty_score
            if pd.isna(nse_val):
                nse_val = self.penalty_score

            return {'kge': float(kge_val), 'nse': float(nse_val)}

        except Exception as e:
            self.logger.error(f"Error calculating HYPE metrics: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
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
        return _evaluate_hype_parameters_worker(task_data)


def _evaluate_hype_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    worker = HYPEWorker()
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
