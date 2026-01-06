"""
NextGen (NGEN) Worker

Worker implementation for NextGen model optimization.
Delegates to existing worker functions while providing BaseWorker interface.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .base_worker import BaseWorker, WorkerTask, WorkerResult
from ..registry import OptimizerRegistry


logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('NGEN')
class NgenWorker(BaseWorker):
    """
    Worker for NextGen (ngen) model calibration.

    Handles parameter application to JSON config files, ngen execution,
    and metric calculation for streamflow calibration.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ngen worker.

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
        Apply parameters to ngen JSON configuration files.

        Parameters use MODULE.param naming convention (e.g., CFE.Kn).

        Args:
            params: Parameter values to apply (MODULE.param format)
            settings_dir: Ngen settings directory (isolated for parallel workers)
            **kwargs: Additional arguments including 'config'

        Returns:
            True if successful
        """
        try:
            # Use settings_dir directly - BaseOptimizer ensures this path is correct for the worker
            # NGEN settings are typically organized by module subdirectories within settings_dir
            ngen_setup_dir = settings_dir

            # Group parameters by module
            module_params: Dict[str, Dict[str, float]] = {}
            for param_name, value in params.items():
                if '.' in param_name:
                    module, param = param_name.split('.', 1)
                    if module not in module_params:
                        module_params[module] = {}
                    module_params[module][param] = value
                else:
                    self.logger.warning(f"Parameter {param_name} missing module prefix")

            # Update each module's config file
            for module, module_param_dict in module_params.items():
                config_file = ngen_setup_dir / module / f"{module.lower()}_config.json"

                if not config_file.exists():
                    self.logger.warning(f"Config file not found: {config_file}")
                    continue

                # Read existing config
                with open(config_file, 'r') as f:
                    cfg = json.load(f)

                # Update parameters
                updated = False
                for param, value in module_param_dict.items():
                    if param in cfg:
                        cfg[param] = value
                        updated = True
                    else:
                        self.logger.warning(f"Parameter {param} not found in {module} config")

                if updated:
                    # Write updated config
                    with open(config_file, 'w') as f:
                        json.dump(cfg, f, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Error applying ngen parameters: {e}")
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run ngen model.

        Supports both serial and parallel execution modes.

        Args:
            config: Configuration dictionary
            settings_dir: Ngen settings directory
            output_dir: Output directory
            **kwargs: Additional arguments including parallel config keys

        Returns:
            True if model ran successfully
        """
        try:
            # Use a dictionary for local modifications to avoid SymfluenceConfig immutability/subscriptability issues
            if hasattr(config, 'to_dict'):
                parallel_config = config.to_dict(flatten=True)
            else:
                parallel_config = config.copy()

            # Ensure runner uses isolated directories
            parallel_config['_ngen_output_dir'] = str(output_dir)
            parallel_config['_ngen_settings_dir'] = str(settings_dir)

            # Import NgenRunner
            from symfluence.models.ngen import NgenRunner

            experiment_id = parallel_config.get('EXPERIMENT_ID')

            # Initialize and run
            runner = NgenRunner(parallel_config, self.logger)
            success = runner.run_ngen(experiment_id)

            return success

        except FileNotFoundError as e:
            error_msg = f"Required ngen input file not found: {e}"
            self.logger.error(error_msg)
            self._last_error = error_msg
            return False
        except Exception as e:
            error_msg = f"Error running ngen: {e}"
            self.logger.error(error_msg)
            import traceback
            self._last_error = error_msg + "\n" + traceback.format_exc()
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate metrics from ngen output.

        Args:
            output_dir: Directory containing model outputs (isolated)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            # Try to use calibration target
            from ..calibration_targets import NgenStreamflowTarget

            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"

            # Create calibration target
            target = NgenStreamflowTarget(config, project_dir, self.logger)

            # Calculate metrics using isolated output_dir
            # NgenStreamflowTarget needs to be aware of the isolated directory
            metrics = target.calculate_metrics(experiment_id=experiment_id, output_dir=output_dir)

            # Normalize metric keys to lowercase
            return {k.lower(): float(v) for k, v in metrics.items()}

        except ImportError:
            # Fallback: Calculate metrics directly
            return self._calculate_metrics_direct(output_dir, config)

        except Exception as e:
            self.logger.error(f"Error calculating ngen metrics: {e}")
            return {'kge': self.penalty_score}

    def _calculate_metrics_direct(
        self,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate metrics directly from ngen output files.

        Args:
            output_dir: Output directory (isolated)
            config: Configuration dictionary

        Returns:
            Dictionary of metrics
        """
        try:
            import pandas as pd
            from symfluence.evaluation.metrics import kge, nse

            domain_name = config.get('DOMAIN_NAME')
            
            # Find ngen output in isolated output_dir
            output_files = list(output_dir.glob('*.csv')) + list(output_dir.glob('*.nc'))

            if not output_files:
                return {'kge': self.penalty_score, 'error': 'No output files found'}

            # Read simulation
            if output_files[0].suffix == '.csv':
                sim_df = pd.read_csv(output_files[0], index_col=0, parse_dates=True)
                if 'q_cms' in sim_df.columns:
                    sim = sim_df['q_cms'].values
                else:
                    sim = sim_df.iloc[:, 0].values
            else:
                import xarray as xr
                with xr.open_dataset(output_files[0]) as ds:
                    # Generic extraction - pick first data variable
                    var = next(iter(ds.data_vars))
                    sim = ds[var].values.flatten()

            # Load observations
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"
            obs_file = (project_dir / 'observations' / 'streamflow' / 'preprocessed' /
                       f'{domain_name}_streamflow_processed.csv')

            if not obs_file.exists():
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
            
            # Simple alignment (actual implementation might need more robustness)
            # This is a fallback so we keep it simple
            min_len = min(len(sim), len(obs_df)) # <--- Here it uses obs_df
            sim_vals = sim[:min_len]
            obs_vals = obs_df['discharge_cms'].values[:min_len]

            kge_val = kge(obs_vals, sim_vals, transfo=1)
            nse_val = nse(obs_vals, sim_vals, transfo=1)

            return {'kge': float(kge_val), 'nse': float(nse_val)}

        except Exception as e:
            self.logger.error(f"Error in direct ngen metrics calculation: {e}")
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
        return _evaluate_ngen_parameters_worker(task_data)


def _evaluate_ngen_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    worker = NgenWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
