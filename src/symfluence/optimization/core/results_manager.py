"""
Optimization results tracking, history management, and visualization.

Handles storage, retrieval, and plotting of optimization trial history
for calibration experiments.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import ValidationError

from symfluence.core.mixins import ConfigMixin

class ResultsManager(ConfigMixin):
    """Handles optimization results, history tracking, and visualization"""

    def __init__(self, config: Dict, logger: logging.Logger, output_dir: Path, reporting_manager: Optional[Any] = None):
        # Import here to avoid circular imports

        from symfluence.core.config.models import SymfluenceConfig



        # Auto-convert dict to typed config for backward compatibility

        if isinstance(config, dict):

            try:

                self._config = SymfluenceConfig(**config)

            except (ValidationError, TypeError):

                # Fallback for partial configs (e.g., in tests)

                self._config = config

        else:

            self._config = config
        self.logger = logger
        self.output_dir = output_dir
        self.reporting_manager = reporting_manager
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

    def save_results(self, best_params: Dict, best_score: float, history: List[Dict],
                    final_result: Optional[Dict] = None) -> bool:
        """Save optimization results to files"""
        try:
            # Save best parameters to CSV
            self._save_best_parameters_csv(best_params)

            # Save history to CSV
            self._save_history_csv(history)

            # Save metadata
            self._save_metadata(best_score, len(history), final_result)

            # Create visualization plots
            self._create_plots(history, best_params)

            return True

        except (IOError, OSError, ValueError) as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return False

    def _save_best_parameters_csv(self, best_params: Dict) -> None:
        """Save best parameters to CSV file"""
        param_data = []

        for param_name, values in best_params.items():
            if isinstance(values, np.ndarray):
                if len(values) == 1:
                    param_data.append({
                        'parameter': param_name,
                        'value': values[0],
                        'type': 'scalar'
                    })
                else:
                    param_data.append({
                        'parameter': param_name,
                        'value': np.mean(values),
                        'type': 'array_mean',
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values)
                    })
            else:
                param_data.append({
                    'parameter': param_name,
                    'value': values,
                    'type': 'scalar'
                })

        param_df = pd.DataFrame(param_data)
        param_csv_path = self.output_dir / "best_parameters.csv"
        param_df.to_csv(param_csv_path, index=False)

        self.logger.info(f"Saved best parameters to: {param_csv_path}")

    def _save_history_csv(self, history: List[Dict]) -> None:
        """Save optimization history to CSV"""
        if not history:
            return

        history_data = []
        for gen_data in history:
            row = {
                'generation': gen_data.get('generation', 0),
                'best_score': gen_data.get('best_score'),
                'mean_score': gen_data.get('mean_score'),
                'std_score': gen_data.get('std_score'),
                'valid_individuals': gen_data.get('valid_individuals', 0)
            }

            # Add best parameters if available
            if gen_data.get('best_params'):
                for param_name, values in gen_data['best_params'].items():
                    if isinstance(values, np.ndarray):
                        row[f'best_{param_name}'] = np.mean(values) if len(values) > 1 else values[0]
                    else:
                        row[f'best_{param_name}'] = values

            history_data.append(row)

        history_df = pd.DataFrame(history_data)
        history_csv_path = self.output_dir / "optimization_history.csv"
        history_df.to_csv(history_csv_path, index=False)

        self.logger.info(f"Saved optimization history to: {history_csv_path}")

    def _save_metadata(self, best_score: float, num_generations: int, final_result: Optional[Dict]) -> None:
        """Save optimization metadata"""
        metadata = {
            'algorithm': 'Differential Evolution',
            'domain_name': self.domain_name,
            'experiment_id': self.experiment_id,
            'calibration_variable': self._get_config_value(lambda: self.config.optimization.calibration_variable, default='streamflow', dict_key='CALIBRATION_VARIABLE'),
            'target_metric': self._get_config_value(lambda: self.config.optimization.metric, default='KGE', dict_key='OPTIMIZATION_METRIC'),
            'best_score': best_score,
            'num_generations': num_generations,
            'population_size': self._get_config_value(lambda: self.config.optimization.population_size, default=50, dict_key='POPULATION_SIZE'),
            'F': self._get_config_value(lambda: self.config.optimization.de.scaling_factor, default=0.5, dict_key='DE_SCALING_FACTOR'),
            'CR': self._get_config_value(lambda: self.config.optimization.de.crossover_rate, default=0.9, dict_key='DE_CROSSOVER_RATE'),
            'parallel_processes': self._get_config_value(lambda: self.config.system.num_processes, default=1, dict_key='NUM_PROCESSES'),
            'completed_at': datetime.now().isoformat()
        }

        if final_result:
            metadata.update(final_result)

        metadata_df = pd.DataFrame([metadata])
        metadata_csv_path = self.output_dir / "optimization_metadata.csv"
        metadata_df.to_csv(metadata_csv_path, index=False)

        self.logger.info(f"Saved metadata to: {metadata_csv_path}")

    def _create_plots(self, history: List[Dict], best_params: Dict) -> None:
        """Create optimization progress plots"""
        if not self.reporting_manager:
            return

        calibration_variable = self._get_config_value(lambda: self.config.optimization.calibration_variable, default="streamflow", dict_key='CALIBRATION_VARIABLE')
        metric = self._get_config_value(lambda: self.config.optimization.metric, default='KGE', dict_key='OPTIMIZATION_METRIC')
        self.reporting_manager.visualize_optimization_progress(history, self.output_dir, calibration_variable, metric)

        # Parameter evolution plots for depth parameters
        if self._get_config_value(lambda: self.config.model.summa.calibrate_depth, default=False, dict_key='CALIBRATE_DEPTH'):
            self._create_depth_parameter_plots(history, self.output_dir)

    def _create_depth_parameter_plots(self, history: List[Dict], plots_dir: Path) -> None:
        """Create depth parameter evolution plots"""
        if not self.reporting_manager:
            return

        self.reporting_manager.visualize_optimization_depth_parameters(history, self.output_dir)
