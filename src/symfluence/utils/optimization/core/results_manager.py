import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

class ResultsManager:
    """Handles optimization results, history tracking, and visualization"""
    
    def __init__(self, config: Dict, logger: logging.Logger, output_dir: Path, reporting_manager: Optional[Any] = None):
        self.config = config
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
            
        except Exception as e:
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
            'calibration_variable': self.config.get('CALIBRATION_VARIABLE', 'streamflow'),
            'target_metric': self.config.get('OPTIMIZATION_METRIC', 'KGE'),
            'best_score': best_score,
            'num_generations': num_generations,
            'population_size': self.config.get('POPULATION_SIZE', 50),
            'F': self.config.get('DE_SCALING_FACTOR', 0.5),
            'CR': self.config.get('DE_CROSSOVER_RATE', 0.9),
            'parallel_processes': self.config.get('MPI_PROCESSES', 1),
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

        calibration_variable = self.config.get("CALIBRATION_VARIABLE", "streamflow")
        metric = self.config.get('OPTIMIZATION_METRIC', 'KGE')
        self.reporting_manager.visualize_optimization_progress(history, self.output_dir, calibration_variable, metric)
        
        # Parameter evolution plots for depth parameters
        if self.config.get('CALIBRATE_DEPTH', False):
            self._create_depth_parameter_plots(history, self.output_dir)
    
    def _create_depth_parameter_plots(self, history: List[Dict], plots_dir: Path) -> None:
        """Create depth parameter evolution plots"""
        if not self.reporting_manager:
            return
            
        self.reporting_manager.visualize_optimization_depth_parameters(history, self.output_dir)
