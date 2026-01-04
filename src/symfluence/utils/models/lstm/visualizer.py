"""
LSTM Model Visualizer.

Provides model-specific visualization registration for LSTM.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from symfluence.utils.models.registry import ModelRegistry

@ModelRegistry.register_visualizer('LSTM')
def visualize_lstm(reporting_manager: Any, config: Dict[str, Any], project_dir: Path, experiment_id: str, workflow: List[str]):
    """
    Visualize LSTM model outputs.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running LSTM visualizer for experiment {experiment_id}")
    
    try:
        # LSTM results are consolidated into the main results file by the postprocessor.
        reporting_manager.visualize_timeseries_results()
        
    except Exception as e:
        logger.error(f"Error during LSTM visualization: {str(e)}")
