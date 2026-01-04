"""
GR Model Visualizer.

Provides model-specific visualization registration for GR.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from symfluence.models.registry import ModelRegistry

@ModelRegistry.register_visualizer('GR')
def visualize_gr(reporting_manager: Any, config: Dict[str, Any], project_dir: Path, experiment_id: str, workflow: List[str]):
    """
    Visualize GR model outputs.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running GR visualizer for experiment {experiment_id}")
    
    try:
        # GR results are consolidated into the main results file by the postprocessor.
        reporting_manager.visualize_timeseries_results()
        
    except Exception as e:
        logger.error(f"Error during GR visualization: {str(e)}")
