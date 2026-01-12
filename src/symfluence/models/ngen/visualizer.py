"""
NGen Model Visualizer.

Provides model-specific visualization registration for NGen.
"""

from pathlib import Path
from typing import Dict, Any, List
import logging

from symfluence.models.registry import ModelRegistry

@ModelRegistry.register_visualizer('NGEN')
def visualize_ngen(reporting_manager: Any, config: Dict[str, Any], project_dir: Path, experiment_id: str, workflow: List[str]):
    """
    Visualize NGen model outputs.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running NGen visualizer for experiment {experiment_id}")
    
    try:
        # NGen results are consolidated into the main results file by the postprocessor.
        reporting_manager.visualize_timeseries_results()
        
    except Exception as e:
        logger.error(f"Error during NGen visualization: {str(e)}")
