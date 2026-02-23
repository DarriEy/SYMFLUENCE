"""
MESH Model Visualizer.

Provides model-specific visualization registration for MESH.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_visualizer('MESH')
def visualize_mesh(reporting_manager: Any, config: Dict[str, Any], project_dir: Path, experiment_id: str, workflow: List[str]):
    """
    Visualize MESH model outputs.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running MESH visualizer for experiment {experiment_id}")

    try:
        # MESH results are consolidated into the main results file by the postprocessor.
        reporting_manager.visualize_timeseries_results()

    except Exception as e:  # noqa: BLE001 — model execution resilience
        logger.error(f"Error during MESH visualization: {str(e)}")
