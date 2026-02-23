"""
FUSE Model Visualizer.

Provides model-specific visualization registration for FUSE.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_visualizer('FUSE')
def visualize_fuse(reporting_manager: Any, config: Dict[str, Any], project_dir: Path, experiment_id: str, workflow: List[str]):
    """
    Visualize FUSE model outputs.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running FUSE visualizer for experiment {experiment_id}")

    domain_name = config.get('DOMAIN_NAME')

    model_outputs = [("FUSE", str(project_dir / "simulations" / experiment_id / "FUSE" / f"{domain_name}_{experiment_id}_runs_best.nc"))]
    obs_files = [('Observed', str(resolve_data_subdir(project_dir, 'observations') / "streamflow" / "preprocessed" / f"{domain_name}_streamflow_processed.csv"))]
    reporting_manager.visualize_fuse_outputs(model_outputs, obs_files)
