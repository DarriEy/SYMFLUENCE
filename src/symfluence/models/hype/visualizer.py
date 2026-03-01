# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HYPE Model Visualizer.

Provides model-specific visualization registration for HYPE.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_visualizer('HYPE')
def visualize_hype(reporting_manager: Any, config: Dict[str, Any], project_dir: Path, experiment_id: str, workflow: List[str]):
    """
    Visualize HYPE model outputs.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running HYPE visualizer for experiment {experiment_id}")

    try:
        # HYPE results are typically consolidated into the main results file
        # by the postprocessor. We can use the standard timeseries visualization.
        reporting_manager.visualize_timeseries_results()

        # If specific HYPE plotting is needed beyond standard timeseries:
        # (This is what plot_hype_results does, but it requires explicit dataframes)
        # For now, timeseries_results covers the streamflow comparison which is the primary goal.

    except Exception as e:  # noqa: BLE001 — model execution resilience
        logger.error(f"Error during HYPE visualization: {str(e)}")
