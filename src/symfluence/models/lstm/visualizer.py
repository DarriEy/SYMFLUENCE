# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
LSTM Model Visualizer.

Provides model-specific visualization registration for LSTM.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from symfluence.models.registry import ModelRegistry


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

    except Exception as e:  # noqa: BLE001 — model execution resilience
        logger.error(f"Error during LSTM visualization: {str(e)}")
