# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NGen Model Visualizer.

Provides model-specific visualization registration for NGen.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from symfluence.core.registries import R


@R.visualizers.add('NGEN')
def visualize_ngen(reporting_manager: Any, config: Dict[str, Any], project_dir: Path, experiment_id: str, workflow: List[str]):
    """
    Visualize NGen model outputs.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running NGen visualizer for experiment {experiment_id}")

    try:
        # NGen results are consolidated into the main results file by the postprocessor.
        reporting_manager.visualize_timeseries_results()

    except Exception as e:  # noqa: BLE001 — model execution resilience
        logger.error(f"Error during NGen visualization: {str(e)}", exc_info=True)
