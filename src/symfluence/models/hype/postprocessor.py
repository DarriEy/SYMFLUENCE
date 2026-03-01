# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HYPE model postprocessor.

Handles output extraction, processing, and analysis for HYPE model outputs.
Migrated to use StandardModelPostprocessor for reduced boilerplate (Phase 1.2).
"""

from pathlib import Path
from typing import Dict

from ..base import StandardModelPostprocessor
from ..registry import ModelRegistry


@ModelRegistry.register_postprocessor('HYPE')
class HYPEPostProcessor(StandardModelPostprocessor):
    """
    Postprocessor for HYPE model outputs within SYMFLUENCE.

    Handles output extraction, processing, and analysis.
    Uses StandardModelPostprocessor with configuration-based extraction.

    HYPE outputs already-routed discharge at each subbasin outlet in timeCOUT.txt,
    so mizuRoute is not needed (HYPE's internal routing is sufficient).

    Attributes:
        model_name: "HYPE"
        output_file_pattern: "timeCOUT.txt"
        text_file_separator: Tab-separated
        outlet_selection_method: Falls back to highest discharge if config ID not found
    """

    # Model identification
    model_name = "HYPE"

    # Output file configuration
    output_file_pattern = "timeCOUT.txt"

    # Text file parsing
    text_file_separator = "\t"
    text_file_skiprows = 1  # Skip header row
    text_file_date_column = "DATE"
    text_file_flow_column = "config:SIM_REACH_ID"  # Get column name from config

    # Outlet selection: try config first, fall back to highest discharge
    outlet_selection_method = "highest_discharge"

    # Streamflow is already in cms from HYPE
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        """Return model name for HYPE."""
        return "HYPE"

    def extract_results(self) -> Dict[str, Path]:
        """
        Extract and process all HYPE results.

        Returns:
            Dict[str, Path]: Paths to processed result files
        """
        self.logger.info("Extracting HYPE results")
        results = {}

        try:
            # Process streamflow using StandardModelPostprocessor
            streamflow_path = self.extract_streamflow()
            if streamflow_path:
                results['streamflow'] = streamflow_path
                self.logger.info("Streamflow extracted successfully")

            return results

        except Exception as e:  # noqa: BLE001 — wrap-and-raise to domain error
            self.logger.error(f"Error extracting HYPE results: {str(e)}")
            raise
