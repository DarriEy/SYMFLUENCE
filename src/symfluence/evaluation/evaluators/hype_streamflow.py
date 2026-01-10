#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HYPE Streamflow Evaluator
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from symfluence.evaluation.registry import EvaluationRegistry
from symfluence.evaluation.output_file_locator import OutputFileLocator
from .streamflow import StreamflowEvaluator

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


@EvaluationRegistry.register('HYPE_STREAMFLOW')
class HYPEStreamflowEvaluator(StreamflowEvaluator):
    """Streamflow evaluator for HYPE models"""

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """Get HYPE output files (timeCOUT.txt or mizuRoute)."""
        locator = OutputFileLocator(self.logger)
        return locator.find_hype_output(sim_dir, 'streamflow')

    def extract_simulated_data(self, sim_files: List[Path], **kwargs) -> pd.Series:
        """Extract streamflow data from HYPE timeCOUT.txt"""
        sim_file = sim_files[0]
        self.logger.info(f"Extracting HYPE streamflow from: {sim_file}")

        # Check if it's a NetCDF file (mizuRoute output)
        if sim_file.suffix == '.nc':
            return self._extract_mizuroute_streamflow(sim_file)

        # Otherwise, process HYPE timeCOUT.txt
        return self._extract_hype_streamflow(sim_file)

    def _extract_hype_streamflow(self, sim_file: Path) -> pd.Series:
        """
        Extract streamflow from HYPE timeCOUT.txt file.

        HYPE outputs streamflow for all subbasins in timeCOUT.txt.
        We auto-select the outlet subbasin (highest mean flow) for evaluation.
        """
        try:
            # Read timeCOUT.txt (tab-separated, skip first comment line)
            df = pd.read_csv(sim_file, sep='\t', skiprows=1)

            # Parse dates
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'], format='%Y-%m-%d')
                df = df.set_index('DATE')
            elif 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')

            # Get all subbasin columns (numeric column names)
            subbasin_cols = [col for col in df.columns if col not in ['DATE', 'time']]

            if len(subbasin_cols) == 0:
                self.logger.error(f"No subbasin columns found in {sim_file}")
                return None

            # Auto-select outlet subbasin (highest mean flow)
            if len(subbasin_cols) > 1:
                subbasin_means = df[subbasin_cols].mean()
                outlet_col = subbasin_means.idxmax()
                self.logger.info(f"Auto-selected HYPE outlet subbasin {outlet_col} (mean flow: {subbasin_means[outlet_col]:.2f} m3/s)")
                self.logger.debug(f"All subbasin mean flows: {dict(sorted(subbasin_means.items(), key=lambda x: x[1], reverse=True)[:5])}")
                streamflow = df[outlet_col]
            else:
                # Single subbasin - use it directly
                outlet_col = subbasin_cols[0]
                self.logger.info(f"Using single HYPE subbasin: {outlet_col}")
                streamflow = df[outlet_col]

            # Convert to numeric and handle errors
            streamflow = pd.to_numeric(streamflow, errors='coerce')

            # HYPE timeCOUT.txt is already in m3/s - no unit conversion needed
            return streamflow

        except Exception as e:
            self.logger.error(f"Error extracting HYPE streamflow from {sim_file}: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
