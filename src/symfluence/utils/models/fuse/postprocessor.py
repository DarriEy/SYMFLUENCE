import csv
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from shutil import rmtree, copyfile
from typing import Dict, Any, Optional, List, Tuple

import geopandas as gpd # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import rasterio # type: ignore
import xarray as xr # type: ignore
from scipy import ndimage

from ..base import BaseModelPreProcessor, BaseModelPostProcessor
from ..mixins import PETCalculatorMixin
from ..registry import ModelRegistry
from symfluence.utils.common.constants import UnitConversion
from symfluence.utils.common.metrics import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE
from symfluence.utils.data.utilities.variable_utils import VariableHandler # type: ignore


@ModelRegistry.register_postprocessor('FUSE')
class FUSEPostprocessor(BaseModelPostProcessor):
    """
    Postprocessor for FUSE (Framework for Understanding Structural Errors) model outputs.
    Handles extraction, processing, and saving of simulation results.

    Attributes:
        config (Dict[str, Any]): Configuration settings for FUSE
        logger (Any): Logger object for recording processing information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """

    def _get_model_name(self) -> str:
        """Return the model name."""
        return "FUSE"

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract simulated streamflow from FUSE output and save to CSV.
        Converts units from mm/day to m3/s (cms) using catchment area.

        Returns:
            Optional[Path]: Path to the saved CSV file if successful, None otherwise
        """
        try:
            self.logger.info("Extracting FUSE streamflow results")

            # Define paths
            sim_path = self.sim_dir / f"{self.domain_name}_{self.experiment_id}_runs_best.nc"

            # Use inherited NetCDF reader with selections
            q_sim = self.read_netcdf_streamflow(
                sim_path,
                'q_routed',
                param_set=0,
                latitude=0,
                longitude=0
            )

            # Use inherited unit converter (mm/day → cms)
            q_sim_cms = self.convert_mm_per_day_to_cms(q_sim)

            # Use inherited save method
            return self.save_streamflow_to_results(q_sim_cms)

        except Exception as e:
            self.logger.error(f"Error extracting streamflow: {str(e)}")
            raise

