"""
RHESSys Calibration Targets

RHESSys-specific calibration target implementations for final evaluation.
Reads RHESSys text-based output format (rhessys_basin.daily).
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from symfluence.evaluation.evaluators.base import ModelEvaluator


class RHESSysStreamflowTarget(ModelEvaluator):
    """
    Streamflow calibration target for RHESSys model.

    Reads RHESSys basin daily output and calculates streamflow metrics.
    RHESSys outputs mm/day which is converted to m³/s for comparison.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        logger: logging.Logger
    ):
        """
        Initialize RHESSys streamflow target.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            logger: Logger instance
        """
        super().__init__(config, project_dir, logger)
        self.variable = 'streamflow'
        self._area_m2 = None

    def get_simulation_files(self, sim_dir: Path) -> List[Path]:
        """
        Get RHESSys simulation output files.

        Args:
            sim_dir: Simulation output directory

        Returns:
            List of output file paths
        """
        output_file = sim_dir / 'rhessys_basin.daily'
        if output_file.exists():
            return [output_file]

        # Also check for RHESSys subdirectory
        rhessys_dir = sim_dir / 'RHESSys'
        if rhessys_dir.exists():
            output_file = rhessys_dir / 'rhessys_basin.daily'
            if output_file.exists():
                return [output_file]

        self.logger.error(f"No simulation files found in {sim_dir}")
        return []

    def extract_simulated_data(
        self,
        sim_files: List[Path],
        **kwargs
    ) -> pd.Series:
        """
        Extract streamflow from RHESSys output.

        Args:
            sim_files: List of simulation output files

        Returns:
            Streamflow series in m³/s with datetime index
        """
        if not sim_files:
            return pd.Series(dtype=float)

        try:
            # Read RHESSys basin output (whitespace-separated)
            sim_df = pd.read_csv(sim_files[0], sep=r'\s+', header=0)

            # Get streamflow in mm/day
            if 'streamflow' not in sim_df.columns:
                self.logger.error("'streamflow' column not found in RHESSys output")
                return pd.Series(dtype=float)

            streamflow_mm = sim_df['streamflow'].values

            # Convert to m³/s
            area_m2 = self._get_catchment_area()
            # Q (m³/s) = Q (mm/day) * area (m²) / 86400 / 1000
            streamflow_m3s = streamflow_mm * area_m2 / 86400 / 1000

            # Create datetime index
            dates = pd.to_datetime(
                sim_df.apply(
                    lambda r: f"{int(r['year'])}-{int(r['month']):02d}-{int(r['day']):02d}",
                    axis=1
                )
            )

            return pd.Series(streamflow_m3s, index=dates, name='streamflow_cms')

        except Exception as e:
            self.logger.error(f"Error extracting RHESSys streamflow: {e}")
            return pd.Series(dtype=float)

    def _get_catchment_area(self) -> float:
        """Get catchment area in m²."""
        if self._area_m2 is not None:
            return self._area_m2

        try:
            import geopandas as gpd

            domain_name = self.config.get('DOMAIN_NAME')
            catchment_dir = self.project_dir / 'shapefiles' / 'catchment'
            catchment_files = list(catchment_dir.glob('*.shp'))

            if catchment_files:
                gdf = gpd.read_file(catchment_files[0])
                # Project to UTM for accurate area calculation
                if gdf.crs and gdf.crs.is_geographic:
                    centroid = gdf.geometry.centroid.iloc[0]
                    lon = centroid.x
                    utm_zone = int((lon + 180) / 6) + 1
                    utm_crs = f"EPSG:{32600 + utm_zone}"
                    gdf = gdf.to_crs(utm_crs)
                self._area_m2 = gdf.geometry.area.sum()
                return self._area_m2

        except Exception as e:
            self.logger.warning(f"Could not calculate area: {e}")

        # Default to 100 km²
        self._area_m2 = 1e8
        return self._area_m2

    def get_observed_data_path(self) -> Path:
        """Get path to observed streamflow data."""
        domain_name = self.config.get('DOMAIN_NAME')
        return (
            self.project_dir / 'observations' / 'streamflow' / 'preprocessed' /
            f'{domain_name}_streamflow_processed.csv'
        )

    def _get_observed_data_column(self, columns: List[str]) -> Optional[str]:
        """Get the column name for observed discharge data."""
        for col in columns:
            if 'discharge' in col.lower() or 'flow' in col.lower():
                return col
        return None

    def needs_routing(self) -> bool:
        """RHESSys handles its own routing."""
        return False


__all__ = ['RHESSysStreamflowTarget']
