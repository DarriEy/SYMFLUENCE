"""
IGNACIO Result Extractor.

Handles extraction of simulation results from IGNACIO fire model outputs.
IGNACIO outputs are shapefiles (fire perimeters) and JSON summaries.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from symfluence.models.base import ModelResultExtractor


class IGNACIOResultExtractor(ModelResultExtractor):
    """IGNACIO-specific result extraction.

    Handles IGNACIO's output characteristics:
    - File formats: Shapefiles (.shp), JSON (.json)
    - Variable types: burned_area, fire_intensity, rate_of_spread
    - Returns event metrics (single-value or per-timestep) rather than
      continuous hydrological time series.
    """

    def __init__(self, model_name: str = 'IGNACIO'):
        super().__init__(model_name)

    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for IGNACIO outputs."""
        return {
            'burned_area': [
                '**/fire_*.shp',
                '**/perimeters/*.shp',
                '*.shp',
            ],
            'summary': [
                'ignacio_summary.json',
                '**/ignacio_summary.json',
            ],
            'rate_of_spread': [
                '**/ros_grid*.tif',
                '**/ros_*.nc',
            ],
        }

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get IGNACIO variable names for different types."""
        variable_mapping = {
            'burned_area': ['total_area_ha', 'area_ha', 'burned_area'],
            'fire_intensity': ['hfi', 'head_fire_intensity', 'fire_intensity'],
            'rate_of_spread': ['ros', 'rate_of_spread', 'head_ros'],
            'iou': ['iou', 'intersection_over_union'],
            'dice': ['dice', 'sorensen_dice'],
        }
        return variable_mapping.get(variable_type, [variable_type])

    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract variable from IGNACIO output.

        Supports two extraction modes:
        1. Shapefile: compute burned area from geometries
        2. JSON summary: read pre-computed metrics

        Args:
            output_file: Path to output file (shapefile or JSON)
            variable_type: Type of variable to extract
            **kwargs: Additional options

        Returns:
            Series with extracted metric values.
        """
        suffix = output_file.suffix.lower()

        if suffix == '.json':
            return self._extract_from_json(output_file, variable_type)
        elif suffix == '.shp':
            return self._extract_from_shapefile(output_file, variable_type)
        else:
            raise ValueError(
                f"Unsupported IGNACIO output format: {suffix}. "
                "Expected .shp or .json"
            )

    def _extract_from_json(
        self, json_path: Path, variable_type: str
    ) -> pd.Series:
        """Extract metrics from IGNACIO summary JSON."""
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)

        var_names = self.get_variable_names(variable_type)

        # Search in statistics section
        stats = data.get('statistics', {})
        for var_name in var_names:
            if var_name in stats:
                return pd.Series(
                    [stats[var_name]],
                    index=[pd.Timestamp.now()],
                    name=variable_type,
                )

        # Search in validation section
        validation = data.get('observed_validation', {})
        for var_name in var_names:
            if var_name in validation:
                return pd.Series(
                    [validation[var_name]],
                    index=[pd.Timestamp.now()],
                    name=variable_type,
                )

        raise ValueError(
            f"Variable type '{variable_type}' not found in {json_path}. "
            f"Tried: {var_names}"
        )

    def _extract_from_shapefile(
        self, shp_path: Path, variable_type: str
    ) -> pd.Series:
        """Extract metrics from IGNACIO perimeter shapefile."""
        import geopandas as gpd

        gdf = gpd.read_file(shp_path)

        if variable_type in ('burned_area', 'total_area_ha', 'area_ha'):
            # Ensure projected CRS for accurate area
            if gdf.crs and gdf.crs.is_geographic:
                centroid = gdf.geometry.unary_union.centroid
                utm_zone = int((centroid.x + 180) / 6) + 1
                utm_crs = (
                    f"EPSG:{32600 + utm_zone}"
                    if centroid.y >= 0
                    else f"EPSG:{32700 + utm_zone}"
                )
                gdf = gdf.to_crs(utm_crs)

            area_ha = gdf.geometry.area.sum() / 10000.0
            return pd.Series(
                [area_ha],
                index=[pd.Timestamp.now()],
                name='burned_area_ha',
            )

        # For other variables, check attributes
        var_names = self.get_variable_names(variable_type)
        for var_name in var_names:
            if var_name in gdf.columns:
                return pd.Series(
                    gdf[var_name].values,
                    name=variable_type,
                )

        raise ValueError(
            f"Variable type '{variable_type}' not found in {shp_path}. "
            f"Available columns: {list(gdf.columns)}"
        )

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """IGNACIO outputs don't require unit conversion."""
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """IGNACIO uses union for spatial aggregation of perimeters."""
        return 'sum'
