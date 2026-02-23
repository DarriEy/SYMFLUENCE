"""
GNN Model Postprocessor.

Handles result saving, formatting, and standardized streamflow plotting for the GNN model.
"""

from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base.standard_postprocessor import StandardModelPostprocessor
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_postprocessor('GNN')
class GNNPostprocessor(StandardModelPostprocessor):
    """
    Handles post-processing of GNN model results.

    Extends StandardModelPostprocessor with GNN-specific outlet selection logic.
    Supports three-level fallback for outlet identification:
    1. Config-specified outlet HRU IDs (from NetCDF attributes)
    2. Pour point shapefile intersection with river network
    3. Highest mean discharge inference

    Class Attributes:
        model_name: "GNN"
        output_file_pattern: Pattern for GNN output NetCDF files
        streamflow_variable: Name of streamflow variable in output
        streamflow_unit: Unit of streamflow (cms)
        outlet_selection_method: "config" with custom fallback logic
    """

    # StandardModelPostprocessor configuration
    model_name = "GNN"
    output_file_pattern = "{experiment}_GNN_output.nc"
    streamflow_variable = "predicted_streamflow"
    streamflow_unit = "cms"
    outlet_selection_method = "config"

    def _get_model_name(self) -> str:
        """Return the model name."""
        return "GNN"

    def _map_outlet_ids_to_indices(
        self,
        ds: xr.Dataset,
        outlet_ids: List[int],
        spatial_dim: Optional[str] = None
    ) -> List[int]:
        """
        Map outlet HRU IDs to array indices in the dataset.

        Args:
            ds: xarray Dataset containing the streamflow data
            outlet_ids: List of outlet HRU IDs to find
            spatial_dim: Optional name of spatial dimension

        Returns:
            List of indices corresponding to the outlet IDs
        """
        if not outlet_ids:
            return []

        # Try to use gruId coordinate if available
        if 'gruId' in ds.variables:
            gru_ids = ds['gruId'].values
            return [idx for idx, gru_id in enumerate(gru_ids) if int(gru_id) in outlet_ids]

        # Fall back to direct index interpretation
        dim_name = spatial_dim
        if dim_name is None:
            non_time_dims = [dim for dim in ds.dims if dim != 'time']
            dim_name = non_time_dims[0] if non_time_dims else None
        max_index = ds.sizes.get(dim_name, 0) - 1 if dim_name else -1
        indices = [int(val) for val in outlet_ids if 0 <= int(val) <= max_index]
        return indices

    def _infer_outlet_indices_from_pour_point(self, ds: xr.Dataset) -> List[int]:
        """
        Infer outlet indices from pour point shapefile.

        Uses spatial intersection between pour point and river network
        to identify the outlet reach/HRU.

        Args:
            ds: xarray Dataset for coordinate reference

        Returns:
            List of outlet indices, or empty list if inference fails
        """
        default_name = f"{self.domain_name}_pourPoint.shp"
        pour_point_path = self._get_file_path(
            path_key='POUR_POINT_SHP_PATH',
            name_key='POUR_POINT_SHP_NAME',
            default_subpath='shapefiles/pour_point',
            default_name=default_name,
            must_exist=False
        )

        if not pour_point_path.exists():
            return []

        pour_points = gpd.read_file(pour_point_path)
        if pour_points.empty:
            return []

        # Try to get outlet IDs directly from pour point attributes
        outlet_ids: List[int] = []
        id_columns = ['GRU_ID', 'HRU_ID', 'hruId', 'gruId']
        for col in id_columns:
            if col in pour_points.columns:
                outlet_ids = [int(val) for val in pour_points[col].dropna().unique().tolist()]
                break

        if outlet_ids:
            return self._map_outlet_ids_to_indices(ds, outlet_ids)

        # Fall back to spatial intersection with river network
        point_geom = pour_points.geometry.iloc[0]
        method_suffix = self._get_method_suffix()
        river_name = f"{self.domain_name}_riverNetwork_{method_suffix}.shp"
        river_path = self._get_file_path(
            path_key='RIVER_NETWORK_SHP_PATH',
            name_key='RIVER_NETWORK_SHP_NAME',
            default_subpath='shapefiles/river_network',
            default_name=river_name,
            must_exist=False
        )
        if not river_path.exists():
            return []

        river_gdf = gpd.read_file(river_path)
        if river_gdf.empty:
            return []

        # Ensure same CRS
        if river_gdf.crs and pour_points.crs and river_gdf.crs != pour_points.crs:
            river_gdf = river_gdf.to_crs(pour_points.crs)

        # Find nearest river segment
        distances = river_gdf.geometry.distance(point_geom)
        nearest_idx = int(distances.idxmin())

        gru_id_col = self._get_config_value(lambda: None, default='GRU_ID', dict_key='RIVER_BASIN_SHP_RM_GRUID')
        if gru_id_col not in river_gdf.columns:
            return []

        outlet_id = int(river_gdf.loc[nearest_idx, gru_id_col])
        return self._map_outlet_ids_to_indices(ds, [outlet_id])

    def _infer_outlet_indices_from_discharge(self, data: xr.DataArray) -> List[int]:
        """
        Infer outlet index from highest mean discharge.

        This is the last-resort fallback when neither config nor shapefiles
        provide outlet information.

        Args:
            data: xarray DataArray of streamflow with spatial dimension

        Returns:
            List containing single index of highest-discharge location
        """
        if data.ndim <= 1 or 'time' not in data.dims:
            return []

        spatial_dims = [dim for dim in data.dims if dim != 'time']
        spatial_dim = spatial_dims[0] if spatial_dims else data.dims[-1]

        mean_vals = data.mean(dim=[dim for dim in data.dims if dim != spatial_dim])
        best_index = int(mean_vals.argmax(dim=spatial_dim).values)
        return [best_index]

    def _select_outlet_data(
        self,
        ds: xr.Dataset,
        data: xr.DataArray,
        spatial_dim: str
    ) -> xr.DataArray:
        """
        Select outlet data using three-level fallback.

        Fallback order:
        1. Config-specified outlet IDs from NetCDF attributes
        2. Pour point shapefile intersection
        3. Highest mean discharge inference

        Args:
            ds: Full xarray Dataset
            data: Streamflow DataArray to select from
            spatial_dim: Name of the spatial dimension

        Returns:
            DataArray with outlet data selected (summed if multiple outlets)
        """
        # Level 1: Try config-specified outlet IDs from attributes
        outlet_ids_raw = ds.attrs.get('outlet_hru_ids')
        outlet_ids: List[int] = []
        if outlet_ids_raw is not None:
            if isinstance(outlet_ids_raw, (list, tuple, np.ndarray)):
                outlet_ids = [int(val) for val in outlet_ids_raw]
            elif isinstance(outlet_ids_raw, str):
                outlet_ids = [
                    int(token) for token in outlet_ids_raw.replace('[', '').replace(']', '').split(',')
                    if token.strip().isdigit()
                ]

        outlet_indices = self._map_outlet_ids_to_indices(ds, outlet_ids, spatial_dim=spatial_dim)

        # Level 2: Try pour point shapefile
        if not outlet_indices:
            outlet_indices = self._infer_outlet_indices_from_pour_point(ds)

        # Level 3: Fall back to highest discharge
        if not outlet_indices:
            outlet_indices = self._infer_outlet_indices_from_discharge(data)

        # Apply selection
        if outlet_indices:
            data = data.isel({spatial_dim: outlet_indices})
            if len(outlet_indices) > 1:
                data = data.sum(dim=spatial_dim)
        else:
            # Ultimate fallback: last index
            data = data.isel({spatial_dim: -1})

        return data

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from GNN output NetCDF.

        Uses StandardModelPostprocessor pattern with custom outlet selection
        logic for distributed GNN outputs.

        Returns:
            Path to saved results CSV, or None if extraction fails
        """
        try:
            self.logger.info(f"Extracting {self.model_name} streamflow results")

            # Use parent's pattern formatting
            output_file = self._get_output_file()
            if not output_file.exists():
                self.logger.warning(f"{self.model_name} output file not found: {output_file}")
                return None

            with xr.open_dataset(output_file) as ds:
                if self.streamflow_variable not in ds:
                    self.logger.error(f"Variable '{self.streamflow_variable}' not found in {output_file}")
                    return None

                data = ds[self.streamflow_variable]

                # Handle multi-dimensional output (distributed mode)
                if data.ndim > 1:
                    spatial_dims = [dim for dim in data.dims if dim != 'time']
                    spatial_dim = spatial_dims[0] if spatial_dims else data.dims[-1]
                    data = self._select_outlet_data(ds, data, spatial_dim)

                # Save to standard results format
                return self.save_streamflow_to_results(data.to_pandas())

        except Exception as e:
            self.logger.error(f"Error extracting {self.model_name} streamflow: {str(e)}")
            return None

    def save_results(
        self,
        results_df: pd.DataFrame,
        hru_ids: Optional[List[int]] = None,
        outlet_hru_ids: Optional[List[int]] = None,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save the simulation results to CSV and NetCDF.

        Handles both lumped and distributed modes with appropriate formatting.

        Args:
            results_df: DataFrame with MultiIndex (time, hruId) for distributed
                mode, or simple time index for lumped mode. Should have
                'predicted_streamflow' column.
            hru_ids: Optional list of HRU IDs for consistent column ordering.
            outlet_hru_ids: Optional list of outlet HRU IDs for hydrograph comparison.
            output_dir: Optional directory for legacy CSV/NetCDF outputs.

        Returns:
            Path to the saved NetCDF file.
        """
        self.logger.info("Saving GNN model results")
        self.sim_dir.mkdir(parents=True, exist_ok=True)

        if output_dir is None:
            output_dir = self.project_dir / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Legacy CSV output
        csv_path = output_dir / 'gnn_output.csv'
        results_df.to_csv(csv_path)
        self.logger.info(f"Saved GNN results to {csv_path}")

        is_distributed = isinstance(results_df.index, pd.MultiIndex)
        output_file = self.sim_dir / f"{self.experiment_id}_GNN_output.nc"

        if is_distributed:
            # Distributed mode: pivot to (time, hru) layout
            pivot_df = results_df.reset_index().pivot(
                index='time', columns='hruId', values='predicted_streamflow'
            )
            if hru_ids:
                pivot_df = pivot_df[hru_ids]

            ds = xr.Dataset(
                coords={
                    'time': pivot_df.index,
                    'gru': np.arange(len(pivot_df.columns))
                }
            )
            ds['gruId'] = (['gru'], np.array(pivot_df.columns, dtype='int32'))
            ds['predicted_streamflow'] = (['time', 'gru'], pivot_df.values)
            ds['predicted_streamflow'].attrs['units'] = 'm3 s-1'

            # Extract outlet streamflow for standard results
            if outlet_hru_ids:
                outlet_series = pivot_df[outlet_hru_ids].sum(axis=1)
            else:
                outlet_series = pivot_df.iloc[:, -1]

            self.save_streamflow_to_results(
                outlet_series,
                model_column_name=f"{self.model_name}_discharge_cms"
            )
        else:
            # Lumped mode: simple time series
            ds = xr.Dataset(
                data_vars={
                    "predicted_streamflow": (["time"], results_df['predicted_streamflow'].values)
                },
                coords={
                    "time": results_df.index
                }
            )
            ds.predicted_streamflow.attrs['units'] = 'm3 s-1'

            self.save_streamflow_to_results(
                results_df['predicted_streamflow'],
                model_column_name=f"{self.model_name}_discharge_cms"
            )

        # Add metadata
        ds.attrs['model'] = 'GNN'
        ds.attrs['experiment_id'] = self.experiment_id
        if outlet_hru_ids:
            ds.attrs['outlet_hru_ids'] = [int(val) for val in outlet_hru_ids]

        ds.to_netcdf(output_file)
        self.logger.info(f"GNN results saved to {output_file}")

        # Legacy NetCDF output
        legacy_nc_path = output_dir / 'gnn_output.nc'
        ds.to_netcdf(legacy_nc_path)
        self.logger.info(f"Saved GNN NetCDF to {legacy_nc_path}")

        return output_file
