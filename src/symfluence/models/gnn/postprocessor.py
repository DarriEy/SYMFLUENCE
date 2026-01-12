
"""
GNN Model Postprocessor.

Handles result saving, formatting, and standardized streamflow plotting for the GNN model.
"""

from typing import Optional, List
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
import geopandas as gpd

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelPostProcessor


@ModelRegistry.register_postprocessor('GNN')
class GNNPostprocessor(BaseModelPostProcessor):
    """
    Handles post-processing of GNN model results.
    """

    def _get_model_name(self) -> str:
        return "GNN"

    def _map_outlet_ids_to_indices(
        self,
        ds: xr.Dataset,
        outlet_ids: List[int],
        spatial_dim: Optional[str] = None
    ) -> List[int]:
        if not outlet_ids:
            return []

        if 'gruId' in ds.variables:
            gru_ids = ds['gruId'].values
            return [idx for idx, gru_id in enumerate(gru_ids) if int(gru_id) in outlet_ids]

        dim_name = spatial_dim
        if dim_name is None:
            non_time_dims = [dim for dim in ds.dims if dim != 'time']
            dim_name = non_time_dims[0] if non_time_dims else None
        max_index = ds.sizes.get(dim_name, 0) - 1 if dim_name else -1
        indices = [int(val) for val in outlet_ids if 0 <= int(val) <= max_index]
        return indices

    def _infer_outlet_indices_from_pour_point(self, ds: xr.Dataset) -> List[int]:
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

        outlet_ids: List[int] = []
        id_columns = ['GRU_ID', 'HRU_ID', 'hruId', 'gruId']
        for col in id_columns:
            if col in pour_points.columns:
                outlet_ids = [int(val) for val in pour_points[col].dropna().unique().tolist()]
                break

        if outlet_ids:
            return self._map_outlet_ids_to_indices(ds, outlet_ids)

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

        if river_gdf.crs and pour_points.crs and river_gdf.crs != pour_points.crs:
            river_gdf = river_gdf.to_crs(pour_points.crs)

        distances = river_gdf.geometry.distance(point_geom)
        nearest_idx = int(distances.idxmin())

        gru_id_col = self.config_dict.get('RIVER_BASIN_SHP_RM_GRUID', 'GRU_ID')
        if gru_id_col not in river_gdf.columns:
            return []

        outlet_id = int(river_gdf.loc[nearest_idx, gru_id_col])
        return self._map_outlet_ids_to_indices(ds, [outlet_id])

    def _infer_outlet_indices_from_discharge(self, data: xr.DataArray) -> List[int]:
        if data.ndim <= 1 or 'time' not in data.dims:
            return []

        spatial_dims = [dim for dim in data.dims if dim != 'time']
        spatial_dim = spatial_dims[0] if spatial_dims else data.dims[-1]

        mean_vals = data.mean(dim=[dim for dim in data.dims if dim != spatial_dim])
        best_index = int(mean_vals.argmax(dim=spatial_dim).values)
        return [best_index]

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from GNN output NetCDF, if present.
        """
        try:
            output_file = self.sim_dir / f"{self.experiment_id}_GNN_output.nc"
            if not output_file.exists():
                return None

            with xr.open_dataset(output_file) as ds:
                if 'predicted_streamflow' in ds:
                    data = ds['predicted_streamflow']
                    if data.ndim > 1:
                        spatial_dims = [dim for dim in data.dims if dim != 'time']
                        spatial_dim = spatial_dims[0] if spatial_dims else data.dims[-1]

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
                        if not outlet_indices:
                            outlet_indices = self._infer_outlet_indices_from_pour_point(ds)
                        if not outlet_indices:
                            outlet_indices = self._infer_outlet_indices_from_discharge(data)

                        if outlet_indices:
                            data = data.isel({spatial_dim: outlet_indices})
                            if len(outlet_indices) > 1:
                                data = data.sum(dim=spatial_dim)
                        else:
                            data = data.isel({spatial_dim: -1})

                    return self.save_streamflow_to_results(data.to_pandas())
            return None
        except Exception as e:
            self.logger.error(f"Error extracting GNN streamflow: {str(e)}")
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

        Args:
            results_df: DataFrame with MultiIndex (time, hruId) and columns like 'predicted_streamflow'.
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

        csv_path = output_dir / 'gnn_output.csv'
        results_df.to_csv(csv_path)
        self.logger.info(f"Saved GNN results to {csv_path}")

        is_distributed = isinstance(results_df.index, pd.MultiIndex)
        output_file = self.sim_dir / f"{self.experiment_id}_GNN_output.nc"

        if is_distributed:
            pivot_df = results_df.reset_index().pivot(index='time', columns='hruId', values='predicted_streamflow')
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

            if outlet_hru_ids:
                outlet_series = pivot_df[outlet_hru_ids].sum(axis=1)
            else:
                outlet_series = pivot_df.iloc[:, -1]

            self.save_streamflow_to_results(
                outlet_series,
                model_column_name=f"{self.model_name}_discharge_cms"
            )
        else:
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

        ds.attrs['model'] = 'GNN'
        ds.attrs['experiment_id'] = self.experiment_id
        if outlet_hru_ids:
            ds.attrs['outlet_hru_ids'] = [int(val) for val in outlet_hru_ids]

        ds.to_netcdf(output_file)
        self.logger.info(f"GNN results saved to {output_file}")

        legacy_nc_path = output_dir / 'gnn_output.nc'
        ds.to_netcdf(legacy_nc_path)
        self.logger.info(f"Saved GNN NetCDF to {legacy_nc_path}")

        return output_file
