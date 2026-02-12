"""
TRoute Model Postprocessor.

Simplified implementation using StandardModelPostprocessor.
Handles extraction of routed streamflow from t-route NetCDF outputs.
"""

from ..base import StandardModelPostprocessor
from ..registry import ModelRegistry


@ModelRegistry.register_postprocessor('TROUTE')
class TRoutePostProcessor(StandardModelPostprocessor):
    """
    Postprocessor for TRoute model outputs.

    TRoute outputs NetCDF files with routed streamflow in m³/s (cms).
    Output files are typically named with timestamps and contain
    segment-based flow routing results.
    """

    model_name = "TROUTE"
    output_file_pattern = "*.nc"  # TRoute outputs multiple timestamped NC files
    streamflow_variable = "flowveldepth"  # TRoute flow variable
    streamflow_unit = "cms"  # TRoute outputs in m³/s
    netcdf_selections = {}  # Will be determined dynamically

    # TRoute typically outputs to a specific directory
    output_dir_override = "TROUTE"

    # Dynamic outlet selection: use segment with highest discharge
    outlet_selection_method = "highest_discharge"

    def _get_output_file(self):
        """
        Get the most recent TRoute output file.

        TRoute creates multiple timestamped output files. This method
        finds and returns the most recent one.
        """

        output_dir = self._get_output_dir()

        # Look for NetCDF files in TRoute output directory
        nc_files = list(output_dir.glob("*.nc"))

        if not nc_files:
            # Also check subdirectories
            nc_files = list(output_dir.glob("**/*.nc"))

        if not nc_files:
            self.logger.error(f"No TRoute output files found in {output_dir}")
            return output_dir / "not_found.nc"

        # Return the most recently modified file
        most_recent = max(nc_files, key=lambda f: f.stat().st_mtime)
        self.logger.info(f"Using TRoute output file: {most_recent}")
        return most_recent

    def _extract_from_netcdf(self, file_path):
        """
        Extract streamflow from TRoute NetCDF output.

        TRoute outputs have segment/feature_id dimensions for multi-reach
        routing. This extracts the outlet reach flow.
        """
        import xarray as xr
        import pandas as pd
        from typing import cast

        try:
            ds = xr.open_dataset(file_path)

            # Try different possible variable names for flow
            flow_vars = ['flowveldepth', 'flow', 'q', 'streamflow', 'discharge']
            flow_var = None

            for var in flow_vars:
                if var in ds.data_vars:
                    flow_var = var
                    break

            if flow_var is None:
                self.logger.error(
                    f"No flow variable found in TRoute output. "
                    f"Available: {list(ds.data_vars)}"
                )
                ds.close()
                return None

            data = ds[flow_var]

            # Handle segment/feature_id dimension
            # TRoute uses feature_id for reach identification
            if 'feature_id' in data.dims:
                # Try to get configured reach ID
                sim_reach_id = self.config_dict.get('SIM_REACH_ID')

                if sim_reach_id is not None:
                    sim_reach_id = int(sim_reach_id)
                    if 'feature_id' in ds.coords:
                        # Select by feature_id value
                        try:
                            data = data.sel(feature_id=sim_reach_id)
                        except KeyError:
                            self.logger.warning(
                                f"Reach ID {sim_reach_id} not found. "
                                "Using segment with highest mean flow."
                            )
                            data = self._select_outlet_segment(data)
                    else:
                        data = data.isel(feature_id=-1)
                else:
                    # Auto-select outlet (highest flow)
                    data = self._select_outlet_segment(data)

            # Handle other spatial dimensions
            non_time_dims = [d for d in data.dims if d not in ('time', 'reference_time')]
            for dim in non_time_dims:
                data = data.isel({dim: 0})

            streamflow = cast(pd.Series, data.to_pandas())
            ds.close()

            return streamflow

        except Exception as e:
            self.logger.error(f"Error extracting TRoute streamflow: {e}")
            return None

    def _select_outlet_segment(self, data):
        """
        Select the outlet segment based on highest mean discharge.

        Args:
            data: xarray DataArray with feature_id dimension

        Returns:
            DataArray with feature_id dimension removed (outlet selected)
        """
        if 'feature_id' not in data.dims:
            return data

        # Calculate mean flow for each segment
        mean_flow = data.mean(dim='time')

        # Find segment with highest mean flow (likely the outlet)
        outlet_idx = int(mean_flow.argmax())

        self.logger.info(
            f"Auto-selected outlet segment at index {outlet_idx} "
            f"(highest mean flow)"
        )

        return data.isel(feature_id=outlet_idx)
