"""
dRoute model postprocessor.

Handles extraction and processing of dRoute routing simulation results.
Uses StandardModelPostprocessor for reduced boilerplate.
"""

from pathlib import Path
from typing import Optional

from ..base import StandardModelPostprocessor
from ..registry import ModelRegistry


@ModelRegistry.register_postprocessor('DROUTE')
class DRoutePostProcessor(StandardModelPostprocessor):
    """
    Postprocessor for the dRoute routing model.

    dRoute outputs routed streamflow to NetCDF with variables:
    - outletStreamflow: Routed flow at outlet (m³/s)
    - routedRunoff: Routed flow at all segments (m³/s)

    Units are already m³/s so no conversion is needed.

    Attributes:
        model_name: "DROUTE"
        output_file_pattern: "*_droute_output.nc"
        streamflow_variable: "outletStreamflow"
        streamflow_unit: "cms"
    """

    model_name = "DROUTE"

    output_file_pattern = "*_droute_output.nc"

    streamflow_variable = "outletStreamflow"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "DROUTE"

    def _setup_model_specific_paths(self) -> None:
        """Set up dRoute-specific paths."""
        self.droute_output_dir = (
            self.project_dir / 'simulations' / self.experiment_id / 'dRoute'
        )

    def _get_output_dir(self) -> Path:
        """dRoute outputs to standard simulation directory."""
        return self.project_dir / 'simulations' / self.experiment_id / 'dRoute'

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract routed streamflow from dRoute outputs.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        self.logger.info("Extracting streamflow from dRoute outputs")

        output_dir = self._get_output_dir()

        # Find output file
        output_file = None
        for pattern in ['*_droute_output.nc', '*droute*.nc']:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break

        if output_file is None:
            self.logger.error(f"dRoute output not found in {output_dir}")
            return None

        try:
            import xarray as xr

            ds = xr.open_dataset(output_file)

            # Try outlet streamflow first, then routed runoff
            streamflow = None
            for var in ['outletStreamflow', 'routedRunoff']:
                if var in ds.data_vars:
                    data = ds[var]
                    if var == 'routedRunoff' and 'seg' in data.dims:
                        # Select outlet segment (first or max mean)
                        mean_flow = data.mean(dim='time')
                        outlet_idx = int(mean_flow.argmax().values)
                        data = data.isel(seg=outlet_idx)
                    streamflow = data.to_series()
                    break

            ds.close()

            if streamflow is None:
                self.logger.error("No streamflow variable found in dRoute output")
                return None

            # Already in m³/s, no conversion needed

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='DROUTE_discharge_cms'
            )

        except Exception as e:
            import traceback
            self.logger.error(f"Error extracting dRoute streamflow: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
