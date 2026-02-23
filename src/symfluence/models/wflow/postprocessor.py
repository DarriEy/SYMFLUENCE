"""Wflow model postprocessor."""
from pathlib import Path
from typing import Optional

from ..base import StandardModelPostprocessor
from ..registry import ModelRegistry


@ModelRegistry.register_postprocessor('WFLOW')
class WflowPostProcessor(StandardModelPostprocessor):
    """Postprocessor for the Wflow model. Wflow outputs Q in m3/s."""

    model_name = "WFLOW"
    output_file_pattern = "output*.nc"
    streamflow_variable = "Q"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "WFLOW"

    def _setup_model_specific_paths(self) -> None:
        self.wflow_output_dir = self.project_dir / 'simulations' / self.experiment_id / 'WFLOW'
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        return self.project_dir / 'simulations' / self.experiment_id / 'WFLOW'

    def extract_streamflow(self) -> Optional[Path]:
        self.logger.info("Extracting streamflow from Wflow outputs")
        output_dir = self._get_output_dir()
        output_file = None
        for pattern in ['output*.nc', '*output*.nc']:
            matches = list(output_dir.glob(pattern))
            if matches:
                output_file = matches[0]
                break
        if output_file is None:
            self.logger.error(f"Wflow output not found in {output_dir}")
            return None
        try:
            import xarray as xr
            ds = xr.open_dataset(output_file)
            q_var = None
            for var in ['Q', 'Q_av', 'q_av', 'q_river']:
                if var in ds.data_vars:
                    q_var = ds[var]
                    break
            if q_var is None:
                self.logger.error("No streamflow variable found in Wflow output")
                ds.close()
                return None
            spatial_dims = [d for d in q_var.dims if d not in ['time']]
            if spatial_dims:
                streamflow = q_var.max(dim=spatial_dims)
            else:
                streamflow = q_var
            streamflow = streamflow.to_series()
            ds.close()
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()
            return self.save_streamflow_to_results(streamflow, model_column_name='WFLOW_discharge_cms')
        except Exception as e:  # noqa: BLE001
            import traceback
            self.logger.error(f"Error extracting Wflow streamflow: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
