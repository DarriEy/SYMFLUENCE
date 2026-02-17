"""
WRF-Hydro model postprocessor.

Handles extraction and processing of WRF-Hydro model simulation results.
Uses StandardModelPostprocessor for reduced boilerplate.
"""

from pathlib import Path
from typing import Optional

from ..registry import ModelRegistry
from ..base import StandardModelPostprocessor


@ModelRegistry.register_postprocessor('WRFHYDRO')
class WRFHydroPostProcessor(StandardModelPostprocessor):
    """
    Postprocessor for the WRF-Hydro model.

    WRF-Hydro outputs streamflow in CHRTOUT NetCDF files with variable
    'streamflow' in cubic meters per second (cms). No unit conversion
    is needed for channel output.

    Attributes:
        model_name: "WRFHYDRO"
        output_file_pattern: "*CHRTOUT*.nc"
        streamflow_variable: "streamflow"
        streamflow_unit: "cms"
    """

    model_name = "WRFHYDRO"

    output_file_pattern = "*CHRTOUT*.nc"

    streamflow_variable = "streamflow"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "WRFHYDRO"

    def _setup_model_specific_paths(self) -> None:
        """Set up WRF-Hydro-specific paths."""
        self.wrfhydro_output_dir = self.project_dir / 'simulations' / self.experiment_id / 'WRFHYDRO'
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """WRF-Hydro outputs to standard simulation directory."""
        return self.project_dir / 'simulations' / self.experiment_id / 'WRFHYDRO'

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from WRF-Hydro CHRTOUT outputs.

        WRF-Hydro CHRTOUT files contain streamflow in cms (m3/s) at
        channel routing points. No unit conversion needed.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        self.logger.info("Extracting streamflow from WRF-Hydro CHRTOUT outputs")

        output_dir = self._get_output_dir()

        # Find CHRTOUT files
        output_files = sorted(output_dir.glob('*CHRTOUT*.nc'))

        if not output_files:
            self.logger.error(f"WRF-Hydro CHRTOUT files not found in {output_dir}")
            return None

        try:
            import xarray as xr
            import pandas as pd

            streamflow_series = []

            for fpath in output_files:
                ds = xr.open_dataset(fpath)

                # Find streamflow variable
                flow = None
                for var in ['streamflow', 'q_lateral', 'qSfcLatRunoff']:
                    if var in ds.data_vars:
                        flow = ds[var]
                        break

                if flow is not None:
                    # Take the outlet point (last feature or max flow)
                    if 'feature_id' in flow.dims:
                        # Use the last feature_id (typically outlet)
                        flow = flow.isel(feature_id=-1)

                    if 'time' in flow.dims:
                        streamflow_series.append(flow.to_series())
                    else:
                        # Single timestep - get time from filename or attributes
                        time_val = ds.attrs.get('model_output_valid_time',
                                                fpath.stem[:10])
                        try:
                            t = pd.Timestamp(time_val)
                        except Exception:
                            t = pd.Timestamp.now()
                        streamflow_series.append(
                            pd.Series([float(flow.values)], index=[t])
                        )
                ds.close()

            if not streamflow_series:
                self.logger.error("No streamflow data found in CHRTOUT files")
                return None

            streamflow = pd.concat(streamflow_series).sort_index()

            # Streamflow is already in cms, no conversion needed

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='WRFHYDRO_discharge_cms'
            )

        except Exception as e:
            import traceback
            self.logger.error(f"Error extracting WRF-Hydro streamflow: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None
