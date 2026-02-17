"""
PRMS model postprocessor.

Handles extraction and processing of PRMS model simulation results.
Uses StandardModelPostprocessor for reduced boilerplate.
"""

from pathlib import Path
from typing import Optional

from ..registry import ModelRegistry
from ..base import StandardModelPostprocessor


@ModelRegistry.register_postprocessor('PRMS')
class PRMSPostProcessor(StandardModelPostprocessor):
    """
    Postprocessor for the PRMS model.

    PRMS outputs streamflow as seg_outflow in the statvar output file.
    Units are cubic meters per second (cms). The seg_outflow variable
    represents segment outflow at the basin outlet.

    Attributes:
        model_name: "PRMS"
        output_file_pattern: "statvar*"
        streamflow_variable: "seg_outflow"
        streamflow_unit: "cms"
    """

    model_name = "PRMS"

    output_file_pattern = "statvar*"

    streamflow_variable = "seg_outflow"
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        return "PRMS"

    def _setup_model_specific_paths(self) -> None:
        """Set up PRMS-specific paths."""
        self.prms_output_dir = self.project_dir / 'simulations' / self.experiment_id / 'PRMS'
        catchment_file = self._get_catchment_file_path()
        self.catchment_path = catchment_file.parent
        self.catchment_name = catchment_file.name

    def _get_output_dir(self) -> Path:
        """PRMS outputs to standard simulation directory."""
        return self.project_dir / 'simulations' / self.experiment_id / 'PRMS'

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from PRMS statvar output.

        PRMS statvar files contain seg_outflow in cms (m3/s).
        Supports both CSV and space-delimited formats.

        Returns:
            Path to processed streamflow file, or None if extraction fails.
        """
        self.logger.info("Extracting streamflow from PRMS statvar output")

        output_dir = self._get_output_dir()

        # Find statvar file (check both output dir and settings dir)
        output_file = None
        search_dirs = [output_dir, self.project_dir / "PRMS_input" / "settings"]
        for search_dir in search_dirs:
            for pattern in ['statvar.dat', 'statvar*.csv', 'statvar*.nc']:
                matches = list(search_dir.glob(pattern))
                if matches:
                    output_file = matches[0]
                    break
            if output_file:
                break

        if output_file is None:
            self.logger.error("PRMS statvar output not found")
            return None

        try:

            if output_file.suffix == '.nc':
                return self._extract_from_netcdf(output_file)

            # Parse statvar text file
            streamflow = self._parse_statvar_file(output_file)

            if streamflow is None or streamflow.empty:
                self.logger.error("No streamflow data found in PRMS statvar output")
                return None

            # Streamflow is already in cms, no conversion needed

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='PRMS_discharge_cms'
            )

        except Exception as e:
            import traceback
            self.logger.error(f"Error extracting PRMS streamflow: {str(e)}")
            self.logger.debug(traceback.format_exc())
            return None

    def _parse_statvar_file(self, statvar_path: Path):
        """Parse PRMS statvar.dat file and extract seg_outflow."""
        import pandas as pd

        try:
            # PRMS statvar files are space-delimited with a header section
            # Format: date columns followed by variable values
            lines = statvar_path.read_text().strip().split('\n')

            # Find the data start (after header)
            data_start = 0
            header_vars = []
            for i, line in enumerate(lines):
                if line.strip().startswith('####'):
                    data_start = i + 1
                    break
                # Collect variable names from header
                parts = line.strip().split()
                if parts and not parts[0].isdigit():
                    header_vars.append(parts[0])

            if data_start == 0:
                # No header marker, try reading as CSV
                df = pd.read_csv(statvar_path, sep=r'\s+', comment='#')
                if 'seg_outflow' in df.columns:
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                    return df['seg_outflow']

            # Parse data lines
            dates = []
            values = []
            for line in lines[data_start:]:
                parts = line.strip().split()
                if len(parts) >= 7:  # year month day hour min sec + values
                    try:
                        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                        date = pd.Timestamp(year=year, month=month, day=day)
                        dates.append(date)
                        # seg_outflow is typically the first variable after date
                        values.append(float(parts[6]))
                    except (ValueError, IndexError):
                        continue

            if dates:
                return pd.Series(values, index=dates, name='seg_outflow')

            return None

        except Exception as e:
            self.logger.warning(f"Error parsing statvar file: {e}")
            return None

    def _extract_from_netcdf(self, nc_path: Path):
        """Extract streamflow from NetCDF statvar output."""
        import xarray as xr

        ds = xr.open_dataset(nc_path)

        for var in ['seg_outflow', 'streamflow', 'basin_cfs']:
            if var in ds.data_vars:
                flow = ds[var]
                if 'nsegment' in flow.dims:
                    flow = flow.isel(nsegment=-1)
                series = flow.to_series()
                ds.close()
                return series

        ds.close()
        return None
