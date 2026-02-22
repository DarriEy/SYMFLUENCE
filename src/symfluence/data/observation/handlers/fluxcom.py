"""
FLUXCOM evapotranspiration observation handler.

Provides acquisition and preprocessing of FLUXCOM machine learning-based
evapotranspiration products for model validation and calibration.
"""

import pandas as pd
import xarray as xr
from pathlib import Path
from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

@ObservationRegistry.register('fluxcom_et')
class FLUXCOMETHandler(BaseObservationHandler):
    """
    Handles FLUXCOM Evapotranspiration data.
    """

    obs_type = "et"
    source_name = "FLUXCOM"

    def acquire(self) -> Path:
        """Locate FLUXCOM ET data."""
        et_path_cfg = self._get_config_value(lambda: None, default=None, dict_key='FLUXCOM_ET_PATH')
        if et_path_cfg and str(et_path_cfg).lower() != 'default':
            et_dir = Path(et_path_cfg)
            et_dir.mkdir(parents=True, exist_ok=True)
            return et_dir

        et_dir = self.project_observations_dir / "et" / "fluxcom"
        et_dir.mkdir(parents=True, exist_ok=True)

        try:
            from symfluence.data.acquisition.handlers.observation_acquirers import FLUXCOMETAcquirer
            acquirer = FLUXCOMETAcquirer(self.config, self.logger)
            acquirer.download(self.project_observations_dir)
        except Exception as exc:
            self.logger.warning(f"FLUXCOM ET acquisition failed: {exc}")

        return et_dir

    def process(self, input_path: Path) -> Path:
        """Process FLUXCOM ET NetCDF data."""
        self.logger.info(f"Processing FLUXCOM ET for domain: {self.domain_name}")

        nc_files = list(input_path.glob("*.nc"))
        if not nc_files:
            self.logger.warning("No FLUXCOM ET NetCDF files found")
            return input_path

        results = []
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                # FLUXCOM ET variable is usually 'ET'
                if 'ET' not in ds.data_vars:
                    continue

                # Spatial average
                mean_et = ds['ET'].mean(dim=[d for d in ds['ET'].dims if d != 'time'])
                df_ts = mean_et.to_dataframe().reset_index()
                results.append(df_ts)

        if not results:
            self.logger.warning("No FLUXCOM ET data could be extracted")
            return input_path

        df = pd.concat(results).sort_values('time').set_index('time')

        output_dir = self.project_observations_dir / "et" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_fluxcom_et_processed.csv"
        df.to_csv(output_file)

        self.logger.info(f"FLUXCOM ET processing complete: {output_file}")
        return output_file
