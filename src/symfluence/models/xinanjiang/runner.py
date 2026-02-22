"""
Xinanjiang Model Runner.

Handles Xinanjiang model execution and output processing. Lumped mode only.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.base import BaseModelRunner
from symfluence.models.registry import ModelRegistry
from symfluence.models.execution import SpatialOrchestrator
from symfluence.models.mixins import SpatialModeDetectionMixin, ObservationLoaderMixin
from symfluence.geospatial.geometry_utils import calculate_catchment_area_km2
from symfluence.data.utils.netcdf_utils import create_netcdf_encoding
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler


@ModelRegistry.register_runner('XINANJIANG', method_name='run_xinanjiang')
class XinanjiangRunner(  # type: ignore[misc]
    BaseModelRunner,
    SpatialOrchestrator,
    SpatialModeDetectionMixin,
    ObservationLoaderMixin
):
    """Runner for the Xinanjiang rainfall-runoff model."""

    MODEL_NAME = "XINANJIANG"

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None,
        settings_dir: Optional[Path] = None
    ):
        self.settings_dir = Path(settings_dir) if settings_dir else None
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self._external_params: Optional[Dict[str, float]] = None
        self.spatial_mode = self.detect_spatial_mode('XINANJIANG')

        self.warmup_days = self._get_config_value(
            lambda: self.config.model.xinanjiang.warmup_days
            if self.config.model and hasattr(self.config.model, 'xinanjiang') and self.config.model.xinanjiang
            else None,
            365
        )

        backend_str = self._get_config_value(
            lambda: self.config.model.xinanjiang.backend
            if self.config.model and hasattr(self.config.model, 'xinanjiang') and self.config.model.xinanjiang
            else None,
            'jax'
        )
        self.use_jax = (backend_str == 'jax')

        self.snow_module = self._get_config_value(
            lambda: self.config.model.xinanjiang.snow_module
            if self.config.model and hasattr(self.config.model, 'xinanjiang') and self.config.model.xinanjiang
            else None,
            default='none',
            dict_key='XINANJIANG_SNOW_MODULE'
        )

        self.si = float(self._get_config_value(
            lambda: None, default=100.0, dict_key='XINANJIANG_SI'))
        self.latitude = float(self._get_config_value(
            lambda: None,
            default=float(self._get_config_value(lambda: None, default=45.0, dict_key='LATITUDE')),
            dict_key='XINANJIANG_LATITUDE'
        ))

    def _setup_model_specific_paths(self) -> None:
        if hasattr(self, 'settings_dir') and self.settings_dir:
            self.xinanjiang_setup_dir = self.settings_dir
        else:
            self.xinanjiang_setup_dir = self.project_dir / "settings" / "XINANJIANG"
        self.xinanjiang_forcing_dir = self.project_forcing_dir / 'XINANJIANG_input'

    def _get_output_dir(self) -> Path:
        return self.get_experiment_output_dir()

    def _get_catchment_area(self) -> float:
        """Get total catchment area in m²."""
        try:
            import geopandas as gpd
            catchment_dir = self.project_dir / 'shapefiles' / 'catchment'
            discretization = self._get_config_value(
                lambda: self.config.domain.discretization, 'GRUs'
            )
            possible_paths = [
                catchment_dir / f"{self.domain_name}_HRUs_{discretization}.shp",
                catchment_dir / self.spatial_mode / self.experiment_id / f"{self.domain_name}_HRUs_{discretization}.shp",
                catchment_dir / self.spatial_mode / f"{self.domain_name}_HRUs_{discretization}.shp",
            ]
            for path in possible_paths:
                if path.exists():
                    gdf = gpd.read_file(path)
                    try:
                        area_km2 = calculate_catchment_area_km2(gdf, logger=self.logger)
                        return float(area_km2 * 1e6)
                    except Exception:
                        area_cols = [c for c in gdf.columns if 'area' in c.lower()]
                        if area_cols:
                            total_area = gdf[area_cols[0]].sum()
                            if 'km' in area_cols[0].lower():
                                return float(total_area * 1e6)
                            return float(total_area)
        except ImportError:
            pass
        except Exception as e:
            self.logger.debug(f"Could not read catchment shapefile: {e}")

        area_km2 = self._get_config_value(
            lambda: self.config.domain.catchment_area_km2, None
        )
        if area_km2:
            return area_km2 * 1e6

        raise ValueError(
            "Catchment area could not be determined. "
            "Provide via shapefile or CATCHMENT_AREA_KM2 config."
        )

    def run_xinanjiang(self, params: Optional[Dict[str, float]] = None) -> Optional[Path]:
        """Run the Xinanjiang model."""
        self.logger.info("Starting Xinanjiang model run (lumped mode)")

        if params:
            self._external_params = params

        with symfluence_error_handler(
            "Xinanjiang model execution", self.logger, error_type=ModelExecutionError
        ):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            success = self._execute_lumped()

            if success:
                self.logger.info("Xinanjiang model run completed successfully")
                self._calculate_and_log_metrics()
                return self.output_dir
            else:
                self.logger.error("Xinanjiang model run failed")
                return None

    def _execute_lumped(self) -> bool:
        """Execute Xinanjiang in lumped mode."""
        try:
            from .model import simulate
            from .parameters import DEFAULT_PARAMS, split_params

            forcing = self._load_forcing()
            precip = forcing['precip'].flatten()
            pet = forcing['pet'].flatten()
            time_index = forcing['time']

            params = self._external_params if self._external_params else DEFAULT_PARAMS.copy()

            self.logger.info(f"Running Xinanjiang for {len(precip)} timesteps")

            if self.snow_module == 'snow17':
                # Coupled Snow-17 + XAJ mode
                temp = forcing.get('temp')
                if temp is None:
                    raise ValueError(
                        "Temperature data required for snow_module='snow17'. "
                        "Ensure 'tas' or 'temp' variable exists in forcing."
                    )
                temp = temp.flatten()

                doy = forcing.get('day_of_year')
                if doy is None:
                    doy = np.array([d.timetuple().tm_yday for d in time_index])

                xaj_dict, snow17_dict = split_params(params)

                self.logger.info(
                    f"Running coupled Snow-17 + XAJ (lat={self.latitude}, si={self.si})"
                )
                runoff, final_state = simulate(
                    precip, pet,
                    params=xaj_dict,
                    warmup_days=self.warmup_days,
                    use_jax=self.use_jax,
                    temp=temp,
                    day_of_year=doy,
                    snow17_params=snow17_dict,
                    latitude=self.latitude,
                    si=self.si,
                )
            else:
                # Standalone XAJ mode
                runoff, final_state = simulate(
                    precip, pet,
                    params=params,
                    warmup_days=self.warmup_days,
                    use_jax=self.use_jax,
                )

            self._save_lumped_results(runoff, time_index)
            return True

        except FileNotFoundError as e:
            self.logger.error(f"Missing forcing data: {e}")
            return False
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid data in Xinanjiang execution: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _load_forcing(self) -> Dict[str, np.ndarray]:
        """Load forcing data from preprocessed files."""
        nc_file = self.xinanjiang_forcing_dir / f"{self.domain_name}_xinanjiang_forcing.nc"
        if nc_file.exists():
            ds = xr.open_dataset(nc_file)
            forcing = {
                'precip': ds['pr'].values,
                'pet': ds['pet'].values,
                'time': pd.to_datetime(ds.time.values),
            }
            # Load temperature if available (needed for snow coupling)
            for temp_var in ('tas', 'temp', 'temperature', 'airtemp'):
                if temp_var in ds.variables:
                    forcing['temp'] = ds[temp_var].values
                    break
            ds.close()
            return forcing

        raise FileNotFoundError(f"Xinanjiang forcing not found: {nc_file}")

    def _save_lumped_results(self, runoff: np.ndarray, time_index) -> None:
        """Save lumped simulation results."""
        area_m2 = self._get_catchment_area()

        # mm/day to m³/s
        streamflow_cms = runoff * area_m2 / (1000.0 * 86400.0)

        results_df = pd.DataFrame({
            'datetime': time_index,
            'streamflow_mm_day': runoff,
            'streamflow_cms': streamflow_cms,
        })

        csv_file = self.output_dir / f"{self.domain_name}_xinanjiang_output.csv"
        results_df.to_csv(csv_file, index=False)

        ds = xr.Dataset(
            data_vars={
                'streamflow': (['time'], streamflow_cms),
                'runoff': (['time'], runoff),
            },
            coords={'time': time_index},
            attrs={
                'model': 'Xinanjiang',
                'spatial_mode': 'lumped',
                'domain': self.domain_name,
                'experiment_id': self.experiment_id,
                'catchment_area_m2': area_m2,
            }
        )
        ds['streamflow'].attrs = {'units': 'm3/s', 'long_name': 'Streamflow'}
        ds['runoff'].attrs = {'units': 'mm/day', 'long_name': 'Runoff depth'}

        nc_file = self.output_dir / f"{self.domain_name}_xinanjiang_output.nc"
        encoding = create_netcdf_encoding(ds, compression=True)
        ds.to_netcdf(nc_file, encoding=encoding)
        self.logger.info(f"Saved Xinanjiang output to: {nc_file}")

    def _calculate_and_log_metrics(self) -> None:
        """Calculate and log performance metrics."""
        try:
            from symfluence.evaluation.metrics import kge, nse

            output_file = self.output_dir / f"{self.domain_name}_xinanjiang_output.nc"
            if not output_file.exists():
                return

            ds = xr.open_dataset(output_file)
            sim = ds['streamflow'].values
            sim_time = pd.to_datetime(ds.time.values)
            ds.close()

            obs_file = (self.project_observations_dir / 'streamflow' /
                        'preprocessed' / f"{self.domain_name}_streamflow_processed.csv")
            if not obs_file.exists():
                return

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)
            sim_series = pd.Series(sim, index=sim_time)
            obs_series = obs_df.iloc[:, 0]

            if len(sim_series) > self.warmup_days:
                sim_series = sim_series.iloc[self.warmup_days:]

            common_idx = sim_series.index.intersection(obs_series.index)
            if len(common_idx) < 10:
                return

            sim_aligned = sim_series.loc[common_idx].values
            obs_aligned = obs_series.loc[common_idx].values
            valid_mask = ~(np.isnan(sim_aligned) | np.isnan(obs_aligned))
            sim_aligned = sim_aligned[valid_mask]
            obs_aligned = obs_aligned[valid_mask]

            if len(sim_aligned) == 0:
                return

            kge_val = kge(obs_aligned, sim_aligned, transfo=1)
            nse_val = nse(obs_aligned, sim_aligned, transfo=1)

            self.logger.info("=" * 40)
            self.logger.info("Xinanjiang Model Performance (lumped)")
            self.logger.info(f"   KGE: {kge_val:.4f}")
            self.logger.info(f"   NSE: {nse_val:.4f}")
            self.logger.info("=" * 40)

        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {e}")
