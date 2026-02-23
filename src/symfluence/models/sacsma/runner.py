"""
SAC-SMA Model Runner.

Handles SAC-SMA + Snow-17 model execution and output processing.
Lumped mode only.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.data.utils.netcdf_utils import create_netcdf_encoding
from symfluence.geospatial.geometry_utils import calculate_catchment_area_km2
from symfluence.models.base import BaseModelRunner
from symfluence.models.execution import SpatialOrchestrator
from symfluence.models.mixins import ObservationLoaderMixin, SpatialModeDetectionMixin
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_runner('SACSMA', method_name='run_sacsma')
class SacSmaRunner(  # type: ignore[misc]
    BaseModelRunner,
    SpatialOrchestrator,
    SpatialModeDetectionMixin,
    ObservationLoaderMixin
):
    """Runner for the SAC-SMA + Snow-17 coupled model."""

    MODEL_NAME = "SACSMA"

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
        self.spatial_mode = self.detect_spatial_mode('SACSMA')

        self.warmup_days = self._get_config_value(
            lambda: self.config.model.sacsma.warmup_days
            if self.config.model and hasattr(self.config.model, 'sacsma') and self.config.model.sacsma
            else None,
            365
        )

        self.latitude = self._get_config_value(
            lambda: self.config.model.sacsma.latitude
            if self.config.model and hasattr(self.config.model, 'sacsma') and self.config.model.sacsma
            else None,
            45.0
        )

        self.si = self._get_config_value(
            lambda: self.config.model.sacsma.si
            if self.config.model and hasattr(self.config.model, 'sacsma') and self.config.model.sacsma
            else None,
            100.0
        )

        self.snow_module = self._get_config_value(
            lambda: self.config.model.sacsma.snow_module
            if self.config.model and hasattr(self.config.model, 'sacsma') and self.config.model.sacsma
            else None,
            'snow17'
        )

        self.backend = self._get_config_value(
            lambda: self.config.model.sacsma.backend
            if self.config.model and hasattr(self.config.model, 'sacsma') and self.config.model.sacsma
            else None,
            'numpy'
        )

    def _setup_model_specific_paths(self) -> None:
        if hasattr(self, 'settings_dir') and self.settings_dir:
            self.sacsma_setup_dir = self.settings_dir
        else:
            self.sacsma_setup_dir = self.project_dir / "settings" / "SACSMA"
        self.sacsma_forcing_dir = self.project_forcing_dir / 'SACSMA_input'

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

    def run_sacsma(self, params: Optional[Dict[str, float]] = None) -> Optional[Path]:
        """Run the SAC-SMA + Snow-17 model."""
        self.logger.info("Starting SAC-SMA model run (lumped mode)")

        if params:
            self._external_params = params

        with symfluence_error_handler(
            "SAC-SMA model execution", self.logger, error_type=ModelExecutionError
        ):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            success = self._execute_lumped()

            if success:
                self.logger.info("SAC-SMA model run completed successfully")
                self._calculate_and_log_metrics()
                return self.output_dir
            else:
                self.logger.error("SAC-SMA model run failed")
                return None

    def _execute_lumped(self) -> bool:
        """Execute SAC-SMA in lumped mode."""
        try:
            from .model import simulate
            from .parameters import DEFAULT_PARAMS

            # Load forcing
            forcing = self._load_forcing()
            precip = forcing['precip'].flatten()
            temp = forcing['temp'].flatten()
            pet = forcing['pet'].flatten()
            time_index = forcing['time']

            # Get parameters
            params = self._external_params if self._external_params else DEFAULT_PARAMS.copy()

            # Create day_of_year array
            day_of_year = pd.to_datetime(time_index).dayofyear.values

            # Run simulation
            self.logger.info(f"Running SAC-SMA for {len(precip)} timesteps")
            runoff, final_state = simulate(
                precip, temp, pet,
                params=params,
                day_of_year=day_of_year,
                warmup_days=self.warmup_days,
                latitude=self.latitude,
                si=self.si,
                use_jax=(self.backend == 'jax'),
                snow_module=self.snow_module,
            )

            # Save results
            self._save_lumped_results(runoff, time_index)
            return True

        except FileNotFoundError as e:
            self.logger.error(f"Missing forcing data: {e}")
            return False
        except (ValueError, TypeError) as e:
            self.logger.error(f"Invalid data in SAC-SMA execution: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _load_forcing(self) -> Dict[str, np.ndarray]:
        """Load forcing data from preprocessed files."""
        nc_file = self.sacsma_forcing_dir / f"{self.domain_name}_sacsma_forcing.nc"
        if nc_file.exists():
            ds = xr.open_dataset(nc_file)
            forcing = {
                'precip': ds['pr'].values,
                'temp': ds['temp'].values,
                'pet': ds['pet'].values,
                'time': pd.to_datetime(ds.time.values),
            }
            ds.close()
            return forcing

        raise FileNotFoundError(f"SAC-SMA forcing not found: {nc_file}")

    def _save_lumped_results(self, runoff: np.ndarray, time_index) -> None:
        """Save lumped simulation results."""
        area_m2 = self._get_catchment_area()

        # mm/day to m³/s
        streamflow_cms = runoff * area_m2 / (1000.0 * 86400.0)

        # Create DataFrame
        results_df = pd.DataFrame({
            'datetime': time_index,
            'streamflow_mm_day': runoff,
            'streamflow_cms': streamflow_cms,
        })

        csv_file = self.output_dir / f"{self.domain_name}_sacsma_output.csv"
        results_df.to_csv(csv_file, index=False)

        # Save NetCDF
        ds = xr.Dataset(
            data_vars={
                'streamflow': (['time'], streamflow_cms),
                'runoff': (['time'], runoff),
            },
            coords={'time': time_index},
            attrs={
                'model': 'SAC-SMA + Snow-17',
                'spatial_mode': 'lumped',
                'domain': self.domain_name,
                'experiment_id': self.experiment_id,
                'catchment_area_m2': area_m2,
            }
        )
        ds['streamflow'].attrs = {'units': 'm3/s', 'long_name': 'Streamflow'}
        ds['runoff'].attrs = {'units': 'mm/day', 'long_name': 'Runoff depth'}

        nc_file = self.output_dir / f"{self.domain_name}_sacsma_output.nc"
        encoding = create_netcdf_encoding(ds, compression=True)
        ds.to_netcdf(nc_file, encoding=encoding)
        self.logger.info(f"Saved SAC-SMA output to: {nc_file}")

    def _calculate_and_log_metrics(self) -> None:
        """Calculate and log performance metrics."""
        try:
            from symfluence.evaluation.metrics import kge, nse

            output_file = self.output_dir / f"{self.domain_name}_sacsma_output.nc"
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
            self.logger.info("SAC-SMA Model Performance (lumped)")
            self.logger.info(f"   KGE: {kge_val:.4f}")
            self.logger.info(f"   NSE: {nse_val:.4f}")
            self.logger.info("=" * 40)

        except Exception as e:
            self.logger.warning(f"Error calculating metrics: {e}")
