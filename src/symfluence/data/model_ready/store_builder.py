# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Orchestrator for building the complete model-ready data store.

Coordinates ``ForcingsStoreBuilder``, ``ObservationsNetCDFBuilder``,
and ``AttributesNetCDFBuilder`` into a single ``build_all()`` entry point.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from .attributes_builder import AttributesNetCDFBuilder
from .forcings_builder import ForcingsStoreBuilder
from .observations_builder import ObservationsNetCDFBuilder

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig

logger = logging.getLogger(__name__)


class ModelReadyStoreBuilder:
    """Orchestrate building the complete model-ready data store.

    Parameters
    ----------
    project_dir : Path
        Root of the SYMFLUENCE domain directory.
    domain_name : str
        Name of the hydrological domain.
    config : SymfluenceConfig or dict, optional
        Typed config or legacy flat dict.
    config_dict : dict, optional
        Deprecated. Use ``config`` instead.
    """

    def __init__(
        self,
        project_dir: Path,
        domain_name: str,
        config: Optional[Union['SymfluenceConfig', dict]] = None,
        config_dict: Optional[dict] = None,
    ) -> None:
        """Initialise the store builder.

        Args:
            project_dir: Root of the SYMFLUENCE domain directory.
            domain_name: Name of the hydrological domain.
            config: Typed config or legacy flat dict.
            config_dict: Deprecated. Use *config* instead.
        """
        self.project_dir = project_dir
        self.domain_name = domain_name
        # Accept either typed config or legacy dict
        self._config = config if config is not None else config_dict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_all(self) -> None:
        """Build forcings, observations, and attributes stores."""
        logger.info("Building model-ready data store for %s", self.domain_name)

        self.build_forcings()
        self._compute_climate_statistics()
        self.build_observations()
        self.build_attributes()

        logger.info("Model-ready data store build complete")

    def _cfg(self, key: str, default=None):
        """Get config value from typed config or legacy dict."""
        cfg = self._config
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        # Typed SymfluenceConfig — use .get() backward-compat layer
        return cfg.get(key, default)

    def build_forcings(self) -> Optional[Path]:
        """Build the forcings section. Skip if no basin_averaged_data."""
        forcing_dataset = self._cfg('FORCING_DATASET', 'ERA5')
        strategy = self._cfg('MODEL_READY_FORCING_STRATEGY', 'symlink')

        builder = ForcingsStoreBuilder(
            project_dir=self.project_dir,
            domain_name=self.domain_name,
            forcing_dataset=forcing_dataset,
            strategy=strategy,
        )
        return builder.build()

    def build_observations(self) -> Optional[Path]:
        """Build the observations section. Skip if no observations dir."""
        builder = ObservationsNetCDFBuilder(
            project_dir=self.project_dir,
            domain_name=self.domain_name,
            config=self._config,
        )
        return builder.build()

    def _compute_climate_statistics(self) -> None:
        """Compute per-HRU climate statistics from basin-averaged forcing.

        Derives mean precipitation, temperature, aridity index, and snow
        fraction from the forcing files and writes them to
        ``attributes/climate/climate_statistics.csv`` so the attributes
        builder can include a ``/climate/`` group in the model-ready store.
        """
        import glob

        import numpy as np
        import pandas as pd

        forcing_dir = self.project_dir / 'data' / 'forcing' / 'basin_averaged_data'
        if not forcing_dir.exists():
            return

        files = sorted(glob.glob(str(forcing_dir / '*.nc')))
        if not files:
            return

        try:
            import xarray as xr

            ds = xr.open_mfdataset(
                files, combine='nested', concat_dim='time',
                coords='minimal', compat='override', data_vars='minimal',
            ).sortby('time')

            n_hru = ds.sizes.get('hru', 1)
            if n_hru <= 1:
                ds.close()
                return

            # Identify forcing variable names (CFIF or standard)
            precip_var = next((v for v in ['precipitation_flux', 'pptrate'] if v in ds), None)
            temp_var = next((v for v in ['air_temperature', 'airtemp'] if v in ds), None)

            if precip_var is None or temp_var is None:
                logger.debug("Climate stats: missing precip or temp variable")
                ds.close()
                return

            rows = []
            for i in range(n_hru):
                p_mean = float(ds[precip_var].isel(hru=i).mean())
                p_mm_yr = p_mean * 86400 * 365.25

                t_vals = ds[temp_var].isel(hru=i).values
                t_c = t_vals - 273.15 if np.nanmean(t_vals) > 100 else t_vals
                t_mean = float(np.nanmean(t_c))

                pet_daily = np.where(t_c > -5, 0.4 * (t_c + 5), 0.0)
                pet_annual = float(np.nanmean(pet_daily)) * 365.25
                aridity = pet_annual / max(p_mm_yr, 1.0)

                p_vals = ds[precip_var].isel(hru=i).values
                wet = p_vals > 1e-7
                snow_frac = float(np.sum((t_c < 0) & wet) / max(np.sum(wet), 1))

                rows.append({
                    'precip_mm_yr': p_mm_yr,
                    'temp_C': t_mean,
                    'aridity': aridity,
                    'snow_frac': snow_frac,
                })

            ds.close()

            # ── Enrich with mean elevation from catchment shapefile ──
            elev_values = [float('nan')] * n_hru
            try:
                import geopandas as gpd

                shp_root = self.project_dir / 'shapefiles' / 'catchment'
                shp_file = None
                if shp_root.exists():
                    for dirpath, _dirnames, filenames in os.walk(shp_root):
                        for fn in filenames:
                            if fn.endswith('.shp') and 'HRUs' in fn:
                                shp_file = Path(dirpath) / fn
                                break
                        if shp_file is not None:
                            break

                if shp_file is not None:
                    gdf = gpd.read_file(shp_file)
                    if 'elev_mean' in gdf.columns and len(gdf) == n_hru:
                        elev_values = gdf['elev_mean'].tolist()
                        logger.debug("Loaded elev_mean from %s", shp_file)
                    else:
                        logger.debug(
                            "Shapefile found but elev_mean missing or row count "
                            "mismatch (%d vs %d HRUs)", len(gdf), n_hru,
                        )
            except Exception:  # noqa: BLE001
                logger.debug("Could not load elevation from shapefile")

            for row, elev in zip(rows, elev_values):
                row['elev_m'] = elev

            out_dir = self.project_dir / 'data' / 'attributes' / 'climate'
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / 'climate_statistics.csv', index=False)
            logger.info("Climate statistics computed for %d HRUs", n_hru)

        except Exception as e:  # noqa: BLE001
            logger.debug("Could not compute climate statistics: %s", e)

    def build_attributes(self) -> Optional[Path]:
        """Build the attributes section. Skip if no intersection shapefiles."""
        builder = AttributesNetCDFBuilder(
            project_dir=self.project_dir,
            domain_name=self.domain_name,
            config=self._config,
        )
        return builder.build()

    def is_store_complete(self) -> bool:
        """Check if all available data has been materialized."""
        store_dir = self.project_dir / 'data' / 'model_ready'
        if not store_dir.exists():
            return False

        has_forcings = any((store_dir / 'forcings').glob('*.nc')) if (store_dir / 'forcings').exists() else False
        has_obs = (store_dir / 'observations' / f'{self.domain_name}_observations.nc').exists()
        has_attrs = (store_dir / 'attributes' / f'{self.domain_name}_attributes.nc').exists()

        # Consider complete if at least forcings are present (obs/attrs optional)
        return has_forcings or has_obs or has_attrs

    def migrate_from_legacy(self) -> None:
        """Build model-ready store from existing legacy domain directory."""
        logger.info("Migrating legacy domain '%s' to model-ready store", self.domain_name)
        self.build_all()
