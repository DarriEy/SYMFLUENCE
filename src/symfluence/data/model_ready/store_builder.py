"""
Orchestrator for building the complete model-ready data store.

Coordinates ``ForcingsStoreBuilder``, ``ObservationsNetCDFBuilder``,
and ``AttributesNetCDFBuilder`` into a single ``build_all()`` entry point.
"""

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union

from .forcings_builder import ForcingsStoreBuilder
from .observations_builder import ObservationsNetCDFBuilder
from .attributes_builder import AttributesNetCDFBuilder

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
