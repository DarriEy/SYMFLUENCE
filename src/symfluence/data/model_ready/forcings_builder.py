"""
Forcings store builder for the model-ready data store.

Symlinks (or copies) basin-averaged forcing NetCDF files into
``data/model_ready/forcings/`` and enriches them with CF-1.8 global
attributes and per-variable source metadata.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from symfluence.core.mixins.project import resolve_data_subdir

from .cf_conventions import CF_STANDARD_NAMES, build_global_attrs
from .source_metadata import SourceMetadata

logger = logging.getLogger(__name__)


class ForcingsStoreBuilder:
    """Build the forcings section of the model-ready data store.

    Parameters
    ----------
    project_dir : Path
        Root of the SYMFLUENCE domain directory.
    domain_name : str
        Name of the hydrological domain.
    forcing_dataset : str
        Name of the forcing dataset (e.g. ``'ERA5'``, ``'RDRS'``).
    strategy : str
        Either ``'symlink'`` (default) or ``'copy'``.  Symlinks are
        preferred; copies are needed on HPC file-systems that forbid
        symlinks across partitions.
    """

    def __init__(
        self,
        project_dir: Path,
        domain_name: str,
        forcing_dataset: str = 'ERA5',
        strategy: str = 'symlink',
    ) -> None:
        """Initialise the forcings builder.

        Args:
            project_dir: Root of the SYMFLUENCE domain directory.
            domain_name: Name of the hydrological domain.
            forcing_dataset: Forcing dataset identifier (e.g. ``'ERA5'``).
            strategy: ``'symlink'`` (default) or ``'copy'``.
        """
        self.project_dir = project_dir
        self.domain_name = domain_name
        self.forcing_dataset = forcing_dataset
        self.strategy = strategy

        self.source_dir = resolve_data_subdir(project_dir, 'forcing') / 'basin_averaged_data'
        self.target_dir = project_dir / 'data' / 'model_ready' / 'forcings'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> Optional[Path]:
        """Build the forcings store.

        Returns the target directory on success, or ``None`` if the source
        directory does not exist or contains no NetCDF files.
        """
        if not self.source_dir.exists():
            logger.info(
                "Skipping forcings store: source dir does not exist: %s",
                self.source_dir,
            )
            return None

        nc_files = list(self.source_dir.glob('*.nc'))
        if not nc_files:
            logger.info("No NetCDF files in %s, skipping forcings store", self.source_dir)
            return None

        self.target_dir.mkdir(parents=True, exist_ok=True)

        self._create_links(nc_files)
        self._enrich_metadata(nc_files)

        logger.info(
            "Forcings store built: %d files in %s", len(nc_files), self.target_dir
        )
        return self.target_dir

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _create_links(self, nc_files: list) -> None:
        """Create symlinks or copies from source to target directory."""
        for src_file in nc_files:
            dst = self.target_dir / src_file.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()

            if self.strategy == 'copy':
                shutil.copy2(str(src_file), str(dst))
                logger.debug("Copied %s -> %s", src_file.name, dst)
            else:
                os.symlink(src_file.resolve(), dst)
                logger.debug("Symlinked %s -> %s", src_file.name, dst)

    def _enrich_metadata(self, nc_files: list) -> None:
        """Add CF-1.8 global attrs and per-variable source attrs.

        Operates on the *original* files so that metadata is visible
        through symlinks and persists across rebuilds.
        """
        try:
            import netCDF4  # noqa: N813
        except ImportError:
            logger.warning("netCDF4 not available; skipping metadata enrichment")
            return

        global_attrs = build_global_attrs(
            domain_name=self.domain_name,
            title=f'{self.domain_name} forcing data',
            history=f'Enriched by ForcingsStoreBuilder from {self.forcing_dataset}',
        )

        source_meta = SourceMetadata(
            source=self.forcing_dataset,
            processing='EASMORE area-weighted remapping',
        )
        var_attrs = source_meta.to_netcdf_attrs()

        for src_file in nc_files:
            try:
                with netCDF4.Dataset(str(src_file), 'a') as ds:
                    # Global attributes — only set if not already present
                    if 'Conventions' not in ds.ncattrs():
                        ds.setncatts(global_attrs)

                    # Per-variable CF + source attributes
                    for var_name in ds.variables:
                        var = ds.variables[var_name]

                        # CF standard attributes
                        if var_name in CF_STANDARD_NAMES:
                            cf = CF_STANDARD_NAMES[var_name]
                            for attr_key, attr_val in cf.items():
                                if attr_key not in var.ncattrs():
                                    var.setncattr(attr_key, attr_val)

                        # Source provenance (skip coordinate variables)
                        if var_name not in ('time', 'hru', 'gru', 'lat', 'lon'):
                            for ak, av in var_attrs.items():
                                if ak not in var.ncattrs():
                                    var.setncattr(ak, av)

            except Exception as e:  # noqa: BLE001 — preprocessing resilience
                logger.warning("Could not enrich metadata for %s: %s", src_file.name, e)
