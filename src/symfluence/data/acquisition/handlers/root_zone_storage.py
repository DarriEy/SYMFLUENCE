"""
Root Zone Storage Capacity Acquisition Handler

Cloud-based acquisition of the global root zone storage capacity dataset
from Stocker et al. (2023) via Zenodo.

Provides global gridded estimates at 0.5 degree resolution:
- cwdx80: Root zone water storage capacity S_CWDX80 (mm)
- zroot_cwd80: Effective rooting depth z_CWDX80 (mm)

These represent the 80th percentile cumulative water deficit,
i.e., the storage capacity that would be exceeded in 20% of years.

Data Source:
    Zenodo: https://doi.org/10.5281/zenodo.5515246
    No authentication required, CC-BY 4.0 license

References:
    Stocker, B.D., et al. (2023). Global patterns of water storage in
    the rooting zone. Nature Geoscience, 16, 250-256.

Configuration:
    ROOT_ZONE_VARIABLES: ['cwdx80', 'zroot_cwd80'] (default: both)
"""

from pathlib import Path

import numpy as np
import xarray as xr

from ..base import BaseAcquisitionHandler
from ..mixins import RetryMixin
from ..registry import AcquisitionRegistry
from ..utils import create_robust_session, download_file_streaming

# Zenodo download URLs
_ZENODO_FILES = {
    'cwdx80': {
        'url': 'https://zenodo.org/api/records/5515246/files/cwdx80.nc/content',
        'filename': 'cwdx80.nc',
        'description': 'Root zone water storage capacity S_CWDX80',
        'units': 'mm',
        'long_name': 'Root Zone Storage Capacity (80th percentile CWD)',
    },
    'zroot_cwd80': {
        'url': 'https://zenodo.org/api/records/5515246/files/zroot_cwd80.nc/content',
        'filename': 'zroot_cwd80.nc',
        'description': 'Effective rooting depth z_CWDX80',
        'units': 'mm',
        'long_name': 'Effective Rooting Depth (80th percentile CWD)',
    },
}

_DEFAULT_VARIABLES = ['cwdx80', 'zroot_cwd80']


@AcquisitionRegistry.register('ROOT_ZONE_STORAGE')
@AcquisitionRegistry.register('RZSC')
@AcquisitionRegistry.register('ROOTING_DEPTH')
class RootZoneStorageAcquirer(BaseAcquisitionHandler, RetryMixin):
    """
    Stocker et al. (2023) root zone storage capacity acquisition.

    Downloads global NetCDF files from Zenodo and subsets to the
    domain bounding box.

    Output:
        {project_dir}/attributes/soil/root_zone/
            domain_{name}_cwdx80.nc
            domain_{name}_zroot_cwd80.nc
    """

    def download(self, output_dir: Path) -> Path:
        rz_dir = self._attribute_dir("soilclass") / "root_zone"
        rz_dir.mkdir(parents=True, exist_ok=True)

        variables = self._get_config_value(lambda: None, default=_DEFAULT_VARIABLES, dict_key='ROOT_ZONE_VARIABLES')
        variables = [v for v in variables if v in _ZENODO_FILES]

        if not variables:
            raise ValueError(
                f"No valid root zone variables. Choose from: {list(_ZENODO_FILES.keys())}"
            )

        self.logger.info(
            f"Acquiring root zone storage capacity for bbox: {self.bbox}, "
            f"variables: {variables}"
        )

        session = create_robust_session(max_retries=3, backoff_factor=2.0)
        cache_dir = rz_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}
        for var in variables:
            info = _ZENODO_FILES[var]
            out_path = rz_dir / f"domain_{self.domain_name}_{var}.nc"

            if self._skip_if_exists(out_path):
                output_paths[var] = out_path
                continue

            self.logger.info(f"Downloading {var}: {info['description']}")

            # Download global file to cache
            cached_file = cache_dir / info['filename']
            if not cached_file.exists():
                download_file_streaming(
                    info['url'], cached_file, session=session, timeout=300
                )

            # Subset to bbox
            self._subset_netcdf(cached_file, out_path, var, info)
            output_paths[var] = out_path

        if not output_paths:
            raise RuntimeError("No root zone data could be downloaded")

        self.logger.info(
            f"Root zone storage acquisition complete: {len(output_paths)} files"
        )
        return rz_dir

    def _subset_netcdf(self, src_path: Path, dst_path: Path, var: str, info: dict):
        """Subset a global NetCDF to the domain bounding box."""
        ds = xr.open_dataset(src_path)

        # Detect coordinate names (lat/lon or latitude/longitude)
        lat_name = None
        lon_name = None
        for coord in ds.coords:
            if coord.lower() in ('lat', 'latitude', 'y'):
                lat_name = coord
            elif coord.lower() in ('lon', 'longitude', 'x'):
                lon_name = coord

        if lat_name is None or lon_name is None:
            self.logger.warning(f"Could not detect lat/lon coords in {src_path}")
            # Fall back to just copying
            ds.to_netcdf(dst_path)
            ds.close()
            return

        # Handle longitude convention (0-360 vs -180 to 180)
        lon_values = ds[lon_name].values
        if lon_values.max() > 180:
            ds = ds.assign_coords(
                {lon_name: ((ds[lon_name] + 180) % 360) - 180}
            )
            ds = ds.sortby(lon_name)

        # Subset
        lat_slice = slice(
            min(self.bbox['lat_min'], self.bbox['lat_max']),
            max(self.bbox['lat_min'], self.bbox['lat_max'])
        )
        lon_slice = slice(
            min(self.bbox['lon_min'], self.bbox['lon_max']),
            max(self.bbox['lon_min'], self.bbox['lon_max'])
        )

        # Handle both ascending and descending lat
        if ds[lat_name].values[0] > ds[lat_name].values[-1]:
            lat_slice = slice(lat_slice.stop, lat_slice.start)

        subset = ds.sel({lat_name: lat_slice, lon_name: lon_slice})

        # Add metadata
        subset.attrs['title'] = info['long_name']
        subset.attrs['source'] = 'Stocker et al. (2023), doi:10.1038/s41561-023-01125-2'
        subset.attrs['domain'] = self.domain_name

        subset.to_netcdf(dst_path)

        # Log summary
        for data_var in subset.data_vars:
            values = subset[data_var].values
            valid = values[np.isfinite(values)]
            if len(valid) > 0:
                self.logger.info(
                    f"  {data_var}: min={valid.min():.1f}, "
                    f"mean={valid.mean():.1f}, max={valid.max():.1f} {info['units']}"
                )

        ds.close()
        subset.close()
