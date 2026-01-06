"""
Mixin for converting model outputs to routing formats.

Provides utilities for converting model output files to formats compatible
with routing models like mizuRoute.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import shutil

import numpy as np
import xarray as xr


class OutputConverterMixin:
    """
    Mixin for converting model outputs to routing-compatible formats.

    Provides methods for transforming spatial dimensions, adding required
    variables, and preparing outputs for use with routing models.
    """

    def convert_to_mizuroute_format(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        squeeze_dims: Optional[List[str]] = None,
        rename_dims: Optional[Dict[str, str]] = None,
        add_id_var: Optional[str] = None,
        id_source_dim: Optional[str] = None,
        create_backup: bool = True
    ) -> Path:
        """
        Convert model output to mizuRoute-compatible format.

        Performs minimal transformations to make model output compatible
        with mizuRoute routing:
        - Squeeze singleton dimensions
        - Rename spatial dimensions (e.g., latitude → gru)
        - Add required ID variables (e.g., gruId)

        Args:
            input_path: Path to input netCDF file
            output_path: Path for output file (default: overwrite input)
            squeeze_dims: List of dimensions to squeeze if singleton
            rename_dims: Dict mapping old dim names to new names
            add_id_var: Name of ID variable to add (e.g., 'gruId')
            id_source_dim: Dimension to use as source for ID values
            create_backup: Whether to create backup before modifying

        Returns:
            Path to the converted file
        """
        if output_path is None:
            output_path = input_path

        # Create backup if modifying in place
        if create_backup and input_path == output_path:
            backup_path = input_path.with_suffix('.backup.nc')
            if not backup_path.exists():
                shutil.copy2(input_path, backup_path)
                if hasattr(self, 'logger'):
                    self.logger.info(f"Created backup: {backup_path}")

        # Load and transform
        with xr.open_dataset(input_path) as ds_in:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Original dimensions: {dict(ds_in.sizes)}")

            # Load into memory to avoid file lock issues when overwriting
            ds = ds_in.load()

        # Now we can safely modify and write without file locks
        # Squeeze singleton dimensions
        if squeeze_dims:
            for dim in squeeze_dims:
                if dim in ds.sizes and ds.sizes[dim] == 1:
                    ds = ds.squeeze(dim, drop=True)
                    if hasattr(self, 'logger'):
                        self.logger.debug(f"Squeezed {dim} dimension")

        # Rename dimensions
        if rename_dims:
            rename_map = {old: new for old, new in rename_dims.items()
                         if old in ds.sizes}
            if rename_map:
                ds = ds.rename(rename_map)
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Renamed dimensions: {rename_map}")

        # Add ID variable
        if add_id_var and id_source_dim and id_source_dim in ds.sizes:
            if add_id_var not in ds:
                id_values = np.arange(1, ds.sizes[id_source_dim] + 1)
                ds[add_id_var] = xr.DataArray(
                    id_values,
                    dims=[id_source_dim],
                    attrs={'long_name': f'{id_source_dim} identifier'}
                )
                if hasattr(self, 'logger'):
                    self.logger.debug(f"Added {add_id_var} variable")

        # Write output (file is now closed, safe to overwrite)
        ds.to_netcdf(output_path)

        if hasattr(self, 'logger'):
            self.logger.info(f"Converted output saved to: {output_path}")

        return output_path
