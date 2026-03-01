# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE (Framework for Understanding Structural Errors) model postprocessor.

Simplified implementation using StandardModelPostprocessor.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import xarray as xr

from ..base import StandardModelPostprocessor
from ..registry import ModelRegistry


@ModelRegistry.register_postprocessor('FUSE')
class FUSEPostprocessor(StandardModelPostprocessor):
    """
    Postprocessor for FUSE model outputs.

    FUSE outputs NetCDF files with routed streamflow in mm/day.
    This postprocessor extracts the streamflow and converts to cms.
    """

    model_name = "FUSE"
    output_file_pattern = "{domain}_{experiment}_runs_best.nc"
    streamflow_variable = "q_routed"
    streamflow_unit = "mm_per_day"  # Will be converted to cms
    netcdf_selections = {"latitude": 0, "longitude": 0}  # param_set determined dynamically

    def _format_pattern(self, pattern: str) -> str:
        """Format file pattern using FUSE_FILE_ID instead of experiment_id.

        FUSE Fortran uses CHARACTER(LEN=6) for FMODEL_ID, so output files
        use a short file ID (FUSE_FILE_ID) rather than the full experiment_id.
        """
        start_time = self._get_config_value(
            lambda: self.config.domain.time_start,
            default=''
        )
        start_date = start_time.split()[0] if start_time else ''

        fuse_id = self._get_config_value(
            lambda: self.config.model.fuse.file_id,
            default=self.experiment_id
        )
        if len(fuse_id) > 6:
            import hashlib
            fuse_id = hashlib.md5(fuse_id.encode(), usedforsecurity=False).hexdigest()[:6]

        return pattern.format(
            domain=self.domain_name,
            experiment=fuse_id,
            start_date=start_date,
            model=self.model_name.lower()
        )

    def _get_output_file(self) -> Path:
        """Get FUSE output file, falling back from runs_best to runs_def."""
        best_file = super()._get_output_file()
        if best_file.exists():
            return best_file

        # Fallback: try runs_def.nc
        def_pattern = self.output_file_pattern.replace('runs_best', 'runs_def')
        def_file = self._get_output_dir() / self._format_pattern(def_pattern)
        if def_file.exists():
            self.logger.info(f"Using runs_def fallback: {def_file.name}")
            return def_file

        return best_file  # Return original path for error reporting

    def _extract_from_netcdf(self, file_path):
        """Extract streamflow with fallback from q_routed to q_instnt."""
        selections = self._get_netcdf_selections()

        # Try primary variable (q_routed), fall back to q_instnt
        for variable in ['q_routed', 'q_instnt']:
            try:
                result = self.read_netcdf_streamflow(file_path, variable, **selections)
                if result is not None:
                    return result
            except (KeyError, ValueError):
                continue

        self.logger.error(f"No streamflow variable (q_routed/q_instnt) found in {file_path}")
        return None

    def _get_netcdf_selections(self) -> Dict[str, Any]:
        """
        Get NetCDF dimension selections with auto-detected param_set.

        FUSE writes valid data to different param_set indices depending on
        the run mode (e.g., run_def writes to param_set 1, not 0).
        This method auto-detects which param_set has valid (non-NaN) data.
        """
        selections = dict(self.netcdf_selections)

        # Try to auto-detect valid param_set from output file
        try:
            output_file = self._get_output_file()
            if output_file.exists():
                with xr.open_dataset(output_file) as ds:
                    if self.streamflow_variable in ds and 'param_set' in ds[self.streamflow_variable].dims:
                        var = ds[self.streamflow_variable]
                        n_param_sets = var.sizes.get('param_set', 1)
                        valid_param_set = 0
                        for ps in range(n_param_sets):
                            test_slice = var.isel(param_set=ps)
                            if 'latitude' in test_slice.dims:
                                test_slice = test_slice.isel(latitude=0)
                            if 'longitude' in test_slice.dims:
                                test_slice = test_slice.isel(longitude=0)
                            if not np.all(np.isnan(test_slice.values)):
                                valid_param_set = ps
                                break
                        selections['param_set'] = valid_param_set
                        self.logger.debug(f"Auto-detected valid param_set: {valid_param_set}")
                    elif 'param_set' in ds.dims:
                        # Variable doesn't have param_set but dataset does
                        selections['param_set'] = 0
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.debug(f"Could not auto-detect param_set: {e}, defaulting to 0")
            selections['param_set'] = 0

        return selections
