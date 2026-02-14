"""
MESH Parameter Fixer

Fixes parameter files for MESH compatibility and stability.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.mixins import ConfigMixin


from .config_defaults import MESHConfigDefaults


class MESHParameterFixer(ConfigMixin):
    """
    Fixes MESH parameter files for compatibility and stability.

    Handles:
    - Run options variable name fixes
    - Snow/ice parameter fixes for multi-year stability
    - GRU count mismatches between CLASS and DDB
    - CLASS initial conditions for snow simulation
    - Hydrology WF_R2 parameter
    - Safe forcing file creation
    """

    def __init__(
        self,
        forcing_dir: Path,
        setup_dir: Path,
        config: Dict[str, Any],
        logger: logging.Logger = None,
        time_window_func=None
    ):
        """
        Initialize parameter fixer.

        Args:
            forcing_dir: Directory containing MESH files
            setup_dir: Directory containing settings files
            config: Configuration dictionary
            logger: Optional logger instance
            time_window_func: Function to get simulation time window
        """
        self.forcing_dir = forcing_dir
        self.setup_dir = setup_dir
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger or logging.getLogger(__name__)
        self.get_simulation_time_window = time_window_func
        self._actual_spinup_days = None

    @property
    def run_options_path(self) -> Path:
        return self.forcing_dir / "MESH_input_run_options.ini"

    @property
    def class_file_path(self) -> Path:
        return self.forcing_dir / "MESH_parameters_CLASS.ini"

    @property
    def hydro_path(self) -> Path:
        return self.forcing_dir / "MESH_parameters_hydrology.ini"

    @property
    def ddb_path(self) -> Path:
        return self.forcing_dir / "MESH_drainage_database.nc"

    def fix_run_options_var_names(self) -> None:
        """Fix variable names in run options to match forcing file."""
        if not self.run_options_path.exists():
            return

        try:
            with open(self.run_options_path, 'r', encoding='utf-8') as f:
                content = f.read()

            var_replacements = {
                'name_var=SWRadAtm': 'name_var=FSIN',
                'name_var=spechum': 'name_var=QA',
                'name_var=airtemp': 'name_var=TA',
                'name_var=windspd': 'name_var=UV',
                'name_var=pptrate': 'name_var=PRE',
                'name_var=airpres': 'name_var=PRES',
                'name_var=LWRadAtm': 'name_var=FLIN',
            }

            modified = False
            for old_name, new_name in var_replacements.items():
                if old_name in content:
                    content = content.replace(old_name, new_name)
                    modified = True

            if modified:
                with open(self.run_options_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self._update_control_flag_count()
                self.logger.info("Fixed run options variable names")

        except Exception as e:
            self.logger.warning(f"Failed to fix run options variable names: {e}")

    def _get_num_cells(self) -> int:
        """Get number of cells from drainage database."""
        if not self.ddb_path.exists():
            return 1
        try:
            with xr.open_dataset(self.ddb_path) as ds:
                for dim in ['subbasin', 'N']:
                    if dim in ds.sizes:
                        return int(ds.sizes[dim])
        except Exception:
            pass
        return 1

    def fix_run_options_snow_params(self) -> None:
        """Fix run options snow/ice parameters for stable multi-year simulations."""
        if not self.run_options_path.exists():
            return

        try:
            with open(self.run_options_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Get RUNMODE from config (default to 'runrte' for routing)
            runmode = self._get_config_value('MESH_RUNMODE', 'runrte')

            # For single-cell (lumped) domains, use noroute mode.
            # Note: run_def was attempted to enable wf_lzs baseflow (NRVR=1),
            # but in MESH 1.5.6 run_def routes drainage directly to the channel,
            # bypassing the lower zone store (STGGW=0 → LKG=0). The baseflow
            # module requires both NRVR>0 AND drainage→STGGW, which are mutually
            # exclusive for single-cell domains. The extractor compensates by
            # using RFF + DRAINSOL as the total runoff proxy.
            num_cells = self._get_num_cells()
            if num_cells == 1 and runmode != 'noroute':
                self.logger.info(
                    "Single-cell domain detected (lumped mode). "
                    "Using RUNMODE 'noroute' (extractor handles RFF+DRAINSOL)."
                )
                runmode = 'noroute'

            # Determine output flags based on routing mode
            if runmode == 'noroute':
                # In noroute mode, no routing at all (no baseflow either)
                streamflow_flag = 'none'
                outfiles_flag = 'daily'
                basinavgwb_flag = 'daily'
            else:
                # run_def or runrte: enable streamflow + water balance output
                streamflow_flag = 'csv'
                outfiles_flag = 'daily'
                basinavgwb_flag = 'daily'

            # Determine frozen soil flag
            # Default to 0 (OFF) for stability, but allow enabling for cold regions
            enable_frozen = self._get_config_value(lambda: self.config.model.mesh.enable_frozen_soil, default=False, dict_key="MESH_ENABLE_FROZEN_SOIL")
            frozen_flag = "1" if enable_frozen else "0"
            if enable_frozen:
                self.logger.info("FROZENSOILINFILFLAG enabled (1) - calibration of FRZTH is recommended")

            modified = False
            # Snow parameters: SWELIM reduced from 1500 to 500mm for temperate regions
            # 1500mm was unrealistically high and caused multi-year accumulation issues
            # For alpine/polar applications, override via MESH_SWELIM config option
            replacements = [
                (r'FREZTH\s+[-\d.]+', 'FREZTH                0.0'),
                (r'SWELIM\s+[-\d.]+', f'SWELIM                {self._get_config_value(lambda: self.config.model.mesh.swelim, default=800.0, dict_key="MESH_SWELIM")}'),
                (r'SNDENLIM\s+[-\d.]+', 'SNDENLIM              600.0'),
                (r'PBSMFLAG\s+\w+', 'PBSMFLAG              off'),
                (r'FROZENSOILINFILFLAG\s+\d+', f'FROZENSOILINFILFLAG   {frozen_flag}'),
                (r'RUNMODE\s+\w+', f'RUNMODE               {runmode}'),
                (r'METRICSSPINUP\s+\d+', f'METRICSSPINUP         {int(self._get_config_value(lambda: self.config.model.mesh.spinup_days, default=730, dict_key="MESH_SPINUP_DAYS"))}'),
                (r'DIAGNOSEMODE\s+\w+', 'DIAGNOSEMODE          off'),
                (r'SHDFILEFLAG\s+\w+', 'SHDFILEFLAG           nc_subbasin pad_outlets'),
                (r'BASINFORCINGFLAG\s+\w+', 'BASINFORCINGFLAG      nc_subbasin'),
                (r'OUTFILESFLAG\s+\w+', f'OUTFILESFLAG         {outfiles_flag}'),
                (r'OUTFIELDSFLAG\s+\w+', 'OUTFIELDSFLAG        none'),
                (r'STREAMFLOWOUTFLAG\s+\w+', f'STREAMFLOWOUTFLAG     {streamflow_flag}'),
                (r'BASINAVGWBFILEFLAG\s+\w+', f'BASINAVGWBFILEFLAG    {basinavgwb_flag}'),
                (r'PRINTSIMSTATUS\s+\w+', 'PRINTSIMSTATUS        date_monthly'),
            ]

            for pattern, replacement in replacements:
                if re.search(pattern, content):
                    content_new = re.sub(pattern, replacement, content)
                    if content_new != content:
                        content = content_new
                        modified = True

            # Log the routing mode being used
            self.logger.info(
                f"MESH RUNMODE set to '{runmode}' with streamflow output '{streamflow_flag}', "
                f"basin WB output '{basinavgwb_flag}'"
            )

            if modified:
                with open(self.run_options_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self._update_control_flag_count()
                self.logger.info("Fixed run options snow/ice parameters")

        except Exception as e:
            self.logger.warning(f"Failed to fix run options snow parameters: {e}")

    def _update_control_flag_count(self) -> None:
        """Update the number of control flags in MESH_input_run_options.ini."""
        if not self.run_options_path.exists():
            return

        try:
            with open(self.run_options_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            flag_start_idx = -1
            count_line_idx = -1
            for i, line in enumerate(lines):
                if 'Number of control flags' in line:
                    count_line_idx = i
                if line.startswith('----#'):
                    flag_start_idx = i + 1
                    break

            if count_line_idx == -1 or flag_start_idx == -1:
                return

            # Count flags until the next section (starting with #####)
            flag_count = 0
            for i in range(flag_start_idx, len(lines)):
                if lines[i].startswith('#####'):
                    break
                if lines[i].strip() and not lines[i].strip().startswith('#'):
                    flag_count += 1

            # Update the count line
            old_line = lines[count_line_idx]
            match = re.search(r'(\s*)(\d+)(\s*#.*)', old_line)
            if match:
                new_line = f"{match.group(1)}{flag_count:2d}{match.group(3)}\n"
                if new_line != old_line:
                    lines[count_line_idx] = new_line
                    with open(self.run_options_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    self.logger.info(f"Updated control flag count to {flag_count}")

        except Exception as e:
            self.logger.warning(f"Failed to update control flag count: {e}")

    def fix_gru_count_mismatch(self) -> None:
        """Ensure CLASS NM matches GRU count and renormalize GRU fractions.

        MESH has an off-by-one issue where it reads NGRU-1 GRUs from the drainage
        database. This function ensures:
        1. CLASS has NGRU-1 parameter blocks
        2. The first NGRU-1 GRU fractions sum to 1.0 (not all NGRU)

        If MESH_FORCE_SINGLE_GRU is enabled, collapses all GRUs to a single
        dominant land cover class to simplify calibration and avoid numerical
        instabilities in minor GRU classes.

        This function is idempotent - if already aligned, it does nothing.
        """
        # Check if single-GRU mode is requested
        force_single_gru = self._get_config_value(
            lambda: self.config.model.mesh.force_single_gru,
            default=False,
            dict_key='MESH_FORCE_SINGLE_GRU'
        )

        if force_single_gru:
            self._collapse_to_single_gru()
            return

        # First, remove GRUs below the minimum fraction threshold
        # This prevents numerical instability from very small GRU classes
        self._remove_small_grus()

        current_ddb_gru_count = self._get_ddb_gru_count()
        class_block_count = self._get_class_block_count()

        if current_ddb_gru_count is None or class_block_count is None:
            # Fall back to original logic
            keep_mask = self._trim_empty_gru_columns()
            self._fix_class_nm(keep_mask)
            self._ensure_gru_normalization()
            return

        # MESH reads NGRU-1 GRUs, so CLASS should have NGRU-1 blocks
        expected_class_blocks = max(1, current_ddb_gru_count - 1)

        # Check if already aligned
        if class_block_count == expected_class_blocks:
            self.logger.debug(
                f"Already aligned: CLASS has {class_block_count} blocks, "
                f"DDB has {current_ddb_gru_count} NGRU (MESH reads {expected_class_blocks})"
            )
            # Just ensure GRU fractions are normalized for the first NGRU-1 columns
            self._renormalize_mesh_active_grus(expected_class_blocks)
            return

        self.logger.warning(
            f"GRU count mismatch: CLASS has {class_block_count} blocks, "
            f"but MESH will read {expected_class_blocks} GRUs (DDB NGRU={current_ddb_gru_count})"
        )

        # Strategy: adjust CLASS to match MESH's expectations
        if class_block_count > expected_class_blocks:
            # Too many CLASS blocks - trim to match
            self._trim_class_to_count(expected_class_blocks)
            self._update_class_nm(expected_class_blocks)
        elif class_block_count < expected_class_blocks:
            # Too few CLASS blocks - we need to trim DDB to have class_block_count + 1
            # This way MESH reads exactly class_block_count GRUs
            target_ddb_count = class_block_count + 1
            self.logger.info(
                f"Trimming DDB to {target_ddb_count} GRUs so MESH reads {class_block_count}"
            )
            self._trim_ddb_to_active_grus(target_ddb_count)

        # Renormalize GRU fractions for MESH's active GRUs
        final_class_count = self._get_class_block_count() or expected_class_blocks
        self._renormalize_mesh_active_grus(final_class_count)

    def _collapse_to_single_gru(self) -> None:
        """Collapse all GRUs to a single dominant land cover class.

        This simplifies the model by using a single GRU, which:
        - Avoids numerical instabilities in minor GRU classes
        - Simplifies calibration (fewer parameters)
        - Is appropriate for lumped mode where spatial heterogeneity is already ignored

        The dominant GRU (largest fraction) is kept, all others are removed.
        """
        self.logger.info("MESH_FORCE_SINGLE_GRU enabled - collapsing to single GRU")

        # Find dominant GRU from DDB
        if not self.ddb_path.exists():
            self.logger.warning("No DDB found, cannot collapse to single GRU")
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    self.logger.warning("No GRU data in DDB")
                    return

                gru = ds['GRU'].values  # Shape: (subbasin, NGRU)
                ngru = int(ds.sizes['NGRU'])

                # Find index of dominant GRU (largest total fraction)
                gru_sums = gru.sum(axis=0)  # Sum across subbasins
                dominant_idx = int(np.argmax(gru_sums))
                dominant_fraction = gru_sums[dominant_idx]

                self.logger.info(
                    f"Dominant GRU index: {dominant_idx} (fraction: {dominant_fraction:.4f})"
                )

                # Create new DDB with only 2 GRU columns (MESH reads NGRU-1=1)
                # First column gets fraction 1.0, second column gets 0.0
                new_gru = np.zeros((gru.shape[0], 2))
                new_gru[:, 0] = 1.0  # Single GRU with 100% fraction

                # Update DDB
                ds_new = ds.copy()
                # Need to create new dataset with different NGRU dimension
                ds_new = ds_new.drop_dims('NGRU')
                ds_new['GRU'] = xr.DataArray(
                    new_gru,
                    dims=['subbasin', 'NGRU'],
                    coords={'subbasin': ds['subbasin']}
                )

                # Copy other NGRU-dimensioned variables if they exist
                for var in ['LandUse', 'LandClass']:
                    if var in ds:
                        # Keep only dominant GRU's value
                        old_vals = ds[var].values
                        if old_vals.ndim == 1:  # Shape: (NGRU,)
                            new_vals = np.array([old_vals[dominant_idx], 0])
                        else:  # Shape: (subbasin, NGRU)
                            new_vals = np.zeros((old_vals.shape[0], 2))
                            new_vals[:, 0] = old_vals[:, dominant_idx]
                        ds_new[var] = xr.DataArray(
                            new_vals,
                            dims=ds[var].dims if len(new_vals.shape) > 1 else ['NGRU']
                        )

                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds_new.to_netcdf(temp_path)
                os.replace(temp_path, self.ddb_path)

                self.logger.info(
                    f"Collapsed DDB from {ngru} to 2 GRU columns (MESH reads 1)"
                )

        except Exception as e:
            self.logger.warning(f"Failed to collapse DDB to single GRU: {e}")
            return

        # Trim CLASS to single block (keep dominant GRU's parameters)
        self._trim_class_to_count(1)
        self._update_class_nm(1)
        self.logger.info("Single-GRU mode activated: 1 CLASS block, 1 active GRU")

    def _remove_small_grus(self) -> None:
        """Remove GRUs below the minimum fraction threshold.

        Small GRU fractions (e.g., <5%) can cause numerical instability in CLASS
        energy balance calculations because:
        1. Small fractions amplify numerical errors in flux calculations
        2. Minor land cover classes may have poorly-constrained parameters
        3. The GRU's contribution to catchment response is negligible

        This method:
        1. Identifies GRUs below MESH_GRU_MIN_TOTAL threshold
        2. Removes their columns from the DDB
        3. Removes corresponding CLASS parameter blocks
        4. Renormalizes remaining GRU fractions to sum to 1.0
        """
        min_fraction = float(self._get_config_value(
            lambda: self.config.model.mesh.gru_min_total,
            default=0.05,
            dict_key='MESH_GRU_MIN_TOTAL'
        ))

        if not self.ddb_path.exists():
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return

                gru = ds['GRU']
                ngru = int(ds.sizes['NGRU'])

                # Get spatial dimension
                sum_dim = 'N' if 'N' in gru.dims else 'subbasin' if 'subbasin' in gru.dims else None
                if not sum_dim:
                    return

                # Calculate GRU fractions (sum across spatial units, then average)
                # For single-cell (lumped) domains, this is just the GRU values
                gru_fractions = gru.sum(sum_dim).values
                if gru.sizes[sum_dim] > 1:
                    gru_fractions = gru_fractions / gru.sizes[sum_dim]

                # Identify GRUs to keep (above threshold)
                keep_mask = gru_fractions >= min_fraction
                n_keep = int(keep_mask.sum())
                n_remove = ngru - n_keep

                if n_remove == 0:
                    self.logger.debug(
                        f"All {ngru} GRUs above {min_fraction:.1%} threshold"
                    )
                    return

                if n_keep == 0:
                    # Edge case: all GRUs below threshold - keep the largest
                    largest_idx = int(np.argmax(gru_fractions))
                    keep_mask[largest_idx] = True
                    n_keep = 1
                    n_remove = ngru - 1
                    self.logger.warning(
                        f"All GRUs below {min_fraction:.1%} threshold, keeping largest (idx={largest_idx})"
                    )

                # Log which GRUs are being removed
                removed_indices = [i for i, keep in enumerate(keep_mask) if not keep]
                removed_fractions = [gru_fractions[i] for i in removed_indices]
                self.logger.info(
                    f"Removing {n_remove} GRUs below {min_fraction:.1%} threshold: "
                    f"indices {removed_indices} with fractions {[f'{f:.3f}' for f in removed_fractions]}"
                )

                # Subset the DDB to keep only GRUs above threshold
                keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
                ds_filtered = ds.isel(NGRU=keep_indices)

                # Renormalize GRU fractions to sum to 1.0
                if 'GRU' in ds_filtered:
                    gru_sum = ds_filtered['GRU'].sum('NGRU')
                    gru_sum_safe = xr.where(gru_sum == 0, 1.0, gru_sum)
                    ds_filtered['GRU'] = ds_filtered['GRU'] / gru_sum_safe

                    # Log new fractions
                    new_fractions = ds_filtered['GRU'].sum(sum_dim).values
                    if ds_filtered['GRU'].sizes[sum_dim] > 1:
                        new_fractions = new_fractions / ds_filtered['GRU'].sizes[sum_dim]
                    self.logger.debug(
                        f"Renormalized GRU fractions: {[f'{f:.3f}' for f in new_fractions]}"
                    )

                # Save filtered DDB
                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds_filtered.to_netcdf(temp_path)
                os.replace(temp_path, self.ddb_path)

                self.logger.info(
                    f"Removed {n_remove} small GRU(s), {n_keep} remaining"
                )

        except Exception as e:
            self.logger.warning(f"Failed to remove small GRUs from DDB: {e}")
            return

        # Now remove corresponding CLASS parameter blocks
        self._remove_class_blocks_by_mask(keep_mask)

    def _remove_class_blocks_by_mask(self, keep_mask: np.ndarray) -> None:
        """Remove CLASS parameter blocks corresponding to removed GRUs.

        Args:
            keep_mask: Boolean array where True = keep GRU, False = remove
        """
        if not self.class_file_path.exists():
            return

        try:
            with open(self.class_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Find block starts (line 5 of each GRU block)
            block_starts = [i for i, line in enumerate(lines) if '05 5xFCAN/4xLAMX' in line]

            if not block_starts:
                self.logger.debug("No CLASS blocks found (looking for '05 5xFCAN/4xLAMX')")
                return

            n_blocks = len(block_starts)
            n_mask = len(keep_mask)

            # The mask may be longer than blocks (DDB has NGRU, CLASS has NGRU-1)
            # Align from the front - first N blocks correspond to first N mask entries
            effective_mask = keep_mask[:n_blocks] if n_mask >= n_blocks else np.pad(
                keep_mask, (0, n_blocks - n_mask), constant_values=True
            )

            n_keep = int(effective_mask.sum())
            n_remove = n_blocks - n_keep

            if n_remove == 0:
                return

            self.logger.debug(
                f"Removing {n_remove} CLASS blocks (keeping indices {[i for i, k in enumerate(effective_mask) if k]})"
            )

            # Find footer (lines 20, 21, 22)
            footer_start = None
            for i, line in enumerate(lines):
                # Look for line starting with spaces/zeros and containing '20 '
                if re.search(r'^\s*0\s+0\s+0\s+0.*20\s', line):
                    footer_start = i
                    break

            footer = []
            if footer_start is not None:
                footer = lines[footer_start:]
                lines = lines[:footer_start]

            # Identify header (everything before first block)
            header = lines[:block_starts[0]]

            # Identify block boundaries
            block_ends = block_starts[1:] + [len(lines)]
            blocks = [lines[block_starts[i]:block_ends[i]] for i in range(n_blocks)]

            # Keep only blocks where mask is True
            kept_blocks = [blocks[i] for i in range(n_blocks) if effective_mask[i]]

            # Reconstruct file
            new_lines = header + [line for block in kept_blocks for line in block] + footer
            content = '\n'.join(new_lines)
            if not content.endswith('\n'):
                content += '\n'

            with open(self.class_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.logger.info(f"Removed {n_remove} CLASS block(s), {n_keep} remaining")

            # Update NM parameter
            self._update_class_nm(n_keep)

        except Exception as e:
            self.logger.warning(f"Failed to remove CLASS blocks: {e}")

    def _get_ddb_gru_count(self) -> Optional[int]:
        """Get the number of GRU columns in the DDB."""
        if not self.ddb_path.exists():
            return None

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                if 'NGRU' not in ds.dims:
                    return None
                return int(ds.sizes['NGRU'])
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return None

    def _trim_ddb_to_active_grus(self, target_count: int) -> None:
        """Trim DDB GRU columns to exactly target_count, keeping first N columns.

        MESH has an off-by-one issue where it reads NGRU-1 GRUs from the drainage
        database. This function trims the DDB to exactly match what MESH expects.

        We keep the first N GRU columns (not the largest) to maintain correspondence
        with CLASS parameter blocks, which are also indexed 0 to N-1. The GRU
        fractions are renormalized to sum to 1.0.
        """
        if not self.ddb_path.exists():
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return

                current_count = int(ds.sizes['NGRU'])
                if current_count <= target_count:
                    return

                gru = ds['GRU']
                sum_dim = 'N' if 'N' in gru.dims else 'subbasin' if 'subbasin' in gru.dims else None
                if not sum_dim:
                    return

                # Calculate fractions being removed (last columns)
                sums = gru.sum(sum_dim).values
                removed_fractions = sums[target_count:]
                total_removed = sum(removed_fractions)

                self.logger.info(
                    f"Trimming DDB from {current_count} to {target_count} GRU columns "
                    f"(removing last {current_count - target_count} GRUs with total fraction {total_removed:.4f})"
                )

                # Keep first N GRU columns to maintain CLASS block correspondence
                keep_indices = list(range(target_count))
                ds_trim = ds.isel(NGRU=keep_indices)

                # Renormalize GRU fractions to sum to 1.0
                if 'GRU' in ds_trim:
                    sum_per = ds_trim['GRU'].sum('NGRU')
                    sum_safe = xr.where(sum_per == 0, 1.0, sum_per)
                    ds_trim['GRU'] = ds_trim['GRU'] / sum_safe

                    # Log the renormalization
                    new_gru = ds_trim['GRU'].values
                    self.logger.debug(f"Renormalized GRU fractions: {new_gru}")

                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds_trim.to_netcdf(temp_path)
                os.replace(temp_path, self.ddb_path)
                self.logger.info(
                    f"Trimmed DDB to {target_count} GRU column(s) and renormalized fractions to sum to 1.0"
                )
        except Exception as e:
            self.logger.warning(f"Failed to trim DDB to active GRUs: {e}")

    def _renormalize_mesh_active_grus(self, active_count: int) -> None:
        """Renormalize the first N GRU fractions to sum to 1.0.

        MESH reads only the first (NGRU-1) GRU columns due to an off-by-one issue.
        This function renormalizes only those active columns without changing the
        DDB dimension, ensuring the active GRUs sum to 1.0.

        Args:
            active_count: Number of GRUs that MESH will actually read
        """
        if not self.ddb_path.exists():
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return

                ngru_count = int(ds.sizes['NGRU'])
                if active_count >= ngru_count:
                    # No need to renormalize subset - use full normalization
                    self._ensure_gru_normalization()
                    return

                gru = ds['GRU'].values  # Shape: (subbasin, NGRU)

                # Check if first active_count columns already sum to 1.0
                active_sums = gru[:, :active_count].sum(axis=1)
                if np.allclose(active_sums, 1.0, atol=1e-4):
                    self.logger.debug(
                        f"First {active_count} GRU fractions already sum to 1.0"
                    )
                    return

                self.logger.info(
                    f"Renormalizing first {active_count} GRU fractions to sum to 1.0 "
                    f"(current sum: {active_sums[0]:.4f})"
                )

                # Renormalize only the first active_count columns
                for i in range(gru.shape[0]):  # For each subbasin
                    row_sum = gru[i, :active_count].sum()
                    if row_sum > 0:
                        gru[i, :active_count] = gru[i, :active_count] / row_sum
                    else:
                        # If sum is 0, set first GRU to 1.0
                        gru[i, 0] = 1.0

                    # Set remaining columns to 0 (they're ignored by MESH anyway)
                    gru[i, active_count:] = 0.0

                ds['GRU'].values = gru

                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds.to_netcdf(temp_path)
                os.replace(temp_path, self.ddb_path)

                self.logger.debug(f"Renormalized GRU fractions: {gru}")

        except Exception as e:
            self.logger.warning(f"Failed to renormalize MESH active GRUs: {e}")

    def _ensure_gru_normalization(self) -> None:
        """Ensure GRU fractions in DDB sum to 1.0 for every subbasin."""
        if not self.ddb_path.exists():
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return

                # Calculate current sums
                gru = ds['GRU']
                self.logger.debug(f"GRU values before norm: {gru.values}")
                n_dim = self._get_spatial_dim(ds)
                if not n_dim: return

                sums = gru.sum('NGRU')
                self.logger.debug(f"GRU sums: {sums.values}")

                # Identify where sum is not 1.0 (with small tolerance)
                if np.allclose(sums.values, 1.0, atol=1e-4):
                    self.logger.debug("GRU fractions already normalized")
                    return

                self.logger.info("Normalizing GRU fractions in DDB to sum to 1.0")
                # Avoid division by zero
                safe_sums = xr.where(sums == 0, 1.0, sums)
                # If sum was 0, set the first GRU to 1.0 as fallback
                ds['GRU'] = gru / safe_sums

                zero_sum_mask = (sums == 0)
                if zero_sum_mask.any():
                    self.logger.warning(f"Found {int(zero_sum_mask.sum())} subbasins with 0 GRU coverage. Setting first GRU to 1.0.")
                    # Workaround for xarray assignment on slice
                    gru_vals = ds['GRU'].values
                    # n_dim index is the first dimension
                    zero_indices = np.where(zero_sum_mask.values)[0]
                    for idx in zero_indices:
                        gru_vals[idx, 0] = 1.0
                    ds['GRU'].values = gru_vals

                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds.to_netcdf(temp_path)
                os.replace(temp_path, self.ddb_path)

        except Exception as e:
            self.logger.warning(f"Failed to normalize GRUs: {e}")

    def _get_spatial_dim(self, ds: xr.Dataset) -> Optional[str]:
        """Get the spatial dimension name from dataset."""
        if 'N' in ds.dims:
            return 'N'
        elif 'subbasin' in ds.dims:
            return 'subbasin'
        return None

    def _get_mesh_active_gru_count(self) -> Optional[int]:
        """Determine the number of GRUs that MESH will actually read.

        MESH has an off-by-one issue: it reads NGRU-1 GRUs from the drainage database.
        This function returns the count that MESH will actually see.
        """
        if not self.ddb_path.exists():
            return None

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return None

                ngru_dim = int(ds.sizes['NGRU'])

                # MESH has an off-by-one issue: it reads NGRU-1 GRUs.
                # Return what MESH will actually see.
                mesh_gru_count = max(1, ngru_dim - 1) if ngru_dim > 1 else ngru_dim

                self.logger.debug(f"MESH will read {mesh_gru_count} GRU(s) (NGRU dimension = {ngru_dim})")
                return mesh_gru_count

        except Exception as e:
            self.logger.debug(f"Could not determine MESH active GRU count: {e}")
            return None

    def _get_class_block_count(self) -> Optional[int]:
        """Get the number of CLASS parameter blocks."""
        if not self.class_file_path.exists():
            return None

        try:
            with open(self.class_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            block_count = sum(1 for line in lines if 'XSLP/XDRAINH/MANN/KSAT/MID' in line or line.startswith('[GRU_'))
            return block_count if block_count > 0 else None
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return None

    def _trim_class_to_count(self, target_count: int) -> None:
        """Trim CLASS parameter blocks to a specific count, preserving footer."""
        if not self.class_file_path.exists():
            return

        try:
            with open(self.class_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Try both ini-style [GRU_x] and legacy style
            ini_blocks = [i for i, line in enumerate(lines) if line.startswith('[GRU_')]
            legacy_blocks = [i for i, line in enumerate(lines) if '05 5xFCAN/4xLAMX' in line]

            # Find footer lines (lines 20, 21, 22) - they start with 0 and contain "20", "21", "22"
            footer_start = None
            for i, line in enumerate(lines):
                if '20 ' in line or '20\t' in line or line.strip().endswith('20'):
                    footer_start = i
                    break

            footer = []
            if footer_start is not None:
                footer = lines[footer_start:]
                lines = lines[:footer_start]

            if ini_blocks:
                header = lines[:ini_blocks[0]]
                block_starts = ini_blocks + [len(lines)]
                blocks = [lines[block_starts[i]:block_starts[i + 1]] for i in range(len(block_starts) - 1)]
            elif legacy_blocks:
                # Filter legacy_blocks to only those before footer
                legacy_blocks = [i for i in legacy_blocks if i < len(lines)]
                if not legacy_blocks:
                    return
                header = lines[:legacy_blocks[0]]
                block_starts = legacy_blocks + [len(lines)]
                blocks = [lines[block_starts[i]:block_starts[i + 1]] for i in range(len(block_starts) - 1)]
            else:
                return

            # Keep only the first target_count blocks
            kept_blocks = blocks[:target_count]

            if len(kept_blocks) != len(blocks):
                new_lines = header + [line for block in kept_blocks for line in block] + footer
                content = '\n'.join(new_lines)
                if not content.endswith('\n'):
                    content += '\n'
                with open(self.class_file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"Trimmed CLASS parameters to {len(kept_blocks)} GRU block(s)")
        except Exception as e:
            self.logger.warning(f"Failed to trim CLASS to count {target_count}: {e}")

    def _trim_empty_gru_columns(self) -> Optional[list]:
        """Trim empty GRU columns from drainage database."""
        if not self.ddb_path.exists():
            return None

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return None

                gru = ds['GRU']
                sum_dim = 'N' if 'N' in gru.dims else 'subbasin' if 'subbasin' in gru.dims else None
                if not sum_dim:
                    return None

                sums = gru.sum(sum_dim)
                min_total = float(self._get_config_value(lambda: self.config.model.mesh.gru_min_total, default=0.02, dict_key='MESH_GRU_MIN_TOTAL'))
                keep = sums > min_total
                keep_mask = keep.values.tolist()

                if int(keep.sum()) < int(gru.sizes['NGRU']):
                    removed = int(gru.sizes['NGRU'] - keep.sum())
                    ds_trim = ds.isel(NGRU=keep)

                    try:
                        sum_per = ds_trim['GRU'].sum('NGRU')
                        sum_safe = xr.where(sum_per == 0, 1.0, sum_per)
                        ds_trim['GRU'] = ds_trim['GRU'] / sum_safe
                    except Exception as e:
                        self.logger.debug(f"Could not renormalize GRU fractions after trim: {e}")

                    temp_path = self.ddb_path.with_suffix('.tmp.nc')
                    ds_trim.to_netcdf(temp_path)
                    os.replace(temp_path, self.ddb_path)
                    self.logger.info(f"Removed {removed} empty GRU column(s)")

                return keep_mask

        except Exception as e:
            self.logger.warning(f"Failed to trim empty GRU columns: {e}")
            return None

    def _fix_class_nm(self, keep_mask: Optional[list]) -> None:
        """Fix CLASS NM parameter to match block count."""
        if not self.class_file_path.exists():
            return

        try:
            with open(self.class_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            block_count = sum(1 for line in lines if 'XSLP/XDRAINH/MANN/KSAT/MID' in line or line.startswith('[GRU_'))

            # Read current NM
            nm_from_class = self._read_nm_from_lines(lines)

            trimmed_class = False
            if keep_mask is not None:
                trimmed_class = self._trim_class_blocks(lines, keep_mask)
                if trimmed_class:
                    with open(self.class_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                    block_count = sum(1 for line in lines if 'XSLP/XDRAINH/MANN/KSAT/MID' in line or line.startswith('[GRU_'))
                    nm_from_class = self._read_nm_from_lines(lines)

            if nm_from_class != block_count:
                self.logger.warning(f"CLASS NM ({nm_from_class}) != block count ({block_count})")
                self._update_class_nm(block_count)
            else:
                self.logger.debug(f"CLASS NM={nm_from_class} matches {block_count} blocks")

            self._ensure_gru_normalization()
            return
        except Exception as e:
            self.logger.warning(f"Failed to fix GRU count mismatch: {e}")

    def _trim_class_blocks(self, lines: list, keep_mask: list) -> bool:
        """Trim CLASS parameter blocks to match DDB GRU columns."""
        # Try both ini-style [GRU_x] and legacy style
        ini_blocks = [i for i, line in enumerate(lines) if line.startswith('[GRU_')]
        legacy_blocks = [i for i, line in enumerate(lines) if '05 5xFCAN/4xLAMX' in line]

        if ini_blocks:
            header = lines[:ini_blocks[0]]
            block_starts = ini_blocks + [len(lines)]
            blocks = [lines[block_starts[i]:block_starts[i + 1]] for i in range(len(block_starts) - 1)]
        elif legacy_blocks:
            header = lines[:legacy_blocks[0]]
            block_starts = legacy_blocks + [len(lines)]
            blocks = [lines[block_starts[i]:block_starts[i + 1]] for i in range(len(block_starts) - 1)]
        else:
            return False

        max_blocks = min(len(blocks), len(keep_mask))
        kept_blocks = [blocks[i] for i in range(max_blocks) if keep_mask[i]]

        if len(kept_blocks) != len(blocks):
            new_lines = header + [line for block in kept_blocks for line in block]
            content = '\n'.join(new_lines)
            if not content.endswith('\n'):
                content += '\n'
            with open(self.class_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.logger.info(f"Trimmed CLASS parameters to {len(kept_blocks)} GRU block(s)")
            return True

        return False

    def _read_nm_from_lines(self, lines: list) -> Optional[int]:
        """Read NM value from CLASS file lines."""
        for line in lines:
            if '04 DEGLAT' in line or 'NL/NM' in line or line.startswith('NM '):
                parts = line.split()
                if line.startswith('NM '):
                    try:
                        return int(parts[1])
                    except (ValueError, IndexError) as e:
                        self.logger.debug(f"Could not parse NM from '{line.strip()}': {e}")
                else:
                    if len(parts) >= 9:
                        try:
                            return int(parts[8])
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"Could not parse NM from column 9 of '{line.strip()}': {e}")
                break
        return None

    def _update_class_nm(self, new_nm: int) -> None:
        """Update NM in CLASS parameters file."""
        try:
            with open(self.class_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            modified = False
            for i, line in enumerate(lines):
                # Handle NM x style
                if line.startswith('NM '):
                    parts = line.split()
                    old_nm = parts[1]
                    lines[i] = f"NM {new_nm}    ! number of landcover classes (GRUs)\n"
                    modified = True
                    self.logger.info(f"Updated CLASS NM from {old_nm} to {new_nm}")
                    break

                # Handle legacy style
                if '04 DEGLAT' in line or 'NL/NM' in line:
                    parts = line.split()
                    if len(parts) >= 9:
                        old_nm = parts[8]
                        tokens = re.split(r'(\s+)', line)
                        value_count = 0
                        for j, tok in enumerate(tokens):
                            if tok.strip():
                                value_count += 1
                                if value_count == 9:
                                    tokens[j] = str(new_nm)
                                    break
                        lines[i] = ''.join(tokens)
                        modified = True
                        self.logger.info(f"Updated CLASS NM from {old_nm} to {new_nm}")
                    break

            if modified:
                with open(self.class_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

        except Exception as e:
            self.logger.warning(f"Failed to update CLASS NM: {e}")

    def fix_hydrology_wf_r2(self) -> None:
        """Ensure WF_R2 is in the hydrology file.

        Note: WF_R2 (WATFLOOD channel roughness) is DIFFERENT from R2N (overland Manning's n).
        - R2N: Manning's n for overland flow, typically 0.02-0.10
        - WF_R2: Channel roughness coefficient for WATFLOOD routing, typically 0.20-0.40

        This method checks for a configured MESH_WF_R2 value (default 0.30).
        """
        settings_hydro = self.setup_dir / "MESH_parameters_hydrology.ini"

        # Copy from settings if missing or empty
        if not self.hydro_path.exists() or self.hydro_path.stat().st_size == 0:
            if settings_hydro.exists() and settings_hydro.stat().st_size > 0:
                import shutil
                shutil.copy2(settings_hydro, self.hydro_path)
                self.logger.info("Copied hydrology file from settings")
            else:
                self.logger.warning("No valid hydrology file found")
                return

        try:
            with open(self.hydro_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            if not content.strip() and settings_hydro.exists():
                with open(settings_hydro, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                if content.strip():
                    with open(self.hydro_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.logger.info("Restored hydrology file from settings")

            # Get configured WF_R2 value
            # Try config key 'MESH_WF_R2' first, then look in hydrology params dict
            configured_wf_r2 = self._get_config_value('MESH_WF_R2', None)

            if configured_wf_r2 is None:
                # Try to dig into the nested config structure if available
                try:
                    # Accessing via loose dict structure from config
                    if hasattr(self, 'config') and isinstance(self.config, dict):
                         configured_wf_r2 = self.config.get('hydrology_params', {}).get('routing', [{}])[0].get('wf_r2')
                except Exception:
                    pass

            # Default to 0.30 if not configured
            default_wf_r2 = 0.30
            target_wf_r2 = float(configured_wf_r2) if configured_wf_r2 is not None else default_wf_r2

            if 'WF_R2' in content:
                # If it exists, we might want to update it if the user explicitly configured it
                if configured_wf_r2 is not None:
                     self.logger.debug(f"WF_R2 already present, but updating to configured value {target_wf_r2}")
                     # Regex replace to update existing value
                     # Pattern looks for WF_R2 followed by numbers
                     pattern = r'(WF_R2\s+)([\d\.\s]+)(.*)'

                     def replace_wf_r2(match):
                         prefix = match.group(1)
                         suffix = match.group(3)
                         # Count how many values we need (rough heuristic based on existing line)
                         existing_values = match.group(2).split()
                         n_vals = len(existing_values)
                         new_vals = [f"{target_wf_r2:.4f}"] * n_vals
                         return f"{prefix}{'    '.join(new_vals)}{suffix}"

                     content = re.sub(pattern, replace_wf_r2, content)
                     with open(self.hydro_path, 'w', encoding='utf-8') as f:
                         f.write(content)
                     return

                self.logger.debug("WF_R2 already present")
                return

            # Add WF_R2
            new_lines = []
            r2n_found = False
            for line in lines:
                if line.startswith('R2N') and not r2n_found:
                    parts = line.split()
                    if len(parts) >= 2:
                        n_values = len(parts) - 1  # Number of routing classes

                        wf_r2_values = [f"{target_wf_r2:.4f}"] * n_values
                        wf_r2_line = "WF_R2  " + "    ".join(wf_r2_values) + "  # channel roughness (calibratable)"
                        new_lines.append(wf_r2_line)
                        r2n_found = True
                        self.logger.info(f"Added WF_R2={target_wf_r2} for {n_values} routing class(es)")

                        # Update parameter count
                        for j in range(len(new_lines) - 1, -1, -1):
                            if "Number of channel routing parameters" in new_lines[j]:
                                match = re.match(r'\s*(\d+)', new_lines[j])
                                if match:
                                    old_count = int(match.group(1))
                                    new_count = old_count + 1
                                    new_lines[j] = new_lines[j].replace(
                                        str(old_count), str(new_count), 1
                                    )
                                break

                new_lines.append(line)

            if r2n_found:
                with open(self.hydro_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))

        except Exception as e:
            self.logger.warning(f"Failed to add WF_R2: {e}")

    def fix_missing_hydrology_params(self) -> None:
        """Verify and pre-populate hydrology parameters for MESH.

        Ensures that:
        1. Standard routing parameters (R2N, R1N, PWR, FLZ) exist.
        2. Calibratable legacy parameters (RCHARG, FRZTH) are present with
           physically reasonable defaults so the optimizer doesn't start from
           arbitrary mid-range values injected at first iteration.
        """
        if not self.hydro_path.exists():
            self.logger.warning("Hydrology file not found, skipping parameter verification")
            return

        is_noroute = False
        if self.run_options_path.exists():
            with open(self.run_options_path, 'r', encoding='utf-8') as f:
                run_options_content = f.read()
            if re.search(r'RUNMODE\s+noroute', run_options_content):
                is_noroute = True

        try:
            with open(self.hydro_path, 'r', encoding='utf-8') as f:
                content = f.read()

            modified = False

            # Verify standard routing parameters (only relevant when routing)
            if not is_noroute:
                required_params = ['R2N', 'R1N', 'PWR', 'FLZ']
                missing = [p for p in required_params if p not in content]
                if missing:
                    self.logger.warning(f"Missing routing parameters in hydrology file: {missing}")
                else:
                    self.logger.debug("All standard routing parameters present (R2N, R1N, PWR, FLZ)")

            # Pre-populate calibratable parameters with physically reasonable defaults.
            # This prevents the optimizer from auto-injecting at mid-range on first
            # iteration, which wastes early DDS exploration budget.
            # Note: RCHARG and FRZTH are appended after the GRU-dependent section.
            # In noroute mode (single-cell), these are not read by MESH's positional
            # parser but are present for the parameter_manager's regex-based updates.
            # They become functional only in multi-cell run_def/runrte configurations
            # where MESH reads them by keyword after the positional sections.
            calibratable_defaults = {
                'RCHARG': (0.20, 'Recharge fraction to groundwater (typical 0.1-0.3)'),
                'FRZTH': (0.10, 'Frozen soil infiltration threshold (m)'),
            }

            for param_name, (default_val, description) in calibratable_defaults.items():
                # Use word boundary to avoid matching partial names
                if not re.search(rf'\b{param_name}\b', content):
                    if not content.endswith('\n'):
                        content += '\n'
                    content += f"{param_name}  {default_val:.6f}  # {description}\n"
                    modified = True
                    self.logger.info(
                        f"Pre-populated {param_name}={default_val} in {self.hydro_path.name}"
                    )

            if modified:
                with open(self.hydro_path, 'w', encoding='utf-8') as f:
                    f.write(content)

        except Exception as e:
            self.logger.warning(f"Failed to verify hydrology parameters: {e}")

    def fix_run_options_output_dirs(self) -> None:
        """Fix output directory paths in run options file."""
        run_options = self.forcing_dir / "MESH_input_run_options.ini"
        if not run_options.exists():
            return

        try:
            with open(run_options, 'r', encoding='utf-8') as f:
                content = f.read()

            # Replace CLASSOUT with ./ for MESH 1.5 compatibility
            if 'CLASSOUT' in content:
                content = content.replace('CLASSOUT', './' + ' ' * 6)  # Pad to maintain alignment
                with open(run_options, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info("Fixed output directory paths in run options")

        except Exception as e:
            self.logger.warning(f"Failed to fix run options output dirs: {e}")

    def fix_reservoir_file(self) -> None:
        """Fix reservoir input file to match IREACH in drainage database.

        MESH requires the reservoir file to match max(IREACH) from the drainage database.
        When IREACH = 0 (no reservoirs), the reservoir file should have 0 reservoirs.

        The reservoir file format is:
        - Line 1: <n_reservoirs> <n_columns> <header_flag>
        - Lines 2+: reservoir data (if n_reservoirs > 0)
        """
        reservoir_file = self.forcing_dir / "MESH_input_reservoir.txt"

        # Get max IREACH from drainage database
        max_ireach = 0
        if self.ddb_path.exists():
            try:
                with xr.open_dataset(self.ddb_path) as ds:
                    if 'IREACH' in ds:
                        ireach_vals = ds['IREACH'].values
                        # Handle any remaining fill values
                        valid_vals = ireach_vals[ireach_vals >= 0]
                        if len(valid_vals) > 0:
                            max_ireach = int(np.max(valid_vals))
                        self.logger.debug(f"Max IREACH from DDB: {max_ireach}")
            except Exception as e:
                self.logger.debug(f"Could not read IREACH from DDB: {e}")

        # Create or fix reservoir file
        try:
            if max_ireach == 0:
                # No reservoirs - create simple file
                with open(reservoir_file, 'w', encoding='utf-8') as f:
                    f.write("0\n")  # Just the count, no other columns needed
                self.logger.info("Fixed reservoir file: 0 reservoirs")
            else:
                # Reservoirs exist - ensure file has correct count
                if reservoir_file.exists():
                    with open(reservoir_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    first_line = content.split('\n')[0] if content else ""
                    parts = first_line.split()
                    file_count = int(parts[0]) if parts else 0
                    if file_count != max_ireach:
                        self.logger.warning(
                            f"Reservoir file count ({file_count}) != IREACH max ({max_ireach}). "
                            f"Reservoir routing may fail."
                        )
                else:
                    self.logger.warning(
                        f"Reservoir file not found but IREACH max = {max_ireach}. "
                        f"Creating placeholder with 0 reservoirs."
                    )
                    with open(reservoir_file, 'w', encoding='utf-8') as f:
                        f.write("0\n")

        except Exception as e:
            self.logger.warning(f"Failed to fix reservoir file: {e}")

    def configure_lumped_outputs(self) -> None:
        """Configure outputs_balance.txt for lumped mode calibration.

        In lumped (noroute) mode, we need daily runoff output for calibration.
        This method enables daily CSV output of total runoff (RFF).
        """
        outputs_balance = self.forcing_dir / "outputs_balance.txt"
        if not outputs_balance.exists():
            return

        # Check if we're in lumped mode
        num_cells = self._get_num_cells()
        if num_cells > 1:
            return  # Not lumped mode

        try:
            with open(outputs_balance, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            modified = False
            new_lines = []

            for line in lines:
                stripped = line.strip()

                # Enable daily CSV output for runoff (RFF)
                # Change "!RFF D csv" to "RFF D csv" (uncomment)
                # Or add if not present
                if stripped.startswith('!RFF') and ('D' in stripped or 'H' in stripped) and 'csv' in stripped.lower():
                    # Uncomment and ensure daily
                    new_line = stripped[1:].replace(' H ', ' D ')  # Remove ! and change H to D
                    new_lines.append(new_line + '\n')
                    modified = True
                    continue

                new_lines.append(line)

            # If no RFF csv line found, add one
            if not any('RFF' in l and 'csv' in l.lower() and not l.strip().startswith('!')
                      for l in new_lines):
                new_lines.append('RFF     D  csv\n')
                modified = True

            if modified:
                with open(outputs_balance, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                self.logger.info("Configured outputs_balance.txt for lumped mode (daily RFF csv)")

        except Exception as e:
            self.logger.warning(f"Failed to configure lumped outputs: {e}")

    def fix_class_vegetation_parameters(self) -> None:
        """Fix CLASS vegetation parameters for different GRU types.

        The default meshflow template uses the same vegetation parameters for all
        GRU types, which can cause numerical instability in CLASS energy balance
        calculations. This method adjusts key parameters based on the vegetation
        class (from the MID comment in line 13).

        Key parameters adjusted:
        - LNZ0: Log of roughness length - different heights require different z0
        - RSMN: Minimum stomatal resistance - varies by vegetation type

        Reference roughness lengths (z0) by vegetation type:
        - Needleleaf forest: z0 ~ 0.5-1.5m  (LNZ0 ~ -0.7 to 0.4)
        - Broadleaf forest:  z0 ~ 0.5-2.0m  (LNZ0 ~ -0.7 to 0.7)
        - Crops:             z0 ~ 0.05-0.15m (LNZ0 ~ -3.0 to -1.9)
        - Grass:             z0 ~ 0.02-0.08m (LNZ0 ~ -3.9 to -2.5)
        - Barrenland:        z0 ~ 0.001-0.01m (LNZ0 ~ -6.9 to -4.6)
        """
        if not self.class_file_path.exists():
            return

        # Vegetation-specific parameter corrections
        # Format: {keyword: (lnz0, rsmn)}
        # lnz0 is log(z0) where z0 is roughness length in meters
        veg_corrections = {
            'needle': (-0.7, 145),     # Needleleaf forest: z0=0.5m
            'need_fore': (-0.7, 145),
            'broad': (-0.4, 150),      # Broadleaf forest: z0=0.67m
            'shru': (-1.8, 200),       # Shrub: z0=0.17m
            'grass': (-2.5, 200),      # Grassland: z0=0.08m
            'gras': (-2.5, 200),
            'crop': (-2.3, 120),       # Crops: z0=0.10m
            'Crop': (-2.3, 120),
            'barren': (-4.6, 500),     # Barrenland: z0=0.01m
            'urban': (-2.0, 200),      # Urban: z0=0.14m
        }

        try:
            with open(self.class_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # First pass: identify GRU types from MID comments (line 13)
            gru_types = []
            for i, line in enumerate(lines):
                if 'XSLP/XDRAINH/MANN/KSAT/MID' in line:
                    gru_type = None
                    line_lower = line.lower()
                    for veg_key in veg_corrections:
                        if veg_key.lower() in line_lower:
                            gru_type = veg_key
                            break
                    gru_types.append((i, gru_type))

            if not gru_types:
                self.logger.debug("No GRU blocks found in CLASS file")
                return

            self.logger.debug(f"Found {len(gru_types)} GRU blocks: {gru_types}")

            # Second pass: apply corrections to each GRU block
            # GRU blocks are separated by blank lines, with lines 5-13 in sequence
            modified = False
            new_lines = []

            # Find block boundaries (line 13 to next line 5 or end)
            block_starts = []
            for i, line in enumerate(lines):
                if '05 5xFCAN/4xLAMX' in line:
                    block_starts.append(i)

            if len(block_starts) != len(gru_types):
                self.logger.warning(
                    f"Mismatch: {len(block_starts)} block starts vs {len(gru_types)} MID comments"
                )

            # Process each block
            current_block = 0
            for i, line in enumerate(lines):
                # Determine which block we're in
                while current_block < len(block_starts) - 1 and i >= block_starts[current_block + 1]:
                    current_block += 1

                if current_block < len(gru_types):
                    _, gru_type = gru_types[current_block]
                else:
                    gru_type = None

                # Fix LNZ0 in line 6 (5xLNZ0/4xLAMN)
                if '5xLNZ0/4xLAMN' in line and gru_type:
                    if gru_type in veg_corrections:
                        lnz0_target = veg_corrections[gru_type][0]
                        # Replace -1.300 with the target value
                        line = line.replace('-1.300', f'{lnz0_target:.3f}')
                        modified = True
                        self.logger.debug(f"Set LNZ0={lnz0_target:.3f} for {gru_type}")

                # Fix RSMN in line 9 (4xRSMN/4xQA50)
                if '4xRSMN/4xQA50' in line and gru_type:
                    if gru_type in veg_corrections:
                        rsmn_target = veg_corrections[gru_type][1]
                        # Replace 145.000 with the target value
                        line = line.replace('145.000', f'{rsmn_target:.3f}')
                        modified = True
                        self.logger.debug(f"Set RSMN={rsmn_target:.3f} for {gru_type}")

                new_lines.append(line)

            if modified:
                with open(self.class_file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))
                self.logger.info("Fixed CLASS vegetation parameters (LNZ0, RSMN) for different GRU types")

        except Exception as e:
            self.logger.warning(f"Failed to fix CLASS vegetation parameters: {e}")

    def _get_domain_latitude(self) -> Optional[float]:
        """Get representative latitude from drainage database for climate classification."""
        ddb_path = self.forcing_dir / "MESH_drainage_database.nc"
        if not ddb_path.exists():
            return None
        try:
            import xarray as xr
            with xr.open_dataset(ddb_path) as ds:
                if 'lat' in ds:
                    return float(ds['lat'].values.mean())
        except (OSError, ValueError, KeyError) as e:
            self.logger.debug(f"Could not read latitude from drainage database: {e}")
        return None

    def _get_climate_adjusted_snow_params(self, start_month: int, latitude: Optional[float]) -> dict:
        """
        Get snow initial conditions adjusted for climate zone and season.

        Climate zones (by latitude):
        - Temperate: lat < 50° - minimal snow, mild winters
        - Boreal: 50° <= lat < 60° - moderate snow, cold winters
        - Arctic/Alpine: lat >= 60° - heavy snow, very cold winters

        Args:
            start_month: Month of simulation start (1-12)
            latitude: Domain latitude in degrees (None uses temperate defaults)

        Returns:
            Dictionary with SNO, ALBS, RHOS, TSNO, TCAN initial values
        """
        is_winter = start_month in [11, 12, 1, 2, 3, 4]

        # Determine climate zone
        if latitude is None:
            climate = 'temperate'  # Conservative default
        elif abs(latitude) >= 60:
            climate = 'arctic'
        elif abs(latitude) >= 50:
            climate = 'boreal'
        else:
            climate = 'temperate'

        # Snow initial conditions by climate and season
        # Values based on CLASS literature and regional climatology
        params = {
            'arctic': {
                'winter': {'sno': 150.0, 'albs': 0.80, 'rhos': 200.0, 'tsno': -20.0, 'tcan': -15.0},
                'summer': {'sno': 50.0, 'albs': 0.70, 'rhos': 300.0, 'tsno': -5.0, 'tcan': 0.0},
            },
            'boreal': {
                'winter': {'sno': 100.0, 'albs': 0.75, 'rhos': 250.0, 'tsno': -10.0, 'tcan': -5.0},
                'summer': {'sno': 10.0, 'albs': 0.60, 'rhos': 350.0, 'tsno': -1.0, 'tcan': 5.0},
            },
            'temperate': {
                'winter': {'sno': 50.0, 'albs': 0.70, 'rhos': 300.0, 'tsno': -5.0, 'tcan': 0.0},
                'summer': {'sno': 0.0, 'albs': 0.50, 'rhos': 400.0, 'tsno': 0.0, 'tcan': 10.0},
            },
        }

        season = 'winter' if is_winter else 'summer'
        return params[climate][season]

    def fix_class_initial_conditions(self) -> None:
        """Fix CLASS initial conditions for proper snow simulation.

        Uses climate-aware defaults based on domain latitude and simulation start month.
        """
        if not self.class_file_path.exists():
            return

        try:
            with open(self.class_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Determine start month and latitude for climate classification
            time_window = self.get_simulation_time_window() if self.get_simulation_time_window else None
            start_month = time_window[0].month if time_window else 1

            latitude = self._get_domain_latitude()
            snow_params = self._get_climate_adjusted_snow_params(start_month, latitude)

            initial_sno = snow_params['sno']
            initial_albs = snow_params['albs']
            initial_rhos = snow_params['rhos']
            initial_tsno = snow_params['tsno']
            initial_tcan = snow_params['tcan']

            climate_zone = 'arctic' if latitude and abs(latitude) >= 60 else \
                           'boreal' if latitude and abs(latitude) >= 50 else 'temperate'
            self.logger.info(f"Using {climate_zone} snow defaults (lat={latitude:.1f}°)" if latitude else
                             "Using temperate snow defaults (latitude unknown)")

            modified = False
            new_lines = []

            for line in lines:
                # Fix line 17: TBAR/TCAN/TSNO/TPND
                if '17 3xTBAR' in line or ('17' in line and 'TBAR' in line):
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            tbar1 = float(parts[0])
                            tbar2 = float(parts[1])
                            tbar3 = float(parts[2])
                            tpnd = float(parts[5])
                            new_line = (
                                f"  {tbar1:.3f}  {tbar2:.3f}  {tbar3:.3f}  "
                                f"{initial_tcan:.3f}  {initial_tsno:.3f}   {tpnd:.3f}  "
                                f"17 3xTBAR (or more)/TCAN/TSNO/TPND\n"
                            )
                            new_lines.append(new_line)
                            modified = True
                            continue
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"Could not parse TBAR values from '{line.strip()}': {e}")

                # Fix line 19: RCAN/SCAN/SNO/ALBS/RHOS/GRO
                if '19 RCAN/SCAN/SNO/ALBS/RHOS/GRO' in line:
                    parts = line.split()
                    if len(parts) >= 8:
                        try:
                            rcan = float(parts[0])
                            scan = float(parts[1])
                            gro = float(parts[5])
                            new_line = (
                                f"   {rcan:.3f}   {scan:.3f}   {initial_sno:.1f}   "
                                f"{initial_albs:.2f}   {initial_rhos:.1f}   {gro:.3f}  "
                                f"19 RCAN/SCAN/SNO/ALBS/RHOS/GRO\n"
                            )
                            new_lines.append(new_line)
                            modified = True
                            continue
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"Could not parse RCAN/SCAN/SNO values from '{line.strip()}': {e}")

                new_lines.append(line)

            if modified:
                with open(self.class_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                self.logger.info(
                    f"Fixed CLASS initial conditions: SNO={initial_sno}mm, "
                    f"ALBS={initial_albs}, RHOS={initial_rhos}kg/m³"
                )

        except Exception as e:
            self.logger.warning(f"Failed to fix CLASS initial conditions: {e}")

    def create_safe_forcing(self) -> None:
        """Create a trimmed forcing file for the simulation period."""
        forcing_nc = self.forcing_dir / "MESH_forcing.nc"
        safe_forcing_nc = self.forcing_dir / "MESH_forcing_safe.nc"

        if not forcing_nc.exists():
            self.logger.warning("No MESH_forcing.nc found")
            return

        try:
            time_window = self._get_time_window()
            if not time_window:
                self.logger.warning("No simulation time window configured")
                return

            analysis_start, end_time = time_window

            # Get forcing data range
            with xr.open_dataset(forcing_nc) as ds_check:
                forcing_times = pd.to_datetime(ds_check['time'].values)
                forcing_start = forcing_times[0]
                forcing_end = forcing_times[-1]

            # Calculate spinup
            # Use configured value or dynamic default based on climate
            configured_spinup = self._get_config_value(lambda: self.config.model.mesh.spinup_days, default=None, dict_key='MESH_SPINUP_DAYS')

            if configured_spinup is not None:
                spinup_days = int(configured_spinup)
            else:
                # Get domain latitude/elevation for smart default
                latitude = self._get_domain_latitude()
                # Elevation not easily available here without reading more files, so pass None
                spinup_days = MESHConfigDefaults.get_recommended_spinup_days(latitude=latitude)
                self.logger.info(f"Using recommended spinup of {spinup_days} days (based on lat={latitude})")

            from datetime import timedelta
            requested_start = pd.Timestamp(analysis_start - timedelta(days=spinup_days))

            if requested_start < forcing_start:
                actual_spinup_days = (analysis_start - forcing_start).days
                start_time = pd.Timestamp(forcing_start)
                self.logger.warning(
                    f"Limiting spinup to {actual_spinup_days} days"
                )
                self._actual_spinup_days = actual_spinup_days
            else:
                start_time = requested_start
                self._actual_spinup_days = spinup_days

            end_time = pd.Timestamp(end_time)
            if end_time > forcing_end:
                end_time = forcing_end

            end_time_padded = min(end_time + timedelta(days=2), forcing_end)

            # Subset and save using netCDF4 directly for better compatibility
            with xr.open_dataset(forcing_nc) as ds:
                if 'time' not in ds.dims:
                    return

                times = pd.to_datetime(ds['time'].values)
                start_idx, end_idx = 0, len(times)

                for i, t in enumerate(times):
                    if t >= start_time:
                        start_idx = max(0, i - 1)
                        break

                for i, t in enumerate(times):
                    if t > end_time_padded:
                        end_idx = i
                        break

                ds_safe = ds.isel(time=slice(start_idx, end_idx))
                n_timesteps = end_idx - start_idx

                self.logger.info(f"Creating MESH_forcing_safe.nc with {n_timesteps} timesteps")

                # Use netCDF4 directly to ensure proper encoding
                from netCDF4 import Dataset as NC4Dataset
                n_spatial = ds_safe.sizes.get('subbasin', 1)

                with NC4Dataset(safe_forcing_nc, 'w', format='NETCDF4') as ncfile:
                    # Create dimensions
                    ncfile.createDimension('time', None)  # unlimited
                    ncfile.createDimension('subbasin', n_spatial)

                    # Create coordinate variables
                    var_time = ncfile.createVariable('time', 'f8', ('time',))
                    var_time.standard_name = 'time'
                    var_time.long_name = 'time'
                    var_time.axis = 'T'
                    # Use same format as original forcing (MESH_forcing.nc)
                    var_time.units = 'hours since 1900-01-01 00:00:00'
                    var_time.calendar = 'gregorian'  # Match original forcing file

                    # Convert time to hours since 1900-01-01 (same as MESH_forcing.nc)
                    reference = pd.Timestamp('1900-01-01')
                    time_hours = np.array([
                        (pd.Timestamp(t) - reference).total_seconds() / 3600.0
                        for t in ds_safe['time'].values
                    ])
                    var_time[:] = time_hours

                    var_n = ncfile.createVariable('subbasin', 'i4', ('subbasin',))
                    var_n[:] = np.arange(1, n_spatial + 1)

                    # Copy spatial coordinate variables if they exist
                    for coord_var in ['lat', 'lon']:
                        if coord_var in ds_safe:
                            var = ncfile.createVariable(coord_var, 'f8', ('subbasin',))
                            for attr in ds_safe[coord_var].attrs:
                                var.setncattr(attr, ds_safe[coord_var].attrs[attr])
                            var[:] = ds_safe[coord_var].values

                    # Copy CRS if it exists
                    if 'crs' in ds_safe:
                        var_crs = ncfile.createVariable('crs', 'i4')
                        for attr in ds_safe['crs'].attrs:
                            var_crs.setncattr(attr, ds_safe['crs'].attrs[attr])

                    # Copy forcing variables
                    forcing_vars = ['PRES', 'QA', 'TA', 'UV', 'PRE', 'FSIN', 'FLIN']
                    for var_name in forcing_vars:
                        if var_name in ds_safe:
                            var = ncfile.createVariable(
                                var_name, 'f4', ('time', 'subbasin'),
                                fill_value=-9999.0
                            )
                            # Copy attributes
                            for attr in ds_safe[var_name].attrs:
                                if attr != '_FillValue':
                                    var.setncattr(attr, ds_safe[var_name].attrs[attr])
                            var.missing_value = -9999.0

                            # Copy data
                            var[:] = ds_safe[var_name].values

                    # Copy global attributes
                    ncfile.author = "University of Calgary"
                    ncfile.license = "GNU General Public License v3 (or any later version)"
                    ncfile.purpose = "Create forcing .nc file for MESH"
                    ncfile.Conventions = "CF-1.6"
                    if 'history' in ds.attrs:
                        ncfile.history = ds.attrs['history']

            # Update run options
            self._update_run_options_for_safe_forcing(start_time, end_time)

        except Exception as e:
            import traceback
            self.logger.warning(f"Failed to create safe forcing: {e}")
            self.logger.debug(traceback.format_exc())

    def _get_time_window(self) -> Optional[Tuple]:
        """Get simulation time window from config or callback."""
        if self.get_simulation_time_window:
            time_window = self.get_simulation_time_window()
            if time_window:
                return time_window

        # Fallback to calibration/evaluation periods
        cal_period = self._get_config_value(lambda: self.config.domain.calibration_period, dict_key='CALIBRATION_PERIOD')
        eval_period = self._get_config_value(lambda: self.config.domain.evaluation_period, dict_key='EVALUATION_PERIOD')

        if cal_period:
            cal_parts = [p.strip() for p in str(cal_period).split(',')]
            if len(cal_parts) >= 2:
                analysis_start = pd.Timestamp(cal_parts[0])
                if eval_period:
                    eval_parts = [p.strip() for p in str(eval_period).split(',')]
                    end_time = pd.Timestamp(eval_parts[1] if len(eval_parts) >= 2 else eval_parts[0])
                else:
                    end_time = pd.Timestamp(cal_parts[1])
                return (analysis_start, end_time)

        return None

    def _update_run_options_for_safe_forcing(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> None:
        """Update run options for safe forcing file."""
        if not self.run_options_path.exists():
            return

        with open(self.run_options_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        modified = False
        new_lines = []

        for line in lines:
            if 'fname=MESH_forcing' in line and 'fname=MESH_forcing_safe' not in line:
                # Don't add .nc extension - MESH adds it automatically
                line = line.replace('fname=MESH_forcing', 'fname=MESH_forcing_safe')
                modified = True

            if 'start_date=' in line:
                new_start_date = start_time.strftime('%Y%m%d')
                line = re.sub(r'start_date=\d+', f'start_date={new_start_date}', line)
                modified = True

            if 'METRICSSPINUP' in line and self._actual_spinup_days:
                line = re.sub(
                    r'METRICSSPINUP\s+\d+',
                    f'METRICSSPINUP         {self._actual_spinup_days}',
                    line
                )
                modified = True

            new_lines.append(line)

        # Update simulation date lines
        date_line_indices = self._find_date_lines(new_lines)
        if len(date_line_indices) >= 2:
            start_idx = date_line_indices[-2]
            end_idx = date_line_indices[-1]
            new_lines[start_idx] = f"{start_time.year:04d} {start_time.dayofyear:03d}   1   0"
            new_lines[end_idx] = f"{end_time.year:04d} {end_time.dayofyear:03d}  23   0"
            modified = True
            self.logger.info(
                f"Updated simulation dates: {start_time.year:04d}/{start_time.dayofyear:03d} "
                f"to {end_time.year:04d}/{end_time.dayofyear:03d}"
            )

        if modified:
            with open(self.run_options_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))

    def _find_date_lines(self, lines: list) -> list:
        """Find lines that look like date specifications."""
        date_line_indices = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('-'):
                parts = stripped.split()
                if len(parts) >= 4 and parts[0].isdigit() and len(parts[0]) == 4:
                    try:
                        int(parts[0])
                        int(parts[1])
                        int(parts[2])
                        int(parts[3])
                        date_line_indices.append(i)
                    except ValueError as e:
                        self.logger.debug(f"Line does not match date format '{stripped}': {e}")
        return date_line_indices

    def create_elevation_band_class_blocks(self, elevation_info: list) -> bool:
        """Create CLASS parameter blocks for elevation bands.

        For elevation-band discretization, each elevation band becomes a GRU
        with similar vegetation parameters but elevation-adjusted initial conditions.

        Args:
            elevation_info: List of dicts with 'elevation' and 'fraction' for each band

        Returns:
            True if successful, False otherwise
        """
        if not self.class_file_path.exists():
            self.logger.warning("CLASS file not found, cannot create elevation band blocks")
            return False

        if not elevation_info:
            self.logger.warning("No elevation info provided")
            return False

        n_bands = len(elevation_info)
        self.logger.info(f"Creating {n_bands} CLASS blocks for elevation bands")

        try:
            with open(self.class_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # Find header lines (lines 01-04)
            header_lines = []
            block_start = None
            for i, line in enumerate(lines):
                if '04 DEGLAT/DEGLON' in line:
                    header_lines = lines[:i + 1]
                    block_start = i + 1
                    break

            if block_start is None:
                self.logger.warning("Could not find CLASS header (line 04)")
                return False

            # Find footer lines (lines 20-22)
            footer_lines = []
            footer_start = None
            for i, line in enumerate(lines):
                if i > block_start and ('20 ' in line or line.strip().endswith('20')):
                    footer_start = i
                    footer_lines = lines[i:]
                    break

            if footer_start is None:
                self.logger.warning("Could not find CLASS footer (line 20)")
                return False

            # Update NL and NM in header line 04
            # Format: DEGLAT DEGLON ZRFM ZRFH ZBLD GC ILW NL NM  04 comment
            # NL = number of grid cells (subbasins), NM = number of GRU types
            # For multi-subbasin elevation bands, NL must match the DDB subbasin count
            n_cells = self._get_num_cells()
            for i, line in enumerate(header_lines):
                if '04 DEGLAT/DEGLON' in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        # parts[7] = NL, parts[8] = NM, parts[9] = '04', parts[10] = comment
                        parts[7] = str(n_cells)
                        parts[8] = str(n_bands)
                        # Reconstruct with original spacing style
                        comment_idx = line.index('04 DEGLAT')
                        values = parts[:9]
                        header_lines[i] = '  ' + '  '.join(f'{v:>8s}' for v in values) + '       ' + line[comment_idx:]
                    break

            # Get base parameters from first existing GRU block
            # Find first GRU block (lines 05-19)
            block_lines = lines[block_start:footer_start]
            first_block = []
            in_block = False
            for line in block_lines:
                if '05 5xFCAN' in line or in_block:
                    in_block = True
                    first_block.append(line)
                    if '19 RCAN/SCAN' in line:
                        break

            if not first_block or len(first_block) < 15:
                # Use default block if not found
                first_block = self._get_default_class_block()

            # Create blocks for each elevation band
            new_blocks = []
            for i, elev_info in enumerate(elevation_info):
                elevation = elev_info['elevation']
                fraction = elev_info['fraction']

                # Create block with elevation-adjusted parameters
                block = self._create_elevation_adjusted_block(
                    first_block, i, elevation, fraction
                )
                new_blocks.extend(block)
                new_blocks.append('')  # Blank line between blocks

            # Assemble new content
            new_lines = header_lines + [''] + new_blocks + footer_lines

            # Write updated file
            with open(self.class_file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))

            elev_str = ', '.join([f"{e['elevation']:.0f}m" for e in elevation_info])
            self.logger.info(
                f"Created {n_bands} elevation band CLASS blocks (elevations: {elev_str})"
            )
            return True

        except Exception as e:
            self.logger.warning(f"Failed to create elevation band CLASS blocks: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _create_elevation_adjusted_block(
        self, base_block: list, band_index: int, elevation: float, fraction: float
    ) -> list:
        """Create a CLASS block with elevation-adjusted parameters.

        Args:
            base_block: Template CLASS block (lines 05-19)
            band_index: Index of elevation band (0-based)
            elevation: Mean elevation of band (m)
            fraction: Area fraction of band

        Returns:
            List of lines for the CLASS block
        """
        block = []

        # Use grassland/alpine vegetation for all elevation bands
        # Higher elevations have sparser vegetation
        # Cap at 1.0 (for low elevations) and min 0.2 (for very high)
        vegetation_cover = min(1.0, max(0.2, 1.0 - (elevation - 1500) / 3000))

        for line in base_block:
            new_line = line

            # Adjust FCAN (line 05) - vegetation fraction
            if '05 5xFCAN' in line:
                parts = line.split()
                # Set grassland (position 4) as dominant
                fcan_values = [0.0, 0.0, 0.0, vegetation_cover, 0.0]
                new_line = f"   {fcan_values[0]:.3f}   {fcan_values[1]:.3f}   {fcan_values[2]:.3f}   {fcan_values[3]:.3f}   {fcan_values[4]:.3f}   1.450   0.000   0.000   0.000     05 5xFCAN/4xLAMX"

            # Adjust LNZ0 (line 06) - roughness length
            elif '5xLNZ0/4xLAMN' in line:
                # Lower roughness at higher elevations (sparser vegetation)
                lnz0 = -1.8 + (elevation - 1500) / 3000 * 0.5
                new_line = f"   0.000   0.000   0.000  {lnz0:.3f}   0.000   0.000   0.000   0.000   1.200     06 5xLNZ0/4xLAMN"

            # Adjust RSMN (line 09) - stomatal resistance
            elif '4xRSMN/4xQA50' in line:
                # Higher resistance at higher elevations (stressed vegetation)
                rsmn = 200.0 + (elevation - 1500) / 2000 * 100
                new_line = f"   0.000   0.000   0.000 {rsmn:.3f}           0.000   0.000   0.000  36.000     09 4xRSMN/4xQA50"

            # Adjust MID in line 13 - unique identifier for each band
            elif 'XSLP/XDRAINH/MANN/KSAT/MID' in line:
                # Keep parameters, update MID
                mid = 200 + band_index  # 200, 201, 202, etc.
                # Parse existing values
                parts = line.split()
                if len(parts) >= 5:
                    xslp = parts[0]
                    xdrainh = parts[1]
                    mann = parts[2]
                    ksat = parts[3]
                    new_line = f"   {xslp}   {xdrainh}   {mann}   {ksat}   {mid} ElevBand_{band_index+1}_{elevation:.0f}m                        13 XSLP/XDRAINH/MANN/KSAT/MID"

            # Adjust initial snow (line 19) for elevation
            elif '19 RCAN/SCAN' in line:
                # More snow at higher elevations
                sno = 50.0 + (elevation - 1500) / 100 * 5  # 50kg/m2 + 5kg per 100m
                albs = min(0.85, 0.70 + (elevation - 1500) / 5000)  # Higher albedo with elevation
                rhos = 300.0  # Standard snow density
                new_line = f"   0.000   0.000   {sno:.1f}   {albs:.2f}   {rhos:.1f}   1.000  19 RCAN/SCAN/SNO/ALBS/RHOS/GRO"

            block.append(new_line)

        return block

    def _get_default_class_block(self) -> list:
        """Get a default CLASS parameter block for grassland vegetation."""
        return [
            "   0.000   0.000   0.000   1.000   0.000   0.000   0.000   0.000   1.450     05 5xFCAN/4xLAMX",
            "   0.000   0.000   0.000  -1.800   0.000   0.000   0.000   0.000   1.200     06 5xLNZ0/4xLAMN",
            "   0.000   0.000   0.000   0.045   0.000   0.000   0.000   0.000   4.500     07 5xALVC/4xCMAS",
            "   0.000   0.000   0.000   0.160   0.000   0.000   0.000   0.000   1.090     08 5xALIC/4xROOT",
            "   0.000   0.000   0.000 200.000           0.000   0.000   0.000  36.000     09 4xRSMN/4xQA50",
            "   0.000   0.000   0.000   0.800           0.000   0.000   0.000   1.050     10 4xVPDA/4xVPDB",
            "   0.000   0.000   0.000 100.000           0.000   0.000   0.000   5.000     11 4xPSGA/4xPSGB",
            "   1.000   2.500   1.000  50.000                                             12 DRN/SDEP/FARE/DD",
            "   0.030   0.350   0.100   0.050   100 Default                               13 XSLP/XDRAINH/MANN/KSAT/MID",
            "  50.000  50.000  50.000                                                     14 3xSAND (or more)",
            "  20.000  20.000  20.000                                                     15 3xCLAY (or more)",
            "   0.000   0.000   0.000                                                     16 3xORGM (or more)",
            "  4.000  2.000  1.000  -5.000  -10.000   4.000  17 3xTBAR (or more)/TCAN/TSNO/TPND",
            "   0.250   0.150   0.040   0.000   0.000   0.000   0.000                     18 3xTHLQ (or more)/3xTHIC (or more)/ZPND",
            "   0.000   0.000   100.0   0.75   250.0   1.000  19 RCAN/SCAN/SNO/ALBS/RHOS/GRO",
        ]
