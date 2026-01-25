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
        from symfluence.core.config.models import SymfluenceConfig
        if isinstance(config, dict):
            try:
                self._config = SymfluenceConfig(**config)
            except (TypeError, ValueError, KeyError, AttributeError):
                self._config = config
        else:
            self._config = config
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
            with open(self.run_options_path, 'r') as f:
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
                with open(self.run_options_path, 'w') as f:
                    f.write(content)
                self._update_control_flag_count()
                self.logger.info("Fixed run options variable names")

        except Exception as e:
            self.logger.warning(f"Failed to fix run options variable names: {e}")

    def fix_run_options_snow_params(self) -> None:
        """Fix run options snow/ice parameters for stable multi-year simulations."""
        if not self.run_options_path.exists():
            return

        try:
            with open(self.run_options_path, 'r') as f:
                content = f.read()

            # Get RUNMODE from config (default to 'wf_route' for routing)
            runmode = self._get_config_value('MESH_RUNMODE', 'wf_route')

            # Determine output flags based on routing mode
            if runmode == 'noroute':
                streamflow_flag = 'none'
                outfiles_flag = 'none'
            else:
                # Enable streamflow output when routing is enabled
                streamflow_flag = 'csv'
                outfiles_flag = 'default'

            modified = False
            # Snow parameters: SWELIM reduced from 1500 to 500mm for temperate regions
            # 1500mm was unrealistically high and caused multi-year accumulation issues
            # For alpine/polar applications, override via MESH_SWELIM config option
            replacements = [
                (r'FREZTH\s+[-\d.]+', 'FREZTH                -2.0'),
                (r'SWELIM\s+[-\d.]+', 'SWELIM                500.0'),
                (r'SNDENLIM\s+[-\d.]+', 'SNDENLIM              600.0'),
                (r'PBSMFLAG\s+\w+', 'PBSMFLAG              off'),
                (r'FROZENSOILINFILFLAG\s+\d+', 'FROZENSOILINFILFLAG   0'),
                (r'RUNMODE\s+\w+', f'RUNMODE               {runmode}'),
                (r'METRICSSPINUP\s+\d+', 'METRICSSPINUP         730'),
                (r'DIAGNOSEMODE\s+\w+', 'DIAGNOSEMODE          off'),
                (r'SHDFILEFLAG\s+\w+', 'SHDFILEFLAG           nc_subbasin pad_outlets'),
                (r'BASINFORCINGFLAG\s+\w+', 'BASINFORCINGFLAG      nc_subbasin'),
                (r'OUTFILESFLAG\s+\w+', f'OUTFILESFLAG         {outfiles_flag}'),
                (r'OUTFIELDSFLAG\s+\w+', 'OUTFIELDSFLAG        none'),
                (r'STREAMFLOWOUTFLAG\s+\w+', f'STREAMFLOWOUTFLAG     {streamflow_flag}'),
                (r'PRINTSIMSTATUS\s+\w+', 'PRINTSIMSTATUS        date_monthly'),
            ]

            for pattern, replacement in replacements:
                if re.search(pattern, content):
                    content_new = re.sub(pattern, replacement, content)
                    if content_new != content:
                        content = content_new
                        modified = True

            # Log the routing mode being used
            self.logger.info(f"MESH RUNMODE set to '{runmode}' with streamflow output '{streamflow_flag}'")

            if modified:
                with open(self.run_options_path, 'w') as f:
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
            with open(self.run_options_path, 'r') as f:
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
                    with open(self.run_options_path, 'w') as f:
                        f.writelines(lines)
                    self.logger.info(f"Updated control flag count to {flag_count}")

        except Exception as e:
            self.logger.warning(f"Failed to update control flag count: {e}")

    def fix_gru_count_mismatch(self) -> None:
        """Ensure CLASS NM matches parameter block count and trim empty GRU columns."""
        # First, determine which GRUs MESH will actually recognize
        mesh_active_grus = self._get_mesh_active_gru_count()

        # Then get the current DDB GRU count
        current_ddb_gru_count = self._get_ddb_gru_count()

        if mesh_active_grus is not None and current_ddb_gru_count is not None:
            if mesh_active_grus != current_ddb_gru_count:
                self.logger.warning(
                    f"DDB has {current_ddb_gru_count} GRU columns but MESH will only see {mesh_active_grus} active"
                )
                # Trim DDB to match MESH's expectations
                self._trim_ddb_to_active_grus(mesh_active_grus)

            # Now trim CLASS to match (which should equal mesh_active_grus)
            class_block_count = self._get_class_block_count()
            if class_block_count is not None and mesh_active_grus != class_block_count:
                self.logger.warning(
                    f"CLASS blocks ({class_block_count}) don't match MESH active GRUs ({mesh_active_grus})"
                )
                # Trim CLASS to match MESH's expectations
                self._trim_class_to_count(mesh_active_grus)
                # Update NM to reflect the trimmed count
                self._update_class_nm(mesh_active_grus)

            return

        # Fall back to original logic only if we couldn't determine mesh_active_grus
        keep_mask = self._trim_empty_gru_columns()
        self._fix_class_nm(keep_mask)
        self._ensure_gru_normalization()

    def _get_ddb_gru_count(self) -> Optional[int]:
        """Get the number of GRU columns in the DDB."""
        if not self.ddb_path.exists():
            return None

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                if 'NGRU' not in ds.dims:
                    return None
                return int(ds.dims['NGRU'])
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return None

    def _trim_ddb_to_active_grus(self, target_count: int) -> None:
        """Trim DDB GRU columns to only keep the active ones (sum > 0.1)."""
        if not self.ddb_path.exists():
            return

        try:
            with xr.open_dataset(self.ddb_path) as ds:
                if 'GRU' not in ds or 'NGRU' not in ds.dims:
                    return

                if ds.dims['NGRU'] <= target_count:
                    return

                # Determine which GRU columns are actually active (sum > 0.1)
                gru = ds['GRU']
                sum_dim = 'N' if 'N' in gru.dims else 'subbasin' if 'subbasin' in gru.dims else None
                if not sum_dim:
                    return

                sums = gru.sum(sum_dim)
                # Only keep GRU columns with sum > 0.15 (conservative threshold)
                keep_mask = sums > 0.15
                active_indices = [i for i, keep in enumerate(keep_mask.values) if keep]

                if len(active_indices) == 0:
                    self.logger.warning("No GRU columns have sum > 0.15")
                    return

                # Trim to only the active GRUs
                ds_trim = ds.isel(NGRU=active_indices)

                # Renormalize GRU fractions
                if 'GRU' in ds_trim:
                    sum_per = ds_trim['GRU'].sum('NGRU')
                    sum_safe = xr.where(sum_per == 0, 1.0, sum_per)
                    ds_trim['GRU'] = ds_trim['GRU'] / sum_safe

                temp_path = self.ddb_path.with_suffix('.tmp.nc')
                ds_trim.to_netcdf(temp_path)
                os.replace(temp_path, self.ddb_path)
                self.logger.info(f"Trimmed DDB to {len(active_indices)} active GRU column(s) (all with sum > 0.15)")
                self._ensure_gru_normalization()
        except Exception as e:
            self.logger.warning(f"Failed to trim DDB to active GRUs: {e}")

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
        """Determine the number of GRUs that MESH will recognize as active."""
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
                # Threshold for considering a GRU class "active"
                # Reduced from 0.15 to 0.05 to preserve small but hydrologically
                # important classes (e.g., wetlands, urban areas at <15% coverage)
                min_gru_sum = 0.05
                active_count = int((sums > min_gru_sum).sum())

                # Log any classes being excluded
                excluded = sums[sums <= min_gru_sum]
                if len(excluded) > 0:
                    excluded_total = float(excluded.sum())
                    self.logger.warning(
                        f"Excluding {len(excluded)} GRU class(es) below {min_gru_sum:.0%} threshold "
                        f"(total area lost: {excluded_total:.1%})"
                    )

                if active_count > 0:
                    self.logger.debug(f"MESH will recognize {active_count} active GRU(s) (threshold={min_gru_sum:.0%})")
                    return active_count

                return None
        except Exception as e:
            self.logger.debug(f"Could not determine MESH active GRU count: {e}")
            return None

    def _get_class_block_count(self) -> Optional[int]:
        """Get the number of CLASS parameter blocks."""
        if not self.class_file_path.exists():
            return None

        try:
            with open(self.class_file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')

            block_count = sum(1 for line in lines if 'XSLP/XDRAINH/MANN/KSAT/MID' in line or line.startswith('[GRU_'))
            return block_count if block_count > 0 else None
        except (FileNotFoundError, OSError, ValueError, KeyError):
            return None

    def _trim_class_to_count(self, target_count: int) -> None:
        """Trim CLASS parameter blocks to a specific count."""
        if not self.class_file_path.exists():
            return

        try:
            with open(self.class_file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')

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
                return

            # Keep only the first target_count blocks
            kept_blocks = blocks[:target_count]

            if len(kept_blocks) != len(blocks):
                new_lines = header + [line for block in kept_blocks for line in block]
                content = '\n'.join(new_lines)
                if not content.endswith('\n'):
                    content += '\n'
                with open(self.class_file_path, 'w') as f:
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
            with open(self.class_file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')

            block_count = sum(1 for line in lines if 'XSLP/XDRAINH/MANN/KSAT/MID' in line or line.startswith('[GRU_'))

            # Read current NM
            nm_from_class = self._read_nm_from_lines(lines)

            trimmed_class = False
            if keep_mask is not None:
                trimmed_class = self._trim_class_blocks(lines, keep_mask)
                if trimmed_class:
                    with open(self.class_file_path, 'r') as f:
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
            with open(self.class_file_path, 'w') as f:
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
                    except (ValueError, IndexError):
                        pass
                else:
                    if len(parts) >= 9:
                        try:
                            return int(parts[8])
                        except (ValueError, IndexError):
                            pass
                break
        return None

    def _update_class_nm(self, new_nm: int) -> None:
        """Update NM in CLASS parameters file."""
        try:
            with open(self.class_file_path, 'r') as f:
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
                with open(self.class_file_path, 'w') as f:
                    f.writelines(lines)

        except Exception as e:
            self.logger.warning(f"Failed to update CLASS NM: {e}")

    def fix_hydrology_wf_r2(self) -> None:
        """Ensure WF_R2 is in the hydrology file.

        Note: WF_R2 (WATFLOOD channel roughness) is DIFFERENT from R2N (overland Manning's n).
        - R2N: Manning's n for overland flow, typically 0.02-0.10
        - WF_R2: Channel roughness coefficient for WATFLOOD routing, typically 0.20-0.40

        Previously this code incorrectly set WF_R2 = R2N. Now uses appropriate default.
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
            with open(self.hydro_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')

            if not content.strip() and settings_hydro.exists():
                with open(settings_hydro, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                if content.strip():
                    with open(self.hydro_path, 'w') as f:
                        f.write(content)
                    self.logger.info("Restored hydrology file from settings")

            if 'WF_R2' in content:
                self.logger.debug("WF_R2 already present")
                return

            # Add WF_R2 with appropriate default (NOT copying from R2N)
            # WF_R2 = 0.30 is a reasonable default for mixed channel types
            # This can be calibrated during model optimization
            new_lines = []
            r2n_found = False
            for line in lines:
                if line.startswith('R2N') and not r2n_found:
                    parts = line.split()
                    if len(parts) >= 2:
                        n_values = len(parts) - 1  # Number of routing classes
                        # Use default WF_R2 = 0.30 for all classes (appropriate for channels)
                        wf_r2_values = ["0.30"] * n_values
                        wf_r2_line = "WF_R2  " + "    ".join(wf_r2_values) + "  # channel roughness (calibratable)"
                        new_lines.append(wf_r2_line)
                        r2n_found = True
                        self.logger.info(f"Added WF_R2=0.30 (default for {n_values} routing class(es))")

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
                with open(self.hydro_path, 'w') as f:
                    f.write('\n'.join(new_lines))

        except Exception as e:
            self.logger.warning(f"Failed to add WF_R2: {e}")

    def fix_missing_hydrology_params(self) -> None:
        """Verify hydrology parameters are present for MESH routing.

        Note: MESH WATFLOOD routing uses R2N, R1N, PWR, FLZ parameters which are
        generated by meshflow. Legacy parameters like RCHARG, BASEFLW, MANN are
        not supported by MESH and should not be added.
        """
        # Check if routing is enabled in run options
        if self.run_options_path.exists():
            with open(self.run_options_path, 'r') as f:
                run_options_content = f.read()

            # Skip if RUNMODE is set to noroute
            if re.search(r'RUNMODE\s+noroute', run_options_content):
                self.logger.info("RUNMODE is 'noroute', skipping routing parameter additions")
                return

        if not self.hydro_path.exists():
            self.logger.warning("Hydrology file not found, skipping parameter verification")
            return

        # Just verify that standard routing parameters exist (R2N, R1N, PWR, FLZ)
        # These are generated by meshflow - no need to add custom parameters
        try:
            with open(self.hydro_path, 'r') as f:
                content = f.read()

            required_params = ['R2N', 'R1N', 'PWR', 'FLZ']
            missing = [p for p in required_params if p not in content]

            if missing:
                self.logger.warning(f"Missing routing parameters in hydrology file: {missing}")
            else:
                self.logger.debug("All standard routing parameters present (R2N, R1N, PWR, FLZ)")

        except Exception as e:
            self.logger.warning(f"Failed to verify hydrology parameters: {e}")

    def fix_run_options_output_dirs(self) -> None:
        """Fix output directory paths in run options file."""
        run_options = self.forcing_dir / "MESH_input_run_options.ini"
        if not run_options.exists():
            return

        try:
            with open(run_options, 'r') as f:
                content = f.read()

            # Replace CLASSOUT with ./ for MESH 1.5 compatibility
            if 'CLASSOUT' in content:
                content = content.replace('CLASSOUT', './' + ' ' * 6)  # Pad to maintain alignment
                with open(run_options, 'w') as f:
                    f.write(content)
                self.logger.info("Fixed output directory paths in run options")

        except Exception as e:
            self.logger.warning(f"Failed to fix run options output dirs: {e}")

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
            with open(self.class_file_path, 'r') as f:
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
                        except (ValueError, IndexError):
                            pass

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
                        except (ValueError, IndexError):
                            pass

                new_lines.append(line)

            if modified:
                with open(self.class_file_path, 'w') as f:
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
            spinup_days = int(self._get_config_value(lambda: self.config.model.mesh.spinup_days, default=730, dict_key='MESH_SPINUP_DAYS'))
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
                n_spatial = ds_safe.dims.get('subbasin', 1)

                with NC4Dataset(safe_forcing_nc, 'w', format='NETCDF4') as ncfile:
                    # Create dimensions
                    ncfile.createDimension('time', None)  # unlimited
                    ncfile.createDimension('subbasin', n_spatial)

                    # Create coordinate variables
                    var_time = ncfile.createVariable('time', 'f8', ('time',))
                    var_time.standard_name = 'time'
                    var_time.long_name = 'time'
                    var_time.axis = 'T'
                    var_time.units = 'hours since 1979-12-01'
                    var_time.calendar = 'proleptic_gregorian'

                    # Convert time to hours since reference
                    reference = pd.Timestamp('1979-12-01')
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

        with open(self.run_options_path, 'r') as f:
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
            with open(self.run_options_path, 'w') as f:
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
                    except ValueError:
                        pass
        return date_line_indices
