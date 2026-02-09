"""
FUSE Worker

Worker implementation for FUSE model optimization.
Delegates to existing worker functions while providing BaseWorker interface.
"""

import logging
import shutil
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.core.constants import UnitConversion
from symfluence.models.utilities.routing_decider import RoutingDecider
from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.models.fuse.utilities import FuseToMizurouteConverter
from symfluence.models.fuse.calibration.multi_gauge_metrics import MultiGaugeMetrics
from symfluence.models.fuse.calibration.parameter_regionalization import (
    RegionalizationFactory
)

# Suppress xarray FutureWarning about timedelta64 decoding
warnings.filterwarnings('ignore',
                       message='.*decode_timedelta.*',
                       category=FutureWarning,
                       module='xarray.*')

logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('FUSE')
class FUSEWorker(BaseWorker):
    """
    Worker for FUSE model calibration.

    Handles parameter application to NetCDF files, FUSE execution,
    and metric calculation for streamflow calibration.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize FUSE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    # Shared utilities
    _routing_decider = RoutingDecider()
    _streamflow_metrics = StreamflowMetrics()
    _format_converter = FuseToMizurouteConverter()

    def needs_routing(self, config: Dict[str, Any], settings_dir: Optional[Path] = None) -> bool:
        """
        Determine if routing (mizuRoute) is needed for FUSE.

        Delegates to shared RoutingDecider utility.

        Args:
            config: Configuration dictionary
            settings_dir: Optional settings directory to check for mizuRoute control files

        Returns:
            True if routing is needed
        """
        return self._routing_decider.needs_routing(config, 'FUSE', settings_dir)

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to FUSE constraints file AND para_def.nc.

        In run_def mode, FUSE reads parameters directly from para_def.nc.
        We update BOTH the constraints file (for consistency and run_pre mode)
        AND the para_def.nc file (for run_def mode used during calibration).

        Args:
            params: Parameter values to apply
            settings_dir: FUSE settings directory
            **kwargs: Must include 'config' and 'sim_dir' for path resolution

        Returns:
            True if successful
        """

        try:
            config = kwargs.get('config', self.config)
            _sim_dir = kwargs.get('sim_dir')  # noqa: F841

            # Log parameters being applied at DEBUG level to reduce spam
            self.logger.debug(f"APPLY_PARAMS: Applying {len(params)} parameters to {settings_dir}")
            for p, v in list(params.items())[:5]:  # Log first 5 params
                self.logger.debug(f"  PARAM: {p} = {v:.4f}")

            # Handle both cases: settings_dir is already FUSE dir, or contains FUSE subdir
            if settings_dir.name == 'FUSE':
                fuse_settings_dir = settings_dir
            elif (settings_dir / 'FUSE').exists():
                fuse_settings_dir = settings_dir / 'FUSE'
            else:
                fuse_settings_dir = settings_dir

            # =====================================================================
            # Update CONSTRAINTS FILE for consistency and run_pre mode
            # Skip in regionalization mode (coefficients don't map to constraints)
            # =====================================================================
            regionalization_method = config.get('PARAMETER_REGIONALIZATION', 'lumped') if config else 'lumped'
            if config and config.get('USE_TRANSFER_FUNCTIONS', False):
                regionalization_method = 'transfer_function'

            constraints_file = fuse_settings_dir / 'fuse_zConstraints_snow.txt'

            if regionalization_method != 'lumped':
                # In regionalization mode, skip constraints update
                # (coefficients like MAXWATR_1_a don't exist in constraints file)
                self.logger.debug(
                    f"Regionalization mode ({regionalization_method}): "
                    f"skipping constraints file (using para_def.nc)"
                )
            elif constraints_file.exists():
                params_updated_txt = self._update_constraints_file(constraints_file, params)
                if params_updated_txt:
                    sample_params = list(params.items())[:3]
                    self.logger.debug(f"APPLY: Updated {len(params_updated_txt)} params in {constraints_file.name}, sample: {sample_params}")
                else:
                    self.logger.warning("APPLY_PARAMS: No params updated in constraints file")
            else:
                self.logger.error(
                    f"APPLY_PARAMS: Constraints file not found at {constraints_file}. "
                    f"FUSE calibration will not work!"
                )
                return False

            # =====================================================================
            # CRITICAL: Also update para_def.nc for run_def mode
            # In run_def mode, FUSE reads directly from para_def.nc, NOT from
            # constraints. This is the file that run_model() will copy.
            # =====================================================================
            if config:
                domain_name = config.get('DOMAIN_NAME', '')
                experiment_id = config.get('EXPERIMENT_ID', 'run_1')
                fuse_id = config.get('FUSE_FILE_ID', experiment_id)

                # Find and update the para_def.nc file
                para_def_path = fuse_settings_dir / f"{domain_name}_{fuse_id}_para_def.nc"

                # Check for parameter regionalization mode
                regionalization_method = config.get('PARAMETER_REGIONALIZATION', 'lumped')
                # Backward compatibility: USE_TRANSFER_FUNCTIONS overrides
                if config.get('USE_TRANSFER_FUNCTIONS', False):
                    regionalization_method = 'transfer_function'

                if regionalization_method != 'lumped' and para_def_path.exists():
                    # Use regionalization to generate spatially distributed parameters
                    params_updated_nc = self._apply_regionalization(
                        para_def_path, params, config
                    )
                    if params_updated_nc:
                        self.logger.debug(
                            f"APPLY: Updated {len(params_updated_nc)} distributed params "
                            f"via {regionalization_method} regionalization"
                        )
                    else:
                        self.logger.warning(f"APPLY_PARAMS: {regionalization_method} update failed")
                elif para_def_path.exists():
                    params_updated_nc = self._update_para_def_nc(para_def_path, params)
                    if params_updated_nc:
                        self.logger.debug(f"APPLY: Updated {len(params_updated_nc)} params in {para_def_path.name}")
                    else:
                        self.logger.warning("APPLY_PARAMS: No params updated in para_def.nc")
                else:
                    self.logger.warning(f"APPLY_PARAMS: para_def.nc not found at {para_def_path}")

            return True

        except (FileNotFoundError, OSError) as e:
            self.logger.error(f"File error applying FUSE parameters: {e}")
            return False
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data error applying FUSE parameters: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _update_para_def_nc(self, para_def_path: Path, params: Dict[str, float]) -> set:
        """
        Update FUSE para_def.nc file with new parameter values.

        Args:
            para_def_path: Path to para_def.nc file
            params: Parameter values to apply

        Returns:
            Set of parameter names that were updated
        """
        import netCDF4 as nc

        params_updated: set[str] = set()

        try:
            with nc.Dataset(para_def_path, 'r+') as ds:
                # Verify the file structure
                if 'par' not in ds.dimensions:
                    self.logger.error(f"Missing 'par' dimension in {para_def_path}")
                    return params_updated

                par_size = ds.dimensions['par'].size
                if par_size == 0:
                    self.logger.error(f"Empty 'par' dimension in {para_def_path}")
                    return params_updated

                for param_name, value in params.items():
                    if param_name in ds.variables:
                        try:
                            # Always use index 0 for single parameter set
                            before = float(ds.variables[param_name][0])
                            ds.variables[param_name][0] = float(value)
                            after = float(ds.variables[param_name][0])
                            self.logger.debug(f"  NC: {param_name}: {before:.4f} -> {after:.4f}")
                            params_updated.add(param_name)
                        except (IndexError, ValueError, TypeError) as e:
                            self.logger.warning(f"Error updating {param_name} in NetCDF: {e}")
                    else:
                        self.logger.debug(f"  NC: {param_name} not in file (may be structure param)")

                # Force sync to disk
                ds.sync()

        except (OSError, IOError) as e:
            self.logger.error(f"I/O error updating {para_def_path}: {e}")
        except (KeyError, ValueError) as e:
            self.logger.error(f"Data error updating {para_def_path}: {e}")

        # Verify write succeeded by reading back a value
        # Use tolerance of 1e-3 to match FUSE's Fortran fixed-width precision (F9.3 format)
        if params_updated:
            try:
                with nc.Dataset(para_def_path, 'r') as ds:
                    # Read back first updated param to verify
                    first_param = next(iter(params_updated))
                    if first_param in ds.variables:
                        actual_value = float(ds.variables[first_param][0])
                        expected_value = params[first_param]
                        if abs(actual_value - expected_value) > 1e-3:
                            self.logger.warning(
                                f"Parameter write verification: {first_param} expected {expected_value:.6f} "
                                f"but file contains {actual_value:.6f} (diff={abs(actual_value - expected_value):.2e})"
                            )
            except Exception as e:
                self.logger.debug(f"Could not verify para_def.nc write: {e}")

        return params_updated

    def _apply_regionalization(
        self,
        para_def_path: Path,
        calibration_params: Dict[str, float],
        config: Dict[str, Any]
    ) -> set:
        """
        Apply parameter regionalization to generate spatially distributed parameters.

        Supports multiple regionalization methods:
        - lumped: Uniform parameters across all subcatchments
        - transfer_function: Power-law functions based on catchment attributes
        - zones: Shared parameters within predefined zones
        - distributed: Independent parameters per subcatchment

        Args:
            para_def_path: Path to para_def.nc file
            calibration_params: Calibration parameter/coefficient values
            config: Configuration dictionary

        Returns:
            Set of parameter names that were updated
        """
        import netCDF4 as nc
        import pandas as pd

        params_updated: set[str] = set()

        try:
            # Get regionalization method
            method = config.get('PARAMETER_REGIONALIZATION', 'lumped')

            # For backward compatibility
            if config.get('USE_TRANSFER_FUNCTIONS', False) and method == 'lumped':
                method = 'transfer_function'

            # Get original parameter bounds
            param_bounds = config.get('FUSE_PARAM_BOUNDS', {})
            if not param_bounds:
                self.logger.error("FUSE_PARAM_BOUNDS not configured")
                return params_updated

            # Convert bounds from list to tuple format if needed
            param_bounds_tuples = {
                k: tuple(v) if isinstance(v, list) else v
                for k, v in param_bounds.items()
            }

            # Load attributes if needed
            attributes = None
            if method == 'transfer_function':
                attributes_path = config.get('TRANSFER_FUNCTION_ATTRIBUTES')
                if attributes_path:
                    attributes = pd.read_csv(attributes_path)
                else:
                    self.logger.error("TRANSFER_FUNCTION_ATTRIBUTES not configured")
                    return params_updated

            # Determine number of subcatchments
            if attributes is not None:
                n_subcatchments = len(attributes)
            else:
                # Try to get from para_def.nc
                with nc.Dataset(para_def_path, 'r') as ds:
                    n_subcatchments = ds.dimensions['par'].size

            # Create regionalization strategy
            regionalization = RegionalizationFactory.create(
                method=method,
                param_bounds=param_bounds_tuples,
                n_subcatchments=n_subcatchments,
                config=config,
                attributes=attributes,
                logger=self.logger
            )

            self.logger.debug(f"Using '{regionalization.name}' parameter regionalization")

            # Convert calibration parameters to distributed values
            param_array, param_names = regionalization.to_distributed(calibration_params)

            self.logger.debug(
                f"Regionalization: {len(param_names)} params x {n_subcatchments} subcatchments"
            )

            # Update para_def.nc with distributed values
            with nc.Dataset(para_def_path, 'r') as ds:
                par_size = ds.dimensions['par'].size
                # Read existing variable names and attributes
                existing_vars = {}
                for vname in ds.variables:
                    if vname == 'par':
                        continue
                    existing_vars[vname] = {
                        'values': ds.variables[vname][:].copy(),
                        'attrs': {a: ds.variables[vname].getncattr(a)
                                  for a in ds.variables[vname].ncattrs()},
                    }
                global_attrs = {a: ds.getncattr(a) for a in ds.ncattrs()}

            if par_size != n_subcatchments:
                # Recreate para_def.nc with correct par dimension
                self.logger.debug(
                    f"Resizing para_def.nc: par={par_size} -> {n_subcatchments}"
                )
                # Use a temporary file to avoid corruption
                tmp_path = para_def_path.with_suffix('.tmp.nc')
                with nc.Dataset(tmp_path, 'w', format='NETCDF4') as ds_new:
                    ds_new.createDimension('par', n_subcatchments)
                    ds_new.createVariable('par', 'i4', ('par',))
                    ds_new.variables['par'][:] = np.arange(n_subcatchments)
                    for attr_name, attr_val in global_attrs.items():
                        ds_new.setncattr(attr_name, attr_val)

                    for vname, vinfo in existing_vars.items():
                        # Extract _FillValue (must be set at creation time)
                        fill_value = vinfo['attrs'].get('_FillValue', None)
                        ds_new.createVariable(vname, 'f8', ('par',), fill_value=fill_value)
                        # Broadcast first parameter set value to all subcatchments
                        ds_new.variables[vname][:] = np.full(
                            n_subcatchments, float(vinfo['values'][0])
                        )
                        for attr_name, attr_val in vinfo['attrs'].items():
                            if attr_name == '_FillValue':
                                continue  # Already set during creation
                            ds_new.variables[vname].setncattr(attr_name, attr_val)

                import shutil
                shutil.move(str(tmp_path), str(para_def_path))

            # Now write the distributed values
            with nc.Dataset(para_def_path, 'r+') as ds:
                for i, param_name in enumerate(param_names):
                    if param_name not in ds.variables:
                        continue

                    values = param_array[:, i]
                    ds.variables[param_name][:] = values
                    self.logger.debug(
                        f"  {param_name}: distributed [{values.min():.3f}, {values.max():.3f}]"
                    )
                    params_updated.add(param_name)

                # Ensure numerical solver settings are reasonable
                # (para_def.nc can inherit overly tight defaults like 1e-12)
                numerix_defaults = {
                    'SOLUTION': 1.0,       # Explicit Heun (avoid implicit convergence failures)
                    'TIMSTEP_TYP': 1.0,    # Adaptive time steps (original ODE_INT handles this)
                    'ERRITERFUNC': 1e-4,
                    'ERR_ITER_DX': 1e-4,
                    'NITER_TOTAL': 5000.0,
                    'MIN_TSTEP': 0.001 / 1440.0,
                }
                for nvar, nval in numerix_defaults.items():
                    if nvar in ds.variables:
                        cur = float(ds.variables[nvar][0])
                        if nvar in ('SOLUTION', 'TIMSTEP_TYP'):
                            # Always enforce explicit solver + fixed time steps
                            if cur != nval:
                                ds.variables[nvar][:] = nval
                                self.logger.debug(f"  Set {nvar}: {cur:.0f} -> {nval:.0f}")
                        elif cur < nval * 0.01:  # Much tighter than target
                            ds.variables[nvar][:] = nval
                            self.logger.debug(f"  Relaxed {nvar}: {cur:.2e} -> {nval:.2e}")

                ds.sync()

            # Log summary of spatial variation
            if params_updated:
                self.logger.debug(
                    f"Applied regionalization to {len(params_updated)} parameters: "
                    f"{', '.join(sorted(params_updated))}"
                )

        except Exception as e:
            self.logger.error(f"Error applying transfer functions: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

        return params_updated

    def _update_constraints_file(self, constraints_file: Path, params: Dict[str, float]) -> set:
        """
        Update FUSE constraints file with new parameter default values.

        FUSE uses Fortran fixed-width format: (L1,1X,I1,1X,3(F9.3,1X),...)
        The default value column starts at position 4 and is exactly 9 characters.

        Args:
            constraints_file: Path to constraints file
            params: Parameter values to apply

        Returns:
            Set of parameter names that were updated
        """
        params_updated = set()

        try:
            # Read the constraints file with encoding fallback
            try:
                with open(constraints_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                self.logger.warning(
                    f"UTF-8 decode error reading {constraints_file}, falling back to latin-1"
                )
                with open(constraints_file, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

            # Fortran format: (L1,1X,I1,1X,3(F9.3,1X),...)
            # Default value column: position 4-12 (9 chars, F9.3 format)
            DEFAULT_VALUE_START = 4
            DEFAULT_VALUE_WIDTH = 9

            updated_lines = []

            for line in lines:
                # Skip header line (starts with '(') and comment lines
                stripped = line.strip()
                if stripped.startswith('(') or stripped.startswith('*') or stripped.startswith('!'):
                    updated_lines.append(line)
                    continue

                # Check if this line contains any of our parameters
                updated = False
                for param_name, value in params.items():
                    # Match exact parameter name (avoid partial matches)
                    parts = line.split()
                    if len(parts) >= 14 and param_name in parts:
                        # Parameter name is at index 13 in parts
                        if parts[13] == param_name:
                            # Format value to exactly 9 characters (F9.3 format)
                            new_value = f"{value:9.3f}"

                            # Replace the fixed-width column in the line
                            # Position 4-12 is the default value (9 characters)
                            if len(line) > DEFAULT_VALUE_START + DEFAULT_VALUE_WIDTH:
                                new_line = (
                                    line[:DEFAULT_VALUE_START] +
                                    new_value +
                                    line[DEFAULT_VALUE_START + DEFAULT_VALUE_WIDTH:]
                                )
                                updated_lines.append(new_line)
                                params_updated.add(param_name)
                                updated = True
                                break

                if not updated:
                    updated_lines.append(line)

            # Write updated constraints file
            with open(constraints_file, 'w') as f:
                f.writelines(updated_lines)

        except (OSError, IOError) as e:
            self.logger.warning(f"I/O error updating constraints file: {e}")
        except (IndexError, ValueError) as e:
            self.logger.warning(f"Format error updating constraints file: {e}")

        return params_updated

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run FUSE model.

        Args:
            config: Configuration dictionary
            settings_dir: FUSE settings directory
            output_dir: Output directory (not used directly by FUSE)
            **kwargs: Additional arguments including 'mode'

        Returns:
            True if model ran successfully
        """
        try:
            import subprocess

            # Determine FUSE run mode based on regionalization setting
            # - run_def: regenerates para_def.nc from constraints file (default for lumped)
            # - run_pre: reads parameters directly from para_def.nc (required for regionalization)
            # NOTE: run_pre requires FUSE built from martynpclark/fuse bugfix/runpre branch

            # Check for explicit FUSE_RUN_MODE first
            explicit_mode = config.get('FUSE_RUN_MODE')
            if explicit_mode:
                mode = explicit_mode
                self.logger.debug(f"FUSE using explicit run mode from config: {mode}")
            else:
                # Auto-detect based on regionalization
                regionalization_method = config.get('PARAMETER_REGIONALIZATION', 'lumped')
                if config.get('USE_TRANSFER_FUNCTIONS', False):
                    regionalization_method = 'transfer_function'

                if regionalization_method != 'lumped':
                    mode = kwargs.get('mode', 'run_pre')
                    self.logger.info(f"FUSE auto-selected run_pre mode (regionalization={regionalization_method})")
                else:
                    mode = kwargs.get('mode', 'run_def')
                    self.logger.debug("FUSE using run_def mode (lumped regionalization)")

            # Get FUSE executable path
            fuse_install = config.get('FUSE_INSTALL_PATH', 'default')
            if fuse_install == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                fuse_exe = data_dir / 'installs' / 'fuse' / 'bin' / 'fuse.exe'
            else:
                fuse_exe = Path(fuse_install) / 'fuse.exe'

            # Get file manager path using settings_dir
            # Handle both cases: settings_dir already is the FUSE dir, or contains a FUSE subdir
            # Avoid double FUSE (settings/FUSE/FUSE) by checking if we're already in a FUSE dir
            if settings_dir.name == 'FUSE':
                # settings_dir is already the FUSE directory
                filemanager_path = settings_dir / 'fm_catch.txt'
                execution_cwd = settings_dir
            elif (settings_dir / 'FUSE').exists():
                # FUSE subdirectory exists
                filemanager_path = settings_dir / 'FUSE' / 'fm_catch.txt'
                execution_cwd = settings_dir / 'FUSE'
            else:
                # No FUSE subdirectory, use settings_dir directly
                filemanager_path = settings_dir / 'fm_catch.txt'
                execution_cwd = settings_dir

            if not fuse_exe.exists():
                self.logger.error(f"FUSE executable not found: {fuse_exe}")
                return False

            if not filemanager_path.exists():
                self.logger.error(f"FUSE file manager not found: {filemanager_path}")
                return False

            # Use sim_dir for FUSE output (consistent with SUMMA structure)
            # sim_dir = process_X/simulations/run_1/FUSE
            fuse_output_dir = kwargs.get('sim_dir', output_dir)
            if fuse_output_dir:
                Path(fuse_output_dir).mkdir(parents=True, exist_ok=True)

            # Update file manager with isolated paths, experiment_id, and FMODEL_ID
            # We use a short alias 'sim' for the domain ID to avoid Fortran string length limits
            # and create symlinks for the input files in the execution directory
            fuse_run_id = 'sim'

            # Create symlinks for input files
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            domain_name = config.get('DOMAIN_NAME')
            project_dir = data_dir / f"domain_{domain_name}"
            fuse_input_dir = project_dir / 'forcing' / 'FUSE_input'
            experiment_id = config.get('EXPERIMENT_ID', 'run_1')
            fuse_id = config.get('FUSE_FILE_ID', experiment_id)

            # Define input files to symlink
            input_files = [
                (fuse_input_dir / f"{domain_name}_input.nc", f"{fuse_run_id}_input.nc"),
                (fuse_input_dir / f"{domain_name}_elev_bands.nc", f"{fuse_run_id}_elev_bands.nc")
            ]

            # COPY (not symlink!) the parameter file to match the short alias
            # CRITICAL: FUSE overwrites para_def.nc during run_def mode, which would corrupt
            # a symlinked source file. We must copy so FUSE writes to an isolated copy.
            param_file_src = execution_cwd / f"{domain_name}_{fuse_id}_para_def.nc"
            param_file_dst_path = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"
            if param_file_src.exists():
                # Use module-level shutil import (already imported at top of file)
                if param_file_dst_path.exists():
                    param_file_dst_path.unlink()
                shutil.copy2(param_file_src, param_file_dst_path)
                self.logger.debug(f"FUSE para_def copied (not symlinked): {param_file_dst_path.name}")
            else:
                self.logger.warning(f"FUSE para_def source not found: {param_file_src}")

            # Ensure configuration files are present (input_info.txt, fuse_zNumerix.txt, etc.)
            # These should have been copied by the optimizer, but if missing, symlink from main settings
            project_settings_dir = project_dir / 'settings' / 'FUSE'
            self.logger.debug(f"Checking for config files in: {project_settings_dir}")

            config_files = ['input_info.txt', 'fuse_zNumerix.txt']

            # Add decisions file to the list
            # Try to find the specific decisions file for this experiment
            actual_decisions_file = f"fuse_zDecisions_{experiment_id}.txt"
            if (project_settings_dir / actual_decisions_file).exists():
                config_files.append(actual_decisions_file)
            else:
                self.logger.warning(f"Decisions file {actual_decisions_file} not found in {project_settings_dir}")
                # Fallback: find any decisions file
                try:
                    decisions = list(project_settings_dir.glob("fuse_zDecisions_*.txt"))
                    if decisions:
                        actual_decisions_file = decisions[0].name
                        config_files.append(actual_decisions_file)
                        self.logger.warning(f"Using fallback decisions file: {actual_decisions_file}")
                except Exception as e:
                    self.logger.warning(f"Error searching for decisions files: {e}")

            for cfg_file in config_files:
                target_path = execution_cwd / cfg_file
                if not target_path.exists():
                    src_path = project_settings_dir / cfg_file
                    if src_path.exists():
                        input_files.append((src_path, cfg_file))
                        self.logger.warning(f"Restoring missing config file: {cfg_file}")
                    else:
                        self.logger.error(f"Source config file not found: {src_path}")

            # Create symlinks for input files (but NOT para_def.nc which was copied above)
            for src, link_name in input_files:
                if src.exists():
                    link_path = execution_cwd / link_name
                    # Remove existing link/file if it exists
                    if link_path.exists() or link_path.is_symlink():
                        link_path.unlink()
                    try:
                        link_path.symlink_to(src)
                        self.logger.debug(f"Created symlink: {link_path} -> {src}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create symlink {link_path}: {e}")
                else:
                    self.logger.warning(f"Symlink source not found: {src}")

            # Verify para_def copy exists
            if not param_file_dst_path.exists():
                self.logger.error(
                    f"FUSE para_def copy was not created. "
                    f"Expected: {param_file_dst_path}"
                )

            # Pass use_local_input=True and actual decisions file to _update_file_manager
            if not self._update_file_manager(filemanager_path, execution_cwd, fuse_output_dir,
                                          config=config, use_local_input=True,
                                          decisions_file=actual_decisions_file):
                return False

            # List files in execution directory at DEBUG level to reduce spam
            self.logger.debug(f"Files in execution CWD ({execution_cwd}):")
            try:
                for f in execution_cwd.iterdir():
                    if f.is_symlink():
                        self.logger.debug(f"  {f.name} -> {f.resolve()}")
                    else:
                        self.logger.debug(f"  {f.name}")
            except Exception as e:
                self.logger.debug(f"Could not list directory: {e}")

            # Execute FUSE using the short alias
            # cmd = [str(fuse_exe), str(filemanager_path.name), domain_name, mode]
            cmd = [str(fuse_exe), str(filemanager_path.name), fuse_run_id, mode]

            # For run_pre mode (with martynpclark/fuse bugfix/runpre branch):
            # Command: fuse.exe filemanager dom_id run_pre para_def.nc [index]
            # - 4th arg: parameter NetCDF file
            # - 5th arg: parameter set index (optional, defaults to 1)
            if mode == 'run_pre':
                # Use the short alias (fuse_run_id) to match the copied parameter file
                param_file = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"
                if param_file.exists():
                    cmd.append(str(param_file.name))
                    cmd.append('1')  # Parameter set index (always use first/only set)
                else:
                    # Fallback to domain_name version if short alias not found
                    param_file_alt = execution_cwd / f"{domain_name}_{fuse_id}_para_def.nc"
                    if param_file_alt.exists():
                        cmd.append(str(param_file_alt.name))
                        cmd.append('1')
                    else:
                        self.logger.error(f"Parameter file not found for run_pre: tried {param_file} and {param_file_alt}")
                        return False

            # Use execution_cwd as cwd
            # Log command at INFO level for first execution to help debugging
            self.logger.debug(f"Executing FUSE: {' '.join(cmd)} in {execution_cwd}")

            # Verify key files exist before running FUSE
            expected_para_def = execution_cwd / f"{fuse_run_id}_{fuse_id}_para_def.nc"
            if not expected_para_def.exists() and not expected_para_def.is_symlink():
                self.logger.error(f"FUSE parameter file not found: {expected_para_def}")

            result = subprocess.run(
                cmd,
                cwd=str(execution_cwd),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace undecodable bytes to prevent UnicodeDecodeError
                timeout=config.get('FUSE_TIMEOUT', 300)
            )

            if result.returncode != 0:
                self.logger.error(f"FUSE failed with return code {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False

            # Detect Fortran STOP messages (FUSE returns exit code 0 on macOS even on STOP)
            combined_output = (result.stdout or '') + (result.stderr or '')
            if 'STOP' in combined_output and 'failed to converge' in combined_output:
                self.logger.error(
                    f"FUSE hit convergence failure (Fortran STOP with exit code 0). "
                    f"Output: {combined_output[-300:]}"
                )
                return False

            # Log FUSE output only at DEBUG level to reduce spam
            if result.stdout:
                self.logger.debug(f"FUSE stdout (last 500 chars): {result.stdout[-500:]}")
            if result.stderr:
                self.logger.debug(f"FUSE stderr: {result.stderr}")

            # Validate that FUSE actually produced output (FUSE can return 0 but fail silently)
            # Output will now use the short alias 'sim' and the FMODEL_ID from file manager
            fuse_id = config.get('FUSE_FILE_ID', config.get('EXPERIMENT_ID'))

            # The output filename format is {domain_id}_{fmodel_id}_{suffix}
            # FUSE writes to execution_cwd because we set OUTPUT_PATH to ./
            # The suffix depends on the run mode: run_def -> runs_def.nc, run_pre -> runs_pre.nc
            run_suffix = 'runs_def' if mode == 'run_def' else 'runs_pre'
            local_output_filename = f"{fuse_run_id}_{fuse_id}_{run_suffix}.nc"
            local_output_path = execution_cwd / local_output_filename

            # Final destination - use consistent naming regardless of mode
            final_output_path = fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc"

            if local_output_path.exists():
                try:
                    # Move to final destination and rename
                    if final_output_path.exists():
                        final_output_path.unlink()
                    shutil.move(str(local_output_path), str(final_output_path))
                    self.logger.debug(f"Moved output from {local_output_path} to {final_output_path}")
                except Exception as e:
                    self.logger.error(f"Failed to move output file: {e}")
                    return False
            else:
                self.logger.error(f"FUSE returned success but local output file not created: {local_output_path}")
                # Only log first and last 1000 chars of stdout to avoid massive log spam
                if result.stdout:
                    stdout_lines = result.stdout.split('\n')
                    if len(stdout_lines) > 20:
                        self.logger.error(f"FUSE stdout (first 10 lines): {chr(10).join(stdout_lines[:10])}")
                        self.logger.error(f"FUSE stdout (last 10 lines): {chr(10).join(stdout_lines[-10:])}")
                    else:
                        self.logger.error(f"FUSE stdout: {result.stdout}")
                return False

            # Validate FUSE output has actual data (Fortran STOP returns exit code 0 on macOS)
            try:
                import xarray as xr
                with xr.open_dataset(final_output_path, decode_times=False) as ds_check:
                    time_dim = 'time' if 'time' in ds_check.dims else None
                    if time_dim and ds_check.sizes[time_dim] == 0:
                        self.logger.error(
                            f"FUSE output has 0 time steps — model likely crashed silently "
                            f"(Fortran STOP returns exit code 0). "
                            f"Stderr: {(result.stderr or '')[:500]}"
                        )
                        return False
                    n_time = ds_check.sizes.get(time_dim, 0) if time_dim else -1
                    self.logger.debug(f"FUSE output validated: {n_time} time steps in {final_output_path.name}")
            except Exception as e:
                self.logger.warning(f"Could not validate FUSE output time dimension: {e}")

            self.logger.debug(f"FUSE completed successfully, output: {final_output_path}")

            # Run routing if needed
            # Pass settings_dir to check for mizuRoute control files
            needs_routing_check = self.needs_routing(config, settings_dir=settings_dir)
            self.logger.debug(f"Routing check: needs_routing={needs_routing_check}, settings_dir={settings_dir}")

            if needs_routing_check:
                self.logger.debug("Running mizuRoute for FUSE output")

                # Get proc_id for parallel calibration (used for unique filenames)
                proc_id = kwargs.get('proc_id', 0)

                # Determine output directories
                sim_dir = kwargs.get('sim_dir')
                if sim_dir:
                    mizuroute_dir = Path(sim_dir).parent / 'mizuRoute'
                else:
                    mizuroute_dir = Path(fuse_output_dir).parent / 'mizuRoute'

                mizuroute_dir.mkdir(parents=True, exist_ok=True)

                # Clean stale mizuRoute output from previous iterations/runs
                # to prevent calculate_metrics() from picking up old results
                stale_mizu_files = list(mizuroute_dir.glob("proc_*.nc")) + list(mizuroute_dir.glob("*.h.*.nc"))
                if stale_mizu_files:
                    self.logger.debug(f"Cleaning {len(stale_mizu_files)} stale mizuRoute output file(s)")
                    for stale_f in stale_mizu_files:
                        try:
                            stale_f.unlink()
                        except OSError:
                            pass

                # Convert FUSE output to mizuRoute format
                # Pass proc_id for correct filename generation in parallel calibration
                if not self._convert_fuse_to_mizuroute_format(
                    fuse_output_dir, config, execution_cwd, proc_id=proc_id
                ):
                    self.logger.error("Failed to convert FUSE output to mizuRoute format")
                    return False

                # Run mizuRoute
                # Pass settings_dir explicitly since it's a positional arg, not in kwargs
                # Remove keys that are passed explicitly to avoid duplicate argument errors
                keys_to_remove = {'proc_id', 'mizuroute_dir', 'settings_dir'}
                kwargs_filtered = {k: v for k, v in kwargs.items() if k not in keys_to_remove}
                if not self._run_mizuroute_for_fuse(
                    config, fuse_output_dir, mizuroute_dir,
                    settings_dir=settings_dir, proc_id=proc_id, **kwargs_filtered
                ):
                    if mode == 'run_pre':
                        # Calibration mode: routing is required for multi-gauge metrics
                        self.logger.error("Routing failed during calibration — returning failure")
                        return False
                    else:
                        self.logger.warning("Routing failed, but FUSE succeeded (non-calibration mode)")

            return True

        except subprocess.TimeoutExpired:
            self.logger.error("FUSE execution timed out")
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Required file not found for FUSE: {e}")
            return False
        except (OSError, IOError) as e:
            self.logger.error(f"I/O error running FUSE: {e}")
            return False
        except (subprocess.SubprocessError, RuntimeError) as e:
            # Catch subprocess errors and runtime issues during model execution
            self.logger.error(f"Error running FUSE: {e}")
            return False

    def _update_file_manager(self, filemanager_path: Path, settings_dir: Path, output_dir: Path,
                              experiment_id: str = None, config: Dict[str, Any] = None,
                              use_local_input: bool = False, decisions_file: str = None) -> bool:
        """
        Update FUSE file manager with isolated paths for parallel execution.

        Args:
            filemanager_path: Path to fm_catch.txt
            settings_dir: Isolated settings directory (where input files are)
            output_dir: Isolated output directory
            experiment_id: Experiment ID to use for FMODEL_ID and decisions file
            config: Configuration dictionary
            use_local_input: If True, set INPUT_PATH to ./ and expect files to be symlinked
            decisions_file: Actual decisions filename to use (if known from pre-check)

        Returns:
            True if successful
        """
        try:
            # Use encoding with error handling to prevent UnicodeDecodeError
            # Some file managers may contain non-UTF-8 characters (e.g., from paths)
            try:
                with open(filemanager_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError as ue:
                self.logger.warning(
                    f"UTF-8 decode error reading {filemanager_path} at position {ue.start}: "
                    f"falling back to latin-1 encoding"
                )
                with open(filemanager_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

            # Get experiment_id from config if not provided
            if experiment_id is None and config:
                experiment_id = config.get('EXPERIMENT_ID', 'run_1')
            elif experiment_id is None:
                experiment_id = 'run_1'

            # Get fuse_id for output files
            fuse_id = experiment_id
            if config:
                fuse_id = config.get('FUSE_FILE_ID', experiment_id)

            updated_lines = []

            # Use relative paths where possible to avoid Fortran string length limits (often 128 chars)
            # FUSE execution CWD is settings_dir (or settings_dir/FUSE)
            execution_cwd = filemanager_path.parent

            try:
                # SETNGS_PATH is the execution directory itself
                settings_path_str = "./"

                # Use local output path to avoid FUSE path length/symlink issues
                output_path_str = "./"

                self.logger.debug(f"Using paths - Settings: {settings_path_str}, Output: {output_path_str}")
            except Exception as e:
                self.logger.warning(f"Error setting paths: {e}")
                settings_path_str = "./"
                output_path_str = "./"
                self.logger.warning(f"Could not calculate relative paths: {e}. Falling back to absolute.")
                settings_path_str = str(settings_dir)
                if not settings_path_str.endswith('/'):
                    settings_path_str += '/'
                output_path_str = str(output_dir)
                if not output_path_str.endswith('/'):
                    output_path_str += '/'

            # Get input path from config (forcing directory)
            if use_local_input:
                input_path_str = "./"
            else:
                input_path_str = None
                if config:
                    data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                    domain_name = config.get('DOMAIN_NAME', '')
                    project_dir = data_dir / f"domain_{domain_name}"
                    fuse_input_dir = project_dir / 'forcing' / 'FUSE_input'
                    if fuse_input_dir.exists():
                        input_path_str = str(fuse_input_dir)
                        if not input_path_str.endswith('/'):
                            input_path_str += '/'

            # Get simulation dates - prefer actual forcing file dates over config
            sim_start = None
            sim_end = None
            eval_start = None
            eval_end = None

            if config:
                # First, try to read actual dates from forcing file
                # This handles the case where daily resampling shifts the start date
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                domain_name = config.get('DOMAIN_NAME', '')
                project_dir = data_dir / f"domain_{domain_name}"
                forcing_file = project_dir / 'forcing' / 'FUSE_input' / f"{domain_name}_input.nc"

                if forcing_file.exists():
                    try:
                        import xarray as xr
                        import pandas as pd
                        import numpy as np
                        with xr.open_dataset(forcing_file) as ds:
                            time_vals = ds['time'].values
                            if len(time_vals) > 0:
                                # Check type of time values
                                first_val = time_vals[0]
                                last_val = time_vals[-1]

                                # Check if datetime64 type
                                if np.issubdtype(type(first_val), np.datetime64):
                                    # Already datetime64
                                    forcing_start = pd.Timestamp(first_val)
                                    forcing_end = pd.Timestamp(last_val)
                                elif isinstance(first_val, (int, float, np.integer, np.floating)):
                                    # Numeric - assume 'days since 1970-01-01'
                                    forcing_start = pd.Timestamp('1970-01-01') + pd.Timedelta(days=float(first_val))
                                    forcing_end = pd.Timestamp('1970-01-01') + pd.Timedelta(days=float(last_val))
                                else:
                                    # Try direct conversion
                                    forcing_start = pd.Timestamp(first_val)
                                    forcing_end = pd.Timestamp(last_val)

                                sim_start = forcing_start.strftime('%Y-%m-%d')
                                sim_end = forcing_end.strftime('%Y-%m-%d')
                                self.logger.debug(f"Using forcing file dates: {sim_start} to {sim_end}")
                    except Exception as e:
                        self.logger.warning(f"Could not read forcing file dates: {e}")

                # Fallback to config dates if forcing file not available
                if sim_start is None:
                    exp_start = config.get('EXPERIMENT_TIME_START', '')
                    if exp_start:
                        sim_start = str(exp_start).split()[0]
                if sim_end is None:
                    exp_end = config.get('EXPERIMENT_TIME_END', '')
                    if exp_end:
                        sim_end = str(exp_end).split()[0]

                # Calibration period from config
                calib_period = config.get('CALIBRATION_PERIOD', '')
                if calib_period and ',' in str(calib_period):
                    parts = str(calib_period).split(',')
                    eval_start = parts[0].strip()
                    eval_end = parts[1].strip()

            for line in lines:
                stripped = line.strip()
                # Only match actual path lines (start with quote), not comment lines
                if stripped.startswith("'") and 'SETNGS_PATH' in line:
                    # Replace path inside single quotes
                    updated_lines.append(f"'{settings_path_str}'     ! SETNGS_PATH\n")
                elif stripped.startswith("'") and 'INPUT_PATH' in line:
                    if input_path_str:
                        updated_lines.append(f"'{input_path_str}'        ! INPUT_PATH\n")
                    else:
                        updated_lines.append(line)  # Keep original if not found
                elif stripped.startswith("'") and 'OUTPUT_PATH' in line:
                    updated_lines.append(f"'{output_path_str}'       ! OUTPUT_PATH\n")
                elif stripped.startswith("'") and 'M_DECISIONS' in line:
                    # Use the passed decisions_file if provided, otherwise try to find one
                    actual_decisions = decisions_file
                    if not actual_decisions:
                        actual_decisions = f"fuse_zDecisions_{experiment_id}.txt"
                        # Check if file exists, if not use what's available
                        if not (execution_cwd / actual_decisions).exists():
                            # Find any decisions file
                            found_files = list(execution_cwd.glob('fuse_zDecisions_*.txt'))
                            if found_files:
                                actual_decisions = found_files[0].name
                                self.logger.debug(f"Using available decisions file: {actual_decisions}")
                    updated_lines.append(f"'{actual_decisions}'        ! M_DECISIONS        = definition of model decisions\n")
                elif stripped.startswith("'") and 'FMODEL_ID' in line:
                    # Update FMODEL_ID to match fuse_id (used in output filename)
                    updated_lines.append(f"'{fuse_id}'                            ! FMODEL_ID          = string defining FUSE model, only used to name output files\n")
                elif stripped.startswith("'") and 'FORCING INFO' in line:
                    # Ensure input_info.txt doesn't have trailing spaces
                    updated_lines.append("'input_info.txt'                 ! FORCING INFO       = definition of the forcing file\n")
                elif stripped.startswith("'") and 'date_start_sim' in line and sim_start:
                    updated_lines.append(f"'{sim_start}'                     ! date_start_sim     = date start simulation\n")
                elif stripped.startswith("'") and 'date_end_sim' in line and sim_end:
                    updated_lines.append(f"'{sim_end}'                     ! date_end_sim       = date end simulation\n")
                elif stripped.startswith("'") and 'date_start_eval' in line and eval_start:
                    updated_lines.append(f"'{eval_start}'                     ! date_start_eval    = date start evaluation period\n")
                elif stripped.startswith("'") and 'date_end_eval' in line and eval_end:
                    updated_lines.append(f"'{eval_end}'                     ! date_end_eval      = date end evaluation period\n")
                else:
                    updated_lines.append(line)

            with open(filemanager_path, 'w') as f:
                f.writelines(updated_lines)

            self.logger.debug(f"Updated file manager: decisions={experiment_id}, fmodel_id={fuse_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update FUSE file manager: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from FUSE output.

        Args:
            output_dir: Directory containing model outputs (not used directly)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            import xarray as xr
            import pandas as pd

            # Get paths
            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"

            # Read observed streamflow
            obs_file_path = config.get('OBSERVATIONS_PATH', 'default')
            if obs_file_path == 'default':
                obs_file_path = (project_dir / 'observations' / 'streamflow' / 'preprocessed' /
                                f"{domain_name}_streamflow_processed.csv")
            else:
                obs_file_path = Path(obs_file_path)

            if not obs_file_path.exists():
                self.logger.error(f"Observation file not found: {obs_file_path}")
                return {'kge': self.penalty_score}

            # Read observations
            df_obs = pd.read_csv(obs_file_path, index_col='datetime', parse_dates=True, dayfirst=True)

            # Ensure DatetimeIndex for resampling (fallback if parse_dates failed)
            if not isinstance(df_obs.index, pd.DatetimeIndex):
                try:
                    df_obs.index = pd.to_datetime(df_obs.index)
                    self.logger.debug("Converted observation index to DatetimeIndex")
                except Exception as e:
                    self.logger.error(f"Failed to convert observation time index to DatetimeIndex: {e}")
                    return {'kge': self.penalty_score}

            observed_streamflow = df_obs['discharge_cms'].resample('D').mean()

            # Check if routing was used - prioritize routed output over direct FUSE output
            mizuroute_dir = kwargs.get('mizuroute_dir')
            proc_id = kwargs.get('proc_id', 0)
            use_routed_output = False

            if mizuroute_dir and Path(mizuroute_dir).exists():
                mizuroute_dir = Path(mizuroute_dir)
                # Look for mizuRoute output - mizuRoute uses {case_name}.h.{start_time}.nc pattern
                # For parallel calibration, case_name = proc_{proc_id:02d}_{experiment_id}
                case_name = f"proc_{proc_id:02d}_{experiment_id}"

                # Try to find the output file using glob pattern
                mizuroute_dir / f"{case_name}.h.*.nc"
                mizu_output_files = list(mizuroute_dir.glob(f"{case_name}.h.*.nc"))

                if mizu_output_files:
                    # Sort by file size (largest first) to get file with actual data
                    # Empty/incomplete files will be smaller
                    mizu_output_files.sort(key=lambda f: f.stat().st_size, reverse=True)
                    sim_file_path = mizu_output_files[0]
                    use_routed_output = True
                    self.logger.debug(f"Using mizuRoute output for metrics calculation: {sim_file_path} (size: {sim_file_path.stat().st_size} bytes)")
                else:
                    # Fallback to non-prefixed pattern (for backward compatibility / default runs)
                    mizu_output_files_fallback = list(mizuroute_dir.glob(f"{experiment_id}.h.*.nc"))
                    if mizu_output_files_fallback:
                        # Sort by file size to get file with actual data
                        mizu_output_files_fallback.sort(key=lambda f: f.stat().st_size, reverse=True)
                        sim_file_path = mizu_output_files_fallback[0]
                        use_routed_output = True
                        self.logger.debug(f"Using mizuRoute output for metrics calculation: {sim_file_path} (size: {sim_file_path.stat().st_size} bytes)")
                    else:
                        # Also try the older timestep naming convention
                        old_pattern = mizuroute_dir / f"{experiment_id}_timestep.nc"
                        if old_pattern.exists():
                            sim_file_path = old_pattern
                            use_routed_output = True
                            self.logger.debug(f"Using mizuRoute output for metrics calculation: {sim_file_path}")

            # If no routed output, use FUSE output
            if not use_routed_output:
                # Read FUSE simulation output from sim_dir (or fallback to output_dir)
                # sim_dir = process_X/simulations/run_1/FUSE (consistent with SUMMA structure)
                fuse_id = config.get('FUSE_FILE_ID', experiment_id)
                fuse_output_dir = kwargs.get('sim_dir', output_dir)
                if fuse_output_dir:
                    fuse_output_dir = Path(fuse_output_dir)
                else:
                    fuse_output_dir = output_dir

                # FUSE runs in 'run_pre' mode for calibration (reads para_def.nc without regenerating it)
                # Output is renamed to runs_def.nc for consistency. Try runs_def first, then other files
                sim_file_path = None
                candidates = [
                    fuse_output_dir / f"{domain_name}_{fuse_id}_runs_def.nc",   # run_def mode (default)
                    fuse_output_dir / f"{domain_name}_{fuse_id}_runs_best.nc",  # run_best mode
                    fuse_output_dir / f"{domain_name}_{fuse_id}_runs_pre.nc",   # run_pre mode (legacy)
                    fuse_output_dir.parent / f"{domain_name}_{fuse_id}_runs_def.nc",
                    output_dir.parent / f"{domain_name}_{fuse_id}_runs_pre.nc",
                ]
                for cand in candidates:
                    if cand.exists():
                        sim_file_path = cand
                        break

                if sim_file_path is None or not sim_file_path.exists():
                    self.logger.error(f"Simulation file not found. Searched: {[str(c) for c in candidates]}")
                    return {'kge': self.penalty_score}

                self.logger.debug("Using FUSE output for metrics calculation")

            # Check for multi-gauge calibration mode
            multi_gauge_enabled = config.get('MULTI_GAUGE_CALIBRATION', False)
            if multi_gauge_enabled and use_routed_output:
                # Remove project_dir from kwargs if present (already passed explicitly)
                kwargs_clean = {k: v for k, v in kwargs.items() if k != 'project_dir'}
                return self._calculate_multi_gauge_metrics(
                    config=config,
                    mizuroute_output_path=sim_file_path,
                    project_dir=project_dir,
                    **kwargs_clean
                )

            # Read simulations
            # Explicitly decode times to ensure proper DatetimeIndex conversion
            with xr.open_dataset(sim_file_path, decode_times=True, decode_timedelta=True) as ds:
                if use_routed_output:
                    # mizuRoute output is already in m³/s
                    # Get the segment index for the configured reach ID
                    sim_reach_id = config.get('SIM_REACH_ID')
                    seg_idx = 0  # Default to first segment

                    if sim_reach_id is not None and sim_reach_id != 'default':
                        # Find the segment index matching the reach ID
                        if 'reachID' in ds.variables:
                            reach_ids = ds['reachID'].values
                            matches = np.where(reach_ids == int(sim_reach_id))[0]
                            if len(matches) > 0:
                                seg_idx = int(matches[0])
                                self.logger.debug(f"Using segment index {seg_idx} for reach ID {sim_reach_id}")
                            else:
                                self.logger.warning(f"Reach ID {sim_reach_id} not found in mizuRoute output, using segment 0")
                        else:
                            self.logger.warning("No reachID variable in mizuRoute output, using segment 0")

                    if 'IRFroutedRunoff' in ds.variables:
                        simulated = ds['IRFroutedRunoff'].isel(seg=seg_idx)
                    elif 'dlayRunoff' in ds.variables:
                        simulated = ds['dlayRunoff'].isel(seg=seg_idx)
                    else:
                        self.logger.error(f"No routed runoff variable in mizuRoute output. Variables: {list(ds.variables.keys())}")
                        return {'kge': self.penalty_score}

                    simulated_streamflow = simulated.to_pandas()

                    # Ensure DatetimeIndex for resampling (fallback if xarray decoding failed)
                    if not isinstance(simulated_streamflow.index, pd.DatetimeIndex):
                        simulated_streamflow.index = pd.to_datetime(simulated_streamflow.index)

                    # mizuRoute output is already in m³/s, no conversion needed
                    # Just resample to daily if needed
                    simulated_streamflow = simulated_streamflow.resample('D').mean()

                else:
                    # FUSE dimensions: (time, latitude, longitude) or (time, param_set, latitude, longitude)
                    # In distributed mode, subcatchments can be on latitude or longitude dimension
                    spatial_mode = config.get('FUSE_SPATIAL_MODE', 'lumped')
                    subcatchment_dim = config.get('FUSE_SUBCATCHMENT_DIM', 'latitude')

                    if 'q_routed' in ds.variables:
                        runoff_var = ds['q_routed']
                        var_name = 'q_routed'
                    elif 'q_instnt' in ds.variables:
                        runoff_var = ds['q_instnt']
                        var_name = 'q_instnt'
                    else:
                        self.logger.error(f"No runoff variable found in FUSE output. Variables: {list(ds.variables.keys())}")
                        return {'kge': self.penalty_score}

                    # Log actual dimensions for debugging
                    self.logger.debug(f"FUSE output dimensions: {runoff_var.dims}, sizes: {dict(runoff_var.sizes)}")

                    # Diagnostic: Log raw FUSE output statistics
                    raw_mean = float(runoff_var.mean())
                    raw_max = float(runoff_var.max())
                    if raw_mean < 1e-10 and raw_max < 1e-10:
                        self.logger.warning(
                            f"FUSE output {var_name} is all zeros! Raw mean={raw_mean:.6f}, max={raw_max:.6f}. "
                            f"This may indicate FUSE is not reading calibration parameters correctly."
                        )

                    # Determine which dimension has the subcatchments
                    has_param_set = 'param_set' in runoff_var.dims
                    n_subcatchments = runoff_var.sizes.get(subcatchment_dim, 1)

                    # Handle distributed mode: sum across all subcatchments
                    # FUSE output in mm/day represents depth over each subcatchment
                    if spatial_mode == 'distributed' and n_subcatchments > 1:
                        # For distributed mode without routing, we need to aggregate subcatchments
                        # Sum the volumetric runoff (convert each subcatchment from mm/day to m3/s, then sum)
                        self.logger.debug(f"Distributed mode: aggregating {n_subcatchments} subcatchments on '{subcatchment_dim}' dimension")

                        # Get individual subcatchment areas if available, otherwise assume equal distribution
                        total_area_km2 = self._get_catchment_area(config, project_dir)

                        # Select indices for non-subcatchment dimensions
                        isel_kwargs = {}
                        if has_param_set:
                            # Find the param_set with valid (non-NaN) data
                            n_param_sets = runoff_var.sizes.get('param_set', 1)
                            valid_param_set = 0
                            for ps in range(n_param_sets):
                                test_vals = runoff_var.isel(param_set=ps).values
                                if not np.all(np.isnan(test_vals)):
                                    valid_param_set = ps
                                    break
                            isel_kwargs['param_set'] = valid_param_set
                        # Select index 0 for the non-subcatchment spatial dimension
                        other_spatial_dim = 'latitude' if subcatchment_dim == 'longitude' else 'longitude'
                        if other_spatial_dim in runoff_var.dims:
                            isel_kwargs[other_spatial_dim] = 0

                        runoff_selected = runoff_var.isel(**isel_kwargs) if isel_kwargs else runoff_var

                        # For equal-area subcatchments: total flow = sum of individual flows
                        # Each subcatchment's mm/day * (total_area/n_subcatchments) / 86.4 gives m3/s
                        # Sum gives total m3/s
                        subcatchment_area = total_area_km2 / n_subcatchments

                        # Convert each subcatchment to m3/s and sum
                        simulated_cms = (runoff_selected * subcatchment_area / UnitConversion.MM_DAY_TO_CMS).sum(dim=subcatchment_dim)
                        simulated_streamflow = simulated_cms.to_pandas()
                        self.logger.debug(f"Aggregated distributed output: mean flow = {simulated_streamflow.mean():.2f} m³/s")
                    else:
                        # Lumped mode or single subcatchment
                        isel_kwargs = {}
                        if has_param_set:
                            # Find the param_set with valid (non-NaN) data
                            # FUSE run_def writes to param_set 1, not 0
                            n_param_sets = runoff_var.sizes.get('param_set', 1)
                            valid_param_set = 0
                            for ps in range(n_param_sets):
                                test_vals = runoff_var.isel(param_set=ps, latitude=0, longitude=0).values
                                if not np.all(np.isnan(test_vals)):
                                    valid_param_set = ps
                                    break
                            isel_kwargs['param_set'] = valid_param_set
                            self.logger.debug(f"Using param_set {valid_param_set} (has valid data)")
                        if 'latitude' in runoff_var.dims:
                            isel_kwargs['latitude'] = 0
                        if 'longitude' in runoff_var.dims:
                            isel_kwargs['longitude'] = 0

                        simulated = runoff_var.isel(**isel_kwargs) if isel_kwargs else runoff_var
                        simulated_streamflow = simulated.to_pandas()

                        # Get catchment area for unit conversion
                        area_km2 = self._get_catchment_area(config, project_dir)

                        # DEBUG: Log raw values before conversion
                        self.logger.debug(
                            f"DEBUG: raw sim mean={simulated_streamflow.mean():.4f} mm/day, "
                            f"area={area_km2:.2f} km2, sim_file={sim_file_path}"
                        )

                        # Convert FUSE output from mm/day to cms
                        # Q(cms) = Q(mm/day) * Area(km2) / 86.4
                        simulated_streamflow = simulated_streamflow * area_km2 / UnitConversion.MM_DAY_TO_CMS
                        self.logger.debug(f"DEBUG: converted sim mean={simulated_streamflow.mean():.4f} m3/s")

            # Ensure simulated_streamflow has a DatetimeIndex (fallback if xarray decoding failed)
            if not isinstance(simulated_streamflow.index, pd.DatetimeIndex):
                try:
                    simulated_streamflow.index = pd.to_datetime(simulated_streamflow.index)
                    self.logger.debug("Converted simulated streamflow index to DatetimeIndex")
                except Exception as e:
                    self.logger.error(f"Failed to convert time index to DatetimeIndex: {e}")
                    return {'kge': self.penalty_score}

            # Align time series
            common_index = observed_streamflow.index.intersection(simulated_streamflow.index)
            if len(common_index) == 0:
                self.logger.error("No overlapping time period")
                return {'kge': self.penalty_score}

            obs_aligned = observed_streamflow.loc[common_index].dropna()
            sim_aligned = simulated_streamflow.loc[common_index].dropna()

            # Filter to calibration period if specified
            calib_period = config.get('CALIBRATION_PERIOD', '')
            if calib_period and ',' in str(calib_period):
                try:
                    calib_start, calib_end = [s.strip() for s in str(calib_period).split(',')]
                    calib_start = pd.Timestamp(calib_start)
                    calib_end = pd.Timestamp(calib_end)

                    # Filter to calibration period
                    mask_obs = (obs_aligned.index >= calib_start) & (obs_aligned.index <= calib_end)
                    mask_sim = (sim_aligned.index >= calib_start) & (sim_aligned.index <= calib_end)

                    obs_aligned = obs_aligned[mask_obs]
                    sim_aligned = sim_aligned[mask_sim]

                    self.logger.debug(f"Filtered to calibration period {calib_start} to {calib_end}: {len(obs_aligned)} points")
                except Exception as e:
                    self.logger.warning(f"Could not parse calibration period '{calib_period}': {e}")

            common_index = obs_aligned.index.intersection(sim_aligned.index)
            obs_values = obs_aligned.loc[common_index].values
            sim_values = sim_aligned.loc[common_index].values

            if len(obs_values) == 0:
                self.logger.error("No valid data points")
                return {'kge': self.penalty_score}

            # Calculate metrics using shared utility
            metrics = self._streamflow_metrics.calculate_metrics(
                obs_values, sim_values, metrics=['kge', 'nse', 'rmse', 'mae']
            )

            # Debug: Log computed metrics and diagnostic info to trace score flow
            self.logger.debug(
                f"FUSE metrics: KGE={metrics['kge']:.4f}, NSE={metrics['nse']:.4f}, "
                f"n_pts={len(obs_values)}, sim_mean={sim_values.mean():.2f}, obs_mean={obs_values.mean():.2f}"
            )

            return metrics

        except FileNotFoundError as e:
            self.logger.error(f"Output or observation file not found: {e}")
            return {'kge': self.penalty_score}
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data error calculating FUSE metrics: {e}")
            return {'kge': self.penalty_score}
        except (ImportError, OSError) as e:
            # Catch xarray/pandas import issues or I/O errors
            self.logger.error(f"Error calculating FUSE metrics: {e}")
            return {'kge': self.penalty_score}

    def _get_catchment_area(self, config: Dict[str, Any], project_dir: Path) -> float:
        """
        Get catchment area for FUSE unit conversion. Delegates to shared utility.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path

        Returns:
            Catchment area in km2
        """
        domain_name = config.get('DOMAIN_NAME')
        return self._streamflow_metrics.get_catchment_area(config, project_dir, domain_name)

    def _calculate_multi_gauge_metrics(
        self,
        config: Dict[str, Any],
        mizuroute_output_path: Path,
        project_dir: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics across multiple stream gauges.

        This method is used when MULTI_GAUGE_CALIBRATION is enabled in the config.
        It extracts simulated streamflow at multiple gauge locations from mizuRoute
        output and calculates aggregated KGE across all gauges.

        Args:
            config: Configuration dictionary
            mizuroute_output_path: Path to mizuRoute output NetCDF
            project_dir: Project directory path
            **kwargs: Additional arguments (settings_dir, etc.)

        Returns:
            Dictionary with aggregated metrics and per-gauge details
        """
        try:
            # Get multi-gauge configuration
            gauge_mapping_path = config.get('GAUGE_SEGMENT_MAPPING')
            obs_dir = config.get('MULTI_GAUGE_OBS_DIR')
            gauge_ids = config.get('MULTI_GAUGE_IDS')  # None means all gauges
            exclude_ids = config.get('MULTI_GAUGE_EXCLUDE_IDS', [])  # Gauges to exclude
            aggregation = config.get('MULTI_GAUGE_AGGREGATION', 'mean')
            min_gauges = config.get('MULTI_GAUGE_MIN_GAUGES', 5)

            if not gauge_mapping_path:
                self.logger.error("GAUGE_SEGMENT_MAPPING not configured for multi-gauge calibration")
                return {'kge': self.penalty_score}

            if not obs_dir:
                self.logger.error("MULTI_GAUGE_OBS_DIR not configured for multi-gauge calibration")
                return {'kge': self.penalty_score}

            gauge_mapping_path = Path(gauge_mapping_path)
            obs_dir = Path(obs_dir)

            # Get topology file for segment lookup
            settings_dir = kwargs.get('settings_dir')
            topology_path = None
            if settings_dir:
                settings_dir = Path(settings_dir)
                # Check for topology file in mizuRoute settings
                mizu_settings = settings_dir / 'mizuRoute'
                if mizu_settings.exists():
                    topology_candidates = [
                        mizu_settings / 'topology.nc',
                        mizu_settings / 'network_topology.nc',
                    ]
                    for t in topology_candidates:
                        if t.exists():
                            topology_path = t
                            break

            # Get calibration period
            calib_period = config.get('CALIBRATION_PERIOD', '')
            start_date = None
            end_date = None
            if calib_period and ',' in str(calib_period):
                try:
                    start_date, end_date = [s.strip() for s in str(calib_period).split(',')]
                except (ValueError, AttributeError):
                    pass

            # Create multi-gauge metrics calculator
            multi_gauge = MultiGaugeMetrics(
                gauge_segment_mapping_path=gauge_mapping_path,
                obs_data_dir=obs_dir,
                logger=self.logger
            )

            # Apply exclusion list if gauge_ids not explicitly set
            if gauge_ids is None and exclude_ids:
                all_gauge_ids = multi_gauge.gauge_mapping['id'].tolist()
                gauge_ids = [gid for gid in all_gauge_ids if gid not in exclude_ids]
                self.logger.debug(f"Excluded {len(exclude_ids)} gauges, using {len(gauge_ids)} gauges")

            # Build quality filter config from config keys
            filter_config = {}
            max_dist = config.get('MULTI_GAUGE_MAX_DISTANCE')
            if max_dist is not None:
                filter_config['max_distance'] = float(max_dist)
            min_cv = config.get('MULTI_GAUGE_MIN_OBS_CV')
            if min_cv is not None:
                filter_config['min_obs_cv'] = float(min_cv)
            min_sq = config.get('MULTI_GAUGE_MIN_SPECIFIC_Q')
            if min_sq is not None:
                filter_config['min_specific_q'] = float(min_sq)

            # Minimum overlap days (gauges with less are skipped as invalid)
            min_overlap = int(config.get('MULTI_GAUGE_MIN_OVERLAP_DAYS', 10))

            # KGE floor: cap negative KGE values before aggregation to prevent
            # structurally unfittable gauges from dominating the objective function
            kge_floor = config.get('MULTI_GAUGE_KGE_FLOOR')
            if kge_floor is not None:
                kge_floor = float(kge_floor)

            # Calculate metrics across all gauges
            results = multi_gauge.calculate_multi_gauge_metrics(
                mizuroute_output_path=mizuroute_output_path,
                gauge_ids=gauge_ids,
                start_date=start_date,
                end_date=end_date,
                topology_path=topology_path,
                min_gauges=min_gauges,
                aggregation=aggregation,
                filter_config=filter_config if filter_config else None,
                min_overlap_days=min_overlap,
                kge_floor=kge_floor
            )

            # Log summary
            self.logger.debug(
                f"Multi-gauge calibration: KGE={results['kge']:.4f} "
                f"({results['n_valid_gauges']}/{results['n_total_gauges']} valid gauges)"
            )

            # Return metrics in expected format
            return {
                'kge': results['kge'],
                'kge_std': results.get('kge_std', 0.0),
                'kge_min': results.get('kge_min', results['kge']),
                'kge_max': results.get('kge_max', results['kge']),
                'n_gauges': results['n_valid_gauges'],
                'multi_gauge_details': results.get('per_gauge', {})
            }

        except FileNotFoundError as e:
            self.logger.error(f"Multi-gauge file not found: {e}")
            return {'kge': self.penalty_score}
        except Exception as e:
            self.logger.error(f"Error in multi-gauge metrics calculation: {e}")
            return {'kge': self.penalty_score}

    def _convert_fuse_to_mizuroute_format(
        self,
        fuse_output_dir: Path,
        config: Dict[str, Any],
        settings_dir: Path,
        proc_id: int = 0
    ) -> bool:
        """
        Convert FUSE distributed output to mizuRoute-compatible format.

        Delegates to FuseToMizurouteConverter for the actual conversion.

        Args:
            fuse_output_dir: Directory containing FUSE output
            config: Configuration dictionary
            settings_dir: Settings directory (unused, kept for API compatibility)
            proc_id: Process ID for parallel calibration (used in filename)

        Returns:
            True if conversion successful
        """
        # Use instance logger for the converter
        converter = FuseToMizurouteConverter(logger=self.logger)
        return converter.convert(fuse_output_dir, config, proc_id)

    def _run_mizuroute_for_fuse(
        self,
        config: Dict[str, Any],
        fuse_output_dir: Path,
        mizuroute_dir: Path,
        **kwargs
    ) -> bool:
        """
        Execute mizuRoute for FUSE output.

        Args:
            config: Configuration dictionary
            fuse_output_dir: Directory containing FUSE output
            mizuroute_dir: Output directory for mizuRoute
            **kwargs: Additional arguments (settings_dir, sim_dir, etc.)

        Returns:
            True if mizuRoute ran successfully
        """
        try:
            import subprocess

            # Get mizuRoute executable
            mizuroute_install = config.get('MIZUROUTE_INSTALL_PATH', 'default')
            if mizuroute_install == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                mizuroute_exe = data_dir / 'installs' / 'mizuRoute' / 'route' / 'bin' / 'mizuRoute.exe'
            else:
                mizuroute_exe = Path(mizuroute_install) / 'mizuRoute.exe'

            if not mizuroute_exe.exists():
                self.logger.error(f"mizuRoute executable not found: {mizuroute_exe}")
                return False

            # Get process-specific control file
            # The optimizer should have copied and configured mizuRoute settings
            # to the process-specific settings directory
            # settings_dir structure: .../process_N/settings/FUSE/
            # mizuRoute settings are at: .../process_N/settings/mizuRoute/

            # Try to get from kwargs first (set by BaseModelOptimizer)
            mizuroute_settings_dir = kwargs.get('mizuroute_settings_dir')
            if mizuroute_settings_dir:
                control_file = Path(mizuroute_settings_dir) / 'mizuroute.control'
            else:
                settings_dir_path = Path(kwargs.get('settings_dir', Path('.')))
                # Check both in settings_dir and settings_dir.parent to handle FUSE subdirectory
                control_file = settings_dir_path / 'mizuRoute' / 'mizuroute.control'
                if not control_file.exists() and settings_dir_path.name == 'FUSE':
                    control_file = settings_dir_path.parent / 'mizuRoute' / 'mizuroute.control'

            # Fallback to main control file (default runs)
            if not control_file or not control_file.exists():
                domain_name = config.get('DOMAIN_NAME')
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                project_dir = data_dir / f"domain_{domain_name}"
                control_file = project_dir / 'settings' / 'mizuRoute' / 'mizuroute.control'

            if not control_file.exists():
                self.logger.error(f"mizuRoute control file not found: {control_file}")
                return False

            self.logger.debug(f"Using mizuRoute control file: {control_file}")

            # Execute mizuRoute
            cmd = [str(mizuroute_exe), str(control_file)]

            self.logger.debug(f"Executing mizuRoute: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace undecodable bytes to prevent UnicodeDecodeError
                timeout=config.get('MIZUROUTE_TIMEOUT', 600)
            )

            if result.returncode != 0:
                self.logger.error(f"mizuRoute failed with return code {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False

            # Log stdout on success at debug level
            if result.stdout:
                stdout_lines = result.stdout.strip().split('\n')
                self.logger.debug(f"mizuRoute completed, stdout lines: {len(stdout_lines)}")
                if len(stdout_lines) <= 20:
                    for line in stdout_lines:
                        if line.strip():
                            self.logger.debug(f"  mizuRoute: {line}")
                else:
                    for line in stdout_lines[:5]:
                        if line.strip():
                            self.logger.debug(f"  mizuRoute: {line}")
                    self.logger.debug(f"  ... ({len(stdout_lines) - 10} more lines) ...")
                    for line in stdout_lines[-5:]:
                        if line.strip():
                            self.logger.debug(f"  mizuRoute: {line}")
            else:
                self.logger.debug("mizuRoute completed successfully (no stdout)")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error("mizuRoute execution timed out")
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Required file not found for mizuRoute: {e}")
            return False
        except (OSError, subprocess.SubprocessError) as e:
            self.logger.error(f"Error running mizuRoute: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_fuse_parameters_worker(task_data)


def _evaluate_fuse_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    worker = FUSEWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
