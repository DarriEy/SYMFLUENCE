"""
FUSE Runner Module

Refactored to use the Unified Model Execution Framework:
- UnifiedModelExecutor: Combined execution and spatial orchestration
"""

import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import xarray as xr

from ..base import BaseModelRunner
from ..mixins import OutputConverterMixin, SpatialModeDetectionMixin
from ..execution import UnifiedModelExecutor
from ..mizuroute.mixins import MizuRouteConfigMixin
from ..registry import ModelRegistry
from .subcatchment_processor import SubcatchmentProcessor
from symfluence.core.exceptions import (
    ModelExecutionError,
    symfluence_error_handler
)


@ModelRegistry.register_runner('FUSE', method_name='run_fuse')
class FUSERunner(BaseModelRunner, UnifiedModelExecutor, OutputConverterMixin, MizuRouteConfigMixin, SpatialModeDetectionMixin):  # type: ignore[misc]
    """
    Runner class for the FUSE (Framework for Understanding Structural Errors) model.
    Handles model execution, output processing, and file management.

    Now uses the Unified Model Execution Framework for:
    - Subprocess execution (via ModelExecutor)
    - Spatial mode handling and routing (via SpatialOrchestrator)
    - Output format conversion (via OutputConverterMixin)

    Attributes:
        config (Dict[str, Any]): Configuration settings for FUSE
        logger (Any): Logger object for recording run information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """

    def __init__(self, config, logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the FUSE runner.

        Sets up FUSE execution environment including spatial mode detection,
        routing requirements check, and lazy initialization of subcatchment
        processor for distributed runs.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                FUSE model settings, paths, and execution parameters.
            logger: Logger instance for status messages and debugging output.
            reporting_manager: Optional reporting manager for experiment tracking.

        Note:
            Uses Unified Model Execution Framework mixins for subprocess execution,
            spatial orchestration, output conversion, and mizuRoute integration.
        """
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

        # FUSE-specific initialization - determine spatial mode using mixin
        self.spatial_mode = self.detect_spatial_mode('FUSE')

        self.needs_routing = self._check_routing_requirements()
        self._subcatchment_processor = None

    @property
    def subcatchment_processor(self) -> SubcatchmentProcessor:
        """Lazy-loaded subcatchment processor for distributed runs."""
        if self._subcatchment_processor is None:
            self._subcatchment_processor = SubcatchmentProcessor(
                project_dir=self.project_dir,
                domain_name=self.domain_name,
                experiment_id=self.experiment_id,
                config_dict=self.config_dict,
                setup_dir=self.setup_dir,
                output_path=self.output_path,
                fuse_exe=self.fuse_exe,
                logger=self.logger,
                config=self.config  # Pass typed config
            )
        assert self._subcatchment_processor is not None
        return self._subcatchment_processor

    def _get_fuse_file_id(self) -> str:
        """Return a short file ID for FUSE outputs/settings."""
        fuse_id: str = self.config_dict.get('FUSE_FILE_ID', self.experiment_id)
        return fuse_id

    def _setup_model_specific_paths(self) -> None:
        """Set up FUSE-specific paths."""
        self.setup_dir = self.project_dir / "settings" / "FUSE"
        self.forcing_fuse_path = self.project_dir / 'forcing' / 'FUSE_input'

        # FUSE executable path (installation dir + exe name)
        self.fuse_exe = self.get_model_executable(
            install_path_key='FUSE_INSTALL_PATH',
            default_install_subpath='installs/fuse/bin',
            exe_name_key='FUSE_EXE',
            default_exe_name='fuse.exe',
            must_exist=True
        )
        self.output_path = self.get_config_path('EXPERIMENT_OUTPUT_FUSE', f"simulations/{self.experiment_id}/FUSE")

        # FUSE-specific: result_dir is an alias for output_dir (backward compatibility)
        self.output_dir = self.get_experiment_output_dir()
        self.setup_path_aliases({'result_dir': 'output_dir'})

    def _get_model_name(self) -> str:
        """Return model name for FUSE."""
        return "FUSE"

    def _get_output_dir(self) -> Path:
        """FUSE uses custom result_dir naming."""
        return self.get_experiment_output_dir()

    def _convert_fuse_distributed_to_mizuroute_format(self):
        """
        Convert FUSE spatial dimensions to mizuRoute format.

        Uses OutputConverterMixin for the core conversion:
        - Squeezes singleton latitude dimension
        - Renames longitude → gru
        - Adds gruId variable
        - Filters out coastal GRUs and sets gruId to match topology hruId
        """
        import pandas as pd

        experiment_id = self.experiment_id
        fuse_id = self._get_fuse_file_id()
        domain = self.domain_name

        fuse_out_dir = self.project_dir / "simulations" / experiment_id / "FUSE"

        # Find FUSE output file
        target_files = [
            fuse_out_dir / f"{domain}_{fuse_id}_runs_def.nc",
            fuse_out_dir / f"{domain}_{fuse_id}_runs_best.nc"
        ]

        target = None
        for file_path in target_files:
            if file_path.exists():
                target = file_path
                break

        if target is None:
            raise FileNotFoundError(f"FUSE output not found. Tried: {[str(f) for f in target_files]}")

        self.logger.debug(f"Converting FUSE spatial dimensions: {target}")

        # Use generic mixin method with FUSE-specific parameters
        self.convert_to_mizuroute_format(
            input_path=target,
            squeeze_dims=['latitude'],
            rename_dims={'longitude': 'gru'},
            add_id_var='gruId',
            id_source_dim='gru',
            create_backup=True
        )

        # Filter coastal GRUs using mapping file if available
        mapping_file = self.project_dir / 'settings' / 'mizuRoute' / 'fuse_to_routing_mapping.csv'
        if mapping_file.exists():
            ds = xr.open_dataset(target)
            ds = ds.load()
            ds.close()

            mapping = pd.read_csv(mapping_file)
            non_coastal = mapping[~mapping['is_coastal']]
            fuse_indices = non_coastal['fuse_gru_idx'].values
            gru_ids = non_coastal['gru_to_seg'].values.astype(np.int32)

            n_total = ds.sizes.get('gru', 0)
            n_keep = len(non_coastal)
            self.logger.debug(f"Filtering coastal GRUs: {n_total} total → {n_keep} non-coastal")

            ds = ds.isel(gru=fuse_indices)
            ds = ds.assign_coords(gru=np.arange(n_keep))
            ds['gruId'] = xr.DataArray(gru_ids, dims=['gru'],
                                       attrs={'long_name': 'gru identifier'})

            # Convert FUSE runoff from mm/day to m/s for mizuRoute
            routing_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
            routing_units = self.config_dict.get('SETTINGS_MIZU_ROUTING_UNITS', 'm/s')
            if routing_units == 'm/s' and routing_var in ds:
                fuse_timestep = int(self.config_dict.get('FUSE_OUTPUT_TIMESTEP_SECONDS', 86400))
                conversion_factor = 1.0 / (1000.0 * fuse_timestep)
                ds[routing_var] = ds[routing_var] * conversion_factor
                ds[routing_var].attrs['units'] = 'm/s'
                self.logger.debug(f"Converted {routing_var}: mm/day → m/s (factor={conversion_factor:.2e})")

            ds.to_netcdf(target)
            self.logger.debug(f"Saved filtered output with {n_keep} GRUs to {target}")
        else:
            self.logger.warning(f"Mapping file not found at {mapping_file}, skipping coastal GRU filtering")
            # Still need to convert units from mm/day to m/s for mizuRoute
            routing_var = self.config_dict.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
            routing_units = self.config_dict.get('SETTINGS_MIZU_ROUTING_UNITS', 'm/s')
            if routing_units == 'm/s':
                ds = xr.open_dataset(target)
                ds = ds.load()
                ds.close()
                if routing_var in ds:
                    fuse_timestep = int(self.config_dict.get('FUSE_OUTPUT_TIMESTEP_SECONDS', 86400))
                    conversion_factor = 1.0 / (1000.0 * fuse_timestep)
                    ds[routing_var] = ds[routing_var] * conversion_factor
                    ds[routing_var].attrs['units'] = 'm/s'
                    ds.to_netcdf(target)
                    self.logger.debug(f"Converted {routing_var}: mm/day → m/s (factor={conversion_factor:.2e})")

        # Ensure _runs_def.nc exists if we processed a different file
        def_file = fuse_out_dir / f"{domain}_{fuse_id}_runs_def.nc"
        if target != def_file and not def_file.exists():
            shutil.copy2(target, def_file)
            self.logger.info(f"Created runs_def file: {def_file}")

    def run_fuse(self) -> Optional[Path]:
        """
        Run FUSE model with distributed support.

        Returns:
            Path to output directory on success, None on failure

        Raises:
            ModelExecutionError: If model execution fails
        """
        self.logger.debug(f"Starting FUSE model run in {self.spatial_mode} mode")

        with symfluence_error_handler(
            "FUSE model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Create output directory
            self.output_path.mkdir(parents=True, exist_ok=True)

            # Run FUSE simulations
            success = self._execute_fuse_workflow()

            if success:
                # Handle routing if needed
                if self.needs_routing:
                    self._convert_fuse_distributed_to_mizuroute_format()
                    success = self._run_distributed_routing()

                if success:
                    self._process_outputs()
                    self.logger.debug("FUSE run completed successfully")
                    return self.output_path
                else:
                    self.logger.error("FUSE routing failed")
                    return None
            else:
                self.logger.error("FUSE simulation failed")
                return None

    def _check_routing_requirements(self) -> bool:
        """
        Check if distributed routing is needed for the current configuration.

        Determines whether mizuRoute should be executed after FUSE based on
        the spatial mode and routing integration settings. Routing is needed
        for distributed/semi-distributed modes or when routing delineation
        uses river_network in lumped mode.

        Returns:
            bool: True if mizuRoute routing should be executed, False otherwise.
        """
        routing_integration = self.config_dict.get('FUSE_ROUTING_INTEGRATION', 'none')

        if routing_integration == 'mizuRoute':
            if self.spatial_mode in ['semi_distributed', 'distributed']:
                return True
            elif self.spatial_mode == 'lumped' and self.config_dict.get('ROUTING_DELINEATION') == 'river_network':
                return True

        return False

    def _execute_fuse_workflow(self) -> bool:
        """
        Execute the main FUSE workflow based on spatial mode.

        Routes to either lumped or distributed execution workflow based on
        the configured spatial mode. Lumped mode runs a single catchment
        simulation while distributed mode processes the full multi-HRU dataset.

        Returns:
            bool: True if FUSE execution completed successfully, False otherwise.
        """
        if self.spatial_mode == 'lumped':
            # Original lumped workflow
            return self._run_lumped_fuse()
        else:
            # Distributed workflow
            return self._run_distributed_fuse()

    def _run_distributed_fuse(self) -> bool:
        """Run FUSE in distributed mode - always process the full dataset at once"""
        self.logger.debug("Running distributed FUSE workflow with full dataset")

        try:
            # Run FUSE once with the complete distributed forcing file
            return self._run_multidimensional_fuse()

        except (subprocess.CalledProcessError, OSError, FileNotFoundError) as e:
            self.logger.error(f"Error in distributed FUSE execution: {str(e)}")
            return False

    def _run_multidimensional_fuse(self) -> bool:
        """Run FUSE once with the full distributed forcing file"""

        try:
            self.logger.debug("Running FUSE with complete distributed forcing dataset")

            # Run FUSE with the distributed forcing file (all HRUs at once)
            success = self._execute_fuse_distributed()

            if success:
                self.logger.debug("Distributed FUSE run completed successfully")
                return True
            else:
                self.logger.error("Distributed FUSE run failed")
                return False

        except (subprocess.CalledProcessError, OSError) as e:
            self.logger.error(f"Error in multidimensional FUSE execution: {str(e)}")
            return False

    def _execute_fuse_distributed(self) -> bool:
        """Execute FUSE with the complete distributed forcing file"""

        try:
            # Use the main file manager (points to distributed forcing file)
            control_file = self.setup_dir / 'fm_catch.txt'

            # Run FUSE once for the entire distributed domain
            command = [
                str(self.fuse_exe),
                str(control_file),
                self.domain_name,  # Use original domain name
                "run_def"  # Run with default parameters
            ]

            # Create log file
            log_file = self.output_path / 'fuse_distributed_run.log'

            self.logger.debug(f"Executing distributed FUSE: {' '.join(command)}")

            with open(log_file, 'w', encoding='utf-8', errors='replace') as f:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    cwd=str(self.setup_dir)
                )

            if result.returncode == 0:
                self.logger.debug("Distributed FUSE execution completed successfully")
                return True
            else:
                self.logger.error(f"FUSE failed with return code {result.returncode}")
                return False

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FUSE execution failed: {str(e)}")
            return False
        except (OSError, FileNotFoundError) as e:
            self.logger.error(f"Error executing distributed FUSE: {str(e)}")
            return False


    def _create_subcatchment_settings(self, subcat_id: int, index: int) -> Path:
        """Create subcatchment-specific settings files. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.create_subcatchment_settings(subcat_id, index)

    def _execute_fuse_subcatchment(self, subcat_id: int, forcing_file: Path, settings_dir: Path) -> Optional[Path]:
        """Execute FUSE for a specific subcatchment. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.execute_fuse_subcatchment(subcat_id, forcing_file, settings_dir)

    def _ensure_best_output_file(self):
        """Ensure the expected 'best' output file exists by copying from 'def' output if needed"""
        fuse_id = self._get_fuse_file_id()
        def_file = self.output_path / f"{self.domain_name}_{fuse_id}_runs_def.nc"
        best_file = self.output_path / f"{self.domain_name}_{fuse_id}_runs_best.nc"

        if def_file.exists() and not best_file.exists():
            self.logger.info(f"Copying {def_file.name} to {best_file.name} for compatibility")
            shutil.copy2(def_file, best_file)

        return best_file if best_file.exists() else def_file

    def _extract_subcatchment_forcing(self, subcat_id: int, index: int) -> Path:
        """Extract forcing data for a specific subcatchment. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.extract_subcatchment_forcing(subcat_id, index)

    def _combine_subcatchment_outputs(self, outputs: List[Tuple[int, Path]]):
        """Combine outputs from all subcatchments. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.combine_subcatchment_outputs(outputs)

    def _load_subcatchment_info(self):
        """Load subcatchment information for distributed mode. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.load_subcatchment_info(self.catchment_name_col)

    def _run_individual_subcatchments(self, subcatchments) -> bool:
        """Run FUSE separately for each subcatchment. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.run_individual_subcatchments(subcatchments)

    def _create_subcatchment_elevation_bands(self, subcat_id: int) -> Path:
        """Create elevation bands file for a specific subcatchment. Delegates to SubcatchmentProcessor."""
        return self.subcatchment_processor.create_subcatchment_elevation_bands(subcat_id)

    def _run_distributed_routing(self) -> bool:
        """Run mizuRoute routing for distributed FUSE output.

        Uses SpatialOrchestrator._run_mizuroute() for unified routing integration.
        """
        self.logger.debug("Starting mizuRoute routing for distributed FUSE")

        # Update config for FUSE-mizuRoute integration
        self._setup_fuse_mizuroute_config()

        # Use orchestrator method (creates control file and runs mizuRoute)
        spatial_config = self.get_spatial_config('FUSE')
        result = self._run_mizuroute(spatial_config, model_name='fuse')

        return result is not None

    def _convert_fuse_to_mizuroute_format(self) -> bool:
        """
        Convert FUSE distributed output to the mizuRoute input format *in place*
        so it matches what the FUSE-specific mizu control file expects:
        - dims: (time, gru)
        - var:  <routing_var> = config['SETTINGS_MIZU_ROUTING_VAR']
        - id:   gruId (int)
        """
        try:
            # 1) Locate the FUSE output that the control file points to
            #    Control uses: <fname_qsim> DOMAIN_EXPERIMENT_runs_def.nc
            #    Prefer runs_def; fall back to runs_best if needed.
            out_dir = self.project_dir / "simulations" / self.experiment_id / "FUSE"
            fuse_id = self._get_fuse_file_id()
            base = f"{self.domain_name}_{fuse_id}"
            candidates = [
                out_dir / f"{base}_runs_def.nc",
                out_dir / f"{base}_runs_best.nc",
            ]
            fuse_output_file = next((p for p in candidates if p.exists()), None)
            if fuse_output_file is None:
                self.logger.error(f"FUSE output file not found. Tried: {candidates}")
                return False

            # 2) Open and convert
            with xr.open_dataset(fuse_output_file) as ds:
                mizu_ds = self._create_mizuroute_forcing_dataset(ds)

            # 3) Overwrite in place so mizuRoute reads exactly what control declares
            #    If the in-use file was runs_best, still write the converted data
            #    back to _runs_def.nc since that's what the control file names.
            write_target = out_dir / f"{base}_runs_def.nc"
            mizu_ds.to_netcdf(write_target, format="NETCDF4")
            self.logger.info(f"Converted FUSE output → mizuRoute format: {write_target}")
            return True

        except (FileNotFoundError, OSError, ValueError, KeyError) as e:
            self.logger.error(f"Error converting FUSE output: {e}")
            return False


    def _create_mizuroute_forcing_dataset(self, fuse_ds: xr.Dataset) -> xr.Dataset:
        """
        Build a mizuRoute-compatible dataset from distributed FUSE output.

        Transforms FUSE spatial output (latitude/longitude dimensions) to
        mizuRoute format (time, gru dimensions). Automatically detects which
        spatial coordinate holds multiple subcatchments and reshapes accordingly.

        Args:
            fuse_ds: FUSE output dataset with dimensions (time, latitude, longitude)
                where one spatial dimension contains subcatchment data.

        Returns:
            xr.Dataset: mizuRoute-compatible dataset with:
                - dims: (time, gru)
                - vars: routing variable (from SETTINGS_MIZU_ROUTING_VAR)
                - gruId: Integer GRU identifiers from spatial coordinates

        Raises:
            ModelExecutionError: If no suitable runoff variable found in FUSE output.
            ValueError: If spatial dimensions cannot be mapped to subcatchments.
        """
        # --- Choose runoff variable (prefer q_routed, else sensible fallbacks)
        routing_var_name = self.mizu_routing_var
        candidates = [
            'q_routed', 'q_instnt', 'qsim', 'runoff',
            # fallbacks by substring
            *[v for v in fuse_ds.data_vars if v.lower().startswith("q_")],
            *[v for v in fuse_ds.data_vars if "runoff" in v.lower()],
        ]
        runoff_src = next((v for v in candidates if v in fuse_ds.data_vars), None)
        if runoff_src is None:
            raise ModelExecutionError(f"No suitable runoff variable found in FUSE output. "
                            f"Available: {list(fuse_ds.data_vars)}")

        # --- Identify spatial axis (one of latitude/longitude must have length > 1)
        lat_len = fuse_ds.sizes.get('latitude', 0)
        lon_len = fuse_ds.sizes.get('longitude', 0)

        if lat_len > 1 and (lon_len in (0, 1)):
            # (time, latitude, 1)
            data = fuse_ds[runoff_src].squeeze('longitude', drop=True).transpose('time', 'latitude')
            spatial_name = 'latitude'
            ids = fuse_ds[spatial_name].values
        elif lon_len > 1 and (lat_len in (0, 1)):
            # (time, 1, longitude)
            data = fuse_ds[runoff_src].squeeze('latitude', drop=True).transpose('time', 'longitude')
            spatial_name = 'longitude'
            ids = fuse_ds[spatial_name].values
        else:
            # If both >1 (unlikely for your setup) or neither, fail loudly
            raise ValueError(f"Could not infer subcatchment axis from dims: {fuse_ds.dims}")

        # --- Rename spatial dimension to 'gru'
        data = data.rename({data.dims[1]: 'gru'})

        # --- Build output dataset
        mizu = xr.Dataset()
        # copy/forward the time coordinate as-is
        mizu['time'] = fuse_ds['time']
        mizu['time'].attrs.update(fuse_ds['time'].attrs)

        # Add gruId from the spatial coordinate; cast to int32 if possible
        try:
            gid = ids.astype('int32')
        except (ValueError, TypeError):
            gid = ids
        mizu['gru'] = xr.DataArray(range(data.sizes['gru']), dims=('gru',))
        mizu['gruId'] = xr.DataArray(gid, dims=('gru',), attrs={
            'long_name': 'ID of grouped response unit', 'units': '-'
        })

        # Ensure variable is named exactly as control expects
        if runoff_src != routing_var_name:
            data = data.rename(routing_var_name)
        mizu[routing_var_name] = data
        # Add/normalize attrs (units default to m/s unless overridden)
        units = self.mizu_routing_units
        mizu[routing_var_name].attrs.update({'long_name': 'FUSE runoff for mizuRoute routing',
                                            'units': units})

        # Preserve some useful globals if present
        mizu.attrs.update({k: v for k, v in fuse_ds.attrs.items()})

        return mizu


    def _setup_fuse_mizuroute_config(self):
        """Update configuration for FUSE-mizuRoute integration"""

        # Update input file name for mizuRoute
        self.config_dict['EXPERIMENT_ID_TEMP'] = self.experiment_id  # Backup

        # Set mizuRoute to look for FUSE output instead of SUMMA

    def _is_snow_optimization(self) -> bool:
        """Check if this is a snow optimization run by examining the forcing data."""
        try:
            # Check if q_obs contains only dummy values
            forcing_file = self.forcing_fuse_path / f"{self.domain_name}_input.nc"

            if forcing_file.exists():
                with xr.open_dataset(forcing_file) as ds:
                    if 'q_obs' in ds.variables:
                        q_obs_values = ds['q_obs'].values
                        # If all values are -9999 or very close to it, it's dummy data
                        if np.all(np.abs(q_obs_values + 9999) < 0.1):
                            return True

            # Also check optimization target from config
            optimization_target = self.config_dict.get('OPTIMIZATION_TARGET', 'streamflow')
            if optimization_target in ['swe', 'sca', 'snow_depth', 'snow']:
                return True

            return False

        except (FileNotFoundError, KeyError, ValueError) as e:
            self.logger.warning(f"Could not determine if snow optimization: {str(e)}")
            # Fall back to checking config
            optimization_target = self.config_dict.get('OPTIMIZATION_TARGET', 'streamflow')
            return optimization_target in ['swe', 'sca', 'snow_depth', 'snow']

    def _copy_default_to_best_params(self):
        """Copy default parameter file to best parameter file for snow optimization."""
        try:
            fuse_id = self._get_fuse_file_id()
            default_params = self.output_path / f"{self.domain_name}_{fuse_id}_para_def.nc"
            best_params = self.output_path / f"{self.domain_name}_{fuse_id}_para_sce.nc"

            if default_params.exists():
                # Use module-level shutil import (already imported at top of file)
                shutil.copy2(default_params, best_params)
                self.logger.info("Copied default parameters to best parameters file for snow optimization")
            else:
                self.logger.warning("Default parameter file not found - snow optimization may fail")

        except Exception as e:
            self.logger.error(f"Error copying default to best parameters: {str(e)}")

    def _add_elevation_params_to_constraints(self) -> bool:
        """
        Add elevation band parameters to the FUSE constraints file as FIXED parameters.

        FUSE reads its parameters from the constraints file. If elevation band
        parameters (N_BANDS, Z_FORCING, Z_MID01, AF01) are missing, FUSE defaults
        them to zeros which breaks snow aggregation - snowmelt never becomes
        effective precipitation.

        This method reads the elevation bands file and adds the parameters to
        the constraints file BEFORE running FUSE, so they are used correctly.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Read elevation bands file
            elev_bands_path = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
            if not elev_bands_path.exists():
                self.logger.warning(f"Elevation bands file not found: {elev_bands_path}")
                return False

            with xr.open_dataset(elev_bands_path) as eb_ds:
                n_bands = eb_ds.sizes.get('elevation_band', 1)
                mean_elevs = eb_ds['mean_elev'].values.flatten()
                area_fracs = eb_ds['area_frac'].values.flatten()
                z_forcing = float(np.sum(mean_elevs * area_fracs))

            # Find constraints file
            constraints_file = self.setup_dir / 'fuse_zConstraints_snow.txt'
            if not constraints_file.exists():
                self.logger.warning(f"Constraints file not found: {constraints_file}")
                return False

            # Read existing constraints
            with open(constraints_file, 'r') as f:
                lines = f.readlines()

            # Check if elevation params already exist
            existing_params = set()
            for line in lines:
                for param in ['N_BANDS', 'Z_FORCING', 'Z_MID01', 'AF01']:
                    if param in line and not line.strip().startswith('!'):
                        existing_params.add(param)

            if existing_params:
                self.logger.debug(f"Elevation params already in constraints: {existing_params}")
                # Update existing values instead of adding duplicates
                updated_lines = []
                for line in lines:
                    if 'N_BANDS' in line and not line.strip().startswith('!'):
                        line = f"F 0 {float(n_bands):9.3f} {float(n_bands):9.3f} {float(n_bands):9.3f} .10   1.0 0 0 0  0 0 0 N_BANDS   NO_CHILD1 NO_CHILD2 ! number of elevation bands\n"
                    elif 'Z_FORCING' in line and not line.strip().startswith('!'):
                        line = f"F 0 {z_forcing:9.3f} {z_forcing:9.3f} {z_forcing:9.3f} .10   1.0 0 0 0  0 0 0 Z_FORCING NO_CHILD1 NO_CHILD2 ! forcing elevation (m)\n"
                    elif 'Z_MID01' in line and not line.strip().startswith('!'):
                        line = f"F 0 {mean_elevs[0]:9.3f} {mean_elevs[0]:9.3f} {mean_elevs[0]:9.3f} .10   1.0 0 0 0  0 0 0 Z_MID01   NO_CHILD1 NO_CHILD2 ! band 1 elevation (m)\n"
                    elif 'AF01' in line and not line.strip().startswith('!'):
                        line = f"F 0 {area_fracs[0]:9.3f} {area_fracs[0]:9.3f} {area_fracs[0]:9.3f} .10   1.0 0 0 0  0 0 0 AF01      NO_CHILD1 NO_CHILD2 ! band 1 area fraction\n"
                    updated_lines.append(line)
                lines = updated_lines

            # Add elevation params if not present
            params_to_add = []
            if 'N_BANDS' not in existing_params:
                params_to_add.append(f"F 0 {float(n_bands):9.3f} {float(n_bands):9.3f} {float(n_bands):9.3f} .10   1.0 0 0 0  0 0 0 N_BANDS   NO_CHILD1 NO_CHILD2 ! number of elevation bands\n")
            if 'Z_FORCING' not in existing_params:
                params_to_add.append(f"F 0 {z_forcing:9.3f} {z_forcing:9.3f} {z_forcing:9.3f} .10   1.0 0 0 0  0 0 0 Z_FORCING NO_CHILD1 NO_CHILD2 ! forcing elevation (m)\n")
            if 'Z_MID01' not in existing_params:
                params_to_add.append(f"F 0 {mean_elevs[0]:9.3f} {mean_elevs[0]:9.3f} {mean_elevs[0]:9.3f} .10   1.0 0 0 0  0 0 0 Z_MID01   NO_CHILD1 NO_CHILD2 ! band 1 elevation (m)\n")
            if 'AF01' not in existing_params:
                params_to_add.append(f"F 0 {area_fracs[0]:9.3f} {area_fracs[0]:9.3f} {area_fracs[0]:9.3f} .10   1.0 0 0 0  0 0 0 AF01      NO_CHILD1 NO_CHILD2 ! band 1 area fraction\n")

            if params_to_add:
                # Find position to insert (before the description section)
                insert_pos = len(lines)
                for i, line in enumerate(lines):
                    if '*****' in line or 'description' in line.lower():
                        insert_pos = i
                        break

                # Insert the new parameters
                for param_line in reversed(params_to_add):
                    lines.insert(insert_pos, param_line)

            # Write back
            with open(constraints_file, 'w') as f:
                f.writelines(lines)

            self.logger.info(f"Added elevation params to constraints: N_BANDS={n_bands}, "
                           f"Z_FORCING={z_forcing:.1f}, Z_MID01={mean_elevs[0]:.1f}, AF01={area_fracs[0]:.3f}")
            return True

        except (FileNotFoundError, OSError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to add elevation params to constraints: {e}")
            return False

    def _fix_elevation_band_params_in_para_def(self) -> bool:
        """
        Fix elevation band parameters in para_def.nc after FUSE run_def creates it.

        FUSE's run_def mode generates para_def.nc with zeros for elevation band
        parameters (N_BANDS, Z_FORCING, Z_MID01, AF01), which causes snow
        aggregation to fail - snowmelt never becomes effective precipitation.

        This method reads the elevation bands file and updates para_def.nc with
        the correct values so that calib_sce and run_best will work properly.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            fuse_id = self._get_fuse_file_id()
            para_def_path = self.output_path / f"{self.domain_name}_{fuse_id}_para_def.nc"

            if not para_def_path.exists():
                self.logger.warning(f"para_def.nc not found: {para_def_path}")
                return False

            # Read elevation bands file to get correct values
            elev_bands_path = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
            if not elev_bands_path.exists():
                self.logger.warning(f"Elevation bands file not found: {elev_bands_path}")
                return False

            with xr.open_dataset(elev_bands_path) as eb_ds:
                n_bands = eb_ds.sizes.get('elevation_band', 1)
                mean_elevs = eb_ds['mean_elev'].values.flatten()
                area_fracs = eb_ds['area_frac'].values.flatten()

                # Calculate Z_FORCING as area-weighted mean elevation
                z_forcing = float(np.sum(mean_elevs * area_fracs))

            # Update para_def.nc with correct elevation band parameters
            # FUSE creates para_def.nc with 2 param sets (bounds), but set 1 often has NaN values
            # We keep only the first valid parameter set to avoid FUSE failing on NaN values
            with xr.open_dataset(para_def_path) as ds:
                ds_attrs = dict(ds.attrs)
                # Only keep the first parameter set (index 0) which has valid values
                ds_dict = {}
                for var in ds.data_vars:
                    vals = ds[var].values
                    if vals.ndim > 0 and len(vals) > 0:
                        # Keep only first value, as array for consistency
                        ds_dict[var] = np.array([vals[0]])
                    else:
                        ds_dict[var] = vals

            # Update the elevation band parameters
            ds_dict['N_BANDS'] = np.array([float(n_bands)])
            ds_dict['Z_FORCING'] = np.array([z_forcing])

            for i in range(n_bands):
                z_mid_name = f'Z_MID{i+1:02d}'
                af_name = f'AF{i+1:02d}'
                ds_dict[z_mid_name] = np.array([float(mean_elevs[i])])
                ds_dict[af_name] = np.array([float(area_fracs[i])])

            # Write back to NetCDF with single parameter set
            new_ds = xr.Dataset(
                {var: (['par'], vals) for var, vals in ds_dict.items()},
                coords={'par': [0]},
                attrs=ds_attrs
            )
            new_ds.to_netcdf(para_def_path, mode='w')

            self.logger.info(f"Fixed elevation band params in para_def.nc (reduced to 1 param set): "
                           f"N_BANDS={n_bands}, Z_FORCING={z_forcing:.1f}, Z_MID01={mean_elevs[0]:.1f}, AF01={area_fracs[0]:.3f}")
            return True

        except (FileNotFoundError, OSError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to fix elevation band params: {e}")
            return False

    def _fix_elevation_band_params_in_para_sce(self) -> bool:
        """
        Fix elevation band parameters in para_sce.nc after FUSE calibration.

        Similar to _fix_elevation_band_params_in_para_def, but for the calibration
        output file. FUSE's calib_sce creates para_sce.nc with zeros for elevation
        band parameters, causing run_best to fail in aggregating snowmelt.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            fuse_id = self._get_fuse_file_id()
            para_sce_path = self.output_path / f"{self.domain_name}_{fuse_id}_para_sce.nc"

            if not para_sce_path.exists():
                self.logger.warning(f"para_sce.nc not found: {para_sce_path}")
                return False

            # Read elevation bands file to get correct values
            elev_bands_path = self.forcing_fuse_path / f"{self.domain_name}_elev_bands.nc"
            if not elev_bands_path.exists():
                self.logger.warning(f"Elevation bands file not found: {elev_bands_path}")
                return False

            with xr.open_dataset(elev_bands_path) as eb_ds:
                n_bands = eb_ds.sizes.get('elevation_band', 1)
                mean_elevs = eb_ds['mean_elev'].values.flatten()
                area_fracs = eb_ds['area_frac'].values.flatten()
                z_forcing = float(np.sum(mean_elevs * area_fracs))

            # Update para_sce.nc with correct elevation band parameters
            # Need to handle the 'par' dimension which may have multiple parameter sets
            with xr.open_dataset(para_sce_path) as ds:
                par_size = ds.sizes.get('par', 1)
                ds_dict = {}
                for var in ds.data_vars:
                    ds_dict[var] = ds[var].values.copy()
                ds_attrs = dict(ds.attrs)
                ds_coords = {c: ds.coords[c].values for c in ds.coords}

            # Update elevation band parameters - broadcast to all parameter sets
            ds_dict['N_BANDS'] = np.full(par_size, float(n_bands))
            ds_dict['Z_FORCING'] = np.full(par_size, z_forcing)

            for i in range(n_bands):
                z_mid_name = f'Z_MID{i+1:02d}'
                af_name = f'AF{i+1:02d}'
                ds_dict[z_mid_name] = np.full(par_size, float(mean_elevs[i]))
                ds_dict[af_name] = np.full(par_size, float(area_fracs[i]))

            # Write back to NetCDF
            new_ds = xr.Dataset(
                {var: (['par'], vals) for var, vals in ds_dict.items()},
                coords={'par': ds_coords.get('par', np.arange(par_size))},
                attrs=ds_attrs
            )
            new_ds.to_netcdf(para_sce_path, mode='w')

            self.logger.info(f"Fixed elevation band params in para_sce.nc: N_BANDS={n_bands}, "
                           f"Z_FORCING={z_forcing:.1f}, Z_MID01={mean_elevs[0]:.1f}, AF01={area_fracs[0]:.3f}")
            return True

        except (FileNotFoundError, OSError, KeyError, ValueError) as e:
            self.logger.error(f"Failed to fix elevation band params in para_sce.nc: {e}")
            return False

    def _update_filemanager_for_run(self) -> bool:
        """
        Update file manager with current experiment settings before running FUSE.

        Ensures OUTPUT_PATH and FMODEL_ID match the current experiment configuration,
        allowing the same preprocessed setup to be used for different experiment runs.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            fuse_fm = self.config_dict.get('SETTINGS_FUSE_FILEMANAGER', 'fm_catch.txt')
            if fuse_fm == 'default':
                fuse_fm = 'fm_catch.txt'
            fm_path = self.project_dir / 'settings' / 'FUSE' / fuse_fm

            if not fm_path.exists():
                self.logger.warning(f"File manager not found: {fm_path}")
                return False

            # Read current file manager with encoding fallback
            try:
                with open(fm_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                self.logger.warning(f"UTF-8 decode error reading {fm_path}, falling back to latin-1")
                with open(fm_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()

            # Get current settings
            fuse_id = self.config_dict.get('FUSE_FILE_ID', self.experiment_id)
            output_path = str(self.output_path) + '/'

            # Find actual decisions file
            settings_dir = self.project_dir / 'settings' / 'FUSE'
            decisions_file = f"fuse_zDecisions_{fuse_id}.txt"
            if not (settings_dir / decisions_file).exists():
                # Find any decisions file
                decisions = list(settings_dir.glob("fuse_zDecisions_*.txt"))
                if decisions:
                    decisions_file = decisions[0].name
                    self.logger.debug(f"Using available decisions file: {decisions_file}")

            # Update relevant lines
            updated_lines = []
            input_path_str = str(self.forcing_fuse_path) + '/'

            for line in lines:
                stripped = line.strip()
                if stripped.startswith("'") and 'OUTPUT_PATH' in line:
                    updated_lines.append(f"'{output_path}'       ! OUTPUT_PATH\n")
                elif stripped.startswith("'") and 'INPUT_PATH' in line:
                     updated_lines.append(f"'{input_path_str}'       ! INPUT_PATH\n")
                elif stripped.startswith("'") and 'FMODEL_ID' in line:
                    updated_lines.append(f"'{fuse_id}'                            ! FMODEL_ID          = string defining FUSE model, only used to name output files\n")
                elif stripped.startswith("'") and 'M_DECISIONS' in line:
                    updated_lines.append(f"'{decisions_file}'        ! M_DECISIONS        = definition of model decisions\n")
                else:
                    updated_lines.append(line)

            # Write updated file manager
            with open(fm_path, 'w') as f:
                f.writelines(updated_lines)

            self.logger.debug(f"Updated file manager: OUTPUT_PATH={output_path}, INPUT_PATH={input_path_str}, FMODEL_ID={fuse_id}, M_DECISIONS={decisions_file}")
            return True

        except (FileNotFoundError, OSError, PermissionError) as e:
            self.logger.error(f"Failed to update file manager: {e}")
            return False

    def _execute_fuse(self, mode: str, para_file: Optional[Path] = None) -> bool:
        """
        Execute the FUSE model with specified run mode.

        Constructs and executes the FUSE command with the given mode,
        capturing output to a log file. Uses ModelExecutor mixin for
        subprocess management.

        Args:
            mode: FUSE run mode, one of:
                - 'run_def': Run with default parameters
                - 'calib_sce': Run SCE-UA calibration
                - 'run_best': Run with calibrated parameters
                - 'run_pre': Run with provided parameter file
            para_file: Path to parameter file for 'run_pre' mode (optional).

        Returns:
            bool: True if execution was successful, False otherwise.
        """
        # Update file manager with current experiment settings
        self._update_filemanager_for_run()

        self.logger.debug("Executing FUSE model")

        # Construct command
        fuse_fm = self.config_dict.get('SETTINGS_FUSE_FILEMANAGER')
        if fuse_fm == 'default':
            fuse_fm = 'fm_catch.txt'

        control_file = self.project_dir / 'settings' / 'FUSE' / fuse_fm

        command = [
            str(self.fuse_exe),
            str(control_file),
            self.domain_name,
            mode
        ]
            # ADD THIS: Add parameter file for run_pre mode
        if mode == 'run_pre' and para_file:
            command.append(str(para_file))

        # Create log file path - use mode-specific log files to avoid overwriting
        log_file = self.get_log_path() / f'fuse_{mode}.log'

        try:
            result = self.execute_model_subprocess(
                command,
                log_file,
                cwd=self.setup_dir,  # Run from settings directory where input_info.txt lives
                check=False,  # Don't raise, we'll return boolean
                success_message="FUSE execution completed"
            )
            self.logger.info(f"FUSE return code: {result.returncode}")
            return result.returncode == 0

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FUSE execution failed with error: {str(e)}")
            return False

    def _process_outputs(self):
        """Process and organize FUSE output files."""
        self.logger.debug("Processing FUSE outputs")

        output_dir = self.output_path / 'output'

        # Read and process streamflow output
        q_file = output_dir / 'streamflow.nc'
        if q_file.exists():
            with xr.open_dataset(q_file) as ds:
                # Add metadata
                ds.attrs['model'] = 'FUSE'
                ds.attrs['domain'] = self.domain_name
                ds.attrs['experiment_id'] = self.experiment_id
                ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save processed output
                processed_file = self.output_path / f"{self.experiment_id}_streamflow.nc"
                ds.to_netcdf(processed_file)
                self.logger.debug(f"Processed streamflow output saved to: {processed_file}")

        # Process state variables if they exist
        state_file = output_dir / 'states.nc'
        if state_file.exists():
            with xr.open_dataset(state_file) as ds:
                # Add metadata
                ds.attrs['model'] = 'FUSE'
                ds.attrs['domain'] = self.domain_name
                ds.attrs['experiment_id'] = self.experiment_id
                ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Save processed output
                processed_file = self.output_path / f"{self.experiment_id}_states.nc"
                ds.to_netcdf(processed_file)
                self.logger.info(f"Processed state variables saved to: {processed_file}")


    def _run_lumped_fuse(self) -> bool:
        """Run FUSE in lumped mode using the original workflow"""
        self.logger.info("Running lumped FUSE workflow")

        try:
            # Check if this is a snow optimization case
            if self._is_snow_optimization():
                self.logger.info("Snow optimization detected - copying default to best parameters")
                self._copy_default_to_best_params()

            # Note: Adding elevation params to constraints was causing FUSE crashes
            # The underlying issue is in how FUSE handles single-band elevation aggregation
            # TODO: Investigate FUSE source code for proper fix

            # Run FUSE with default parameters
            success = self._execute_fuse('run_def')

            # Also fix para_def.nc for any downstream uses
            self._fix_elevation_band_params_in_para_def()

            # Check if FUSE internal calibration should run (independent of external optimization)
            run_internal_calibration = self._get_config_value(
                lambda: self.config.model.fuse.run_internal_calibration if self.config.model and self.config.model.fuse else None,
                self.config_dict.get('FUSE_RUN_INTERNAL_CALIBRATION', True)
            )

            if run_internal_calibration:
                try:
                    # Run FUSE internal SCE-UA calibration as benchmark
                    self.logger.info("Running FUSE internal calibration (calib_sce) as benchmark")
                    success = self._execute_fuse('calib_sce')

                    # CRITICAL: Fix elevation band parameters in para_sce.nc too
                    # FUSE calib_sce creates para_sce.nc with zeros for Z_FORCING, Z_MID, AF
                    self._fix_elevation_band_params_in_para_sce()

                    # Run FUSE with best parameters from internal calibration
                    success = self._execute_fuse('run_best')
                except (subprocess.CalledProcessError, OSError, RuntimeError) as e:
                    self.logger.warning(f'FUSE internal calibration failed: {e}')
            else:
                self.logger.info("FUSE internal calibration disabled (FUSE_RUN_INTERNAL_CALIBRATION=false)")

            if success:
                # Ensure the expected output file exists
                self._ensure_best_output_file()
                self.logger.debug("Lumped FUSE run completed successfully")
                return True
            else:
                self.logger.error("Lumped FUSE run failed")
                return False

        except Exception as e:
            self.logger.error(f"Error in lumped FUSE execution: {str(e)}")
            return False

    def backup_run_files(self):
        """Backup important run files for reproducibility."""
        self.logger.info("Backing up run files")

        backup_dir = self.output_path / 'run_settings'
        backup_dir.mkdir(exist_ok=True)

        files_to_backup = [
            self.output_path / 'settings' / 'control.txt',
            self.output_path / 'settings' / 'structure.txt',
            self.output_path / 'settings' / 'params.txt'
        ]

        for file in files_to_backup:
            if file.exists():
                shutil.copy2(file, backup_dir / file.name)
                self.logger.info(f"Backed up {file.name}")
