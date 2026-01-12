"""
FUSE Runner Module

Refactored to use the Unified Model Execution Framework:
- ModelExecutor: For subprocess execution
- SpatialOrchestrator: For routing integration and spatial mode handling
"""

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List, Tuple

import numpy as np
import xarray as xr

from ..base import BaseModelRunner
from ..mixins import OutputConverterMixin
from ..execution import ModelExecutor, SpatialOrchestrator
from ..mizuroute.mixins import MizuRouteConfigMixin
from ..registry import ModelRegistry
from .subcatchment_processor import SubcatchmentProcessor
from symfluence.core.exceptions import (
    ModelExecutionError,
    symfluence_error_handler
)


@ModelRegistry.register_runner('FUSE', method_name='run_fuse')
class FUSERunner(BaseModelRunner, ModelExecutor, SpatialOrchestrator, OutputConverterMixin, MizuRouteConfigMixin):
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

    def __init__(self, config, logger: Any, reporting_manager: Optional[Any] = None):
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

        # FUSE-specific initialization
        self.spatial_mode = self._get_config_value(
            lambda: self.config.model.fuse.spatial_mode if self.config.model and self.config.model.fuse else None,
            'lumped'
        )

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
        return self._subcatchment_processor

    def _get_fuse_file_id(self) -> str:
        """Return a short file ID for FUSE outputs/settings."""
        return self.config_dict.get('FUSE_FILE_ID', self.experiment_id)

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
        - Squeezes singleton longitude dimension
        - Renames latitude → gru
        - Adds gruId variable
        """
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
            squeeze_dims=['longitude'],
            rename_dims={'latitude': 'gru'},
            add_id_var='gruId',
            id_source_dim='gru',
            create_backup=True
        )

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
                    
        except Exception as e:
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
                
        except Exception as e:
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
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
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
        except Exception as e:
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

        except Exception as e:
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
        lat_len = fuse_ds.dims.get('latitude', 0)
        lon_len = fuse_ds.dims.get('longitude', 0)

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
        except Exception:
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
        mizuroute_input_file = f"{self.experiment_id}_fuse_runoff.nc"

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
            
        except Exception as e:
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
                import shutil
                shutil.copy2(default_params, best_params)
                self.logger.info("Copied default parameters to best parameters file for snow optimization")
            else:
                self.logger.warning("Default parameter file not found - snow optimization may fail")
                
        except Exception as e:
            self.logger.error(f"Error copying default to best parameters: {str(e)}")
            

    def _execute_fuse(self, mode, para_file=None) -> bool:
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
        
        # Create log file path
        log_file = self.get_log_path() / 'fuse_run.log'

        try:
            result = self.execute_model_subprocess(
                command,
                log_file,
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
            
            # Run FUSE with default parameters
            success = self._execute_fuse('run_def')

            # Check if external optimization is enabled
            opt_methods = self.config_dict.get('OPTIMIZATION_METHODS', [])
            if isinstance(opt_methods, str):
                opt_methods = opt_methods.split(',')
            
            is_external_calibration = 'iteration' in [m.strip() for m in opt_methods]

            if is_external_calibration:
                self.logger.info("External optimization enabled - skipping FUSE internal calibration (calib_sce)")
            else:
                try:
                    # Run FUSE and calibrate with sce
                    success = self._execute_fuse('calib_sce')

                    # Run FUSE with best parameters
                    success = self._execute_fuse('run_best')
                except: 
                    self.logger.warning('FUSE internal calibration failed')

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
