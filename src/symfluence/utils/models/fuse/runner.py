import os
import sys
import time
import subprocess
from shutil import rmtree, copyfile
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import shutil
from datetime import datetime
import rasterio # type: ignore
from scipy import ndimage
import csv
import itertools
import matplotlib.pyplot as plt # type: ignore
import xarray as xr # type: ignore
from typing import Dict, List, Tuple, Any
from ..registry import ModelRegistry
from ..base import BaseModelPreProcessor
from ..mixins import PETCalculatorMixin

sys.path.append(str(Path(__file__).resolve().parent.parent))
from symfluence.utils.evaluation.calculate_sim_stats import get_KGE, get_KGEp, get_NSE, get_MAE, get_RMSE # type: ignore
from symfluence.utils.data.utilities.variable_utils import VariableHandler # type: ignore


@ModelRegistry.register_runner('FUSE', method_name='run_fuse')
class FUSERunner:
    """
    Runner class for the FUSE (Framework for Understanding Structural Errors) model.
    Handles model execution, output processing, and file management.
    
    Attributes:
        config (Dict[str, Any]): Configuration settings for FUSE
        logger (Any): Logger object for recording run information
        project_dir (Path): Directory for the current project
        domain_name (str): Name of the domain being processed
    """
    
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.result_dir = self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "FUSE"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize required paths
        self.fuse_path = self._get_install_path()
        self.output_path = self._get_output_path()
        self.setup_dir = self.project_dir / "settings" / "FUSE"
        self.forcing_fuse_path = self.project_dir / 'forcing' / 'FUSE_input'
        
        # Spatial mode
        self.spatial_mode = self.config.get('FUSE_SPATIAL_MODE', 'lumped')
        self.needs_routing = self._check_routing_requirements()

    def _convert_fuse_distributed_to_mizuroute_format(self):
        """
        Convert FUSE spatial dimensions to mizuRoute format.
        MINIMAL changes only: latitude->gru, add gruId, squeeze longitude.
        Preserves ALL original data and time coordinates unchanged.
        """
        import xarray as xr
        import numpy as np
        import shutil
        import tempfile
        import os

        experiment_id = self.config.get('EXPERIMENT_ID')
        domain = self.domain_name
        
        fuse_out_dir = self.project_dir / "simulations" / experiment_id / "FUSE"
        
        # Find FUSE output file
        target_files = [
            fuse_out_dir / f"{domain}_{experiment_id}_runs_def.nc",
            fuse_out_dir / f"{domain}_{experiment_id}_runs_best.nc"
        ]
        
        target = None
        for file_path in target_files:
            if file_path.exists():
                target = file_path
                break
        
        if target is None:
            raise FileNotFoundError(f"FUSE output not found. Tried: {[str(f) for f in target_files]}")

        self.logger.debug(f"Converting FUSE spatial dimensions: {target}")

        # Create backup
        backup_file = target.with_suffix('.backup.nc')
        if not backup_file.exists():
            shutil.copy2(target, backup_file)
            self.logger.info(f"Created backup: {backup_file}")

        # Load, modify, and immediately close the dataset
        with xr.open_dataset(target) as ds:
            self.logger.debug(f"Original dimensions: {dict(ds.sizes)}")
            
            # Step 1: Remove singleton longitude dimension if it exists
            if 'longitude' in ds.sizes and ds.sizes['longitude'] == 1:
                ds = ds.squeeze('longitude', drop=True)
                self.logger.debug("Squeezed longitude dimension")
            
            # Step 2: Rename latitude dimension to gru
            if 'latitude' in ds.sizes:
                ds = ds.rename({'latitude': 'gru'})
                self.logger.debug("Renamed latitude -> gru")
                
                # Step 3: Create gruId variable from gru coordinates
                if 'gru' in ds.coords:
                    gru_values = ds.coords['gru'].values
                    try:
                        # Try to convert to integers
                        gru_ids = gru_values.astype('int32')
                    except (ValueError, TypeError):
                        # If conversion fails, use sequential IDs
                        gru_ids = np.arange(1, len(gru_values) + 1, dtype='int32')
                        self.logger.warning(f"Using sequential GRU IDs 1-{len(gru_values)}")
                    
                    ds['gruId'] = xr.DataArray(
                        gru_ids,
                        dims=('gru',),
                        attrs={'long_name': 'ID of grouped response unit', 'units': '-'}
                    )
                    
                    self.logger.debug(f"Created gruId variable with {len(gru_ids)} GRUs")
                else:
                    raise ValueError("No gru coordinate found after renaming")
            else:
                raise ValueError("No latitude dimension found in FUSE output")
            
            self.logger.debug(f"Final dimensions: {dict(ds.sizes)}")
            
            # Load all data into memory before closing
            ds = ds.load()
        
        # Now the original file is closed, we can write to a temp file and replace
        try:
            # Make sure target file is writable
            try:
                os.chmod(target, 0o664)
            except Exception as e:
                self.logger.warning(f"Could not change file permissions: {e}")
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc', dir=target.parent) as tmp_file:
                temp_path = tmp_file.name
            
            # Write the modified dataset to temp file
            ds.to_netcdf(temp_path, format='NETCDF4')
            
            # Replace original with temp file
            shutil.move(temp_path, str(target))
            self.logger.debug(f"Spatial conversion completed: {target}")
            
            # Ensure _runs_def.nc exists if we processed a different file
            def_file = fuse_out_dir / f"{domain}_{experiment_id}_runs_def.nc"
            if target != def_file and not def_file.exists():
                shutil.copy2(target, def_file)
                self.logger.info(f"Created runs_def file: {def_file}")
                
        except Exception as e:
            # Clean up temp file if it exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise

    def run_fuse(self) -> Optional[Path]:
        """Run FUSE model with distributed support"""
        self.logger.debug(f"Starting FUSE model run in {self.spatial_mode} mode")
        
        try:
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
                
        except Exception as e:
            self.logger.error(f"Error during FUSE run: {str(e)}")
            raise

    def _check_routing_requirements(self) -> bool:
        """Check if distributed routing is needed"""
        routing_integration = self.config.get('FUSE_ROUTING_INTEGRATION', 'none')
        
        if routing_integration == 'mizuRoute':
            if self.spatial_mode in ['semi_distributed', 'distributed']:
                return True
            elif self.spatial_mode == 'lumped' and self.config.get('ROUTING_DELINEATION') == 'river_network':
                return True
        
        return False

    def _execute_fuse_workflow(self) -> bool:
        """Execute the main FUSE workflow based on spatial mode"""
        
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
            fuse_exe = self.fuse_path / self.config.get('FUSE_EXE', 'fuse.exe')
            control_file = self.setup_dir / 'fm_catch.txt'
            
            # Run FUSE once for the entire distributed domain
            command = [
                str(fuse_exe),
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
        """Create subcatchment-specific settings files"""
        
        try:
            # Create subcatchment-specific settings directory
            subcat_settings_dir = self.setup_dir / f"subcat_{subcat_id}"
            subcat_settings_dir.mkdir(exist_ok=True)
            
            # Copy base settings files
            base_settings_dir = self.setup_dir
            
            for file in base_settings_dir.glob("*.txt"):
                if "subcat_" not in file.name:  # Don't copy other subcatchment files
                    dest_file = subcat_settings_dir / file.name
                    shutil.copy2(file, dest_file)
            
            # Update file manager for this subcatchment
            fm_file = subcat_settings_dir / 'fm_catch.txt'
            if fm_file.exists():
                with open(fm_file, 'r') as f:
                    content = f.read()
                
                # Update paths to point to subcatchment-specific files
                content = content.replace(
                    f"{self.domain_name}_input.nc",
                    f"subcat_{subcat_id}_input.nc"
                )
                content = content.replace(
                    f"/{self.config.get('EXPERIMENT_ID')}/FUSE/",
                    f"/{self.config.get('EXPERIMENT_ID')}/FUSE/subcat_{subcat_id}/"
                )
                
                with open(fm_file, 'w') as f:
                    f.write(content)
            
            return subcat_settings_dir
            
        except Exception as e:
            self.logger.error(f"Error creating subcatchment settings for {subcat_id}: {str(e)}")
            raise

    def _execute_fuse_subcatchment(self, subcat_id: int, forcing_file: Path, settings_dir: Path) -> Optional[Path]:
        """Execute FUSE for a specific subcatchment"""
        
        try:
            # Create subcatchment output directory
            subcat_output_dir = self.output_path / f"subcat_{subcat_id}"
            subcat_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create elevation bands file for this subcatchment
            self._create_subcatchment_elevation_bands(subcat_id)
            
            # Run FUSE with subcatchment-specific settings
            fuse_exe = self.fuse_path / self.config.get('FUSE_EXE', 'fuse.exe')
            control_file = settings_dir / 'fm_catch.txt'
            
            command = [
                str(fuse_exe),
                str(control_file),
                f"{self.domain_name}_subcat_{subcat_id}",
                "run_def"  # Run with default parameters for distributed mode
            ]
            
            # Create log file for this subcatchment
            log_file = subcat_output_dir / 'fuse_run.log'
            
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(settings_dir)
                )
            
            if result.returncode == 0:
                # Find and return the output file
                output_files = list(subcat_output_dir.glob("*_runs_best.nc"))
                if output_files:
                    return output_files[0]
                else:
                    self.logger.warning(f"No output file found for subcatchment {subcat_id}")
                    return None
            else:
                self.logger.error(f"FUSE failed for subcatchment {subcat_id} with return code {result.returncode}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing FUSE for subcatchment {subcat_id}: {str(e)}")
            return None

    def _ensure_best_output_file(self):
        """Ensure the expected 'best' output file exists by copying from 'def' output if needed"""
        
        def_file = self.output_path / f"{self.domain_name}_{self.config.get('EXPERIMENT_ID')}_runs_def.nc"
        best_file = self.output_path / f"{self.domain_name}_{self.config.get('EXPERIMENT_ID')}_runs_best.nc"
        
        if def_file.exists() and not best_file.exists():
            self.logger.info(f"Copying {def_file.name} to {best_file.name} for compatibility")
            shutil.copy2(def_file, best_file)
        
        return best_file if best_file.exists() else def_file

    def _extract_subcatchment_forcing(self, subcat_id: int, index: int) -> Path:
        """Extract forcing data for a specific subcatchment while preserving proper netCDF structure"""
        
        # Load distributed forcing data
        forcing_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_input.nc"
        ds = xr.open_dataset(forcing_file)
        
        # Extract data for this subcatchment based on coordinate system
        subcatchment_dim = self.config.get('FUSE_SUBCATCHMENT_DIM', 'latitude')
        
        try:
            if subcatchment_dim == 'latitude':
                # Find the index of this subcatchment ID in the latitude coordinates
                lat_coords = ds.latitude.values
                try:
                    subcat_idx = list(lat_coords).index(float(subcat_id))
                except ValueError:
                    # If exact match not found, use the index directly
                    if index < len(lat_coords):
                        subcat_idx = index
                    else:
                        raise ValueError(f"Subcatchment index {index} out of range")
                
                # Extract data for this subcatchment but preserve the dimensional structure
                subcat_data = ds.isel(latitude=slice(subcat_idx, subcat_idx + 1))
                
            else:
                # Similar logic for longitude dimension
                lon_coords = ds.longitude.values
                try:
                    subcat_idx = list(lon_coords).index(float(subcat_id))
                except ValueError:
                    if index < len(lon_coords):
                        subcat_idx = index
                    else:
                        raise ValueError(f"Subcatchment index {index} out of range")
                
                subcat_data = ds.isel(longitude=slice(subcat_idx, subcat_idx + 1))
            
            # Now subcat_data should have the same dimensional structure as the original
            # but with latitude=1 (or longitude=1) instead of latitude=49
            
            # Verify the structure
            expected_dims = ['time', 'latitude', 'longitude']
            for var in ['pr', 'temp', 'pet', 'q_obs']:
                if var in subcat_data:
                    actual_dims = list(subcat_data[var].dims)
                    if actual_dims != expected_dims:
                        self.logger.error(f"Dimension mismatch for {var}: got {actual_dims}, expected {expected_dims}")
                        raise ValueError(f"Dimension structure incorrect for {var}")
            
            # Preserve all attributes
            for var in subcat_data.data_vars:
                if var in ds:
                    subcat_data[var].attrs = ds[var].attrs.copy()
            
            for coord in subcat_data.coords:
                if coord in ds.coords:
                    subcat_data[coord].attrs = ds[coord].attrs.copy()
            
            subcat_data.attrs = ds.attrs.copy()
            subcat_data.attrs['subcatchment_id'] = subcat_id
            
            # Save with proper encoding
            subcat_forcing_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_subcat_{subcat_id}_input.nc"
            
            encoding = {}
            for var in subcat_data.data_vars:
                encoding[var] = {
                    '_FillValue': -9999.0,
                    'dtype': 'float32'
                }
            
            for coord in subcat_data.coords:
                if coord == 'time':
                    encoding[coord] = {'dtype': 'float64'}
                else:
                    encoding[coord] = {'dtype': 'float64'}
            
            subcat_data.to_netcdf(
                subcat_forcing_file,
                encoding=encoding,
                format='NETCDF4',
                unlimited_dims=['time']
            )
            
            ds.close()
            subcat_data.close()
            
            self.logger.info(f"Created forcing file for subcatchment {subcat_id}: {subcat_forcing_file}")
            return subcat_forcing_file
            
        except Exception as e:
            self.logger.error(f"Error extracting forcing for subcatchment {subcat_id}: {str(e)}")
            ds.close()
            raise

    def _combine_subcatchment_outputs(self, outputs: List[Tuple[int, Path]]):
        """Combine outputs from all subcatchments into distributed format"""
        
        self.logger.info(f"Combining outputs from {len(outputs)} subcatchments")
        
        combined_outputs = {}
        
        # Load and combine all subcatchment outputs
        for subcat_id, output_file in outputs:
            try:
                ds = xr.open_dataset(output_file)
                
                # Store with subcatchment identifier
                for var_name in ds.data_vars:
                    if var_name not in combined_outputs:
                        combined_outputs[var_name] = {}
                    combined_outputs[var_name][subcat_id] = ds[var_name]
                
                ds.close()
                
            except Exception as e:
                self.logger.warning(f"Error loading output for subcatchment {subcat_id}: {str(e)}")
                continue
        
        # Create combined dataset and save
        if combined_outputs:
            self._create_combined_dataset(combined_outputs)

    def _create_combined_dataset(self, combined_outputs):
        """Create a combined dataset from subcatchment outputs"""
        
        try:
            self.logger.info("Creating combined dataset from subcatchment outputs")
            
            if not combined_outputs:
                self.logger.warning("No outputs to combine")
                return
            
            # Get list of subcatchment IDs and variables
            first_var = list(combined_outputs.keys())[0]
            subcatchment_ids = list(combined_outputs[first_var].keys())
            variable_names = list(combined_outputs.keys())
            
            self.logger.info(f"Combining {len(subcatchment_ids)} subcatchments with {len(variable_names)} variables")
            
            # Create the combined dataset
            combined_ds = xr.Dataset()
            
            # Add subcatchment coordinate
            combined_ds.coords['subcatchment'] = ('subcatchment', subcatchment_ids)
            
            # Process each variable
            for var_name in variable_names:
                self.logger.debug(f"Processing variable: {var_name}")
                
                # Collect data arrays for this variable from all subcatchments
                var_arrays = []
                reference_da = None
                
                for subcat_id in subcatchment_ids:
                    if subcat_id in combined_outputs[var_name]:
                        da = combined_outputs[var_name][subcat_id]
                        var_arrays.append(da)
                        if reference_da is None:
                            reference_da = da
                    else:
                        self.logger.warning(f"Missing data for variable {var_name} in subcatchment {subcat_id}")
                
                if var_arrays:
                    try:
                        # Concatenate along new subcatchment dimension
                        combined_var = xr.concat(var_arrays, dim='subcatchment')
                        
                        # Assign subcatchment coordinates
                        combined_var = combined_var.assign_coords(subcatchment=subcatchment_ids)
                        
                        # Copy attributes from reference data array
                        if reference_da is not None:
                            combined_var.attrs = reference_da.attrs.copy()
                        
                        # Add to combined dataset
                        combined_ds[var_name] = combined_var
                        
                        self.logger.debug(f"Combined {var_name} with shape: {combined_var.shape}")
                        
                    except Exception as e:
                        self.logger.error(f"Error combining variable {var_name}: {str(e)}")
                        continue
            
            # Add global attributes
            combined_ds.attrs.update({
                'model': 'FUSE',
                'spatial_mode': 'distributed',
                'domain': self.domain_name,
                'experiment_id': self.config.get('EXPERIMENT_ID'),
                'n_subcatchments': len(subcatchment_ids),
                'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Combined FUSE distributed simulation results'
            })
            
            # Add subcatchment coordinate attributes
            combined_ds.subcatchment.attrs = {
                'long_name': 'Subcatchment identifier',
                'description': 'Unique identifier for each subcatchment in the distributed model'
            }
            
            # Save the combined dataset
            combined_file = self.output_path / f"{self.domain_name}_{self.config.get('EXPERIMENT_ID')}_distributed_results.nc"
            
            # Define encoding for better compression and compatibility
            encoding = {}
            for var_name in combined_ds.data_vars:
                encoding[var_name] = {
                    'zlib': True,
                    'complevel': 4,
                    'shuffle': True,
                    '_FillValue': -9999.0,
                    'dtype': 'float32'
                }
            
            # Add coordinate encoding
            encoding['subcatchment'] = {'dtype': 'int32'}
            if 'time' in combined_ds.coords:
                encoding['time'] = {'dtype': 'float64'}
            if 'param_set' in combined_ds.coords:
                encoding['param_set'] = {'dtype': 'int32'}
            
            # Save to netCDF
            combined_ds.to_netcdf(
                combined_file,
                encoding=encoding,
                format='NETCDF4'
            )
            
            self.logger.info(f"Combined distributed results saved to: {combined_file}")
            
            # Log summary information
            self.logger.info(f"Combined dataset dimensions: {dict(combined_ds.dims)}")
            self.logger.info(f"Combined dataset variables: {list(combined_ds.data_vars.keys())}")
            
            # Also create a simplified streamflow-only file for easier analysis
            if 'q_routed' in combined_ds.data_vars:
                streamflow_file = self.output_path / f"{self.domain_name}_{self.config.get('EXPERIMENT_ID')}_streamflow_distributed.nc"
                streamflow_ds = combined_ds[['q_routed']].copy()
                streamflow_ds.to_netcdf(streamflow_file, encoding={'q_routed': encoding.get('q_routed', {})})
                self.logger.info(f"Streamflow-only file saved to: {streamflow_file}")
            
            combined_ds.close()
            
        except Exception as e:
            self.logger.error(f"Error creating combined dataset: {str(e)}")
            raise


    def _load_subcatchment_info(self):
        """Load subcatchment information for distributed mode"""
        # Check if delineated catchments exist (for distributed routing)
        delineated_path = self.project_dir / 'shapefiles' / 'catchment' / f"{self.domain_name}_catchment_delineated.shp"
        
        if delineated_path.exists():
            self.logger.info("Using delineated subcatchments")
            subcatchments = gpd.read_file(delineated_path)
            return subcatchments['GRU_ID'].values.astype(int)
        else:
            # Use regular HRUs
            catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
            catchment_name = self.config.get('CATCHMENT_SHP_NAME')
            if catchment_name == 'default':
                catchment_name = f"{self.domain_name}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp"
            
            catchment = gpd.read_file(catchment_path / catchment_name)
            if 'GRU_ID' in catchment.columns:
                return catchment['GRU_ID'].values.astype(int)
            else:
                # Create simple subcatchment IDs
                return np.arange(1, len(catchment) + 1)

    def _get_default_path(self, path_key: str, default_subpath: str) -> Path:
        """
        Get a path from config or use a default based on the project directory.
        Helper method for FUSERunner class.
        """
        try:
            path_value = self.config.get(path_key)
            if path_value == 'default' or path_value is None:
                return self.project_dir / default_subpath
            return Path(path_value)
        except KeyError:
            self.logger.error(f"Config key '{path_key}' not found")
            raise

    def _run_individual_subcatchments(self, subcatchments) -> bool:
        """Run FUSE separately for each subcatchment"""
        
        outputs = []
        
        for i, subcat_id in enumerate(subcatchments):
            self.logger.info(f"Running FUSE for subcatchment {subcat_id} ({i+1}/{len(subcatchments)})")
            
            try:
                # Extract forcing for this subcatchment
                subcat_forcing = self._extract_subcatchment_forcing(subcat_id, i)
                
                # Create subcatchment-specific settings
                subcat_settings = self._create_subcatchment_settings(subcat_id, i)
                
                # Run FUSE for this subcatchment
                subcat_output = self._execute_fuse_subcatchment(subcat_id, subcat_forcing, subcat_settings)
                
                if subcat_output:
                    outputs.append((subcat_id, subcat_output))
                else:
                    self.logger.warning(f"FUSE failed for subcatchment {subcat_id}")
                    
            except Exception as e:
                self.logger.error(f"Error running subcatchment {subcat_id}: {str(e)}")
                continue
        
        if outputs:
            # Combine outputs from all subcatchments
            self._combine_subcatchment_outputs(outputs)
            return True
        else:
            self.logger.error("No successful subcatchment runs")
            return False

    def _extract_subcatchment_forcing(self, subcat_id: int, index: int) -> Path:
        """Extract forcing data for a specific subcatchment"""
        
        # Load distributed forcing data
        forcing_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_input.nc"
        ds = xr.open_dataset(forcing_file)
        
        # Extract data for this subcatchment based on coordinate system
        subcatchment_dim = self.config.get('FUSE_SUBCATCHMENT_DIM', 'latitude')
        
        try:
            if subcatchment_dim == 'latitude':
                # Subcatchment IDs are encoded in latitude dimension
                subcat_data = ds.sel(latitude=float(subcat_id))
            else:
                # Subcatchment IDs are encoded in longitude dimension  
                subcat_data = ds.sel(longitude=float(subcat_id))
            
            # Save subcatchment-specific forcing with the correct filename pattern
            subcat_forcing_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_subcat_{subcat_id}_input.nc"
            subcat_data.to_netcdf(subcat_forcing_file)
            
            ds.close()
            return subcat_forcing_file
            
        except Exception as e:
            self.logger.error(f"Error extracting forcing for subcatchment {subcat_id}: {str(e)}")
            ds.close()
            raise

    def _create_subcatchment_elevation_bands(self, subcat_id: int) -> Path:
        """Create elevation bands file for a specific subcatchment"""
        
        try:
            # Source elevation bands file (the main one created during preprocessing)
            source_elev_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_elev_bands.nc"
            
            # Target elevation bands file for this subcatchment
            target_elev_file = self.project_dir / 'forcing' / 'FUSE_input' / f"{self.domain_name}_subcat_{subcat_id}_elev_bands.nc"
            
            if source_elev_file.exists():
                # For now, copy the main elevation bands file for each subcatchment
                # In a more sophisticated implementation, you could extract subcatchment-specific elevation data
                shutil.copy2(source_elev_file, target_elev_file)
                self.logger.debug(f"Created elevation bands file for subcatchment {subcat_id}")
            else:
                self.logger.warning(f"Source elevation bands file not found: {source_elev_file}")
                # Create a simple elevation bands file as fallback
                self._create_simple_elevation_bands(target_elev_file, subcat_id)
            
            return target_elev_file
            
        except Exception as e:
            self.logger.error(f"Error creating elevation bands for subcatchment {subcat_id}: {str(e)}")
            raise

    def _create_simple_elevation_bands(self, target_file: Path, subcat_id: int):
        """Create a simple elevation bands file as fallback"""
        
        # Get catchment centroid for coordinates
        catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        catchment_name = self.config.get('CATCHMENT_SHP_NAME')
        if catchment_name == 'default':
            catchment_name = f"{self.domain_name}_HRUs_{self.config.get('DOMAIN_DISCRETIZATION')}.shp"
        
        catchment = gpd.read_file(catchment_path / catchment_name)
        
        # Calculate centroid
        if catchment.crs is None:
            catchment.set_crs(epsg=4326, inplace=True)
        catchment_geo = catchment.to_crs(epsg=4326)
        bounds = catchment_geo.total_bounds
        lon = (bounds[0] + bounds[2]) / 2
        lat = (bounds[1] + bounds[3]) / 2
        
        # Create simple single elevation band
        ds = xr.Dataset(
            coords={
                'longitude': ('longitude', [lon]),
                'latitude': ('latitude', [lat]),
                'elevation_band': ('elevation_band', [1])
            }
        )
        
        # Add variables (single elevation band covering entire subcatchment)
        for var_name, data, attrs in [
            ('area_frac', [1.0], {'units': '-', 'long_name': 'Fraction of the catchment covered by each elevation band'}),
            ('mean_elev', [1000.0], {'units': 'm asl', 'long_name': 'Mid-point elevation of each elevation band'}),
            ('prec_frac', [1.0], {'units': '-', 'long_name': 'Fraction of catchment precipitation that falls on each elevation band'})
        ]:
            ds[var_name] = xr.DataArray(
                np.array(data).reshape(1, 1, 1),
                dims=['elevation_band', 'latitude', 'longitude'],
                coords=ds.coords,
                attrs=attrs
            )
        
        # Add coordinate attributes
        ds.longitude.attrs = {'units': 'degreesE', 'long_name': 'longitude'}
        ds.latitude.attrs = {'units': 'degreesN', 'long_name': 'latitude'}
        ds.elevation_band.attrs = {'units': '-', 'long_name': 'elevation_band'}
        
        # Save to file
        encoding = {var: {'_FillValue': -9999.0, 'dtype': 'float32'} for var in ds.data_vars}
        ds.to_netcdf(target_file, encoding=encoding)
        
        self.logger.info(f"Created simple elevation bands file for subcatchment {subcat_id}")

    def _combine_subcatchment_outputs(self, outputs: List[Tuple[int, Path]]):
        """Combine outputs from all subcatchments into distributed format"""
        
        self.logger.info(f"Combining outputs from {len(outputs)} subcatchments")
        
        combined_outputs = {}
        
        # Load and combine all subcatchment outputs
        for subcat_id, output_file in outputs:
            try:
                ds = xr.open_dataset(output_file)
                
                # Store with subcatchment identifier
                for var_name in ds.data_vars:
                    if var_name not in combined_outputs:
                        combined_outputs[var_name] = {}
                    combined_outputs[var_name][subcat_id] = ds[var_name]
                
                ds.close()
                
            except Exception as e:
                self.logger.warning(f"Error loading output for subcatchment {subcat_id}: {str(e)}")
                continue
        
        # Create combined dataset
        self._create_combined_dataset(combined_outputs)

    def _run_distributed_routing(self) -> bool:
        """Run mizuRoute routing for distributed FUSE output"""
        
        try:
            self.logger.debug("Starting mizuRoute routing for distributed FUSE")
            
            # Convert FUSE output to mizuRoute input format
            #routing_input = self._convert_fuse_to_mizuroute_format()
            
            #if not routing_input:
            #    return False
            
            # Create FUSE-specific mizuRoute control file
            from symfluence.utils.models.mizuroute_utils import MizuRoutePreProcessor
            mizu_preprocessor = MizuRoutePreProcessor(self.config, self.logger)
            mizu_preprocessor.create_fuse_control_file()
            
            # Run mizuRoute
            from symfluence.utils.models.mizuroute_utils import MizuRouteRunner
            mizuroute_runner = MizuRouteRunner(self.config, self.logger)
            
            # Update config for FUSE-mizuRoute integration
            self._setup_fuse_mizuroute_config()
            
            result = mizuroute_runner.run_mizuroute()
            
            if result:
                self.logger.debug("mizuRoute routing completed successfully")
                return True
            else:
                self.logger.error("mizuRoute routing failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in distributed routing: {str(e)}")
            return False

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
            out_dir = self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "FUSE"
            base = f"{self.domain_name}_{self.config.get('EXPERIMENT_ID')}"
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
        - Detect which spatial coord (latitude/longitude) holds the N>1 groups.
        - Produce dims (time, gru)
        - Add integer gruId from the spatial coordinate values
        - Ensure runoff variable name matches config['SETTINGS_MIZU_ROUTING_VAR']
        """
        # --- Choose runoff variable (prefer q_routed, else sensible fallbacks)
        routing_var_name = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
        candidates = [
            'q_routed', 'q_instnt', 'qsim', 'runoff',
            # fallbacks by substring
            *[v for v in fuse_ds.data_vars if v.lower().startswith("q_")],
            *[v for v in fuse_ds.data_vars if "runoff" in v.lower()],
        ]
        runoff_src = next((v for v in candidates if v in fuse_ds.data_vars), None)
        if runoff_src is None:
            raise ValueError(f"No suitable runoff variable found in FUSE output. "
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
        units = self.config.get('SETTINGS_MIZU_ROUTING_UNITS', 'mm/d')
        mizu[routing_var_name].attrs.update({'long_name': 'FUSE runoff for mizuRoute routing',
                                            'units': units})

        # Preserve some useful globals if present
        mizu.attrs.update({k: v for k, v in fuse_ds.attrs.items()})

        return mizu


    def _setup_fuse_mizuroute_config(self):
        """Update configuration for FUSE-mizuRoute integration"""

        # Update input file name for mizuRoute
        self.config['EXPERIMENT_ID_TEMP'] = self.config.get('EXPERIMENT_ID')  # Backup

        # Set mizuRoute to look for FUSE output instead of SUMMA
        mizuroute_input_file = f"{self.config.get('EXPERIMENT_ID')}_fuse_runoff.nc"

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
            optimization_target = self.config.get('OPTIMIZATION_TARGET', 'streamflow')
            if optimization_target in ['swe', 'sca', 'snow_depth', 'snow']:
                return True
                
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not determine if snow optimization: {str(e)}")
            # Fall back to checking config
            optimization_target = self.config.get('OPTIMIZATION_TARGET', 'streamflow')
            return optimization_target in ['swe', 'sca', 'snow_depth', 'snow']

    def _copy_default_to_best_params(self):
        """Copy default parameter file to best parameter file for snow optimization."""
        try:
            default_params = self.output_path / f"{self.domain_name}_{self.config.get('EXPERIMENT_ID')}_para_def.nc"
            best_params = self.output_path / f"{self.domain_name}_{self.config.get('EXPERIMENT_ID')}_para_sce.nc"
            
            if default_params.exists():
                import shutil
                shutil.copy2(default_params, best_params)
                self.logger.info("Copied default parameters to best parameters file for snow optimization")
            else:
                self.logger.warning("Default parameter file not found - snow optimization may fail")
                
        except Exception as e:
            self.logger.error(f"Error copying default to best parameters: {str(e)}")
            
    def _get_install_path(self) -> Path:
        """Get the FUSE installation path."""
        fuse_path = self.config.get('FUSE_INSTALL_PATH', 'default')
        if fuse_path == 'default':
            return self.data_dir / 'installs' / 'fuse' / 'bin'
        return Path(fuse_path)

    def _get_output_path(self) -> Path:
        """Get the path for FUSE outputs."""
        if self.config.get('EXPERIMENT_OUTPUT_FUSE', 'default') == 'default':
            return self.project_dir / "simulations" / self.config.get('EXPERIMENT_ID') / "FUSE"
        return Path(self.config.get('EXPERIMENT_OUTPUT_FUSE'))

    def _execute_fuse(self, mode, para_file=None) -> bool:
        """
        Execute the FUSE model.
        
        Returns:
            bool: True if execution was successful, False otherwise
        """
        self.logger.debug("Executing FUSE model")
        
        # Construct command
        fuse_fm = self.config.get('SETTINGS_FUSE_FILEMANAGER')
        if fuse_fm == 'default':
            fuse_fm = 'fm_catch.txt'
            
        fuse_exe = self.fuse_path / self.config.get('FUSE_EXE', 'fuse.exe')
        control_file = self.project_dir / 'settings' / 'FUSE' / fuse_fm
        
        command = [
            str(fuse_exe),
            str(control_file),
            self.config.get('DOMAIN_NAME'),
            mode
        ]
            # ADD THIS: Add parameter file for run_pre mode
        if mode == 'run_pre' and para_file:
            command.append(str(para_file))
        
        # Create log directory
        log_dir = self.output_path / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / 'fuse_run.log'
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            self.logger.debug(f"FUSE execution completed with return code: {result.returncode}")
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
                ds.attrs['experiment_id'] = self.config.get('EXPERIMENT_ID')
                ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Save processed output
                processed_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_streamflow.nc"
                ds.to_netcdf(processed_file)
                self.logger.debug(f"Processed streamflow output saved to: {processed_file}")
        
        # Process state variables if they exist
        state_file = output_dir / 'states.nc'
        if state_file.exists():
            with xr.open_dataset(state_file) as ds:
                # Add metadata
                ds.attrs['model'] = 'FUSE'
                ds.attrs['domain'] = self.domain_name
                ds.attrs['experiment_id'] = self.config.get('EXPERIMENT_ID')
                ds.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Save processed output
                processed_file = self.output_path / f"{self.config.get('EXPERIMENT_ID')}_states.nc"
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

