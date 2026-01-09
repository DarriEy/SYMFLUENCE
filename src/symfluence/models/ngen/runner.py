"""
NGen Model Runner.

Manages the execution of the NOAA NextGen Framework (ngen).
Refactored to use the Unified Model Execution Framework.
"""

import os
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelRunner
from symfluence.models.execution import ModelExecutor
from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler


@ModelRegistry.register_runner('NGEN', method_name='run_ngen')
class NgenRunner(BaseModelRunner, ModelExecutor):
    """
    Runner for NextGen Framework simulations.

    Handles execution of ngen with proper paths and error handling.
    Uses the Unified Model Execution Framework for subprocess execution.
    """

    def __init__(self, config: Dict[str, Any], logger: Any, reporting_manager: Optional[Any] = None):
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _setup_model_specific_paths(self) -> None:
        """Set up NGEN-specific paths."""
        # Check if parallel worker has provided isolated settings directory
        if '_ngen_settings_dir' in self.config_dict:
            self.ngen_setup_dir = Path(self.config_dict['_ngen_settings_dir'])
        else:
            self.ngen_setup_dir = self.project_dir / "settings" / "NGEN"

        # Determine absolute ngen base directory
        install_path = self.config_dict.get('NGEN_INSTALL_PATH', 'default')
        if install_path == 'default':
            ngen_base = self.data_dir.parent / 'installs' / 'ngen'
        else:
            p = Path(install_path)
            if p.name == 'cmake_build': ngen_base = p.parent
            else: ngen_base = p

        # Try both root/ngen, root/cmake_build/ngen, root/bin/ngen
        candidates = [
            ngen_base / "ngen",
            ngen_base / "cmake_build" / "ngen",
            ngen_base / "bin" / "ngen"
        ]
        
        self.ngen_exe = None
        for cand in candidates:
            if cand.exists():
                self.ngen_exe = cand
                break
        
        if self.ngen_exe is None:
            # Fallback to get_model_executable if none found
            self.ngen_exe = self.get_model_executable(
                install_path_key='NGEN_INSTALL_PATH',
                default_install_subpath='installs/ngen/cmake_build',
                exe_name_key=None,
                default_exe_name='ngen'
            )

    def _get_model_name(self) -> str:
        """Return model name for NextGen."""
        return "NGEN"

    def _should_create_output_dir(self) -> bool:
        """NGEN creates directories on-demand in run_ngen."""
        return False

    def run_ngen(self, experiment_id: str = None):
        """
        Execute NextGen model simulation.

        Args:
            experiment_id: Optional experiment identifier. If None, uses config value.

        Runs ngen with the prepared catchment, nexus, forcing, and configuration files.
        """
        self.logger.debug("Starting NextGen model run")

        with symfluence_error_handler(
            "NextGen model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # Get experiment info
            if experiment_id is None:
                if self.config:
                    experiment_id = self.config.domain.experiment_id
                else:
                    experiment_id = self.config_dict.get('EXPERIMENT_ID', 'default_run')

            # Check if parallel worker has provided isolated output directory
            if '_ngen_output_dir' in self.config_dict:
                output_dir = Path(self.config_dict['_ngen_output_dir'])
            else:
                output_dir = self.get_experiment_output_dir(experiment_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Setup paths for ngen execution
            if self.config:
                domain_name = self.config.domain.name
            else:
                domain_name = self.config_dict.get('DOMAIN_NAME')
            
            # Force GeoJSON on macOS due to widespread GPKG/SQLite mismatch issues in ngen builds
            import platform
            use_geojson = getattr(self, "_use_geojson_catchments", False)
            if platform.system() == "Darwin":
                self.logger.info("Forcing GeoJSON catchments on macOS for stability")
                use_geojson = True
            
            if use_geojson:
                catchment_file = self.ngen_setup_dir / f"{domain_name}_catchments.geojson"
            else:
                catchment_file = self.ngen_setup_dir / f"{domain_name}_catchments.gpkg"
            fallback_catchment_file = self.ngen_setup_dir / f"{domain_name}_catchments.geojson"
            nexus_file = self.ngen_setup_dir / "nexus.geojson"
            realization_file = self.ngen_setup_dir / "realization_config.json"

            # Verify required files exist
            self.verify_required_files(
                [catchment_file, nexus_file, realization_file],
                context="NextGen model execution"
            )

            # Ensure realization_config uses absolute library paths
            # Patch a copy to avoid side effects if multiple runs use same settings_dir
            patched_realization = output_dir / "realization_config_patched.json"
            import shutil
            shutil.copy(realization_file, patched_realization)
            self._patch_realization_libraries(patched_realization)

            # Build ngen command
            ngen_cmd = [
                str(self.ngen_exe),
                str(catchment_file),
                "all",
                str(nexus_file),
                "all",
                str(patched_realization)
            ]

            self.logger.debug(f"Running command: {' '.join(ngen_cmd)}")

            # Run ngen
            log_file = output_dir / "ngen_log.txt"
            try:
                # Setup environment for NGEN execution
                env = os.environ.copy()

                # Remove PYTHONPATH to avoid version mismatches
                # NGEN is linked to the Homebrew Python, not the venv
                env.pop('PYTHONPATH', None)
                env.pop('PYTHONHOME', None)

                # Ensure library paths are set for BMI modules
                # This is especially important for MPI/multiprocessing workers
                if self.config and self.config.model.ngen:
                    install_path = self.config.model.ngen.install_path
                else:
                    install_path = self.config_dict.get('NGEN_INSTALL_PATH', 'default')
                
                if install_path == 'default':
                    ngen_base = self.data_dir.parent / 'installs' / 'ngen'
                else:
                    p = Path(install_path)
                    if p.name == 'cmake_build': ngen_base = p.parent
                    else: ngen_base = p

                # Try both ngen_base/extern and ngen_base/cmake_build/extern
                lib_paths = []
                for sub in ["extern/sloth/cmake_build", "extern/cfe/cmake_build", 
                            "extern/evapotranspiration/evapotranspiration/cmake_build",
                            "extern/noah-owp-modular/cmake_build"]:
                    p1 = ngen_base / sub
                    p2 = ngen_base / "cmake_build" / sub
                    if p1.exists(): lib_paths.append(str(p1.resolve()))
                    elif p2.exists(): lib_paths.append(str(p2.resolve()))
                
                # Add ngen build dir itself and brew libs
                lib_paths.append(str(self.ngen_exe.parent.resolve()))
                lib_paths.append("/opt/homebrew/lib")

                # Set library path based on OS
                lib_path_str = ':'.join(lib_paths)
                
                # Set both for safety, though only one might be used by the linker
                for var in ['DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']:
                    existing_path = env.get(var, '')
                    env[var] = f"{lib_path_str}:{existing_path}" if existing_path else lib_path_str

                self.logger.debug(f"Executing ngen with DYLD_LIBRARY_PATH={env.get('DYLD_LIBRARY_PATH')}")

                self.execute_model_subprocess(
                    ngen_cmd,
                    log_file,
                    cwd=self.ngen_exe.parent,  # Run from ngen build directory (needed for relative library paths)
                    env=env,  # Use modified environment with library paths
                    success_message="NextGen model run completed successfully"
                )

                # Move outputs from build directory to output directory
                self._move_ngen_outputs(self.ngen_exe.parent, output_dir)

                return True

            except subprocess.CalledProcessError as e:
                # On macOS, code -6 (SIGABRT) is common for GPKG issues if SQLite is missing/mismatched
                is_likely_sqlite_issue = (e.returncode == -6)
                
                if not use_geojson and fallback_catchment_file.exists():
                    try:
                        log_text = log_file.read_text(errors='ignore')
                    except Exception:
                        log_text = ""
                    sqlite_error = "SQLite3 support required to read GeoPackage files"
                    
                    if is_likely_sqlite_issue or sqlite_error in log_text:
                        self.logger.warning(
                            f"NGEN failed (code {e.returncode}); retrying with GeoJSON catchments"
                        )
                        ngen_cmd[1] = str(fallback_catchment_file)
                        try:
                            self.execute_model_subprocess(
                                ngen_cmd,
                                log_file,
                                cwd=self.ngen_exe.parent,
                                env=env,
                                success_message="NextGen model run completed successfully (GeoJSON fallback)"
                            )
                            self._use_geojson_catchments = True
                            self._move_ngen_outputs(self.ngen_exe.parent, output_dir)
                            return True
                        except subprocess.CalledProcessError as retry_error:
                            self.logger.error(
                                f"NextGen model run failed with error code {retry_error.returncode}"
                            )
                            self.logger.error(f"Check log file: {log_file}")
                            return False

                self.logger.error(f"NextGen model run failed with error code {e.returncode}")
                self.logger.error(f"Check log file: {log_file}")
                return False
    
    def _patch_realization_libraries(self, realization_file: Path):
        """Patch realization config to use absolute paths for libraries and init_configs."""
        import json
        try:
            with open(realization_file, 'r') as f:
                data = json.load(f)
            
            changed = False
            # Determine absolute ngen base directory
            if self.config and self.config.model.ngen:
                install_path = self.config.model.ngen.install_path
            else:
                install_path = self.config_dict.get('NGEN_INSTALL_PATH', 'default')
            
            if install_path == 'default':
                ngen_base = self.data_dir.parent / 'installs' / 'ngen'
            else:
                p = Path(install_path)
                if p.name == 'cmake_build': ngen_base = p.parent
                else: ngen_base = p

            if 'global' in data and 'formulations' in data['global']:
                for formulation in data['global']['formulations']:
                    if 'params' in formulation and 'modules' in formulation['params']:
                        for module in formulation['params']['modules']:
                            mod_params = module.get('params', {})
                            
                            # 1. Patch library_file
                            if 'library_file' in mod_params:
                                lib_path = mod_params['library_file']
                                target_subpath = None
                                if 'pet' in lib_path.lower():
                                    target_subpath = "extern/evapotranspiration/evapotranspiration/cmake_build"
                                elif 'cfe' in lib_path.lower():
                                    target_subpath = "extern/cfe/cmake_build"
                                elif 'sloth' in lib_path.lower():
                                    target_subpath = "extern/sloth/cmake_build"
                                elif 'surface' in lib_path.lower() or 'noah' in lib_path.lower():
                                    target_subpath = "extern/noah-owp-modular/cmake_build"
                                
                                if target_subpath:
                                    filename = Path(lib_path).name
                                    # Try both ngen_base/extern and ngen_base/cmake_build/extern
                                    p1 = ngen_base / target_subpath
                                    p2 = ngen_base / "cmake_build" / target_subpath
                                    lib_dir = p1 if p1.exists() else p2
                                    
                                    actual_lib = None
                                    if (lib_dir / filename).exists():
                                        actual_lib = lib_dir / filename
                                    else:
                                        candidates = list(lib_dir.glob(f"{filename.split('.')[0]}*.dylib"))
                                        if candidates:
                                            actual_lib = candidates[0]
                                    
                                    if actual_lib:
                                        abs_lib_path = str(actual_lib.resolve())
                                        if mod_params['library_file'] != abs_lib_path:
                                            mod_params['library_file'] = abs_lib_path
                                            changed = True
                                            self.logger.debug(f"Patched library {lib_path} -> {abs_lib_path}")

                            # 2. Patch init_config
                            if 'init_config' in mod_params:
                                old_path = mod_params['init_config']
                                if old_path and old_path != "/dev/null":
                                    mod_type_name = str(mod_params.get('model_type_name', '')).upper()
                                    target_mod = None
                                    if 'PET' in mod_type_name or 'pet' in old_path.lower(): target_mod = 'PET'
                                    elif 'CFE' in mod_type_name or 'cfe' in old_path.lower(): target_mod = 'CFE'
                                    elif 'NOAH' in mod_type_name or 'noah' in old_path.lower() or '.input' in old_path.lower(): target_mod = 'NOAH'
                                    
                                    if target_mod:
                                        filename = Path(old_path).name
                                        # Replace {{id}} template placeholder with actual catchment ID if present
                                        if '{{id}}' in filename:
                                            # Find first cat-<id>_* file in the target directory
                                            target_dir = self.ngen_setup_dir / target_mod
                                            if target_dir.exists():
                                                candidates = list(target_dir.glob("cat-*"))
                                                if candidates:
                                                    # Use the actual file name pattern from the first matching file
                                                    # For files like "cat-1_pet_config.txt", we want to replace {{id}} with "cat-1"
                                                    first_file = candidates[0].name
                                                    match = re.search(r'(cat-[a-zA-Z0-9_-]+?)(?=_)', first_file)
                                                    if match:
                                                        filename = filename.replace('{{id}}', match.group(1))
                                        # Use the setup dir for THIS runner (might be isolated)
                                        new_path = str((self.ngen_setup_dir / target_mod / filename).resolve())
                                        if old_path != new_path:
                                            mod_params['init_config'] = new_path
                                            changed = True
                                            self.logger.debug(f"Patched {target_mod} init_config to {new_path}")
            
            # 3. Patch forcing file pattern
            if 'global' in data and 'forcing' in data['global']:
                forcing = data['global']['forcing']
                if 'file_pattern' in forcing and '{{id}}' in forcing['file_pattern']:
                    # Find first forcing file to extract the catchment ID
                    forcing_dir = self.ngen_setup_dir.parent / 'forcing' / 'NGEN_input' / 'csv'
                    if forcing_dir.exists():
                        candidates = list(forcing_dir.glob("*_forcing*.csv"))
                        if candidates:
                            first_file = candidates[0].name
                            match = re.search(r'(cat-[a-zA-Z0-9_-]+?)(?=_forcing)', first_file)
                            if match:
                                pattern = forcing['file_pattern']
                                pattern = pattern.replace('{{id}}', match.group(1))
                                forcing['file_pattern'] = pattern
                                changed = True
                                self.logger.debug(f"Patched forcing file pattern to {pattern}")

            # 4. Patch output_root for isolated calibration directories
            if '_ngen_output_dir' in self.config_dict:
                isolated_output_dir = str(Path(self.config_dict['_ngen_output_dir']).resolve())
                if data.get('output_root') != isolated_output_dir:
                    data['output_root'] = isolated_output_dir
                    changed = True
                    self.logger.debug(f"Patched output_root to {isolated_output_dir}")
            
            if changed:
                with open(realization_file, 'w') as f:
                    json.dump(data, f, indent=2)
                self.logger.info(f"Patched absolute paths in realization config copy")
        except Exception as e:
            self.logger.warning(f"Failed to patch realization libraries: {e}")

    def _move_ngen_outputs(self, build_dir: Path, output_dir: Path):
        """
        Move ngen output files from build directory to output directory.
        
        ngen writes outputs to its working directory, so we need to move them
        to the proper experiment output directory.
        
        Args:
            build_dir: ngen build directory where outputs are written
            output_dir: Target output directory for this experiment
        """
        import shutil
        
        # Common ngen output patterns
        output_patterns = [
            'cat-*.csv',      # Catchment outputs
            'nex-*.csv',      # Nexus outputs  
            '*.parquet',      # Parquet outputs
            'cfe_output_*.txt',  # CFE specific outputs
            'noah_output_*.txt', # Noah specific outputs
        ]
        
        moved_files = []
        for pattern in output_patterns:
            for file in build_dir.glob(pattern):
                dest = output_dir / file.name
                shutil.move(str(file), str(dest))
                moved_files.append(file.name)
        
        if moved_files:
            self.logger.debug(f"Moved {len(moved_files)} output files to {output_dir}")
            for f in moved_files[:10]:  # Log first 10
                self.logger.debug(f"  - {f}")
            if len(moved_files) > 10:
                self.logger.debug(f"  ... and {len(moved_files) - 10} more")
        else:
            existing_outputs = []
            for pattern in output_patterns:
                existing_outputs.extend(output_dir.glob(pattern))
            if not existing_outputs:
                self.logger.warning(
                    f"No output files found in {build_dir} or {output_dir}. Check if model ran correctly."
                )
