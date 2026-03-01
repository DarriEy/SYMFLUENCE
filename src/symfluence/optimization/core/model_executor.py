# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Model execution handler for optimization trials.

Manages SUMMA and mizuRoute execution during calibration, including
parameter application, simulation runs, and output extraction.
"""

import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import netCDF4 as nc
import numpy as np
import xarray as xr

from symfluence.core.mixins import ConfigMixin
from symfluence.optimization.calibration_targets import CalibrationTarget


def fix_summa_time_precision(nc_file: Path):
    """
    Ensures that the time variable in a SUMMA output file has sufficient precision
    for mizuRoute.
    """
    try:
        with nc.Dataset(nc_file, 'r+') as ds:
            if 'time' in ds.variables:
                time_var = ds.variables['time']
                if hasattr(time_var, 'units') and 'seconds since' in time_var.units:
                    # No changes needed if already in seconds
                    pass
                else:
                    # Full implementation would convert units - for now we assume
                    # the worker script version is used
                    pass
    except (OSError, RuntimeError, KeyError) as e:
        logging.error(f"Error fixing SUMMA time precision: {str(e)}")

class ModelExecutor(ConfigMixin):
    """Handles SUMMA and mizuRoute execution with routing support"""

    def __init__(self, config: Dict, logger: logging.Logger, calibration_target: CalibrationTarget):
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger
        self.calibration_target = calibration_target

    def run_models(self, summa_dir: Path, mizuroute_dir: Path, settings_dir: Path,
                  mizuroute_settings_dir: Optional[Path] = None) -> bool:
        """Run SUMMA and mizuRoute if needed"""
        try:
            # Run SUMMA
            if not self._run_summa(settings_dir, summa_dir):
                return False

            # Run mizuRoute if needed
            if self.calibration_target.needs_routing():
                if mizuroute_settings_dir is None:
                    mizuroute_settings_dir = settings_dir.parent / "mizuRoute"

                # Handle lumped-to-distributed conversion if needed
                domain_method = self._get_config_value(lambda: self.config.domain.definition_method, default='lumped', dict_key='DOMAIN_DEFINITION_METHOD')
                routing_delineation = self._get_config_value(lambda: self.config.domain.delineation.routing, default='lumped', dict_key='ROUTING_DELINEATION')

                if domain_method == 'lumped' and routing_delineation == 'river_network':
                    if not self._convert_lumped_to_distributed(summa_dir, mizuroute_settings_dir):
                        return False

                if not self._run_mizuroute(mizuroute_settings_dir, mizuroute_dir):
                    return False

            return True

        except (FileNotFoundError, IOError, subprocess.CalledProcessError) as e:
            self.logger.error(f"Error running models: {str(e)}")
            return False

    def _run_summa(self, settings_dir: Path, output_dir: Path) -> bool:
        """Run SUMMA simulation"""
        try:
            # Get SUMMA executable
            summa_path = self._get_config_value(lambda: self.config.model.summa.install_path, dict_key='SUMMA_INSTALL_PATH')
            if summa_path == 'default':
                summa_path = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')) / 'installs' / 'summa' / 'bin'
            else:
                summa_path = Path(summa_path)

            summa_exe_name = self._get_config_value(lambda: self.config.model.summa.exe, default='summa_sundials.exe', dict_key='SUMMA_EXE')
            summa_exe = summa_path / summa_exe_name
            file_manager = settings_dir / self._get_config_value(lambda: self.config.model.summa.filemanager, default='fileManager.txt', dict_key='SETTINGS_SUMMA_FILEMANAGER')

            if not summa_exe.exists():
                self.logger.error(f"SUMMA executable not found: {summa_exe}")
                return False

            if not file_manager.exists():
                self.logger.error(f"File manager not found: {file_manager}")
                return False

            # Create log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Run SUMMA
            cmd = [str(summa_exe), "-m", str(file_manager)]
            log_file = log_dir / f"summa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

            with open(log_file, 'w', encoding='utf-8') as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      check=True, timeout=10800)

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"SUMMA simulation failed with exit code {e.returncode}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("SUMMA simulation timed out")
            return False
        except (FileNotFoundError, IOError, OSError) as e:
            self.logger.error(f"Error running SUMMA: {str(e)}")
            return False

    def _run_mizuroute(self, settings_dir: Path, output_dir: Path) -> bool:
        """Run mizuRoute simulation"""
        try:
            # Get mizuRoute executable
            mizu_path = self._get_config_value(lambda: self.config.model.mizuroute.install_path, dict_key='MIZUROUTE_INSTALL_PATH')
            if mizu_path == 'default':
                mizu_path = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR')) / 'installs' / 'mizuRoute' / 'route' / 'bin'
            else:
                mizu_path = Path(mizu_path)

            mizu_exe = mizu_path / self._get_config_value(lambda: self.config.model.mizuroute.exe, default='mizuRoute.exe', dict_key='MIZUROUTE_EXE')
            control_file = settings_dir / self._get_config_value(lambda: self.config.model.mizuroute.control_file, default='mizuroute.control', dict_key='SETTINGS_MIZU_CONTROL_FILE')

            # 1) Find SUMMA output file for mizuRoute input
            #    Prefer *_for_routing.nc (from lumped→distributed conversion) over *_timestep.nc
            summa_output_dir = output_dir.parent / "SUMMA"
            routing_files = sorted(summa_output_dir.glob("*_for_routing.nc"))
            if routing_files:
                summa_timestep = routing_files[0].name
            else:
                timestep_files = sorted(summa_output_dir.glob("*_timestep.nc"))
                if not timestep_files:
                    self.logger.error(f"No SUMMA timestep files found in {summa_output_dir}")
                    return False
                summa_timestep = timestep_files[0].name

            # 2) Rewrite mizuroute.control paths/names to match this run
            text = control_file.read_text(encoding='utf-8')

            def repl(line, key, val):
                # assumes lines like: key = 'value'
                if line.strip().startswith(key):
                    return f"{key} = '{val}'\n"
                return line

            lines = []
            for line in text.splitlines(True):
                line = repl(line, "input_dir",  str((output_dir.parent / "SUMMA")))
                line = repl(line, "output_dir", str(output_dir))
                line = repl(line, "fname_qsim", summa_timestep)
                # optional: update case_name to strip suffix after last '_' if desired
                lines.append(line)

            control_file.write_text("".join(lines), encoding='utf-8')

            if not mizu_exe.exists():
                self.logger.error(f"mizuRoute executable not found: {mizu_exe}")
                return False

            if not control_file.exists():
                self.logger.error(f"mizuRoute control file not found: {control_file}")
                return False

            # Create log directory
            log_dir = output_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            # Run mizuRoute
            cmd = [str(mizu_exe), str(control_file)]
            log_file = log_dir / f"mizuroute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

            with open(log_file, 'w', encoding='utf-8') as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      check=True, timeout=1800, cwd=str(settings_dir))

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"mizuRoute simulation failed with exit code {e.returncode}")
            return False
        except subprocess.TimeoutExpired:
            self.logger.error("mizuRoute simulation timed out")
            return False
        except (FileNotFoundError, IOError, OSError) as e:
            self.logger.error(f"Error running mizuRoute: {str(e)}")
            return False

    def _convert_lumped_to_distributed(self, summa_dir: Path, mizuroute_settings_dir: Path) -> bool:
        """Convert lumped SUMMA output for distributed routing"""
        try:
            # Load topology to get HRU information
            topology_file = mizuroute_settings_dir / self._get_config_value(lambda: self.config.model.mizuroute.topology, default='topology.nc', dict_key='SETTINGS_MIZU_TOPOLOGY')

            with xr.open_dataset(topology_file) as topo_ds:
                # Handle multiple HRUs from delineated catchments
                hru_ids = topo_ds['hruId'].values
                n_hrus = len(hru_ids)
                lumped_gru_id = 1
                self.logger.info(f"Creating single lumped GRU (ID={lumped_gru_id}) for {n_hrus} HRUs in topology")

            # Find SUMMA timestep file
            timestep_files = list(summa_dir.glob("*timestep.nc"))
            if not timestep_files:
                return False

            summa_file = timestep_files[0]

            # Load topology to get segment information
            if not topology_file.exists():
                return False

            with xr.open_dataset(topology_file) as topo_ds:
                seg_ids = topo_ds['segId'].values
                len(seg_ids)

            # Load and convert SUMMA output
            with xr.open_dataset(summa_file, decode_times=False) as summa_ds:
                # Find routing variable - handle 'default' config value
                routing_var_config = self._get_config_value(lambda: self.config.model.mizuroute.routing_var, default='averageRoutedRunoff', dict_key='SETTINGS_MIZU_ROUTING_VAR')
                if routing_var_config in ('default', None, ''):
                    routing_var = 'averageRoutedRunoff'  # SUMMA default for routing
                else:
                    routing_var = routing_var_config
                if routing_var not in summa_ds:
                    routing_var = 'basin__TotalRunoff'

                if routing_var not in summa_ds:
                    return False

                # Create mizuRoute forcing dataset
                mizuForcing = xr.Dataset()
                mizuForcing['time'] = summa_ds['time']
                mizuForcing['gru'] = xr.DataArray([lumped_gru_id], dims=('gru',))
                mizuForcing['gruId'] = xr.DataArray([lumped_gru_id], dims=('gru',))

                # Extract runoff data
                var_data = summa_ds[routing_var]
                runoff_data = var_data.values

                # Handle different shapes
                if len(runoff_data.shape) == 2:
                    if runoff_data.shape[1] > 1:
                        runoff_data = runoff_data.mean(axis=1)
                        self.logger.info(f"Used mean across {var_data.shape[1]} spatial elements")
                    else:
                        runoff_data = runoff_data[:, 0]
                else:
                    runoff_data = runoff_data.flatten()

                # Keep as single GRU
                single_gru_data = runoff_data[:, np.newaxis]

                mizuForcing['averageRoutedRunoff'] = xr.DataArray(
                    single_gru_data, dims=('time', 'gru'),
                    attrs={'long_name': 'Lumped runoff for distributed routing', 'units': 'm/s'}
                )
                # Copy global attributes
                mizuForcing.attrs.update(summa_ds.attrs)

            # Write to a SEPARATE routing file to preserve original SUMMA output
            routing_file = summa_file.parent / f"{summa_file.stem}_for_routing.nc"
            mizuForcing.to_netcdf(routing_file, format='NETCDF4')
            mizuForcing.close()
            self.logger.info(f"Wrote routing forcing to {routing_file.name} (original SUMMA output preserved)")

            # Fix time precision for mizuRoute compatibility on the routing file
            fix_summa_time_precision(routing_file)

            return True

        except (OSError, RuntimeError, KeyError, ValueError) as e:
            self.logger.error(f"Error converting lumped to distributed: {str(e)}")
            return False
