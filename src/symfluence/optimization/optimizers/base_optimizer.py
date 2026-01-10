"""
Abstract base class for SYMFLUENCE optimization algorithms.

Provides common infrastructure for parameter management, model execution,
results tracking, and trial history for all optimizer implementations.
"""

import os
import numpy as np
import pandas as pd
import shutil
import logging
import random
import time
import subprocess
import pickle
import sys
import gc
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from symfluence.optimization.core.parameter_manager import ParameterManager
from symfluence.optimization.core.model_executor import ModelExecutor
from symfluence.optimization.core.results_manager import ResultsManager
from symfluence.optimization.local_scratch_manager import LocalScratchManager
from symfluence.optimization.core import TransformationManager
from symfluence.optimization.calibration_targets import (
    ETTarget, SnowTarget, GroundwaterTarget, SoilMoistureTarget, StreamflowTarget, CalibrationTarget, TWSTarget
)
from symfluence.optimization.workers.summa_parallel_workers import _evaluate_parameters_worker_safe, _run_dds_instance_worker
from symfluence.core.mixins import ConfigMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseOptimizer(ABC, ConfigMixin):
    """
    Abstract base class for SYMFLUENCE optimizers.

    Inherits from ConfigMixin for typed config access with dict fallback.
    """

    # Convenience properties for commonly accessed config values
    @property
    def domain_name(self) -> str:
        return self._get_config_value(lambda: self.config.domain.name) or self._domain_name

    @domain_name.setter
    def domain_name(self, value: str) -> None:
        self._domain_name = value

    @property
    def experiment_id(self) -> str:
        return self._get_config_value(lambda: self.config.domain.experiment_id) or self._experiment_id

    @experiment_id.setter
    def experiment_id(self, value: str) -> None:
        self._experiment_id = value

    @property
    def summa_filemanager(self) -> str:
        return self._get_config_value(
            lambda: self.config.model.summa.filemanager if self.config.model.summa else None,
            default='fileManager.txt'
        )

    @property
    def optimization_target(self) -> str:
        return self._get_config_value(lambda: self.config.optimization.target, default='streamflow')

    @property
    def calibration_variable(self) -> str:
        return self._get_config_value(lambda: self.config.optimization.calibration_variable, default='streamflow')

    @property
    def config_dict(self) -> Dict[str, Any]:
        """Override to support dict-only initialization."""
        if self._config is not None:
            return self._config.to_dict(flatten=True)
        return self._config_dict or {}

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        output_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None,
        typed_config: Optional['SymfluenceConfig'] = None
    ):
        """Initialize base optimizer with common components."""
        # Set typed config via mixin (accepts SymfluenceConfig or dict)
        if typed_config is not None:
            self._config = typed_config
        elif hasattr(config, 'domain'):  # Already a SymfluenceConfig
            self._config = config
        else:
            self._config = None  # Will use config dict fallback

        self._config_dict = config if isinstance(config, dict) else None
        self._logger = logger
        self.reporting_manager = reporting_manager

        # Setup basic paths
        self.data_dir = Path(self.config_dict.get('SYMFLUENCE_DATA_DIR'))
        self._domain_name = self.config_dict.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self._experiment_id = self.config_dict.get('EXPERIMENT_ID')
        
        # Algorithm-specific directory setup
        self.algorithm_name = self.get_algorithm_name().lower()

        mpi_processes = self._cfg('MPI_PROCESSES', default=1)
        self.use_parallel = mpi_processes > 1
        self.num_processes = max(1, mpi_processes)
        
        # Get MPI rank
        mpi_rank = None
        if self.use_parallel:
            mpi_rank = os.environ.get('PMI_RANK', 0)  # SLURM
            if not mpi_rank:
                mpi_rank = os.environ.get('OMPI_COMM_WORLD_RANK', 0)  # OpenMPI
            mpi_rank = int(mpi_rank) if mpi_rank else 0

        # Initialize scratch manager
        self.scratch_manager = LocalScratchManager(
            config, logger, self.project_dir, self.get_algorithm_name(), mpi_rank
        )

        if self.scratch_manager.use_scratch:
            if mpi_rank and mpi_rank > 0:
                delay = mpi_rank * 2
                self.logger.info(f"Rank {mpi_rank}: Waiting {delay}s before scratch setup...")
                time.sleep(delay)
            
            if self.scratch_manager.setup_scratch_space():
                self.data_dir = self.scratch_manager.get_effective_data_dir()
                self.project_dir = self.scratch_manager.get_effective_project_dir()

        self.optimization_dir = self.project_dir / "simulations" / f"run_{self.algorithm_name}"
        self.logger.info(f"optimization_dir set to: {self.optimization_dir}")
        self.summa_sim_dir = self.optimization_dir / "SUMMA"
        self.mizuroute_sim_dir = self.optimization_dir / "mizuRoute"
        self.optimization_settings_dir = self.optimization_dir / "settings" / "SUMMA"
        # Allow output_dir override, otherwise use default
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = self.project_dir / "optimization" / f"{self.algorithm_name}_{self.experiment_id}"
        
        # Initialize component managers
        self.parameter_manager = ParameterManager(config, logger, self.optimization_settings_dir)
        self.transformation_manager = TransformationManager(config, logger)
        self._setup_optimization_directories()
        
        self.calibration_target = self._create_calibration_target()
        self.model_executor = ModelExecutor(config, logger, self.calibration_target)
        self.results_manager = ResultsManager(config, logger, self.output_dir, self.reporting_manager)
        
        # Common algorithm parameters
        self.max_iterations = self._cfg(
            'NUMBER_OF_ITERATIONS', default=100,
            typed=lambda: self._typed_config.optimization.iterations
        )
        self.target_metric = self._cfg(
            'OPTIMIZATION_METRIC', default='KGE',
            typed=lambda: self._typed_config.optimization.metric
        )

        # Algorithm state variables
        self.best_params = None
        self.best_score = float('-inf')
        self.iteration_history = []

        self.models_to_run = self._cfg(
            'HYDROLOGICAL_MODEL',
            typed=lambda: self._typed_config.model.hydrological_model
        ).split(',')

        # Set random seed for reproducibility
        self.random_seed = self._cfg('RANDOM_SEED', default=None)
        if self.random_seed is not None and self.random_seed != 'None':
            self._set_random_seeds(self.random_seed)
            self.logger.info(f"Random seed set to: {self.random_seed}")

        # Parallel processing setup
        self.parallel_dirs = []
        self._consecutive_parallel_failures = 0
        
        if self.use_parallel:
            self._setup_parallel_processing()

    def _cfg(self, dict_key: str, default: Any = None, typed: Any = None) -> Any:
        """Get config value with typed config fallback to dict.

        Args:
            dict_key: Key to use for dict access
            default: Default value if not found
            typed: Lambda that accesses the typed config value

        Returns:
            Config value from typed config, dict, or default
        """
        if self._config is not None and typed is not None:
            try:
                value = typed()
                if value is not None:
                    return value
            except (AttributeError, TypeError):
                pass
        return self.config_dict.get(dict_key, default)

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Return the name of the optimization algorithm"""
        pass
    
    @abstractmethod
    def _run_algorithm(self) -> Tuple[Dict, float, List]:
        """Run the specific optimization algorithm"""
        pass
    
    def _set_random_seeds(self, seed: int) -> None:
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
    
    def _setup_optimization_directories(self) -> None:
        """Setup directory structure for optimization"""
        # Create all directories
        for directory in [self.optimization_dir, self.summa_sim_dir, self.mizuroute_sim_dir,
                         self.optimization_settings_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create log directories
        (self.summa_sim_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.mizuroute_sim_dir / "logs").mkdir(parents=True, exist_ok=True)
        
        # Copy settings files
        self._copy_settings_files()
        
        # Update file managers for optimization
        self._update_optimization_file_managers()
    
    def _ensure_reproducible_initialization(self):
        """Reset random seeds right before population initialization"""
        if self.random_seed is not None and self.random_seed != 'None':
            self._set_random_seeds(int(self.random_seed))
            self.logger.debug(f"Random state reset for population initialization")
        
    def _copy_settings_files(self) -> None:
        """Copy necessary settings files to optimization directory"""
        source_settings_dir = self.project_dir / "settings" / "SUMMA"

        if not source_settings_dir.exists():
            raise FileNotFoundError(f"Source settings directory not found: {source_settings_dir}")

        # Determine if glacier mode is enabled based on file manager name
        summa_fm = self._cfg('SETTINGS_SUMMA_FILEMANAGER', default='fileManager.txt', typed=lambda: self._typed_config.model.summa.filemanager if self._typed_config.model.summa else None)
        glacier_mode = 'glac' in summa_fm.lower()

        required_files = [
            'modelDecisions.txt', 'outputControl.txt',
            'localParamInfo.txt', 'basinParamInfo.txt',
        ]

        # Add appropriate file manager
        if summa_fm and summa_fm != 'default':
            required_files.append(summa_fm)
        else:
            required_files.append('fileManager.txt')

        # Add attributes and coldState files based on glacier mode
        if glacier_mode:
            # Glacier mode: prefer glacier-specific files, fallback to standard
            attr_files = ['attributes_glac.nc', 'attributes.nc']
            cold_files = ['coldState_glac.nc', 'coldState.nc']
        else:
            attr_files = ['attributes.nc']
            cold_files = ['coldState.nc']

        optional_files = [
            'TBL_GENPARM.TBL', 'TBL_MPTABLE.TBL', 'TBL_SOILPARM.TBL', 'TBL_VEGPARM.TBL',
            'trialParams.nc', 'forcingFileList.txt',
            # Glacier-specific optional files
            'attributes_glacBedTopo.nc', 'coldState_glacSurfTopo.nc',
        ]

        for file_name in required_files:
            source_path = source_settings_dir / file_name
            dest_path = self.optimization_settings_dir / file_name

            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                self.logger.debug(f"Copied required file: {file_name}")
            else:
                raise FileNotFoundError(f"Required SUMMA settings file not found: {source_path}")

        # Copy attributes files (glacier mode tries glacier files first)
        attr_copied = False
        for attr_file in attr_files:
            source_path = source_settings_dir / attr_file
            if source_path.exists():
                dest_path = self.optimization_settings_dir / attr_file
                shutil.copy2(source_path, dest_path)
                self.logger.debug(f"Copied attributes file: {attr_file}")
                attr_copied = True
                break
        if not attr_copied:
            raise FileNotFoundError(f"No attributes file found in {source_settings_dir}")

        # Copy coldState files (glacier mode tries glacier files first)
        cold_copied = False
        for cold_file in cold_files:
            source_path = source_settings_dir / cold_file
            if source_path.exists():
                dest_path = self.optimization_settings_dir / cold_file
                shutil.copy2(source_path, dest_path)
                self.logger.debug(f"Copied coldState file: {cold_file}")
                cold_copied = True
                break
        if not cold_copied:
            self.logger.warning("coldState.nc not found, depth calibration may fail if enabled.")

        for file_name in optional_files:
            source_path = source_settings_dir / file_name
            dest_path = self.optimization_settings_dir / file_name

            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                self.logger.debug(f"Copied optional file: {file_name}")
            else:
                self.logger.debug(f"Optional SUMMA settings file not found: {source_path}")
        
        source_mizu_dir = self.project_dir / "settings" / "mizuRoute"
        dest_mizu_dir = self.optimization_dir / "settings" / "mizuRoute"
        
        if source_mizu_dir.exists():
            dest_mizu_dir.mkdir(parents=True, exist_ok=True)
            for mizu_file in source_mizu_dir.glob("*"):
                if mizu_file.is_file():
                    shutil.copy2(mizu_file, dest_mizu_dir / mizu_file.name)
            self.logger.debug("Copied mizuRoute settings")

    def _update_optimization_file_managers(self) -> None:
        """Update file managers for optimization runs"""
        # Update SUMMA file manager - use config name if specified
        summa_fm_name = self._cfg('SETTINGS_SUMMA_FILEMANAGER', default='fileManager.txt', typed=lambda: self._typed_config.model.summa.filemanager if self._typed_config.model.summa else None)
        file_manager_path = self.optimization_settings_dir / summa_fm_name
        if not file_manager_path.exists():
            # Fallback to default
            file_manager_path = self.optimization_settings_dir / 'fileManager.txt'
        if file_manager_path.exists():
            self._update_summa_file_manager(file_manager_path)
        
        # Update mizuRoute control file if it exists
        mizu_control_path = self.optimization_dir / "settings" / "mizuRoute" / "mizuroute.control"
        if mizu_control_path.exists():
            self._update_mizuroute_control_file(mizu_control_path)

    def _adjust_end_time_for_forcing(self, end_time_str: str) -> str:
        """
        Adjust end time to align with forcing data timestep.
        For sub-daily forcing (e.g., 3-hourly CERRA), ensures end time is a valid timestep.

        Args:
            end_time_str: End time string (e.g., '2019-12-31 23:00')

        Returns:
            Adjusted end time string aligned with forcing timestep
        """
        try:
            forcing_timestep_seconds = self._cfg('FORCING_TIME_STEP_SIZE', default=3600, typed=lambda: self._typed_config.forcing.time_step_size)

            if forcing_timestep_seconds >= 3600:  # Hourly or coarser
                # Parse the end time
                end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')

                # Calculate the last valid hour based on timestep
                forcing_timestep_hours = forcing_timestep_seconds / 3600
                last_hour = int(24 - (24 % forcing_timestep_hours)) - forcing_timestep_hours
                if last_hour < 0:
                    last_hour = 0

                # Adjust if needed
                if end_time.hour > last_hour or (end_time.hour == 23 and last_hour < 23):
                    end_time = end_time.replace(hour=int(last_hour), minute=0)
                    adjusted_str = end_time.strftime('%Y-%m-%d %H:%M')
                    self.logger.info(f"Adjusted end time from {end_time_str} to {adjusted_str} for {forcing_timestep_hours}h forcing")
                    return adjusted_str

            return end_time_str

        except Exception as e:
            self.logger.warning(f"Could not adjust end time: {e}")
            return end_time_str

    def _update_summa_file_manager(self, file_manager_path: Path, use_calibration_period: bool = True) -> None:
        """Update SUMMA file manager with spinup + calibration period"""
        with open(file_manager_path, 'r') as f:
            lines = f.readlines()
        
        if use_calibration_period:
            calibration_period_str = self._cfg('CALIBRATION_PERIOD', default='', typed=lambda: self._typed_config.domain.calibration_period)
            spinup_period_str = self._cfg('SPINUP_PERIOD', default='', typed=lambda: self._typed_config.domain.spinup_period)
            
            if calibration_period_str and spinup_period_str:
                try:
                    spinup_dates = [d.strip() for d in spinup_period_str.split(',')]
                    cal_dates = [d.strip() for d in calibration_period_str.split(',')]
                    
                    if len(spinup_dates) >= 2 and len(cal_dates) >= 2:
                        spinup_start = datetime.strptime(spinup_dates[0], '%Y-%m-%d').replace(hour=1, minute=0)

                        # Adjust end time based on forcing timestep
                        # For sub-daily forcing, we need to align with available timesteps
                        forcing_timestep_seconds = self._cfg('FORCING_TIME_STEP_SIZE', default=3600, typed=lambda: self._typed_config.forcing.time_step_size)
                        if forcing_timestep_seconds >= 3600:  # Hourly or coarser
                            # Calculate the last valid hour of the day
                            forcing_timestep_hours = forcing_timestep_seconds / 3600
                            last_hour = int(24 - (24 % forcing_timestep_hours)) - forcing_timestep_hours
                            if last_hour < 0:
                                last_hour = 0
                            cal_end = datetime.strptime(cal_dates[1], '%Y-%m-%d').replace(hour=int(last_hour), minute=0)
                        else:
                            cal_end = datetime.strptime(cal_dates[1], '%Y-%m-%d').replace(hour=23, minute=0)

                        sim_start = spinup_start.strftime('%Y-%m-%d %H:%M')
                        sim_end = cal_end.strftime('%Y-%m-%d %H:%M')

                        self.logger.info(f"Using spinup + calibration period: {sim_start} to {sim_end}")
                    else:
                        raise ValueError("Invalid period format")
                        
                except Exception as e:
                    self.logger.warning(f"Could not parse spinup+calibration periods: {str(e)}")
                    sim_start = self._cfg('EXPERIMENT_TIME_START', default='1980-01-01 01:00', typed=lambda: str(self._typed_config.domain.time_start))
                    sim_end = self._adjust_end_time_for_forcing(self._cfg('EXPERIMENT_TIME_END', default='2018-12-31 23:00', typed=lambda: str(self._typed_config.domain.time_end)))
            else:
                sim_start = self._cfg('EXPERIMENT_TIME_START', default='1980-01-01 01:00', typed=lambda: str(self._typed_config.domain.time_start))
                sim_end = self._adjust_end_time_for_forcing(self._cfg('EXPERIMENT_TIME_END', default='2018-12-31 23:00', typed=lambda: str(self._typed_config.domain.time_end)))
        else:
            sim_start = self._cfg('EXPERIMENT_TIME_START', default='1980-01-01 01:00', typed=lambda: str(self._typed_config.domain.time_start))
            sim_end = self._adjust_end_time_for_forcing(self._cfg('EXPERIMENT_TIME_END', default='2018-12-31 23:00', typed=lambda: str(self._typed_config.domain.time_end)))
        
        updated_lines = []
        for line in lines:
            if 'simStartTime' in line:
                updated_lines.append(f"simStartTime         '{sim_start}'\n")
            elif 'simEndTime' in line:
                updated_lines.append(f"simEndTime           '{sim_end}'\n")
            elif 'outputPath' in line:
                # Always use summa_sim_dir for SUMMA output - evaluators look there
                output_path = str(self.summa_sim_dir).replace('\\', '/')
                updated_lines.append(f"outputPath           '{output_path}/'\n")
            elif 'settingsPath' in line:
                settings_path = str(self.optimization_settings_dir).replace('\\', '/')
                updated_lines.append(f"settingsPath         '{settings_path}/'\n")
            else:
                updated_lines.append(line)
        
        with open(file_manager_path, 'w') as f:
            f.writelines(updated_lines)
        
    def _update_mizuroute_control_file(self, control_path: Path) -> None:
        """Update mizuRoute control file and ensure comments are present."""
        
        def _normalize_path(path):
            return str(path).replace("\\", "/").rstrip("/") + "/"
        
        with open(control_path, "r") as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines):
            if line.strip().startswith('<input_dir>') :
                # For non-parallel runs, mizuRoute reads from output_dir where SUMMA writes
                if hasattr(self, 'use_parallel') and not self.use_parallel:
                    new_path = _normalize_path(self.output_dir)
                else:
                    new_path = _normalize_path(self.summa_sim_dir)
                if '!' in line:
                    pre_comment = line.split('!')[0].strip()
                    comment = '!' + '!'.join(line.split('!')[1:])
                    lines[i] = f"<input_dir>             {new_path}    {comment}"
                else:
                    lines[i] = f"<input_dir>             {new_path}    ! Folder that contains runoff data from SUMMA\n"

            elif line.strip().startswith('<output_dir>') :
                # For non-parallel runs, mizuRoute writes to output_dir/mizuRoute
                if hasattr(self, 'use_parallel') and not self.use_parallel:
                    new_path = _normalize_path(self.output_dir / "mizuRoute")
                else:
                    new_path = _normalize_path(self.mizuroute_sim_dir)
                if '!' in line:
                    pre_comment = line.split('!')[0].strip()
                    comment = '!' + '!'.join(line.split('!')[1:])
                    lines[i] = f"<output_dir>            {new_path}    {comment}"
                else:
                    lines[i] = f"<output_dir>            {new_path}    ! Folder that will contain mizuRoute simulations\n"
        
        with open(control_path, "w", encoding="ascii", newline="\n") as f:
            f.writelines(lines)

    def _create_calibration_target(self) -> CalibrationTarget:
        """Factory method to create appropriate calibration target"""
        optimization_target = self._cfg('OPTIMIZATION_TARGET', default='streamflow', typed=lambda: self._typed_config.optimization.target).lower()
        calibration_variable = self._cfg('CALIBRATION_VARIABLE', default='streamflow', typed=lambda: self._typed_config.optimization.calibration_variable).lower()

        if optimization_target in ['et', 'latent_heat']:
            return ETTarget(self.config, self.project_dir, self.logger)
        elif (
            optimization_target in ['swe', 'sca', 'snow_depth'] or
            'swe' in calibration_variable or 'snow' in calibration_variable):
            return SnowTarget(self.config, self.project_dir, self.logger)
        elif optimization_target in ['gw_depth', 'gw_grace']:
            return GroundwaterTarget(self.config, self.project_dir, self.logger)
        elif optimization_target in ['sm_point', 'sm_smap', 'sm_esa', 'sm_ismn']:
            return SoilMoistureTarget(self.config, self.project_dir, self.logger)
        elif optimization_target in ['tws', 'grace', 'grace_tws', 'total_storage', 'stor_grace']:
            return TWSTarget(self.config, self.project_dir, self.logger)
        elif optimization_target == 'streamflow' or 'flow' in calibration_variable:
            return StreamflowTarget(self.config, self.project_dir, self.logger)
        else:
            if 'streamflow' in calibration_variable or 'flow' in calibration_variable:
                return StreamflowTarget(self.config, self.project_dir, self.logger)
            else:
                raise ValueError(f"Unsupported optimization target: {optimization_target}")
                
    def _setup_parallel_processing(self) -> None:
        """Setup parallel processing directories and files"""
        self.logger.info(f"Setting up parallel processing with {self.num_processes} processes")
        
        for proc_id in range(self.num_processes):
            proc_base_dir = self.optimization_dir / f"parallel_proc_{proc_id:02d}"
            proc_summa_dir = proc_base_dir / "SUMMA"
            proc_mizuroute_dir = proc_base_dir / "mizuRoute"
            proc_summa_settings_dir = proc_base_dir / "settings" / "SUMMA"
            proc_mizu_settings_dir = proc_base_dir / "settings" / "mizuRoute"
            
            for directory in [proc_base_dir, proc_summa_dir, proc_mizuroute_dir,
                            proc_summa_settings_dir, proc_mizu_settings_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            (proc_summa_dir / "logs").mkdir(parents=True, exist_ok=True)
            (proc_mizuroute_dir / "logs").mkdir(parents=True, exist_ok=True)
            
            self._copy_settings_to_process_dir(proc_summa_settings_dir, proc_mizu_settings_dir)
            self._update_process_file_managers(proc_id, proc_summa_dir, proc_mizuroute_dir,
                                            proc_summa_settings_dir, proc_mizu_settings_dir)
            
            self.parallel_dirs.append({
                'proc_id': proc_id,
                'base_dir': proc_base_dir,
                'summa_dir': proc_summa_dir,
                'mizuroute_dir': proc_mizuroute_dir,
                'summa_settings_dir': proc_summa_settings_dir,
                'mizuroute_settings_dir': proc_mizu_settings_dir
            })
    
    def _copy_settings_to_process_dir(self, proc_summa_settings_dir: Path, proc_mizu_settings_dir: Path) -> None:
        """Copy settings files to process-specific directory"""
        if self.optimization_settings_dir.exists():
            for settings_file in self.optimization_settings_dir.glob("*"):
                if settings_file.is_file():
                    dest_file = proc_summa_settings_dir / settings_file.name
                    shutil.copy2(settings_file, dest_file)
        
        mizu_source_dir = self.optimization_dir / "settings" / "mizuRoute"
        if mizu_source_dir.exists():
            for settings_file in mizu_source_dir.glob("*"):
                if settings_file.is_file():
                    dest_file = proc_mizu_settings_dir / settings_file.name
                    shutil.copy2(settings_file, dest_file)
        
    def _update_process_file_managers(self, proc_id: int, summa_dir: Path, mizuroute_dir: Path,
                                    summa_settings_dir: Path, mizu_settings_dir: Path) -> None:
        """Update file managers for a specific process"""
        summa_fm_name = self._cfg('SETTINGS_SUMMA_FILEMANAGER', default='fileManager.txt', typed=lambda: self._typed_config.model.summa.filemanager if self._typed_config.model.summa else None)
        file_manager = summa_settings_dir / summa_fm_name
        if not file_manager.exists():
            file_manager = summa_settings_dir / 'fileManager.txt'
        if file_manager.exists():
            with open(file_manager, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                if 'outFilePrefix' in line:
                    updated_lines.append(f"outFilePrefix        'proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}'\n")
                elif 'outputPath' in line:
                    output_path = str(summa_dir).replace('\\', '/')
                    updated_lines.append(f"outputPath           '{output_path}/'\n")
                elif 'settingsPath' in line:
                    settings_path = str(summa_settings_dir).replace('\\', '/')
                    updated_lines.append(f"settingsPath         '{settings_path}/'\n")
                else:
                    updated_lines.append(line)
            
            with open(file_manager, 'w') as f:
                f.writelines(updated_lines)
        
        control_file = mizu_settings_dir / 'mizuroute.control'
        def _normalize_path(path):
            return str(path).replace("\\", "/").rstrip("/") + "/"

        if control_file.exists():
            with open(control_file, 'r') as f:
                lines = f.readlines()
            
            updated_lines = []
            for line in lines:
                if '<input_dir>' in line:
                    input_path = _normalize_path(summa_dir)
                    if '!' in line:
                        comment = '!' + '!'.join(line.split('!')[1:])
                        updated_lines.append(f"<input_dir>             {input_path}    {comment}")
                    else:
                        updated_lines.append(f"<input_dir>             {input_path}    ! Folder that contains runoff data from SUMMA\n")
                elif '<output_dir>' in line:
                    output_path = _normalize_path(mizuroute_dir)
                    if '!' in line:
                        comment = '!' + '!'.join(line.split('!')[1:])
                        updated_lines.append(f"<output_dir>            {output_path}    {comment}")
                    else:
                        updated_lines.append(f"<output_dir>            {output_path}    ! Folder that will contain mizuRoute simulations\n")
                elif '<case_name>' in line:
                    if '!' in line:
                        comment = '!' + '!'.join(line.split('!')[1:])
                        updated_lines.append(f"<case_name>             proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}    {comment}")
                    else:
                        updated_lines.append(f"<case_name>             proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}    ! Simulation case name\n")
                elif '<fname_qsim>' in line:
                    if '!' in line:
                        comment = '!' + '!'.join(line.split('!')[1:])
                        updated_lines.append(f"<fname_qsim>            proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}_timestep.nc    {comment}")
                    else:
                        updated_lines.append(f"<fname_qsim>            proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}_timestep.nc    ! netCDF name for HM_HRU runoff\n")
                else:
                    updated_lines.append(line)
            
            with open(control_file, 'w') as f:
                f.writelines(updated_lines)
    
    def _cleanup_parallel_processing(self) -> None:
        """Cleanup parallel processing directories"""
        if not self.use_parallel:
            return
        
        self.logger.info("Cleaning up parallel working directories")
        cleanup_parallel = self._cfg('CLEANUP_PARALLEL_DIRS', default=True, typed=lambda: self._typed_config.optimization.cleanup_parallel_dirs)
        if cleanup_parallel:
            try:
                for proc_dirs in self.parallel_dirs:
                    if proc_dirs['base_dir'].exists():
                        shutil.rmtree(proc_dirs['base_dir'])
            except Exception as e:
                self.logger.warning(f"Error during parallel cleanup: {str(e)}")
    
    def _evaluate_individual(self, normalized_params: np.ndarray) -> float:
        """Evaluate a single parameter set (sequential mode)"""
        try:
            params = self.parameter_manager.denormalize_parameters(normalized_params)
            
            if not self._apply_parameters(params):
                return float('-inf')
            
            if not self.model_executor.run_models(
                self.summa_sim_dir, 
                self.mizuroute_sim_dir, 
                self.optimization_settings_dir
            ):
                return float('-inf')
            
            metrics = self.calibration_target.calculate_metrics(
                self.summa_sim_dir,
                mizuroute_dir=self.mizuroute_sim_dir
            )
            if not metrics:
                return float('-inf')
            
            score = self._extract_target_metric(metrics)
            
            if self.target_metric.upper() in ['RMSE', 'MAE', 'PBIAS']:
                score = -score
            
            return score if score is not None and not np.isnan(score) else float('-inf')
            
        except Exception as e:
            self.logger.debug(f"Parameter evaluation failed: {str(e)}")
            return float('-inf')
    
    def _extract_target_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        """Extract target metric from metrics dictionary"""
        if self.target_metric in metrics:
            return metrics[self.target_metric]
        
        calib_key = f"Calib_{self.target_metric}"
        if calib_key in metrics:
            return metrics[calib_key]
        
        for key, value in metrics.items():
            if key.endswith(f"_{self.target_metric}"):
                return value
        
        return next(iter(metrics.values())) if metrics else None

    def _run_parallel_evaluations(self, evaluation_tasks: List[Dict]) -> List[Dict]:
        """Complete parallel evaluation with MPI framework"""
        
        start_time = time.time()
        num_tasks = len(evaluation_tasks)
        
        if not hasattr(self, '_consecutive_parallel_failures'):
            self._consecutive_parallel_failures = 0
        
        self.logger.info(f"Starting parallel evaluation of {num_tasks} tasks with {self.num_processes} processes")
        
        effective_processes = self.num_processes
        
        worker_tasks = []
        for task in evaluation_tasks:
            proc_dirs = self.parallel_dirs[task['proc_id']]
            
            task_data = {
                'individual_id': task['individual_id'],
                'params': task['params'],
                'proc_id': task['proc_id'],
                'evaluation_id': task['evaluation_id'],
                'multiobjective': task.get('multiobjective', False),
                'objective_names': task.get('objective_names'),
                'summa_exe': str(self._get_summa_exe_path()),
                'file_manager': str(proc_dirs['summa_settings_dir'] / self._cfg('SETTINGS_SUMMA_FILEMANAGER', default='fileManager.txt', typed=lambda: self._typed_config.model.summa.filemanager if self._typed_config.model.summa else None)),
                'summa_dir': str(proc_dirs['summa_dir']),
                'mizuroute_dir': str(proc_dirs['mizuroute_dir']),
                'summa_settings_dir': str(proc_dirs['summa_settings_dir']),
                'mizuroute_settings_dir': str(proc_dirs['mizuroute_settings_dir']),
                # Use flattened config dict for worker compatibility (workers expect dict.get())
                'config': self.config_dict,
                'target_metric': self.target_metric,
                'calibration_variable': self._cfg('CALIBRATION_VARIABLE', default='streamflow', typed=lambda: self._typed_config.optimization.calibration_variable),
                'domain_name': self.domain_name,
                'project_dir': str(self.project_dir),
                'original_depths': self.parameter_manager.original_depths.tolist() if self.parameter_manager.original_depths is not None else None,
                # Multi-target optimization settings (NSGA-II with different targets for each objective)
                'multi_target_mode': self.config_dict.get('NSGA2_MULTI_TARGET', False),
                'primary_target_type': self.config_dict.get('NSGA2_PRIMARY_TARGET', self.config_dict.get('OPTIMIZATION_TARGET', 'streamflow')),
                'secondary_target_type': self.config_dict.get('NSGA2_SECONDARY_TARGET', self.config_dict.get('OPTIMIZATION_TARGET2', 'gw_depth')),
                'primary_metric': self.config_dict.get('NSGA2_PRIMARY_METRIC', self.config_dict.get('OPTIMIZATION_METRIC', 'KGE')),
                'secondary_metric': self.config_dict.get('NSGA2_SECONDARY_METRIC', self.config_dict.get('OPTIMIZATION_METRIC2', 'KGE')),
            }
            if self.random_seed is not None and self.random_seed != 'None':
                task_data['random_seed'] = int(self.random_seed) + task['individual_id'] + 1000
            
            worker_tasks.append(task_data)
            
        results = []
        try:
            batch_size = min(effective_processes, 100)
            for batch_start in range(0, len(worker_tasks), batch_size):
                batch_end = min(batch_start + batch_size, len(worker_tasks))
                batch_tasks = worker_tasks[batch_start:batch_end]
                
                if batch_start > 0:
                    time.sleep(1.0)
                
                batch_results = self._execute_batch_mpi(batch_tasks, len(batch_tasks))
                results.extend(batch_results)
            
            successful_count = sum(1 for r in results if r['score'] is not None)
            success_rate = successful_count / num_tasks if num_tasks > 0 else 0
            
            elapsed = time.time() - start_time
            self.logger.info(f"Parallel evaluation completed: {successful_count}/{num_tasks} successful ({100*success_rate:.1f}%) in {elapsed/60:.1f} minutes")
            
            if success_rate >= 0.7:
                self._consecutive_parallel_failures = 0
            else:
                self._consecutive_parallel_failures += 1
                
            return results
            
        except Exception as e:
            self.logger.error(f"Critical error in parallel evaluation: {str(e)}")
            self._consecutive_parallel_failures += 1
            return [
                {
                    'individual_id': task['individual_id'],
                    'params': task['params'],
                    'score': None,
                    'error': f'Critical parallel evaluation error: {str(e)}',
                    'runtime': None
                }
                for task in evaluation_tasks
            ]

    def _execute_batch_mpi(self, batch_tasks: List[Dict], max_workers: int) -> List[Dict]:
        """Spawn MPI processes internally for parallel execution"""
        try:
            num_processes = min(max_workers, self.num_processes, len(batch_tasks))
            import uuid
            work_dir = Path.cwd()
            unique_id = uuid.uuid4().hex[:8]
            
            tasks_file = work_dir / f'mpi_tasks_{unique_id}.pkl'
            results_file = work_dir / f'mpi_results_{unique_id}.pkl'
            worker_script = work_dir / f'mpi_worker_{unique_id}.py'
            
            try:
                with open(tasks_file, 'wb') as f:
                    pickle.dump(batch_tasks, f)
                
                self._create_mpi_worker_script(worker_script, tasks_file, work_dir)
                os.chmod(worker_script, 0o755)
                
                mpi_cmd = ['mpirun', '-n', str(num_processes), sys.executable, str(worker_script), str(tasks_file), str(results_file)]
                
                result = subprocess.run(mpi_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, env=os.environ.copy())
                
                if results_file.exists():
                    with open(results_file, 'rb') as f:
                        return pickle.load(f)
                    return []
                else:
                    raise RuntimeError("MPI results file not created")
            finally:
                for file_path in [tasks_file, results_file, worker_script]:
                    if file_path.exists():
                        file_path.unlink()
        except Exception as e:
            self.logger.error(f"MPI spawn execution failed: {str(e)}")
            return [{'individual_id': t['individual_id'], 'params': t['params'], 'score': None, 'error': str(e)} for t in batch_tasks]

    def _create_mpi_worker_script(self, script_path: Path, tasks_file: Path, temp_dir: Path) -> None:
        """Create the MPI worker script file"""
        script_content = f'''#!/usr/bin/env python3
import sys
import pickle
import os
from pathlib import Path
from mpi4py import MPI
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

sys.path.append(r"{Path(__file__).parent.parent.parent.parent.parent}") # Add src path

from symfluence.optimization.workers.summa_parallel_workers import _evaluate_parameters_worker_safe

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    tasks_file = Path(sys.argv[1])
    results_file = Path(sys.argv[2])
    
    if rank == 0:
        with open(tasks_file, 'rb') as f:
            all_tasks = pickle.load(f)
        
        tasks_per_rank = len(all_tasks) // size
        extra_tasks = len(all_tasks) % size
        all_results = []
        
        for worker_rank in range(size):
            start_idx = worker_rank * tasks_per_rank + min(worker_rank, extra_tasks)
            end_idx = start_idx + tasks_per_rank + (1 if worker_rank < extra_tasks else 0)
            
            if worker_rank == 0:
                my_tasks = all_tasks[start_idx:end_idx]
            else:
                comm.send(all_tasks[start_idx:end_idx], dest=worker_rank, tag=1)
        
        for task in my_tasks:
            all_results.append(_evaluate_parameters_worker_safe(task))
            
        for worker_rank in range(1, size):
            all_results.extend(comm.recv(source=worker_rank, tag=2))
            
        with open(results_file, 'wb') as f:
            pickle.dump(all_results, f)
    else:
        my_tasks = comm.recv(source=0, tag=1)
        my_results = [_evaluate_parameters_worker_safe(t) for t in my_tasks]
        comm.send(my_results, dest=0, tag=2)

if __name__ == "__main__":
    main()
'''
        with open(script_path, 'w') as f:
            f.write(script_content)

    def _run_final_evaluation(self, best_params: Dict) -> Optional[Dict]:
        """Run final evaluation with best parameters over full period"""
        self.logger.info("Running final evaluation with best parameters")
        
        try:
            self._update_file_manager_for_final_run()
            
            if self._cfg('FINAL_EVALUATION_NUMERICAL_METHOD', default='ida', typed=lambda: self._typed_config.optimization.final_evaluation_numerical_method) == 'ida':
                self._update_model_decisions_for_final_run()
            
            if not self._apply_parameters(best_params):
                return None
            
            self._update_mizuroute_control_file_for_final()
            
            if not self.model_executor.run_models(self.summa_sim_dir, self.mizuroute_sim_dir, self.optimization_settings_dir):
                return None
            
            metrics = self.calibration_target.calculate_metrics(
                self.summa_sim_dir,
                mizuroute_dir=self.mizuroute_sim_dir,
                calibration_only=False
            )
            
            if metrics:
                return {
                    'final_metrics': metrics,
                    'summa_success': True,
                    'mizuroute_success': self.calibration_target.needs_routing(),
                    'calibration_metrics': self._extract_period_metrics(metrics, 'Calib'),
                    'evaluation_metrics': self._extract_period_metrics(metrics, 'Eval'),
                }
            return None
        finally:
            self._restore_model_decisions_for_optimization()
            summa_fm_name = self._cfg('SETTINGS_SUMMA_FILEMANAGER', default='fileManager.txt', typed=lambda: self._typed_config.model.summa.filemanager if self._typed_config.model.summa else None)
            fm_path = self.optimization_settings_dir / summa_fm_name
            if not fm_path.exists():
                fm_path = self.optimization_settings_dir / 'fileManager.txt'
            if fm_path.exists():
                self._update_summa_file_manager(fm_path)

    def _extract_period_metrics(self, all_metrics: Dict, period_prefix: str) -> Dict:
        """Extract metrics for a specific period (Calib or Eval)"""
        period_metrics = {}
        for key, value in all_metrics.items():
            if key.startswith(f"{period_prefix}_"):
                period_metrics[key.replace(f"{period_prefix}_", "")] = value
            elif period_prefix == 'Calib' and not any(key.startswith(p) for p in ['Calib_', 'Eval_']):
                period_metrics[key] = value
        return period_metrics

    def _update_model_decisions_for_final_run(self) -> None:
        """Update modelDecisions.txt to use direct solver for final evaluation"""
        model_decisions_path = self.optimization_settings_dir / 'modelDecisions.txt'
        if not model_decisions_path.exists(): return
        try:
            with open(model_decisions_path, 'r') as f:
                lines = f.readlines()
            updated_lines = []
            for line in lines:
                if line.strip().startswith('num_method') and not line.strip().startswith('!'):
                    updated_lines.append(re.sub(r'(num_method\s+)\w+(\s+.*)', r'\1ida\2', line))
                else:
                    updated_lines.append(line)
            with open(model_decisions_path, 'w') as f:
                f.writelines(updated_lines)
        except Exception as e:
            self.logger.error(f"Error updating modelDecisions.txt: {str(e)}")

    def _restore_model_decisions_for_optimization(self) -> None:
        """Restore modelDecisions.txt to use iterative solver for optimization"""
        model_decisions_path = self.optimization_settings_dir / 'modelDecisions.txt'
        if not model_decisions_path.exists(): return
        try:
            with open(model_decisions_path, 'r') as f:
                lines = f.readlines()
            updated_lines = []
            for line in lines:
                if line.strip().startswith('num_method') and not line.strip().startswith('!'):
                    updated_lines.append(re.sub(r'(num_method\s+)\w+(\s+.*)', r'\1itertive\2', line))
                else:
                    updated_lines.append(line)
            with open(model_decisions_path, 'w') as f:
                f.writelines(updated_lines)
        except Exception as e:
            self.logger.error(f"Error restoring modelDecisions.txt: {str(e)}")

    def _update_mizuroute_control_file_for_final(self) -> None:
        """Update mizuRoute control file for final evaluation"""
        mizu_control_path = self.optimization_settings_dir.parent / "mizuRoute" / "mizuroute.control"
        if not mizu_control_path.exists(): return
        final_prefix = f'run_{self.algorithm_name}_final_{self.experiment_id}'
        with open(mizu_control_path, 'r') as f:
            lines = f.readlines()
        updated_lines = []
        for line in lines:
            if line.strip().startswith('<input_dir>') :
                updated_lines.append(f"<input_dir>             {str(self.summa_sim_dir)}/    ! Folder\n")
            elif line.strip().startswith('<case_name>') :
                updated_lines.append(f"<case_name>             {final_prefix}    ! Name\n")
            elif line.strip().startswith('<fname_qsim>') :
                updated_lines.append(f"<fname_qsim>            {final_prefix}_timestep.nc    ! File\n")
            else:
                updated_lines.append(line)
        with open(mizu_control_path, 'w') as f:
            f.writelines(updated_lines)

    def _apply_parameters(self, best_params: Dict) -> bool:
        """Apply parameters with support for transformations"""
        try:
            if not self.transformation_manager.transform(best_params, self.optimization_settings_dir):
                return False
            
            if (
                hasattr(self.parameter_manager, 'depth_params') and self.parameter_manager.depth_params and 
                'total_mult' in best_params and 'shape_factor' in best_params):
                if not self.parameter_manager._update_soil_depths(best_params):
                    return False
            
            if (hasattr(self.parameter_manager, 'mizuroute_params') and self.parameter_manager.mizuroute_params):
                if not self.parameter_manager._update_mizuroute_parameters(best_params):
                    return False
            
            exclusion_params = []
            if hasattr(self.parameter_manager, 'depth_params'):
                exclusion_params.extend(self.parameter_manager.depth_params)
            if hasattr(self.parameter_manager, 'mizuroute_params'):
                exclusion_params.extend(self.parameter_manager.mizuroute_params)
            
            hydrological_params = {k: v for k, v in best_params.items() if k not in exclusion_params}
            if hydrological_params:
                if not self.parameter_manager._generate_trial_params_file(hydrological_params):
                    return False
            return True
        except Exception:
            return False

    def _update_file_manager_for_final_run(self) -> None:
        """Update file manager to use full experiment period"""
        summa_fm_name = self._cfg('SETTINGS_SUMMA_FILEMANAGER', default='fileManager.txt', typed=lambda: self._typed_config.model.summa.filemanager if self._typed_config.model.summa else None)
        file_manager_path = self.optimization_settings_dir / summa_fm_name
        if not file_manager_path.exists():
            file_manager_path = self.optimization_settings_dir / 'fileManager.txt'
        if not file_manager_path.exists(): return
        with open(file_manager_path, 'r') as f:
            lines = f.readlines()
        sim_start = self._cfg('EXPERIMENT_TIME_START', default='1980-01-01 01:00', typed=lambda: str(self._typed_config.domain.time_start))
        sim_end = self._cfg('EXPERIMENT_TIME_END', default='2018-12-31 23:00', typed=lambda: str(self._typed_config.domain.time_end))
        updated_lines = []
        for line in lines:
            if 'simStartTime' in line:
                updated_lines.append(f"simStartTime         '{sim_start}'\n")
            elif 'simEndTime' in line:
                updated_lines.append(f"simEndTime           '{sim_end}'\n")
            elif 'outFilePrefix' in line:
                updated_lines.append(f"outFilePrefix        'run_{self.algorithm_name}_final_{self.experiment_id}'\n")
            else:
                updated_lines.append(line)
        with open(file_manager_path, 'w') as f:
            f.writelines(updated_lines)
    
    def _save_to_default_settings(self, best_params: Dict) -> bool:
        """Save best parameters to default model settings"""
        try:
            default_settings_dir = self.project_dir / "settings" / "SUMMA"
            if not default_settings_dir.exists(): return False
            hydrological_params = {k: v for k, v in best_params.items() if k not in ['total_mult', 'shape_factor']}
            if hydrological_params:
                param_manager = ParameterManager(self.config, self.logger, default_settings_dir)
                param_manager._generate_trial_params_file(hydrological_params)
            return True
        except Exception:
            return False
    
    def _get_summa_exe_path(self) -> Path:
        """Get SUMMA executable path"""
        summa_path = self._cfg(
            'SUMMA_INSTALL_PATH',
            typed=lambda: self._typed_config.model.summa.install_path if self._typed_config.model.summa else None
        )
        if summa_path == 'default':
            summa_path = Path(self._cfg(
                'SYMFLUENCE_DATA_DIR',
                typed=lambda: self._typed_config.system.data_dir
            )) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)
        summa_exe = self._cfg(
            'SUMMA_EXE',
            typed=lambda: self._typed_config.model.summa.exe if self._typed_config.model.summa else None
        )
        return summa_path / summa_exe
        
    def run_optimization(self) -> Dict[str, Any]:
        """Main optimization workflow"""
        algorithm_name = self.get_algorithm_name()
        try:
            start_time = datetime.now()
            initial_params = self.parameter_manager.get_initial_parameters()
            if not initial_params:
                raise RuntimeError("Failed to get initial parameters")
            
            best_params, best_score, history = self._run_algorithm()
            final_result = self._run_final_evaluation(best_params)
            self.results_manager.save_results(best_params, best_score, history, final_result)
            self._save_to_default_settings(best_params)
            
            duration = datetime.now() - start_time
            if self.scratch_manager.use_scratch:
                self.scratch_manager.stage_results_back()
            
            return {
                'best_parameters': best_params, 'best_score': best_score, 'history': history,
                'final_result': final_result, 'algorithm': algorithm_name, 'duration': str(duration),
                'output_dir': str(self.output_dir)
            }
        finally:
            self._cleanup_parallel_processing()

    def _log_final_optimization_summary(self, algorithm_name: str, best_score: float, 
                                    final_result: Optional[Dict], duration) -> None:
        """Log final summary - kept for compatibility"""
        self.logger.info(f"Optimization {algorithm_name} completed in {duration}")
        if final_result and 'final_metrics' in final_result:
            calib_target = final_result['calibration_metrics'].get(self.target_metric, 0)
            eval_target = final_result['evaluation_metrics'].get(self.target_metric, 0)
            self.logger.info(f"   Final calibration: {self.target_metric} = {calib_target:.6f}")
            self.logger.info(f"   Final evaluation: {self.target_metric} = {eval_target:.6f}")
