"""
Configuration Updater

Updates model configuration files for parallel process directories.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.mixins import ConfigMixin


class ConfigurationUpdater(ConfigMixin):
    """
    Updates model configuration files for parallel execution.

    Handles updating file managers, mizuRoute control files, and other
    model-specific configurations to point to process-specific directories.
    """

    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        """
        Initialize configuration updater.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger or logging.getLogger(__name__)

    def update_file_managers(
        self,
        parallel_dirs: Dict[int, Dict[str, Path]],
        model_name: str,
        experiment_id: str,
        file_manager_name: str = 'fileManager.txt'
    ) -> None:
        """
        Update file manager paths in process-specific directories.

        Updates settingsPath, outputPath, outFilePrefix, and simulation times
        to point to process-specific directories and use calibration period.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
            experiment_id: Experiment identifier
            file_manager_name: Name of the file manager file
        """
        # Get calibration period from config
        cal_start, cal_end = self._get_calibration_period()

        for proc_id, dirs in parallel_dirs.items():
            file_manager_path = dirs['settings_dir'] / file_manager_name

            if not file_manager_path.exists():
                self.logger.warning(
                    f"File manager not found for process {proc_id}: {file_manager_path}"
                )
                continue

            try:
                with open(file_manager_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                updated_lines = self._update_file_manager_lines(
                    lines, dirs, model_name, experiment_id, proc_id, cal_start, cal_end
                )

                with open(file_manager_path, 'w', encoding='utf-8') as f:
                    f.writelines(updated_lines)

                if cal_start and cal_end:
                    self.logger.debug(
                        f"Updated file manager for process {proc_id} with calibration period: "
                        f"{cal_start} to {cal_end}"
                    )
                else:
                    self.logger.debug(
                        f"Updated file manager for process {proc_id}: {file_manager_path}"
                    )

            except (KeyError, ValueError, IOError) as e:
                self.logger.error(
                    f"Failed to update file manager for process {proc_id}: {e}"
                )

    def _get_calibration_period(self) -> tuple:
        """Extract calibration period start and end from config."""
        cal_period = self._get_config_value(lambda: self.config.domain.calibration_period, default='', dict_key='CALIBRATION_PERIOD')
        spinup_period = self._get_config_value(lambda: self.config.domain.spinup_period, default='', dict_key='SPINUP_PERIOD')

        cal_start = None
        cal_end = None

        if spinup_period:
            spinup_parts = [p.strip() for p in spinup_period.split(',')]
            if len(spinup_parts) >= 1:
                cal_start = spinup_parts[0]
                # Preserve time-of-day from EXPERIMENT_TIME_START to avoid
                # SUMMA forcing misalignment (e.g., hourly forcing begins at 01:00)
                exp_start = self._get_config_value(lambda: self.config.domain.time_start, dict_key='EXPERIMENT_TIME_START')
                if exp_start and isinstance(exp_start, str):
                    try:
                        exp_dt = datetime.strptime(exp_start, '%Y-%m-%d %H:%M')
                        start_dt = datetime.strptime(cal_start, '%Y-%m-%d')
                        start_dt = start_dt.replace(hour=exp_dt.hour, minute=exp_dt.minute)
                        cal_start = start_dt.strftime('%Y-%m-%d %H:%M')
                    except (KeyError, ValueError, IOError):
                        # If parsing fails, keep original date string
                        pass

        if cal_period:
            cal_parts = [p.strip() for p in cal_period.split(',')]
            if len(cal_parts) >= 2:
                cal_end = cal_parts[1]

        # Adjust end time to align with forcing timestep
        if cal_end:
            cal_end = self._adjust_end_time_for_forcing(cal_end)

        return cal_start, cal_end

    def _adjust_end_time_for_forcing(self, cal_end: str) -> str:
        """Adjust end time to align with forcing timestep."""
        try:
            forcing_timestep = self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')
            if forcing_timestep >= 3600:  # Hourly or coarser
                end_dt = datetime.strptime(cal_end, '%Y-%m-%d')
                forcing_hours = forcing_timestep / 3600
                last_hour = int(24 - (24 % forcing_hours)) - forcing_hours
                if last_hour < 0:
                    last_hour = 0
                cal_end = end_dt.strftime('%Y-%m-%d') + f' {int(last_hour):02d}:00'
        except (KeyError, ValueError, IOError):
            pass  # Keep original if adjustment fails
        return cal_end

    def _update_file_manager_lines(
        self,
        lines: list,
        dirs: Dict[str, Path],
        model_name: str,
        experiment_id: str,
        proc_id: int,
        cal_start: Optional[str],
        cal_end: Optional[str]
    ) -> list:
        """Update file manager lines with process-specific paths."""
        updated_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('!'):
                updated_lines.append(line)
                continue

            # Split by whitespace, but keep original indentation if possible
            parts = re.split(r'(\s+)', line, maxsplit=1)
            if len(parts) < 2:
                updated_lines.append(line)
                continue

            # parts[0] is the key, parts[1] is the whitespace, parts[2] is the rest
            key = stripped.split()[0]

            # Find the value part (ignoring comments)
            value_match = re.search(r"'(.*?)'", line)
            if not value_match and model_name.upper() != 'HYPE':
                updated_lines.append(line)
                continue

            if key == 'settingsPath':
                new_val = str(dirs['settings_dir']).replace('\\', '/')
                if not new_val.endswith('/'): new_val += '/'
                updated_lines.append(f"{key:20s} '{new_val}'\n")
            elif key == 'outputPath':
                new_val = str(dirs['sim_dir']).replace('\\', '/')
                if not new_val.endswith('/'): new_val += '/'
                updated_lines.append(f"{key:20s} '{new_val}'\n")
            elif key == 'outFilePrefix':
                new_val = f'proc_{proc_id:02d}_{experiment_id}'
                updated_lines.append(f"{key:20s} '{new_val}'\n")
            elif key == 'simStartTime' and cal_start:
                updated_lines.append(f"{key:20s} '{cal_start}'\n")
            elif key == 'simEndTime' and cal_end:
                updated_lines.append(f"{key:20s} '{cal_end}'\n")
            elif model_name.upper() == 'HYPE' and key == 'resultdir':
                new_val = str(dirs['output_dir']).replace('\\', '/').rstrip('/') + '/'
                updated_lines.append(f"resultdir\t{new_val}\n")
            else:
                # PRESERVE UNCHANGED
                updated_lines.append(line)

        return updated_lines

    def update_mizuroute_controls(
        self,
        parallel_dirs: Dict[int, Dict[str, Path]],
        model_name: str,
        experiment_id: str,
        control_file_name: str = 'mizuroute.control'
    ) -> None:
        """
        Update mizuRoute control file paths in process-specific directories.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
            experiment_id: Experiment identifier
            control_file_name: Name of the control file
        """
        for proc_id, dirs in parallel_dirs.items():
            mizu_settings_dir = dirs['settings_dir'].parent / 'mizuRoute'
            control_file_path = mizu_settings_dir / control_file_name

            if not control_file_path.exists():
                self.logger.debug(
                    f"mizuRoute control file not found for process {proc_id}: {control_file_path}"
                )
                continue

            try:
                with open(control_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Get model-specific settings
                mizu_config = self._get_mizuroute_config(model_name, proc_id, experiment_id)

                # Build process-specific paths
                proc_summa_dir = dirs['sim_dir']
                proc_mizu_dir = proc_summa_dir.parent / 'mizuRoute'
                proc_mizu_dir.mkdir(parents=True, exist_ok=True)

                updated_lines = self._update_mizuroute_lines(
                    lines, dirs, mizu_settings_dir, proc_summa_dir,
                    proc_mizu_dir, mizu_config, proc_id, experiment_id
                )

                with open(control_file_path, 'w', encoding='utf-8') as f:
                    f.writelines(updated_lines)

                self.logger.debug(
                    f"Updated mizuRoute control file for process {proc_id}: {control_file_path}"
                )

            except (KeyError, ValueError, IOError) as e:
                self.logger.error(
                    f"Failed to update mizuRoute control file for process {proc_id}: {e}"
                )

    def _get_mizuroute_config(
        self,
        model_name: str,
        proc_id: int,
        experiment_id: str
    ) -> Dict[str, str]:
        """Get model-specific mizuRoute configuration."""
        dt_qsim = self._get_config_value(lambda: self.config.model.mizuroute.routing_dt, default='3600', dict_key='SETTINGS_MIZU_ROUTING_DT')
        if dt_qsim in ('default', None, ''):
            dt_qsim = '3600'

        # Default times for hourly models
        sim_start_time = '01:00'
        sim_end_time = '23:00'

        model_upper = model_name.upper()

        if model_upper == 'SUMMA':
            fname_qsim = f'proc_{proc_id:02d}_{experiment_id}_timestep.nc'
            vname_qsim = 'averageRoutedRunoff'
        elif model_upper == 'FUSE':
            fname_qsim = f'proc_{proc_id:02d}_{experiment_id}_timestep.nc'
            vname_qsim = self._get_config_value(lambda: self.config.model.mizuroute.routing_var, default='q_routed', dict_key='SETTINGS_MIZU_ROUTING_VAR')
            if vname_qsim in ('default', None, ''):
                vname_qsim = 'q_routed'
            # FUSE outputs daily data
            dt_qsim = '86400'
        elif model_upper == 'GR':
            domain_name = self._get_config_value(lambda: self.config.domain.name, dict_key='DOMAIN_NAME')
            fname_qsim = f"{domain_name}_{experiment_id}_runs_def.nc"
            vname_qsim = self._get_config_value(lambda: self.config.model.mizuroute.routing_var, default='q_routed', dict_key='SETTINGS_MIZU_ROUTING_VAR')
            if vname_qsim in ('default', None, ''):
                vname_qsim = 'q_routed'
        elif model_upper == 'HYPE':
            fname_qsim = f'proc_{proc_id:02d}_{experiment_id}_timestep.nc'
            vname_qsim = self._get_config_value(lambda: self.config.model.mizuroute.routing_var, default='q_routed', dict_key='SETTINGS_MIZU_ROUTING_VAR')
            if vname_qsim in ('default', None, ''):
                vname_qsim = 'q_routed'
            # HYPE outputs daily data
            dt_qsim = '86400'
            sim_start_time = '00:00'
            sim_end_time = '00:00'
        else:
            fname_qsim = f'proc_{proc_id:02d}_{experiment_id}_timestep.nc'
            vname_qsim = 'q_routed'

        return {
            'fname_qsim': fname_qsim,
            'vname_qsim': vname_qsim,
            'dt_qsim': dt_qsim,
            'sim_start_time': sim_start_time,
            'sim_end_time': sim_end_time,
            'model_name': model_name,
        }

    def _update_mizuroute_lines(
        self,
        lines: list,
        dirs: Dict[str, Path],
        mizu_settings_dir: Path,
        proc_summa_dir: Path,
        proc_mizu_dir: Path,
        mizu_config: Dict[str, str],
        proc_id: int,
        experiment_id: str
    ) -> list:
        """Update mizuRoute control file lines."""
        def normalize_path(path):
            return str(path).replace('\\', '/').rstrip('/') + '/'

        input_dir = normalize_path(proc_summa_dir)
        output_dir = normalize_path(proc_mizu_dir)
        ancil_dir = normalize_path(mizu_settings_dir)
        case_name = f'proc_{proc_id:02d}_{experiment_id}'

        updated_lines = []

        for line in lines:
            if '<ancil_dir>' in line:
                comment = self._extract_comment(line)
                updated_lines.append(f"<ancil_dir>             {ancil_dir}    {comment}\n")
            elif '<input_dir>' in line:
                comment = self._extract_comment(line)
                updated_lines.append(f"<input_dir>             {input_dir}    {comment}\n")
            elif '<output_dir>' in line:
                comment = self._extract_comment(line)
                updated_lines.append(f"<output_dir>            {output_dir}    {comment}\n")
            elif '<case_name>' in line:
                comment = self._extract_comment(line)
                updated_lines.append(f"<case_name>             {case_name}    {comment}\n")
            elif '<fname_qsim>' in line:
                updated_lines.append(
                    f"<fname_qsim>            {mizu_config['fname_qsim']}    "
                    f"! netCDF name for {mizu_config['model_name']} runoff\n"
                )
            elif '<vname_qsim>' in line:
                updated_lines.append(
                    f"<vname_qsim>            {mizu_config['vname_qsim']}    "
                    f"! Variable name for {mizu_config['model_name']} runoff\n"
                )
            elif '<dt_qsim>' in line:
                updated_lines.append(
                    f"<dt_qsim>               {mizu_config['dt_qsim']}    "
                    f"! Time interval of input runoff in seconds\n"
                )
            elif '<dname_hruid>' in line:
                # FUSE converter outputs 'gru' dimension, SUMMA uses 'hru'
                dim_name = 'gru' if mizu_config['model_name'].upper() == 'FUSE' else 'hru'
                updated_lines.append(
                    f"<dname_hruid>           {dim_name}     ! Dimension name for HM_HRU ID\n"
                )
            elif '<vname_hruid>' in line:
                # FUSE converter outputs 'gruId' variable, SUMMA uses 'hruId'
                var_name = 'gruId' if mizu_config['model_name'].upper() == 'FUSE' else 'hruId'
                updated_lines.append(
                    f"<vname_hruid>           {var_name}   ! Variable name for HM_HRU ID\n"
                )
            elif '<seg_outlet>' in line:
                # Check if multi-gauge calibration is enabled
                multi_gauge = self._get_config_value(
                    lambda: self.config.calibration.multi_gauge,
                    default=False,
                    dict_key='MULTI_GAUGE_CALIBRATION'
                )
                if multi_gauge:
                    # Route all segments for multi-gauge calibration
                    updated_lines.append(
                        "<seg_outlet>            -9999    ! Route all segments for multi-gauge calibration\n"
                    )
                else:
                    updated_lines.append(line)
            elif '<sim_start>' in line:
                match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                if match:
                    sim_date = match.group(1)
                    updated_lines.append(
                        f"<sim_start>             {sim_date} {mizu_config['sim_start_time']}    "
                        f"! Time of simulation start\n"
                    )
                else:
                    updated_lines.append(line)
            elif '<sim_end>' in line:
                match = re.search(r'(\d{4}-\d{2}-\d{2})', line)
                if match:
                    sim_date = match.group(1)
                    updated_lines.append(
                        f"<sim_end>               {sim_date} {mizu_config['sim_end_time']}    "
                        f"! Time of simulation end\n"
                    )
                else:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)

        return updated_lines

    def _extract_comment(self, line: str) -> str:
        """Extract comment from a line if present."""
        if '!' in line:
            return '!' + '!'.join(line.split('!')[1:]).rstrip()
        return ''
