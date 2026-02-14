"""
File Manager Updater

Handles updates to SUMMA file manager for final evaluation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.core.mixins import ConfigMixin


class FileManagerUpdater(ConfigMixin):
    """
    Updates SUMMA file manager settings for final evaluation.

    Handles:
    - Updating simulation time window for full period
    - Updating output path for final evaluation
    - Restoring calibration period settings
    - Adjusting end time for forcing timestep alignment
    """

    def __init__(
        self,
        file_manager_path: Path,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize file manager updater.

        Args:
            file_manager_path: Path to file manager file
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.file_manager_path = file_manager_path
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        self.logger = logger or logging.getLogger(__name__)

    def update_for_full_period(self) -> None:
        """Update file manager to use full experiment period (not just calibration)."""
        if not self.file_manager_path.exists():
            self.logger.warning(f"File manager not found: {self.file_manager_path}")
            return

        try:
            sim_start = self._get_config_value(lambda: self.config.domain.time_start, dict_key='EXPERIMENT_TIME_START')
            sim_end = self._get_config_value(lambda: self.config.domain.time_end, dict_key='EXPERIMENT_TIME_END')

            if not sim_start or not sim_end:
                self.logger.warning("Full experiment period not configured, using current settings")
                return

            # Adjust end time to align with forcing timestep
            sim_end = self._adjust_end_time_for_forcing(sim_end)

            with open(self.file_manager_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                if 'simStartTime' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"simStartTime         '{sim_start}'\n")
                elif 'simEndTime' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"simEndTime           '{sim_end}'\n")
                else:
                    updated_lines.append(line)

            with open(self.file_manager_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            self.logger.debug(f"Updated file manager for full period: {sim_start} to {sim_end}")

        except (FileNotFoundError, IOError, ValueError) as e:
            self.logger.error(f"Failed to update file manager for final run: {e}")

    def update_output_path(self, output_dir: Path) -> None:
        """
        Update file manager with final evaluation output path.

        Args:
            output_dir: Output directory path
        """
        if not self.file_manager_path.exists():
            return

        try:
            with open(self.file_manager_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Ensure path ends with slash
            output_path_str = str(output_dir)
            if not output_path_str.endswith('/'):
                output_path_str += '/'

            updated_lines = []
            for line in lines:
                if 'outputPath' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"outputPath '{output_path_str}' \n")
                else:
                    updated_lines.append(line)

            with open(self.file_manager_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            self.logger.debug(f"Updated output path to: {output_path_str}")

        except (FileNotFoundError, IOError, ValueError) as e:
            self.logger.error(f"Failed to update output path: {e}")

    def restore_calibration_period(self) -> None:
        """Restore file manager to calibration period settings."""
        if not self.file_manager_path.exists():
            return

        try:
            calib_start = self._get_config_value(lambda: self.config.domain.calibration_start_date, dict_key='CALIBRATION_START_DATE')
            calib_end = self._get_config_value(lambda: self.config.domain.calibration_end_date, dict_key='CALIBRATION_END_DATE')

            if not calib_start or not calib_end:
                return

            with open(self.file_manager_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                if 'simStartTime' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"simStartTime         '{calib_start}'\n")
                elif 'simEndTime' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"simEndTime           '{calib_end}'\n")
                else:
                    updated_lines.append(line)

            with open(self.file_manager_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            self.logger.debug("Restored file manager to calibration period")

        except (FileNotFoundError, IOError, ValueError) as e:
            self.logger.error(f"Failed to restore file manager: {e}")

    def _adjust_end_time_for_forcing(self, end_time_str: str) -> str:
        """
        Adjust end time to align with forcing data timestep.

        Args:
            end_time_str: End time string in format 'YYYY-MM-DD HH:MM'

        Returns:
            Adjusted end time string
        """
        try:
            forcing_timestep_seconds = self._get_config_value(lambda: self.config.forcing.time_step_size, default=3600, dict_key='FORCING_TIME_STEP_SIZE')

            if forcing_timestep_seconds >= 3600:  # Hourly or coarser
                end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')

                forcing_timestep_hours = forcing_timestep_seconds / 3600
                last_hour = int(24 - (24 % forcing_timestep_hours)) - forcing_timestep_hours
                if last_hour < 0:
                    last_hour = 0

                if end_time.hour > last_hour or (end_time.hour == 23 and last_hour < 23):
                    end_time = end_time.replace(hour=int(last_hour), minute=0)
                    adjusted_str = end_time.strftime('%Y-%m-%d %H:%M')
                    self.logger.info(
                        f"Adjusted end time from {end_time_str} to {adjusted_str} "
                        f"for {forcing_timestep_hours}h forcing"
                    )
                    return adjusted_str

            return end_time_str

        except (FileNotFoundError, IOError, ValueError) as e:
            self.logger.warning(f"Could not adjust end time: {e}")
            return end_time_str
