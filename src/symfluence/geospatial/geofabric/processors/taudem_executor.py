"""
TauDEM command execution and MPI orchestration.

Provides TauDEM command execution with MPI support and retry logic.
Eliminates code duplication across GeofabricDelineator and LumpedWatershedDelineator.

Handles:
- MPI launcher detection (srun or mpirun)
- Module load commands
- Retry logic on failure
- Command prefixing logic

Refactored from geofabric_utils.py (2026-01-01)
"""

import subprocess
import shutil
import time
import os
from typing import Any, Dict, Optional


class TauDEMExecutor:
    """
    Executes TauDEM commands with MPI support and retry logic.

    This class manages TauDEM command execution across different HPC environments,
    automatically detecting whether to use srun (SLURM) or mpirun (generic MPI).
    """

    def __init__(self, config: Dict[str, Any], logger: Any, taudem_dir: str):
        """
        Initialize TauDEM executor.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            taudem_dir: Path to TauDEM binary directory
        """
        self.config = config
        self.logger = logger
        self.taudem_dir = taudem_dir
        self.mpi_processes = config.get('MPI_PROCESSES', 1)
        self.max_retries = config.get('MAX_RETRIES', 3)
        self.retry_delay = config.get('RETRY_DELAY', 5)

        # Add TauDEM to PATH
        os.environ['PATH'] = f"{os.environ['PATH']}:{taudem_dir}"

    def _get_mpi_command(self) -> Optional[str]:
        """
        Detect available MPI launcher.

        Checks for srun (SLURM) first, then mpirun (generic MPI).

        Returns:
            'srun', 'mpirun', or None if no MPI launcher found
        """
        if shutil.which("srun"):
            return "srun"
        elif shutil.which("mpirun"):
            return "mpirun"
        else:
            return None

    def get_mpi_command(self) -> Optional[str]:
        """
        Get the MPI launcher command for parallel TauDEM execution.

        Detects available MPI implementations (mpiexec, mpirun) on the system
        and returns the appropriate launcher command.

        Returns:
            Optional[str]: MPI launcher command (e.g., 'mpiexec -n 4') or None
                if MPI is not available or configured.
        """
        return self._get_mpi_command()

    def run_command(self, command: str, retry: bool = True) -> None:
        """
        Run a TauDEM command with MPI support and retry logic.

        Handles several scenarios:
        1. Commands with module load (e.g., "module load taudem && pitremove")
        2. Commands already prefixed with mpirun/srun
        3. Regular commands that need MPI prefix

        Args:
            command: TauDEM command to execute
            retry: Enable retry on failure (default: True)

        Raises:
            subprocess.CalledProcessError: If command fails after all retries
        """
        run_cmd = self._get_mpi_command()

        for attempt in range(self.max_retries if retry else 1):
            try:
                # Check if command already has MPI prefix to avoid double prefixing
                has_mpi_prefix = any(cmd in command for cmd in ["mpirun", "srun"])

                if run_cmd and "module load" in command:
                    # Handle commands with module load specially
                    parts = command.split(" && ")
                    if len(parts) == 2:
                        module_part = parts[0]
                        actual_cmd = parts[1]
                        if not has_mpi_prefix:
                            full_command = f"{module_part} && {run_cmd} -n {self.mpi_processes} {actual_cmd}"
                        else:
                            full_command = command
                    else:
                        full_command = command
                elif run_cmd and not has_mpi_prefix:
                    # Add MPI prefix for regular commands
                    full_command = f"{run_cmd} -n {self.mpi_processes} {command}"
                else:
                    # Command already has MPI prefix or no MPI launcher available
                    full_command = command

                self.logger.debug(f"Running command: {full_command}")
                result = subprocess.run(
                    full_command,
                    check=True,
                    shell=True,
                    capture_output=True,
                    text=True
                )
                self.logger.debug(f"Command output: {result.stdout}")
                return

            except subprocess.CalledProcessError as e:
                self.logger.error(f"Error executing command: {full_command}")
                self.logger.error(f"Error details: {e.stderr}")

                if attempt < self.max_retries - 1 and retry:
                    self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                elif run_cmd:
                    self.logger.info(f"Trying without {run_cmd}...")
                    run_cmd = None  # Try without srun/mpirun on the next attempt
                else:
                    raise
