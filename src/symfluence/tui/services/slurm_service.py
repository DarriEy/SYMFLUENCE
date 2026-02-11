"""
SLURM job management service for the TUI.

Wraps squeue/scancel subprocess calls with timeout handling.
Only active when running on an HPC system with SLURM available.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SlurmJob:
    """A single SLURM job entry."""

    job_id: str
    name: str
    status: str
    partition: str
    time: str
    nodes: str


class SlurmService:
    """Interface to SLURM job management."""

    _SQUEUE_FORMAT = "%i|%j|%T|%P|%M|%D"
    _SQUEUE_TIMEOUT = 10  # seconds

    @staticmethod
    def is_hpc() -> bool:
        """Check if we're running on a SLURM-equipped system."""
        # Check for SLURM env vars (set on login/compute nodes)
        if os.environ.get("SLURM_CONF") or os.environ.get("SLURM_JOB_ID"):
            return True
        # Check if squeue binary is available
        try:
            subprocess.run(
                ["squeue", "--version"],
                capture_output=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def list_user_jobs(self) -> List[SlurmJob]:
        """Query squeue for the current user's jobs."""
        try:
            result = subprocess.run(
                ["squeue", "--me", f"--format={self._SQUEUE_FORMAT}", "--noheader"],
                capture_output=True,
                text=True,
                timeout=self._SQUEUE_TIMEOUT,
            )
            if result.returncode != 0:
                logger.debug(f"squeue failed: {result.stderr}")
                return []

            jobs = []
            for line in result.stdout.strip().splitlines():
                parts = line.strip().split("|")
                if len(parts) >= 6:
                    jobs.append(SlurmJob(
                        job_id=parts[0].strip(),
                        name=parts[1].strip(),
                        status=parts[2].strip(),
                        partition=parts[3].strip(),
                        time=parts[4].strip(),
                        nodes=parts[5].strip(),
                    ))
            return jobs
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            logger.debug(f"squeue unavailable: {exc}")
            return []

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a SLURM job. Returns True on success."""
        try:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True,
                timeout=self._SQUEUE_TIMEOUT,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            logger.error(f"scancel failed: {exc}")
            return False

    def submit_job(self, script_path: str) -> Optional[str]:
        """Submit a SLURM batch script. Returns job ID or None."""
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                # Output: "Submitted batch job 12345"
                parts = result.stdout.strip().split()
                return parts[-1] if parts else None
            logger.error(f"sbatch failed: {result.stderr}")
            return None
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            logger.error(f"sbatch failed: {exc}")
            return None
