"""
ModelExecutor - Unified subprocess and SLURM execution framework.

This module consolidates all subprocess execution patterns across model runners,
providing a single, tested implementation for:
- Local subprocess execution with logging
- SLURM job submission and monitoring
- Job array management for parallel execution
- Standardized error handling and result reporting

Usage:
    class MyRunner(BaseModelRunner, ModelExecutor):
        def run_model(self):
            result = self.execute_subprocess(
                command=['./model.exe', '-c', 'config.txt'],
                log_file=self.get_log_path() / 'run.log'
            )
            if result.success:
                return result.output_path
"""

import os
import shutil
import subprocess
import time
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable


class ExecutionMode(Enum):
    """Execution mode for model runs."""
    LOCAL = "local"
    SLURM = "slurm"
    SLURM_ARRAY = "slurm_array"


@dataclass
class SlurmJobConfig:
    """Configuration for SLURM job submission.

    Attributes:
        job_name: Name of the SLURM job
        time_limit: Time limit in format HH:MM:SS
        memory: Memory per node (e.g., '4G', '16G')
        cpus_per_task: Number of CPUs per task
        partition: SLURM partition to submit to (optional)
        account: Account to charge (optional)
        array_size: For job arrays, max array index (0-based)
        output_pattern: Pattern for stdout file (supports %A, %a placeholders)
        error_pattern: Pattern for stderr file
        additional_directives: Extra #SBATCH lines as dict
    """
    job_name: str
    time_limit: str = "03:00:00"
    memory: str = "4G"
    cpus_per_task: int = 1
    partition: Optional[str] = None
    account: Optional[str] = None
    array_size: Optional[int] = None
    output_pattern: Optional[str] = None
    error_pattern: Optional[str] = None
    additional_directives: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of a model execution.

    Attributes:
        success: Whether execution completed successfully
        return_code: Process return code (0 = success)
        output_path: Path to output directory/file if applicable
        log_file: Path to log file
        duration_seconds: Execution duration in seconds
        job_id: SLURM job ID if applicable
        error_message: Error message if execution failed
        metadata: Additional execution metadata
    """
    success: bool
    return_code: int = 0
    output_path: Optional[Path] = None
    log_file: Optional[Path] = None
    duration_seconds: float = 0.0
    job_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelExecutor(ABC):
    """
    Mixin class providing unified execution capabilities for model runners.

    This class consolidates subprocess execution, SLURM job management, and
    parallel execution patterns that were previously duplicated across
    SUMMA, FUSE, GR, and other model runners.

    Designed to be used as a mixin with BaseModelRunner:
        class SummaRunner(BaseModelRunner, ModelExecutor):
            ...

    Key Methods:
        execute_subprocess: Run a command locally with logging
        submit_slurm_job: Submit a single SLURM job
        submit_slurm_array: Submit a SLURM job array for parallel execution
        monitor_slurm_job: Wait for SLURM job completion
        create_slurm_script: Generate SLURM batch script content
    """

    # These should be provided by BaseModelRunner
    logger: Any
    project_dir: Path
    config_dict: Dict[str, Any]

    # =========================================================================
    # Local Subprocess Execution
    # =========================================================================

    def execute_subprocess(
        self,
        command: Union[List[str], str],
        log_file: Path,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        shell: bool = False,
        check: bool = True,
        capture_output: bool = False,
        success_message: Optional[str] = None,
        error_context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute a subprocess with standardized logging and error handling.

        This method consolidates the subprocess execution patterns found across
        all model runners (SUMMA, FUSE, GR, etc.) into a single implementation.

        Args:
            command: Command to execute (list or string)
            log_file: Path to write stdout/stderr
            cwd: Working directory for execution
            env: Environment variables (merged with os.environ)
            timeout: Timeout in seconds (None = no timeout)
            shell: Whether to use shell execution
            check: Raise exception on non-zero exit code
            capture_output: Return stdout/stderr in result metadata
            success_message: Custom message to log on success
            error_context: Additional context to include in error logs

        Returns:
            ExecutionResult with success status, return code, and metadata

        Raises:
            subprocess.TimeoutExpired: If timeout is exceeded
            subprocess.CalledProcessError: If check=True and process fails

        Example:
            result = self.execute_subprocess(
                command=['./summa.exe', '-m', 'fileManager.txt'],
                log_file=self.get_log_path() / 'summa.log',
                env={'LD_LIBRARY_PATH': '/opt/netcdf/lib'},
                timeout=3600
            )
        """
        start_time = time.time()

        # Merge environment variables
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Windows: prepend conda Library\bin to PATH so compiled model
        # executables find the correct DLL versions (netCDF, HDF5,
        # gfortran runtime, openblas, etc.) before any conflicting
        # copies (e.g. Git for Windows bundles its own MinGW runtime)
        if os.name == 'nt':
            conda_prefix = run_env.get('CONDA_PREFIX', '')
            if conda_prefix:
                conda_lib_bin = os.path.join(conda_prefix, 'Library', 'bin')
                current_path = run_env.get('PATH', '')
                # Always prepend to ensure conda DLLs take priority
                run_env['PATH'] = f"{conda_lib_bin}{os.pathsep}{current_path}"

        # Ensure log directory exists
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Log execution start
        cmd_str = command if isinstance(command, str) else ' '.join(command)
        self.logger.debug(f"Executing: {cmd_str}")
        if cwd:
            self.logger.debug(f"Working directory: {cwd}")

        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    command,
                    check=False,  # We handle the check ourselves
                    stdout=f if not capture_output else subprocess.PIPE,
                    stderr=subprocess.STDOUT if not capture_output else subprocess.PIPE,
                    cwd=cwd,
                    env=run_env,
                    shell=shell,  # nosec B602 - shell mode required for model executables
                    text=True,
                    timeout=timeout
                )

            duration = time.time() - start_time

            # Build result
            exec_result = ExecutionResult(
                success=(result.returncode == 0),
                return_code=result.returncode,
                log_file=log_file,
                duration_seconds=duration,
                metadata={}
            )

            if capture_output:
                exec_result.metadata['stdout'] = result.stdout
                exec_result.metadata['stderr'] = result.stderr

            # Log outcome
            if result.returncode == 0:
                msg = success_message or f"Process completed successfully in {duration:.1f}s"
                self.logger.info(msg)
            else:
                self.logger.warning(f"Process exited with code {result.returncode}")
                exec_result.error_message = f"Exit code: {result.returncode}"

                # Log error context
                if error_context:
                    for key, value in error_context.items():
                        self.logger.error(f"  {key}: {value}")

                self.logger.error(f"See log file: {log_file}")

                if check:
                    raise subprocess.CalledProcessError(
                        result.returncode, command
                    )

            return exec_result

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            self.logger.error(f"Process timed out after {timeout}s")
            return ExecutionResult(
                success=False,
                return_code=-1,
                log_file=log_file,
                duration_seconds=duration,
                error_message=f"Timeout after {timeout}s"
            )

    # =========================================================================
    # SLURM Job Management
    # =========================================================================

    def is_slurm_available(self) -> bool:
        """Check if SLURM is available on this system."""
        return shutil.which("sbatch") is not None

    def create_slurm_script(
        self,
        config: SlurmJobConfig,
        commands: List[str],
        setup_commands: Optional[List[str]] = None,
        cleanup_commands: Optional[List[str]] = None
    ) -> str:
        """
        Generate a SLURM batch script.

        This consolidates the SLURM script generation that was duplicated
        in SUMMA's run_summa_parallel and other runners.

        Args:
            config: SlurmJobConfig with job parameters
            commands: Main commands to execute
            setup_commands: Commands to run before main commands
            cleanup_commands: Commands to run after main commands

        Returns:
            Complete SLURM batch script as string

        Example:
            script = self.create_slurm_script(
                config=SlurmJobConfig(
                    job_name='SUMMA-run',
                    time_limit='03:00:00',
                    memory='4G',
                    array_size=99
                ),
                commands=['./summa.exe -g $GRU_ID 1 -m fileManager.txt'],
                setup_commands=['source /opt/modules/init/bash']
            )
        """
        lines = ["#!/bin/bash"]

        # Core directives
        lines.append(f"#SBATCH --job-name={config.job_name}")
        lines.append(f"#SBATCH --time={config.time_limit}")
        lines.append(f"#SBATCH --mem={config.memory}")
        lines.append(f"#SBATCH --cpus-per-task={config.cpus_per_task}")

        # Optional directives
        if config.partition:
            lines.append(f"#SBATCH --partition={config.partition}")
        if config.account:
            lines.append(f"#SBATCH --account={config.account}")
        if config.array_size is not None:
            lines.append(f"#SBATCH --array=0-{config.array_size}")
        if config.output_pattern:
            lines.append(f"#SBATCH --output={config.output_pattern}")
        if config.error_pattern:
            lines.append(f"#SBATCH --error={config.error_pattern}")

        # Additional directives
        for key, value in config.additional_directives.items():
            lines.append(f"#SBATCH --{key}={value}")

        lines.append("")

        # Job info for debugging
        lines.extend([
            "# Print job info for debugging",
            'echo "Starting job at $(date)"',
            'echo "Running on host: $(hostname)"',
            'echo "SLURM Job ID: $SLURM_JOB_ID"',
        ])

        if config.array_size is not None:
            lines.append('echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"')

        lines.append("")

        # Setup commands
        if setup_commands:
            lines.append("# Setup")
            lines.extend(setup_commands)
            lines.append("")

        # Main commands
        lines.append("# Main execution")
        lines.extend(commands)
        lines.append("")

        # Cleanup commands
        if cleanup_commands:
            lines.append("# Cleanup")
            lines.extend(cleanup_commands)
            lines.append("")

        lines.append('echo "Job completed at $(date)"')

        return "\n".join(lines)

    def submit_slurm_job(
        self,
        script_path: Path,
        wait: bool = False,
        poll_interval: int = 60,
        max_wait_time: int = 3600
    ) -> ExecutionResult:
        """
        Submit a SLURM job and optionally wait for completion.

        This consolidates the SLURM submission logic from SUMMA's run_summa_parallel
        (~50 lines of job monitoring code) into a reusable method.

        Args:
            script_path: Path to SLURM batch script
            wait: If True, monitor job until completion
            poll_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait before returning

        Returns:
            ExecutionResult with job_id and completion status

        Raises:
            RuntimeError: If sbatch command not found or submission fails
        """
        if not self.is_slurm_available():
            raise RuntimeError("SLURM 'sbatch' command not found")

        start_time = time.time()

        try:
            # Submit job
            result = subprocess.run(
                ["sbatch", str(script_path)],
                check=True,
                capture_output=True,
                text=True
            )

            # Extract job ID from output like "Submitted batch job 12345"
            job_id = result.stdout.strip().split()[-1]
            self.logger.info(f"Submitted SLURM job: {job_id}")

            if not wait:
                return ExecutionResult(
                    success=True,
                    return_code=0,
                    job_id=job_id,
                    metadata={'status': 'submitted'}
                )

            # Monitor job
            return self.monitor_slurm_job(
                job_id=job_id,
                poll_interval=poll_interval,
                max_wait_time=max_wait_time
            )

        except subprocess.CalledProcessError as e:
            self.logger.error(f"SLURM submission failed: {e.stderr}")
            return ExecutionResult(
                success=False,
                return_code=e.returncode,
                error_message=e.stderr,
                duration_seconds=time.time() - start_time
            )

    def monitor_slurm_job(
        self,
        job_id: str,
        poll_interval: int = 60,
        max_wait_time: int = 3600,
        on_status_change: Optional[Callable[[str, str], None]] = None
    ) -> ExecutionResult:
        """
        Monitor a SLURM job until completion or timeout.

        This replaces the inline job monitoring code in run_summa_parallel.

        Args:
            job_id: SLURM job ID to monitor
            poll_interval: Seconds between status checks
            max_wait_time: Maximum seconds to wait
            on_status_change: Callback(job_id, new_status) on status change

        Returns:
            ExecutionResult with final job status
        """
        start_time = time.time()
        last_status = None

        self.logger.info(f"Monitoring SLURM job {job_id}")

        while (time.time() - start_time) < max_wait_time:
            try:
                # Check if job is still in queue
                queue_result = subprocess.run(
                    ["squeue", "-j", job_id, "-h"],
                    capture_output=True,
                    text=True
                )

                if not queue_result.stdout.strip():
                    # Job no longer in queue, check final status
                    status = self._get_slurm_job_status(job_id)

                    duration = time.time() - start_time

                    if status == "COMPLETED":
                        self.logger.info(f"Job {job_id} completed successfully")
                        return ExecutionResult(
                            success=True,
                            return_code=0,
                            job_id=job_id,
                            duration_seconds=duration,
                            metadata={'slurm_status': status}
                        )
                    elif status in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                        self.logger.error(f"Job {job_id} ended with status: {status}")
                        return ExecutionResult(
                            success=False,
                            return_code=1,
                            job_id=job_id,
                            duration_seconds=duration,
                            error_message=f"SLURM status: {status}",
                            metadata={'slurm_status': status}
                        )
                    else:
                        # Unknown status, consider it done
                        self.logger.warning(f"Job {job_id} has unknown status: {status}")
                        return ExecutionResult(
                            success=True,
                            return_code=0,
                            job_id=job_id,
                            duration_seconds=duration,
                            metadata={'slurm_status': status}
                        )
                else:
                    # Job still running - parse status
                    current_status = self._parse_squeue_output(queue_result.stdout)

                    if current_status != last_status:
                        self.logger.info(f"Job {job_id}: {current_status}")
                        if on_status_change:
                            on_status_change(job_id, current_status)
                        last_status = current_status

            except subprocess.SubprocessError as e:
                self.logger.warning(f"Error checking job status: {e}")

            time.sleep(poll_interval)

        # Timeout
        self.logger.warning(f"Timeout waiting for job {job_id}")
        return ExecutionResult(
            success=False,
            return_code=-1,
            job_id=job_id,
            duration_seconds=time.time() - start_time,
            error_message=f"Timeout after {max_wait_time}s"
        )

    def _get_slurm_job_status(self, job_id: str) -> str:
        """Get final status of a completed SLURM job."""
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "-o", "State", "-n", "--parsable2"],
                capture_output=True,
                text=True
            )
            # First line contains the status
            status = result.stdout.strip().split('\n')[0].split('|')[0]
            return status.strip()
        except (subprocess.SubprocessError, OSError, IndexError):
            return "UNKNOWN"

    def _parse_squeue_output(self, output: str) -> str:
        """Parse squeue output to get human-readable status."""
        lines = output.strip().split('\n')
        pending = sum(1 for line in lines if 'PENDING' in line or ' PD ' in line)
        running = sum(1 for line in lines if 'RUNNING' in line or ' R ' in line)
        return f"{running} running, {pending} pending"

    # =========================================================================
    # Parallel Execution Helpers
    # =========================================================================

    def create_gru_parallel_script(
        self,
        model_exe: Path,
        file_manager: Path,
        log_dir: Path,
        total_grus: int,
        grus_per_job: int,
        job_name: str,
        time_limit: str = "03:00:00",
        memory: str = "4G"
    ) -> str:
        """
        Create a SLURM script for GRU-parallel execution.

        This generalizes SUMMA's parallel execution pattern for use by any model
        that supports GRU-based parallelization.

        Args:
            model_exe: Path to model executable
            file_manager: Path to file manager/control file
            log_dir: Directory for log files
            total_grus: Total number of GRUs to process
            grus_per_job: Number of GRUs per array task
            job_name: SLURM job name
            time_limit: Time limit per task
            memory: Memory per task

        Returns:
            Complete SLURM batch script
        """
        n_array_jobs = max(1, -(-total_grus // grus_per_job)) - 1  # Ceiling div, 0-based

        config = SlurmJobConfig(
            job_name=job_name,
            time_limit=time_limit,
            memory=memory,
            array_size=n_array_jobs,
            output_pattern=str(log_dir / f"{job_name}_%A_%a.out"),
            error_pattern=str(log_dir / f"{job_name}_%A_%a.err")
        )

        setup = [
            f"mkdir -p {log_dir}",
            "",
            "# Calculate GRU range for this task",
            f"gru_start=$(( ({grus_per_job} * $SLURM_ARRAY_TASK_ID) + 1 ))",
            f"gru_end=$(( gru_start + {grus_per_job} - 1 ))",
            "",
            f"if [ $gru_end -gt {total_grus} ]; then",
            f"    gru_end={total_grus}",
            "fi",
            "",
            'echo "Processing GRUs $gru_start to $gru_end"',
        ]

        commands = [
            "for gru in $(seq $gru_start $gru_end); do",
            '    echo "Starting GRU $gru"',
            f"    {model_exe} -g $gru 1 -m {file_manager}",
            "    exit_code=$?",
            "    if [ $exit_code -ne 0 ]; then",
            '        echo "Model failed for GRU $gru with exit code $exit_code"',
            "        exit 1",
            "    fi",
            '    echo "Completed GRU $gru"',
            "done",
        ]

        return self.create_slurm_script(
            config=config,
            commands=commands,
            setup_commands=setup
        )

    def estimate_optimal_grus_per_job(
        self,
        total_grus: int,
        min_jobs: int = 10,
        max_jobs: int = 500,
        ideal_grus_per_job: int = 50
    ) -> int:
        """
        Estimate optimal number of GRUs per SLURM job.

        Balances:
        - Not too many small jobs (queue overhead)
        - Not too few large jobs (poor load balancing)

        Args:
            total_grus: Total GRUs in domain
            min_jobs: Minimum number of jobs to create
            max_jobs: Maximum jobs (prevent queue flooding)
            ideal_grus_per_job: Target GRUs per job

        Returns:
            Optimal GRUs per job
        """
        if total_grus <= min_jobs:
            return 1

        if total_grus <= ideal_grus_per_job * min_jobs:
            return max(1, total_grus // min_jobs)

        ideal_jobs = total_grus // ideal_grus_per_job

        if ideal_jobs <= max_jobs:
            grus_per_job = ideal_grus_per_job
        else:
            grus_per_job = -(-total_grus // max_jobs)

        # Scale up for very large domains
        if total_grus > 10000:
            scale_factor = min(3.0, total_grus / 10000)
            grus_per_job = int(grus_per_job * scale_factor)

        return min(grus_per_job, total_grus)

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def run_with_retry(
        self,
        command: Union[List[str], str],
        log_file: Path,
        max_attempts: int = 3,
        retry_delay: int = 5,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute subprocess with automatic retry on failure.

        Args:
            command: Command to execute
            log_file: Path for log file
            max_attempts: Maximum retry attempts
            retry_delay: Seconds between retries
            **kwargs: Additional arguments for execute_subprocess

        Returns:
            ExecutionResult from final attempt
        """
        # Initialize last_result to handle case where loop might not execute
        last_result = ExecutionResult(
            success=False,
            return_code=-1,
            error_message="Execution failed to start or all attempts failed",
            metadata={'command': str(command)}
        )

        for attempt in range(1, max_attempts + 1):
            self.logger.debug(f"Attempt {attempt}/{max_attempts}")

            # Add attempt number to log file name
            attempt_log = log_file.with_suffix(f".attempt{attempt}.log")

            result = self.execute_subprocess(
                command=command,
                log_file=attempt_log,
                check=False,
                **kwargs
            )

            if result.success:
                return result

            last_result = result

            if attempt < max_attempts:
                self.logger.warning(f"Attempt {attempt} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        return last_result

    def execute_in_mode(
        self,
        mode: ExecutionMode,
        command: Union[List[str], str],
        log_file: Path,
        slurm_config: Optional[SlurmJobConfig] = None,
        wait_for_slurm: bool = True,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute command in specified mode (local or SLURM).

        Provides a unified interface that automatically handles
        the execution mode selected by configuration.

        Args:
            mode: Execution mode (LOCAL or SLURM)
            command: Command to execute
            log_file: Path for log file
            slurm_config: SLURM configuration (required for SLURM mode)
            wait_for_slurm: Wait for SLURM job completion
            **kwargs: Additional arguments for execute_subprocess

        Returns:
            ExecutionResult
        """
        if mode == ExecutionMode.LOCAL:
            return self.execute_subprocess(command, log_file, **kwargs)

        elif mode in (ExecutionMode.SLURM, ExecutionMode.SLURM_ARRAY):
            if not slurm_config:
                raise ValueError("slurm_config required for SLURM execution")

            if not self.is_slurm_available():
                self.logger.warning("SLURM not available, falling back to local execution")
                return self.execute_subprocess(command, log_file, **kwargs)

            # Create script
            cmd_list = command if isinstance(command, list) else [command]
            script_content = self.create_slurm_script(
                config=slurm_config,
                commands=cmd_list
            )

            # Write script
            script_path = log_file.parent / f"{slurm_config.job_name}.sh"
            script_path.write_text(script_content)
            if os.name != 'nt':
                script_path.chmod(0o755)

            # Submit
            return self.submit_slurm_job(
                script_path=script_path,
                wait=wait_for_slurm
            )

        else:
            raise ValueError(f"Unknown execution mode: {mode}")
