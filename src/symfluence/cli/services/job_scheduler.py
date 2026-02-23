"""
Job scheduler service for SLURM and other HPC schedulers.

Handles job script generation, submission, and monitoring.
"""

import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from ..console import Console
from .base import BaseService


class JobSchedulerService(BaseService):
    """
    Service for managing HPC job submissions.

    Handles:
    - SLURM job script generation
    - Job submission
    - Job monitoring
    - Environment detection (HPC vs laptop)
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the JobSchedulerService.

        Args:
            console: Console instance for output. If None, uses global console.
        """
        super().__init__(console=console)

    def handle_slurm_job_submission(
        self,
        execution_plan: Dict[str, Any],
        symfluence_instance: Optional[Any] = None,
    ) -> bool:
        """
        Handle SLURM job submission workflow.

        Args:
            execution_plan: Execution plan from CLI manager
            symfluence_instance: SYMFLUENCE instance (optional)

        Returns:
            bool: Success status
        """
        try:
            slurm_options = execution_plan.get("slurm_options", {})

            if not slurm_options.get("job_account"):
                self._console.warning(
                    "No SLURM account specified. This may be required on your system."
                )
                self._console.indent(
                    "Use --job_account to specify an account if job submission fails."
                )

            result = self.submit_slurm_job(execution_plan, symfluence_instance)

            if result.get("success", False):
                self._console.newline()
                self._console.success("SLURM job submission successful!")
                if not slurm_options.get("submit_and_wait", False):
                    self._console.info("Job is running in background. Monitor with:")
                    self._console.indent(f"squeue -j {result['job_id']}")
                    self._console.indent(f"tail -f SYMFLUENCE_*_{result['job_id']}.out")
                return True
            else:
                self._console.error("Job submission failed")
                return False

        except (KeyError, RuntimeError, OSError, ValueError, TypeError, subprocess.SubprocessError) as e:
            self._console.error(f"Error in SLURM job submission: {str(e)}")
            return False

    def submit_slurm_job(
        self, execution_plan: Dict[str, Any], symfluence_instance=None
    ) -> Dict[str, Any]:
        """
        Submit a SLURM job for the execution plan.

        Args:
            execution_plan: Execution plan dictionary
            symfluence_instance: Optional SYMFLUENCE instance

        Returns:
            Dict with submission results including job_id

        Raises:
            RuntimeError: If SLURM is not available or submission fails
        """
        self._console.panel("Preparing SLURM Job Submission", style="blue")

        if not self._check_slurm_available():
            raise RuntimeError("SLURM commands (sbatch) not available on this system")

        slurm_options = execution_plan.get("slurm_options", {})

        # Auto-generate job name if not provided
        if not slurm_options.get("job_name"):
            if symfluence_instance and hasattr(symfluence_instance, "config"):
                domain = symfluence_instance.config.get("DOMAIN_NAME", "symfluence")
            else:
                domain = "symfluence"

            mode = execution_plan.get("mode", "workflow")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slurm_options["job_name"] = f"{domain}_{mode}_{timestamp}"

        self._console.info(f"Job Name: {slurm_options['job_name']}")
        self._console.info(f"Time Limit: {slurm_options['job_time']}")
        self._console.info(f"Memory: {slurm_options['job_memory']}")
        self._console.info(f"Tasks: {slurm_options['job_ntasks']}")

        if slurm_options.get("job_account"):
            self._console.info(f"Account: {slurm_options['job_account']}")
        if slurm_options.get("job_partition"):
            self._console.info(f"Partition: {slurm_options['job_partition']}")

        config_file = execution_plan.get("config_file", "./config.yaml")

        script_content = self._create_symfluence_slurm_script(
            execution_plan, slurm_options, config_file
        )

        script_path = Path(f"SYMFLUENCE_{slurm_options['job_name']}.sh")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        self._console.info(f"SLURM script created: {script_path}")

        try:
            self._console.newline()
            self._console.info("Submitting job...")
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )

            job_id = result.stdout.strip().split()[-1]

            if not job_id:
                raise RuntimeError("Could not extract job ID from sbatch output")

            self._console.success("Job submitted successfully!")
            self._console.info(f"Job ID: {job_id}")

            submission_result = {
                "success": True,
                "job_id": job_id,
                "script_path": script_path,
                "job_name": slurm_options["job_name"],
                "submission_time": datetime.now().isoformat(),
            }

            if slurm_options.get("submit_and_wait", False):
                self._console.newline()
                self._console.info(f"Waiting for job {job_id} to complete...")
                self._monitor_slurm_job(job_id)

            return submission_result

        except subprocess.CalledProcessError as e:
            self._console.error(f"Error submitting job: {e}")
            self._console.error(f"stderr: {e.stderr}")
            raise RuntimeError(f"Failed to submit SLURM job: {e}") from e

    def _create_symfluence_slurm_script(
        self,
        execution_plan: Dict[str, Any],
        slurm_options: Dict[str, Any],
        config_file: str,
    ) -> str:
        """Create SLURM script content for SYMFLUENCE workflow."""
        job_mode = execution_plan.get("job_mode", "workflow")
        job_steps = execution_plan.get("job_steps", [])
        settings = execution_plan.get("settings", {})

        def _quote_args(args):
            return " ".join(shlex.quote(str(arg)) for arg in args)

        if job_mode == "individual_steps":
            steps = [str(step) for step in job_steps if step]
            if steps:
                if len(steps) == 1:
                    symfluence_cmd = f"symfluence workflow step {shlex.quote(steps[0])}"
                else:
                    symfluence_cmd = f"symfluence workflow steps {_quote_args(steps)}"
            else:
                symfluence_cmd = "symfluence workflow run"

            if config_file:
                symfluence_cmd += f" --config {shlex.quote(config_file)}"

            if settings.get("force_rerun", False):
                symfluence_cmd += " --force-rerun"
        elif job_mode == "pour_point_setup":
            pour_point_info = execution_plan.get("pour_point", {})
            symfluence_cmd = (
                f"symfluence project pour-point {shlex.quote(str(pour_point_info.get('coordinates')))} "
                f"--domain-name {shlex.quote(str(pour_point_info.get('domain_name')))} "
                f"--definition {shlex.quote(str(pour_point_info.get('domain_definition_method')))}"
            )
            if pour_point_info.get("bounding_box_coords"):
                symfluence_cmd += (
                    f" --bounding-box {shlex.quote(str(pour_point_info['bounding_box_coords']))}"
                )
            if pour_point_info.get("experiment_id"):
                symfluence_cmd += (
                    f" --experiment-id {shlex.quote(str(pour_point_info['experiment_id']))}"
                )
        else:
            workflow_args = [str(arg) for arg in job_steps if arg]
            if workflow_args and workflow_args[0] == "symfluence":
                workflow_args = workflow_args[1:]

            if workflow_args:
                symfluence_cmd = f"symfluence {_quote_args(workflow_args)}"
            else:
                symfluence_cmd = "symfluence workflow run"

            if config_file and "--config" not in workflow_args:
                symfluence_cmd += f" --config {shlex.quote(config_file)}"

            if settings.get("force_rerun", False) and "--force-rerun" not in workflow_args:
                symfluence_cmd += " --force-rerun"
            if settings.get("continue_on_error", False) and "--continue-on-error" not in workflow_args:
                symfluence_cmd += " --continue-on-error"

        if settings.get("debug", False) and "--debug" not in symfluence_cmd:
            symfluence_cmd += " --debug"

        script = f"""#!/bin/bash
#SBATCH --job-name={slurm_options['job_name']}
#SBATCH --output=SYMFLUENCE_{slurm_options['job_name']}_%j.out
#SBATCH --error=SYMFLUENCE_{slurm_options['job_name']}_%j.err
#SBATCH --time={slurm_options['job_time']}
#SBATCH --ntasks={slurm_options['job_ntasks']}
#SBATCH --mem={slurm_options['job_memory']} """

        if slurm_options.get("job_account"):
            script += f"\n#SBATCH --account={slurm_options['job_account']}"
        if slurm_options.get("job_partition"):
            script += f"\n#SBATCH --partition={slurm_options['job_partition']}"
        if slurm_options.get("job_nodes") and slurm_options["job_nodes"] > 1:
            script += f"\n#SBATCH --nodes={slurm_options['job_nodes']}"

        script += f"""

# Job information
echo "=========================================="
echo "SYMFLUENCE SLURM Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: {slurm_options['job_name']}"
echo "Node: $HOSTNAME"
echo "Started: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Load modules and environment
echo "Loading environment..."
"""

        if slurm_options.get("job_modules"):
            script += f"module restore {slurm_options['job_modules']}\n"

        if slurm_options.get("conda_env"):
            script += f"conda activate {slurm_options['conda_env']}\n"

        script += f"""
echo "Python environment: $(which python)"
echo "SYMFLUENCE directory: $(pwd)"
echo ""

# Run SYMFLUENCE workflow
echo "Starting SYMFLUENCE workflow..."
echo "Command: {symfluence_cmd}"
echo ""

{symfluence_cmd}

exit_code=$?

echo ""
echo "=========================================="
echo "SYMFLUENCE Job Completed"
echo "Exit code: $exit_code"
echo "Finished: $(date)"
echo "=========================================="

exit $exit_code
"""
        return script

    def _check_slurm_available(self) -> bool:
        """Check if SLURM commands are available."""
        return shutil.which("sbatch") is not None

    def _monitor_slurm_job(self, job_id: str) -> None:
        """Monitor SLURM job until completion."""
        self._console.info(f"Monitoring job {job_id}...")
        while True:
            try:
                result = subprocess.run(
                    ["squeue", "-j", job_id, "-h"],
                    capture_output=True,
                    text=True,
                )
                if not result.stdout.strip():
                    self._console.success(f"Job {job_id} completed!")
                    break
                else:
                    lines = result.stdout.strip().split("\n")
                    if lines:
                        status_info = lines[0].split()
                        if len(status_info) >= 5:
                            status = status_info[4]
                            self._console.info(f"Job status: {status}")
            except subprocess.SubprocessError as e:
                self._console.warning(f"Error checking job status: {e}")
            time.sleep(60)

    def detect_environment(self) -> str:
        """
        Detect whether we're running on HPC or a personal computer.

        Returns:
            str: 'hpc' or 'laptop'
        """
        hpc_schedulers = ["sbatch", "qsub", "bsub"]
        for scheduler in hpc_schedulers:
            if shutil.which(scheduler):
                return "hpc"

        hpc_env_vars = [
            "SLURM_CLUSTER_NAME",
            "SLURM_JOB_ID",
            "PBS_JOBID",
            "SGE_CLUSTER_NAME",
            "LOADL_STEP_ID",
        ]

        for env_var in hpc_env_vars:
            if env_var in os.environ:
                return "hpc"

        if Path("/scratch").exists():
            return "hpc"

        return "laptop"


# Backward compatibility alias
JobScheduler = JobSchedulerService
