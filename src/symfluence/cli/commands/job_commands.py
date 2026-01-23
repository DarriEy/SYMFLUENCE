"""
SLURM job submission command handlers for SYMFLUENCE CLI.

This module implements handlers for submitting workflows as SLURM jobs.
"""

from argparse import Namespace

from .base import BaseCommand, cli_exception_handler
from ..exit_codes import ExitCode


class JobCommands(BaseCommand):
    """Handlers for SLURM job submission commands."""

    @staticmethod
    @cli_exception_handler
    def submit(args: Namespace) -> int:
        """
        Execute: symfluence job submit [workflow command]

        Submit a workflow command as a SLURM job.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        from symfluence.cli.services import JobScheduler

        job_scheduler = JobScheduler()

        BaseCommand._console.info("Submitting SLURM job...")

        # Build SLURM options using safe accessors
        slurm_options = {
            'job_name': BaseCommand.get_arg(args, 'job_name'),
            'job_time': BaseCommand.get_arg(args, 'job_time', '48:00:00'),
            'job_nodes': BaseCommand.get_arg(args, 'job_nodes', 1),
            'job_ntasks': BaseCommand.get_arg(args, 'job_ntasks', 1),
            'job_memory': BaseCommand.get_arg(args, 'job_memory', '50G'),
            'job_account': BaseCommand.get_arg(args, 'job_account'),
            'job_partition': BaseCommand.get_arg(args, 'job_partition'),
            'job_modules': BaseCommand.get_arg(args, 'job_modules', 'symfluence_modules'),
            'conda_env': BaseCommand.get_arg(args, 'conda_env', 'symfluence'),
            'submit_and_wait': BaseCommand.get_arg(args, 'submit_and_wait', False),
            'slurm_template': BaseCommand.get_arg(args, 'slurm_template')
        }

        # Get workflow command from remaining args
        workflow_args = BaseCommand.get_arg(args, 'workflow_args', [])

        BaseCommand._console.indent(f"Job name: {slurm_options['job_name'] or 'auto-generated'}")
        BaseCommand._console.indent(f"Time limit: {slurm_options['job_time']}")
        BaseCommand._console.indent(f"Resources: {slurm_options['job_nodes']} nodes, {slurm_options['job_ntasks']} tasks, {slurm_options['job_memory']}")
        if workflow_args:
            BaseCommand._console.indent(f"Workflow command: symfluence {' '.join(workflow_args)}")

        # Build execution plan
        execution_plan = {
            'config_file': BaseCommand.get_config_path(args),
            'job_mode': 'workflow',
            'job_steps': workflow_args,
            'slurm_options': slurm_options
        }

        # Submit the job
        success = job_scheduler.handle_slurm_job_submission(execution_plan)

        if success:
            BaseCommand._console.success("SLURM job submitted successfully")
            if slurm_options['submit_and_wait']:
                BaseCommand._console.info("Monitoring job execution...")
            return ExitCode.SUCCESS
        else:
            BaseCommand._console.error("SLURM job submission failed")
            return ExitCode.JOB_SUBMIT_ERROR
