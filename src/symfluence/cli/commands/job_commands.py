"""
SLURM job submission command handlers for SYMFLUENCE CLI.

This module implements handlers for submitting workflows as SLURM jobs.
"""

import sys
from argparse import Namespace

from .base import BaseCommand


class JobCommands(BaseCommand):
    """Handlers for SLURM job submission commands."""

    @staticmethod
    def submit(args: Namespace) -> int:
        """
        Execute: symfluence job submit [workflow command]

        Submit a workflow command as a SLURM job.

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code (0 for success, non-zero for failure)
        """
        try:
            from symfluence.utils.cli.job_scheduler import JobScheduler

            job_scheduler = JobScheduler()

            BaseCommand.print_info("📤 Submitting SLURM job...")

            # Build SLURM options
            slurm_options = {
                'job_name': getattr(args, 'job_name', None),
                'job_time': getattr(args, 'job_time', '48:00:00'),
                'job_nodes': getattr(args, 'job_nodes', 1),
                'job_ntasks': getattr(args, 'job_ntasks', 1),
                'job_memory': getattr(args, 'job_memory', '50G'),
                'job_account': getattr(args, 'job_account', None),
                'job_partition': getattr(args, 'job_partition', None),
                'job_modules': getattr(args, 'job_modules', 'symfluence_modules'),
                'conda_env': getattr(args, 'conda_env', 'symfluence'),
                'submit_and_wait': getattr(args, 'submit_and_wait', False),
                'slurm_template': getattr(args, 'slurm_template', None)
            }

            # Get workflow command from remaining args
            workflow_args = getattr(args, 'workflow_args', [])

            BaseCommand.print_info(f"   Job name: {slurm_options['job_name'] or 'auto-generated'}")
            BaseCommand.print_info(f"   Time limit: {slurm_options['job_time']}")
            BaseCommand.print_info(f"   Resources: {slurm_options['job_nodes']} nodes, {slurm_options['job_ntasks']} tasks, {slurm_options['job_memory']}")
            if workflow_args:
                BaseCommand.print_info(f"   Workflow command: symfluence {' '.join(workflow_args)}")

            # Submit the job
            success = job_scheduler.submit_slurm_job(
                config_path=BaseCommand.get_config_path(args),
                workflow_command=workflow_args,
                **slurm_options
            )

            if success:
                BaseCommand.print_success("SLURM job submitted successfully")
                if slurm_options['submit_and_wait']:
                    BaseCommand.print_info("Monitoring job execution...")
                return 0
            else:
                BaseCommand.print_error("SLURM job submission failed")
                return 1

        except Exception as e:
            BaseCommand.print_error(f"Job submission failed: {e}")
            if getattr(args, 'debug', False):
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def execute(args: Namespace) -> int:
        """
        Main execution dispatcher (required by BaseCommand).

        Args:
            args: Parsed arguments namespace

        Returns:
            Exit code
        """
        if hasattr(args, 'func'):
            return args.func(args)
        else:
            BaseCommand.print_error("No job action specified")
            return 1
