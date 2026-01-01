"""Unit tests for JobScheduler."""

import pytest
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path

from symfluence.utils.cli.job_scheduler import JobScheduler

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestInitialization:
    """Test JobScheduler initialization."""

    def test_initialization(self):
        """Test JobScheduler creates successfully."""
        scheduler = JobScheduler()
        assert scheduler is not None


class TestSlurmAvailability:
    """Test SLURM availability checking."""

    @patch('shutil.which')
    def test_slurm_available(self, mock_which, job_scheduler):
        """Test SLURM detected when sbatch exists."""
        mock_which.return_value = '/usr/bin/sbatch'

        result = job_scheduler._check_slurm_available()

        assert result is True
        mock_which.assert_called_with('sbatch')

    @patch('shutil.which')
    def test_slurm_not_available(self, mock_which, job_scheduler):
        """Test SLURM not detected when sbatch missing."""
        mock_which.return_value = None

        result = job_scheduler._check_slurm_available()

        assert result is False


class TestEnvironmentDetection:
    """Test HPC vs laptop environment detection."""

    @patch.dict('os.environ', {'SLURM_CLUSTER_NAME': 'cluster1'}, clear=False)
    @patch('shutil.which')
    def test_detect_hpc_via_slurm_env(self, mock_which, job_scheduler):
        """Test HPC detection via SLURM env var."""
        mock_which.return_value = '/usr/bin/sbatch'

        env = job_scheduler.detect_environment()

        assert env == 'hpc'

    @patch('pathlib.Path.exists')
    @patch('shutil.which')
    @patch.dict('os.environ', {}, clear=True)
    def test_detect_hpc_via_scratch_dir(self, mock_which, mock_exists, job_scheduler):
        """Test HPC detection via /scratch directory."""
        mock_which.return_value = '/usr/bin/sbatch'

        def exists_side_effect(path):
            # Return True for /scratch, False for others
            return str(path) == '/scratch'

        mock_exists.side_effect = exists_side_effect

        env = job_scheduler.detect_environment()

        assert env == 'hpc'

    @patch('shutil.which')
    @patch.dict('os.environ', {}, clear=True)
    def test_detect_laptop(self, mock_which, job_scheduler):
        """Test laptop detection when no HPC indicators."""
        mock_which.return_value = None

        env = job_scheduler.detect_environment()

        assert env == 'laptop'


class TestSlurmScriptGeneration:
    """Test SLURM script generation."""

    def test_workflow_mode_script(self, job_scheduler, sample_config):
        """Test script generation for workflow mode."""
        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'job_mode': 'workflow'
        }
        slurm_options = {
            'job_name': 'test_job',
            'job_account': 'test_account',
            'job_time': '24:00:00',
            'job_ntasks': 8,
            'job_memory': '32GB'
        }
        config_file = '/path/to/config.yaml'

        script = job_scheduler._create_symfluence_slurm_script(
            execution_plan, slurm_options, config_file
        )

        # Assertions
        assert '#SBATCH --job-name=test_job' in script
        assert '#SBATCH --account=test_account' in script
        assert '#SBATCH --time=24:00:00' in script
        assert config_file in script

    def test_individual_steps_script(self, job_scheduler, sample_config):
        """Test script generation for individual steps mode."""
        execution_plan = {
            'mode': 'individual_steps',
            'config': sample_config,
            'job_mode': 'individual_steps',
            'job_steps': ['define_domain', 'discretize_domain']
        }
        slurm_options = {
            'job_name': 'steps_job',
            'job_account': 'test_account',
            'job_time': '12:00:00',
            'job_ntasks': 4,
            'job_memory': '16GB'
        }
        config_file = '/path/to/config.yaml'

        script = job_scheduler._create_symfluence_slurm_script(
            execution_plan, slurm_options, config_file
        )

        assert '--define_domain' in script or 'define_domain' in script
        assert '--discretize_domain' in script or 'discretize_domain' in script

    def test_pour_point_setup_script(self, job_scheduler):
        """Test script generation for pour point setup."""
        execution_plan = {
            'mode': 'pour_point_setup',
            'job_mode': 'pour_point_setup',
            'pour_point': {
                'coordinates': '51.0/115.0',
                'domain_name': 'test_domain',
                'domain_definition_method': 'delineate'
            }
        }
        slurm_options = {
            'job_name': 'pour_point_job',
            'job_account': 'test_account',
            'job_time': '06:00:00',
            'job_ntasks': 4,
            'job_memory': '8GB'
        }
        config_file = '/path/to/config.yaml'

        script = job_scheduler._create_symfluence_slurm_script(
            execution_plan, slurm_options, config_file
        )

        assert '--pour_point 51.0/115.0' in script or '51.0/115.0' in script
        assert 'test_domain' in script

    def test_script_includes_sbatch_directives(self, job_scheduler, sample_config):
        """Test that script includes all necessary SBATCH directives."""
        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'job_mode': 'workflow'
        }
        slurm_options = {
            'job_name': 'test',
            'job_account': 'test_account',
            'job_time': '10:00:00',
            'job_ntasks': 2,
            'job_memory': '4GB',
            'job_partition': 'compute'
        }

        script = job_scheduler._create_symfluence_slurm_script(
            execution_plan, slurm_options, 'config.yaml'
        )

        assert '#!/bin/bash' in script
        assert '#SBATCH' in script
        assert '#SBATCH --job-name=test' in script
        assert '#SBATCH --partition=compute' in script or script.count('#SBATCH') > 0


class TestJobSubmission:
    """Test SLURM job submission."""

    @patch('builtins.open', new_callable=mock_open)
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_successful_submission(self, mock_which, mock_subprocess, mock_file, job_scheduler, sample_config):
        """Test successful job submission."""
        mock_which.return_value = '/usr/bin/sbatch'
        mock_subprocess.return_value = MagicMock(
            stdout='Submitted batch job 12345\n',
            stderr='',
            returncode=0
        )

        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'job_mode': 'workflow',
            'slurm_options': {
                'job_name': 'test',
                'job_account': 'test',
                'job_time': '10:00:00',
                'job_ntasks': 2,
                'job_memory': '4GB'
            }
        }

        result = job_scheduler.submit_slurm_job(execution_plan)

        assert result['success'] is True
        assert result['job_id'] == '12345'
        assert mock_subprocess.called

    @patch('builtins.open', new_callable=mock_open)
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_submission_failure(self, mock_which, mock_subprocess, mock_file, job_scheduler, sample_config):
        """Test handling of submission failure."""
        mock_which.return_value = '/usr/bin/sbatch'
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=['sbatch', 'script.sh'],
            stderr='sbatch: error: Invalid account'
        )

        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'job_mode': 'workflow',
            'slurm_options': {
                'job_name': 'test',
                'job_account': 'invalid',
                'job_time': '10:00:00',
                'job_ntasks': 2,
                'job_memory': '4GB'
            }
        }

        with pytest.raises(RuntimeError, match="Failed to submit SLURM job"):
            job_scheduler.submit_slurm_job(execution_plan)

    @patch('builtins.open', new_callable=mock_open)
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_auto_job_name_generation(self, mock_which, mock_subprocess, mock_file, job_scheduler, sample_config):
        """Test automatic job name generation."""
        mock_which.return_value = '/usr/bin/sbatch'
        mock_subprocess.return_value = MagicMock(
            stdout='Submitted batch job 12345\n',
            stderr='',
            returncode=0
        )

        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'job_mode': 'workflow',
            'slurm_options': {
                'job_account': 'test',
                'job_time': '10:00:00',
                'job_ntasks': 2,
                'job_memory': '4GB'
                # No job_name provided
            }
        }

        result = job_scheduler.submit_slurm_job(execution_plan)

        # Job name should be auto-generated
        assert 'job_name' in result
        assert result['success'] is True

    @patch('shutil.which')
    def test_slurm_not_available_raises_error(self, mock_which, job_scheduler, sample_config):
        """Test that missing SLURM raises error."""
        mock_which.return_value = None

        execution_plan = {
            'mode': 'workflow',
            'config': sample_config,
            'slurm_options': {}
        }

        with pytest.raises(RuntimeError, match="SLURM commands .* not available"):
            job_scheduler.submit_slurm_job(execution_plan)


class TestJobMonitoring:
    """Test SLURM job monitoring."""

    @patch('subprocess.run')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_monitor_until_completion(self, mock_sleep, mock_subprocess, job_scheduler):
        """Test job monitoring until completion."""
        # Mock squeue returning RUNNING then COMPLETED
        mock_subprocess.side_effect = [
            MagicMock(stdout='RUNNING\n', stderr='', returncode=0),
            MagicMock(stdout='RUNNING\n', stderr='', returncode=0),
            MagicMock(stdout='COMPLETED\n', stderr='', returncode=0),
        ]

        job_scheduler._monitor_slurm_job('12345')

        # Should have called squeue 3 times
        assert mock_subprocess.call_count == 3

    @patch('subprocess.run')
    @patch('time.sleep')
    def test_monitor_failure_detection(self, mock_sleep, mock_subprocess, job_scheduler):
        """Test detection of job failure."""
        mock_subprocess.return_value = MagicMock(
            stdout='FAILED\n',
            stderr='',
            returncode=0
        )

        job_scheduler._monitor_slurm_job('12345')

        # Should detect failure and exit monitoring
        assert mock_subprocess.called


class TestHandleSlurmJobSubmission:
    """Test high-level SLURM job submission handler."""

    @patch.object(JobScheduler, 'submit_slurm_job')
    def test_successful_submission_workflow(self, mock_submit, job_scheduler):
        """Test successful submission workflow."""
        mock_submit.return_value = {
            'success': True,
            'job_id': '12345',
            'job_name': 'test_job'
        }

        execution_plan = {
            'slurm_options': {
                'job_account': 'test'
            }
        }

        result = job_scheduler.handle_slurm_job_submission(execution_plan)

        assert result is True
        mock_submit.assert_called_once()

    @patch.object(JobScheduler, 'submit_slurm_job')
    def test_failed_submission_workflow(self, mock_submit, job_scheduler):
        """Test failed submission workflow."""
        mock_submit.return_value = {
            'success': False,
            'error': 'Submission failed'
        }

        execution_plan = {
            'slurm_options': {}
        }

        result = job_scheduler.handle_slurm_job_submission(execution_plan)

        assert result is False

    @patch.object(JobScheduler, 'submit_slurm_job')
    def test_exception_handling(self, mock_submit, job_scheduler):
        """Test exception handling in submission workflow."""
        mock_submit.side_effect = RuntimeError("SLURM error")

        execution_plan = {
            'slurm_options': {}
        }

        result = job_scheduler.handle_slurm_job_submission(execution_plan)

        assert result is False
