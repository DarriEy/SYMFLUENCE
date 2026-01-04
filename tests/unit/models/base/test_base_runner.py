"""
Unit tests for BaseModelRunner.

Tests for the shared model runner infrastructure including:
- Installation path resolution
- Subprocess execution
- File verification
- Configuration path resolution
- Output verification
- Experiment directory management
- Legacy path aliases
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import subprocess
import tempfile
import shutil

from symfluence.models.base.base_runner import BaseModelRunner
from symfluence.core.config.models import SymfluenceConfig


class ConcreteModelRunner(BaseModelRunner):
    """Concrete implementation of BaseModelRunner for testing."""

    def _get_model_name(self) -> str:
        return "TEST_MODEL"


def _create_config(tmp_path, overrides=None):
    """Create a valid SymfluenceConfig with required fields."""
    base = {
        'SYMFLUENCE_DATA_DIR': str(tmp_path / 'data'),
        'SYMFLUENCE_CODE_DIR': str(tmp_path / 'code'),
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'exp_001',
        'EXPERIMENT_TIME_START': '2020-01-01 00:00',
        'EXPERIMENT_TIME_END': '2020-01-02 00:00',
        'DOMAIN_DEFINITION_METHOD': 'lumped',
        'DOMAIN_DISCRETIZATION': 'lumped',
        'HYDROLOGICAL_MODEL': 'SUMMA',
        'FORCING_DATASET': 'ERA5',
    }
    if overrides:
        base.update(overrides)
    return SymfluenceConfig(**base)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


@pytest.fixture
def mock_logger():
    """Create a mock logger."""
    return Mock()


@pytest.fixture
def base_config(temp_dir):
    """Create a base configuration for testing."""
    return _create_config(temp_dir)


@pytest.fixture
def runner(base_config, mock_logger, temp_dir):
    """Create a ConcreteModelRunner instance."""
    # Create required directories
    data_dir = base_config.system.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    return ConcreteModelRunner(base_config, mock_logger)


class TestGetInstallPath:
    """Tests for get_install_path method."""

    def test_default_path_data_dir(self, runner, temp_dir):
        """Test default installation path relative to data_dir."""
        result = runner.get_install_path(
            'TEST_INSTALL_PATH',
            'installs/test_model/bin'
        )

        expected = temp_dir / 'data' / 'installs' / 'test_model' / 'bin'
        assert result.resolve() == expected.resolve()

    def test_default_path_project_dir(self, runner, temp_dir):
        """Test default installation path relative to project_dir."""
        result = runner.get_install_path(
            'TEST_INSTALL_PATH',
            'custom/path',
            relative_to='project_dir'
        )

        expected = temp_dir / 'data' / 'domain_test_domain' / 'custom' / 'path'
        assert result.resolve() == expected.resolve()

    def test_custom_path(self, runner, temp_dir):
        """Test custom installation path from config."""
        custom_path = (temp_dir / 'custom_install').resolve()
        
        # Re-init runner with custom config
        config = _create_config(temp_dir, {'TEST_INSTALL_PATH': str(custom_path)})
        runner = ConcreteModelRunner(config, runner.logger)

        result = runner.get_install_path(
            'TEST_INSTALL_PATH',
            'installs/test_model/bin'
        )

        assert result.resolve() == custom_path

    def test_none_uses_default(self, runner, temp_dir):
        """Test that None config value uses default path."""
        # By default the key is missing in base_config, which is effectively None/default behavior
        # But let's be explicit with None if possible, or just rely on absence
        
        # In SymfluenceConfig, if key is not defined, it won't be in config_dict unless it's a model field
        # For TEST_INSTALL_PATH, it's an extra field.
        
        result = runner.get_install_path(
            'TEST_INSTALL_PATH',
            'installs/test_model/bin'
        )

        expected = temp_dir / 'data' / 'installs' / 'test_model' / 'bin'
        assert result.resolve() == expected.resolve()

    def test_must_exist_valid(self, runner, temp_dir):
        """Test must_exist parameter with existing path."""
        install_path = temp_dir / 'data' / 'installs' / 'test_model' / 'bin'
        install_path.mkdir(parents=True, exist_ok=True)

        result = runner.get_install_path(
            'TEST_INSTALL_PATH',
            'installs/test_model/bin',
            must_exist=True
        )

        assert result.resolve() == install_path.resolve()

    def test_must_exist_raises_error(self, runner):
        """Test must_exist parameter raises error for non-existent path."""
        with pytest.raises(FileNotFoundError) as exc_info:
            runner.get_install_path(
                'TEST_INSTALL_PATH',
                'nonexistent/path',
                must_exist=True
            )

        assert 'Installation path not found' in str(exc_info.value)
        assert 'TEST_INSTALL_PATH' in str(exc_info.value)


class TestExecuteModelSubprocess:
    """Tests for execute_model_subprocess method."""

    def test_success(self, runner, temp_dir, mock_logger):
        """Test successful subprocess execution."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = runner.execute_model_subprocess(
                ['echo', 'test'],
                log_file
            )

            assert result.returncode == 0
            mock_logger.info.assert_called_with("Model execution completed successfully")

    def test_custom_success_message(self, runner, temp_dir, mock_logger):
        """Test custom success message."""
        log_file = temp_dir / 'test.log'
        custom_message = "Custom success!"

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            runner.execute_model_subprocess(
                ['echo', 'test'],
                log_file,
                success_message=custom_message
            )

            mock_logger.info.assert_called_with(custom_message)

    def test_nonzero_return_code_with_check_false(self, runner, temp_dir, mock_logger):
        """Test non-zero return code when check=False."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 1
            mock_run.return_value = mock_result

            result = runner.execute_model_subprocess(
                ['false'],
                log_file,
                check=False
            )

            assert result.returncode == 1
            mock_logger.warning.assert_called_with("Process exited with code 1")

    def test_failure_with_check_true(self, runner, temp_dir, mock_logger):
        """Test subprocess failure when check=True."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'test_cmd')

            with pytest.raises(subprocess.CalledProcessError):
                runner.execute_model_subprocess(
                    ['false'],
                    log_file,
                    check=True
                )

            mock_logger.error.assert_called()

    def test_error_context_logged(self, runner, temp_dir, mock_logger):
        """Test that error context is logged on failure."""
        log_file = temp_dir / 'test.log'
        error_context = {
            'binary_path': '/usr/bin/test',
            'ld_library_path': '/usr/lib'
        }

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, 'test_cmd')

            with pytest.raises(subprocess.CalledProcessError):
                runner.execute_model_subprocess(
                    ['false'],
                    log_file,
                    error_context=error_context
                )

            # Check that error context was logged
            assert mock_logger.error.call_count >= 2

    def test_timeout_expired(self, runner, temp_dir, mock_logger):
        """Test subprocess timeout."""
        log_file = temp_dir / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired('test_cmd', 10)

            with pytest.raises(subprocess.TimeoutExpired):
                runner.execute_model_subprocess(
                    ['sleep', '100'],
                    log_file,
                    timeout=10
                )

            # Verify timeout error was logged
            timeout_logged = any(
                'timeout' in str(call).lower()
                for call in mock_logger.error.call_args_list
            )
            assert timeout_logged

    def test_environment_variables_merged(self, runner, temp_dir):
        """Test that environment variables are properly merged."""
        log_file = temp_dir / 'test.log'
        custom_env = {'CUSTOM_VAR': 'custom_value'}

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            runner.execute_model_subprocess(
                ['echo', 'test'],
                log_file,
                env=custom_env
            )

            # Verify subprocess.run was called with merged environment
            call_kwargs = mock_run.call_args[1]
            assert 'CUSTOM_VAR' in call_kwargs['env']
            assert call_kwargs['env']['CUSTOM_VAR'] == 'custom_value'

    def test_log_directory_created(self, runner, temp_dir):
        """Test that log directory is created if it doesn't exist."""
        log_file = temp_dir / 'subdir' / 'nested' / 'test.log'

        with patch('subprocess.run') as mock_run:
            mock_result = Mock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            runner.execute_model_subprocess(
                ['echo', 'test'],
                log_file
            )

            assert log_file.parent.exists()


class TestVerifyRequiredFiles:
    """Tests for verify_required_files method."""

    def test_all_files_exist(self, runner, temp_dir, mock_logger):
        """Test verification when all files exist."""
        file1 = temp_dir / 'file1.txt'
        file2 = temp_dir / 'file2.txt'
        file1.write_text('test')
        file2.write_text('test')

        # Should not raise
        runner.verify_required_files([file1, file2], context="testing")

        # Should log debug message
        assert mock_logger.debug.called

    def test_single_file_exists(self, runner, temp_dir):
        """Test verification with single file path."""
        file1 = temp_dir / 'file1.txt'
        file1.write_text('test')

        # Should not raise
        runner.verify_required_files(file1, context="testing")

    def test_missing_files_raises_error(self, runner, temp_dir, mock_logger):
        """Test that missing files raise FileNotFoundError."""
        file1 = temp_dir / 'file1.txt'
        file2 = temp_dir / 'file2.txt'  # Does not exist

        with pytest.raises(FileNotFoundError) as exc_info:
            runner.verify_required_files([file1, file2], context="testing")

        error_msg = str(exc_info.value)
        assert 'testing' in error_msg
        assert str(file1) in error_msg or str(file2) in error_msg
        mock_logger.error.assert_called()

    def test_all_missing_files_in_error(self, runner, temp_dir):
        """Test that all missing files are listed in error."""
        file1 = temp_dir / 'file1.txt'
        file2 = temp_dir / 'file2.txt'
        file3 = temp_dir / 'file3.txt'

        with pytest.raises(FileNotFoundError) as exc_info:
            runner.verify_required_files([file1, file2, file3], context="testing")

        error_msg = str(exc_info.value)
        assert str(file1) in error_msg
        assert str(file2) in error_msg
        assert str(file3) in error_msg


class TestGetConfigPath:
    """Tests for get_config_path method."""

    def test_default_path(self, runner, temp_dir):
        """Test config path resolution with default."""
        result = runner.get_config_path(
            'TEST_CONFIG_PATH',
            'settings/test_model'
        )

        expected = temp_dir / 'data' / 'domain_test_domain' / 'settings' / 'test_model'
        assert result.resolve() == expected.resolve()

    def test_custom_path(self, runner, temp_dir):
        """Test config path resolution with custom path."""
        custom_path = (temp_dir / 'custom_settings').resolve()
        
        # Re-init
        config = _create_config(temp_dir, {'TEST_CONFIG_PATH': str(custom_path)})
        runner = ConcreteModelRunner(config, runner.logger)

        result = runner.get_config_path(
            'TEST_CONFIG_PATH',
            'settings/test_model'
        )

        assert result.resolve() == custom_path


class TestVerifyModelOutputs:
    """Tests for verify_model_outputs method."""

    def test_all_outputs_exist(self, runner, temp_dir, mock_logger):
        """Test verification when all output files exist."""
        runner.output_dir = temp_dir / 'output'
        runner.output_dir.mkdir(parents=True, exist_ok=True)

        (runner.output_dir / 'output1.nc').write_text('test')
        (runner.output_dir / 'output2.nc').write_text('test')

        result = runner.verify_model_outputs(['output1.nc', 'output2.nc'])

        assert result is True
        mock_logger.debug.assert_called()

    def test_single_output_exists(self, runner, temp_dir):
        """Test verification with single output file."""
        runner.output_dir = temp_dir / 'output'
        runner.output_dir.mkdir(parents=True, exist_ok=True)
        (runner.output_dir / 'output.nc').write_text('test')

        result = runner.verify_model_outputs('output.nc')

        assert result is True

    def test_missing_outputs_returns_false(self, runner, temp_dir, mock_logger):
        """Test that missing outputs return False."""
        runner.output_dir = temp_dir / 'output'
        runner.output_dir.mkdir(parents=True, exist_ok=True)

        (runner.output_dir / 'output1.nc').write_text('test')
        # output2.nc does not exist

        result = runner.verify_model_outputs(['output1.nc', 'output2.nc'])

        assert result is False
        mock_logger.error.assert_called()

    def test_custom_output_dir(self, runner, temp_dir):
        """Test verification with custom output directory."""
        custom_dir = temp_dir / 'custom_output'
        custom_dir.mkdir(parents=True, exist_ok=True)
        (custom_dir / 'output.nc').write_text('test')

        result = runner.verify_model_outputs('output.nc', output_dir=custom_dir)

        assert result is True


class TestGetExperimentOutputDir:
    """Tests for get_experiment_output_dir method."""

    def test_default_experiment_id(self, runner, temp_dir):
        """Test experiment output directory with default experiment ID."""
        result = runner.get_experiment_output_dir()

        expected = temp_dir / 'data' / 'domain_test_domain' / 'simulations' / 'exp_001' / 'TEST_MODEL'
        assert result.resolve() == expected.resolve()

    def test_custom_experiment_id(self, runner, temp_dir):
        """Test experiment output directory with custom experiment ID."""
        result = runner.get_experiment_output_dir(experiment_id='exp_custom')

        expected = temp_dir / 'data' / 'domain_test_domain' / 'simulations' / 'exp_custom' / 'TEST_MODEL'
        assert result.resolve() == expected.resolve()


class TestSetupPathAliases:
    """Tests for setup_path_aliases method."""

    def test_valid_aliases(self, runner, mock_logger):
        """Test setting up valid path aliases."""
        runner.setup_path_aliases({
            'root_path': 'data_dir',
            'result_dir': 'output_dir'
        })

        assert hasattr(runner, 'root_path')
        assert runner.root_path == runner.data_dir
        assert hasattr(runner, 'result_dir')
        assert runner.result_dir == runner.output_dir

    def test_invalid_source_attribute(self, runner, mock_logger):
        """Test handling of invalid source attribute."""
        runner.setup_path_aliases({
            'test_alias': 'nonexistent_attr'
        })

        assert not hasattr(runner, 'test_alias')
        mock_logger.warning.assert_called()

    def test_multiple_aliases(self, runner):
        """Test setting up multiple aliases at once."""
        runner.setup_path_aliases({
            'alias1': 'data_dir',
            'alias2': 'project_dir',
            'alias3': 'model_name'
        })

        assert runner.alias1 == runner.data_dir
        assert runner.alias2 == runner.project_dir
        assert runner.alias3 == runner.model_name


class TestBaseRunnerIntegration:
    """Integration tests for BaseModelRunner."""

    def test_initialization_sequence(self, base_config, mock_logger, temp_dir):
        """Test complete initialization sequence."""
        data_dir = base_config.system.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        runner = ConcreteModelRunner(base_config, mock_logger)

        # Verify all base attributes are set
        assert runner.data_dir == data_dir.resolve()
        assert runner.domain_name == 'test_domain'
        assert runner.project_dir == data_dir.resolve() / 'domain_test_domain'
        assert runner.model_name == 'TEST_MODEL'
        assert hasattr(runner, 'output_dir')

    def test_backup_settings_integration(self, runner, temp_dir):
        """Test backup_settings method (existing method)."""
        # Create source directory with files
        source_dir = temp_dir / 'settings'
        source_dir.mkdir()
        (source_dir / 'config.txt').write_text('test config')
        (source_dir / 'params.txt').write_text('test params')

        runner.backup_settings(source_dir)

        # Verify backup was created
        backup_path = runner.output_dir / 'run_settings'
        assert backup_path.exists()
        assert (backup_path / 'config.txt').exists()
        assert (backup_path / 'params.txt').exists()

    def test_get_log_path_integration(self, runner):
        """Test get_log_path method (existing method)."""
        log_path = runner.get_log_path()

        assert log_path.exists()
        assert log_path.parent == runner.output_dir
        assert log_path.name == 'logs'
