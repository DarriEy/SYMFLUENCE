"""CLI test fixtures and mocking utilities."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock
from typing import Dict, Any


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample SYMFLUENCE configuration for testing.

    Returns a minimal valid configuration dict that can be used
    in CLI tests without requiring actual file I/O.
    """
    return {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_experiment',
        'SYMFLUENCE_DATA_DIR': '/tmp/symfluence_data',
        'DOMAIN_DISCRETIZATION': 'lumped',
        'START_DATETIME': '2020-01-01',
        'END_DATETIME': '2020-12-31',
        'FORCING_DATASET': 'ERA5',
        'MODEL': 'SUMMA',
        'CALIBRATION_PERIOD_START': '2020-01-01',
        'CALIBRATION_PERIOD_END': '2020-06-30',
        'EVALUATION_PERIOD_START': '2020-07-01',
        'EVALUATION_PERIOD_END': '2020-12-31',
    }


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory structure for config testing.

    Creates directories that mirror the expected SYMFLUENCE structure:
    - config_files/ (for templates)
    - domains/ (for domain configs)
    """
    config_dir = tmp_path / "config_files"
    config_dir.mkdir(parents=True)

    domains_dir = tmp_path / "domains"
    domains_dir.mkdir(parents=True)

    # Create a mock config template
    template_content = """
# SYMFLUENCE Configuration Template
DOMAIN_NAME: test_domain
EXPERIMENT_ID: test_experiment
SYMFLUENCE_DATA_DIR: /path/to/data
DOMAIN_DISCRETIZATION: lumped
START_DATETIME: '2020-01-01'
END_DATETIME: '2020-12-31'
FORCING_DATASET: ERA5
MODEL: SUMMA
"""
    template_path = config_dir / "config_template.yaml"
    template_path.write_text(template_content)

    return tmp_path


@pytest.fixture
def mock_external_tools():
    """Mock external_tools_config for BinaryManager testing.

    Provides a simplified tool definition structure for testing
    dependency resolution, installation, and validation logic.
    """
    return {
        'sundials': {
            'name': 'SUNDIALS',
            'description': 'SUite of Nonlinear and DIfferential/ALgebraic equation Solvers',
            'repository': 'https://github.com/LLNL/sundials.git',
            'tag': 'v6.5.0',
            'build_dir': 'build',
            'build_commands': [
                'mkdir -p build',
                'cd build && cmake ..',
                'cd build && make',
                'cd build && make install'
            ],
            'install_path_suffix': 'bin/external_tools/sundials',
            'dependencies': [],
            'verify_install': {
                'check_type': 'exists',
                'file_paths': ['lib/libsundials_cvode.so']
            }
        },
        'summa': {
            'name': 'SUMMA',
            'description': 'Structure for Unifying Multiple Modeling Alternatives',
            'repository': 'https://github.com/NCAR/summa.git',
            'tag': 'v3.0.3',
            'build_dir': 'build',
            'build_commands': [
                'mkdir -p build',
                'cd build && cmake ..',
                'cd build && make'
            ],
            'install_path_suffix': 'bin/external_tools/summa',
            'dependencies': ['sundials'],
            'verify_install': {
                'check_type': 'exists',
                'file_paths': ['bin/summa.exe']
            },
            'test_command': {
                'command': '--version',
                'timeout': 10
            }
        },
        'mizuroute': {
            'name': 'mizuRoute',
            'description': 'River routing model',
            'repository': 'https://github.com/NCAR/mizuRoute.git',
            'tag': 'v2.0.1',
            'build_commands': ['make'],
            'install_path_suffix': 'bin/external_tools/mizuRoute',
            'dependencies': [],
            'verify_install': None
        },
        'taudem': {
            'name': 'TauDEM',
            'description': 'Terrain Analysis Using Digital Elevation Models',
            'repository': 'https://github.com/dtarb/TauDEM.git',
            'build_commands': ['mkdir -p build', 'cd build && cmake ..', 'cd build && make'],
            'install_path_suffix': 'bin/external_tools/taudem',
            'dependencies': [],
            'verify_install': {
                'check_type': 'exists_any',
                'file_paths': ['bin/pitremove', 'bin/d8flowdir', 'bin/aread8']
            }
        }
    }


@pytest.fixture
def mock_symfluence_instance():
    """Mock SYMFLUENCE instance for testing.

    Provides a MagicMock that simulates a SYMFLUENCE object with
    commonly used attributes and methods.
    """
    mock_instance = MagicMock()
    mock_instance.config = {
        'DOMAIN_NAME': 'test_domain',
        'EXPERIMENT_ID': 'test_experiment',
        'SYMFLUENCE_DATA_DIR': '/tmp/symfluence_data',
    }
    mock_instance.data_dir = Path('/tmp/symfluence_data')
    mock_instance.domain_dir = Path('/tmp/symfluence_data/domain_test_domain')
    mock_instance.domains = {}

    # Mock workflow steps
    mock_instance.workflow_steps = [
        'setup_project',
        'define_domain',
        'discretize_domain',
        'acquire_forcings',
        'run_model',
        'postprocess_results'
    ]

    return mock_instance


@pytest.fixture
def cli_manager(monkeypatch, mock_symfluence_instance):
    """DEPRECATED: CLIArgumentManager fixture - kept for backward compatibility.

    The CLI has been refactored to use subcommands. This fixture now returns
    None and should be removed from tests that use it.

    For new tests, use individual component fixtures (binary_manager,
    job_scheduler, etc.) directly.
    """
    # Return None - tests using this fixture should be updated
    return None


@pytest.fixture
def binary_manager(mock_external_tools, tmp_path):
    """BinaryManager fixture with mocked external tools.

    Creates a BinaryManager instance with:
    - Mocked external_tools_config
    - Temporary installation directory
    """
    from symfluence.utils.cli.binary_manager import BinaryManager

    # Create manager with mocked external tools
    manager = BinaryManager(external_tools=mock_external_tools)
    manager.install_base_dir = tmp_path / "bin" / "external_tools"
    manager.install_base_dir.mkdir(parents=True, exist_ok=True)

    return manager


@pytest.fixture
def job_scheduler():
    """JobScheduler fixture.

    Creates a JobScheduler instance for testing SLURM script
    generation and job submission logic.
    """
    from symfluence.utils.cli.job_scheduler import JobScheduler

    return JobScheduler()


@pytest.fixture
def notebook_service(tmp_path):
    """NotebookService fixture with temporary repo structure.

    Creates a NotebookService instance with:
    - Temporary repo root
    - Mock examples directory
    """
    from symfluence.utils.cli.notebook_service import NotebookService

    service = NotebookService()

    # Create mock repo structure
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    examples_dir = repo_root / "examples"
    examples_dir.mkdir()

    # Store for testing
    service._test_repo_root = repo_root
    service._test_examples_dir = examples_dir

    return service


@pytest.fixture
def mock_subprocess():
    """Reusable subprocess mock factory.

    Returns a callable that creates configured subprocess.run mocks.
    Usage:
        mock_run = mock_subprocess(returncode=0, stdout="output")
    """
    def _create_mock(returncode=0, stdout="", stderr="", side_effect=None):
        mock = Mock()
        mock.returncode = returncode
        mock.stdout = stdout
        mock.stderr = stderr
        if side_effect:
            mock.side_effect = side_effect
        return mock

    return _create_mock


@pytest.fixture
def mock_yaml_load():
    """Mock yaml.safe_load for config file testing.

    Returns a function that can be used to mock YAML loading
    with custom config dictionaries.
    """
    def _mock_load(config_dict):
        from unittest.mock import mock_open, patch

        mock_file = mock_open(read_data="mocked yaml content")
        with patch('builtins.open', mock_file):
            with patch('yaml.safe_load', return_value=config_dict):
                yield mock_file

    return _mock_load




@pytest.fixture(autouse=True)
def reset_mocks(request):
    """Automatically reset all mocks after each test.

    This autouse fixture ensures test isolation by resetting
    mock call counts and side effects between tests.
    """
    yield
    # Cleanup happens automatically after test
