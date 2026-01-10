# CLI Unit Tests

This directory contains unit tests for the SYMFLUENCE Command-Line Interface (CLI) components.

## Overview

The CLI test suite provides comprehensive coverage of the CLI components that manage user interactions, workflow execution, binary management, job scheduling, and example notebook launching.

## Test Organization

```
tests/unit/cli/
├── __init__.py                      # Package marker
├── conftest.py                       # CLI-specific fixtures
├── test_cli_argument_manager.py     # CLIArgumentManager tests
├── test_binary_manager.py           # BinaryManager tests
├── test_job_scheduler.py            # JobScheduler tests
├── test_notebook_service.py         # NotebookService tests
└── README.md                        # This file
```

## Components Tested

### 1. CLIArgumentManager (`test_cli_argument_manager.py`)

Tests for command-line argument parsing, validation, and execution plan generation.

**Test Coverage:**
- Argument parsing and validation
- Coordinate and bounding box validation
- Execution plan generation for different modes
- Configuration overrides
- Pour point workflow setup
- Status information printing

**Key Test Classes:**
- `TestInitialization` - Component initialization
- `TestCoordinateValidation` - Coordinate parsing and validation
- `TestArgumentParsing` - Command-line argument parsing
- `TestArgumentValidation` - Argument logic validation
- `TestExecutionPlanGeneration` - Execution mode determination
- `TestConfigOverrides` - Configuration override logic
- `TestPourPointSetup` - Pour point workflow configuration
- `TestPrintStatusInformation` - Status reporting

### 2. BinaryManager (`test_binary_manager.py`)

Tests for external tool installation, validation, and dependency management.

**Test Coverage:**
- Binary installation and validation
- Dependency resolution
- Build command execution
- Path validation
- System diagnostics (doctor)
- Tool information display

**Key Test Classes:**
- `TestInitialization` - Manager initialization
- `TestHandleBinaryManagement` - Operation dispatcher
- `TestGetExecutables` - Tool installation
- `TestBinaryValidation` - Binary validation logic
- `TestDoctorDiagnostics` - System diagnostics
- `TestDependencyResolution` - Dependency ordering
- `TestBuildCommands` - Build execution

### 3. JobScheduler (`test_job_scheduler.py`)

Tests for SLURM job submission and monitoring.

**Test Coverage:**
- SLURM availability checking
- Environment detection (HPC vs laptop)
- SLURM script generation
- Job submission
- Job monitoring

**Key Test Classes:**
- `TestSlurmAvailability` - SLURM detection
- `TestEnvironmentDetection` - HPC/laptop detection
- `TestSlurmScriptGeneration` - Script content generation
- `TestJobSubmission` - Job submission logic
- `TestJobMonitoring` - Job status monitoring
- `TestHandleSlurmJobSubmission` - High-level submission workflow

### 4. NotebookService (`test_notebook_service.py`)

Tests for Jupyter notebook launching and kernel management.

**Test Coverage:**
- Example ID parsing
- Notebook discovery
- Virtual environment detection
- Python executable selection
- Kernel registration
- Jupyter launcher selection

**Key Test Classes:**
- `TestExampleIDParsing` - Example ID pattern matching
- `TestNotebookDiscovery` - Notebook file discovery
- `TestVirtualEnvironmentDetection` - Venv detection
- `TestPythonExecutableSelection` - Python path resolution
- `TestKernelRegistration` - Ipykernel installation
- `TestJupyterLaunch` - Jupyter launcher selection
- `TestErrorCases` - Error handling

## Fixtures

CLI-specific fixtures are defined in `conftest.py`:

### Core Fixtures

- **`sample_config`** - Sample SYMFLUENCE configuration dictionary
- **`temp_config_dir`** - Temporary directory with config templates
- **`mock_external_tools`** - Mock tool definitions for BinaryManager
- **`mock_symfluence_instance`** - Mock SYMFLUENCE instance

### Component Fixtures

- **`cli_manager`** - CLIArgumentManager with mocked submanagers
- **`binary_manager`** - BinaryManager with mock tools
- **`job_scheduler`** - JobScheduler instance
- **`notebook_service`** - NotebookService instance

### Utility Fixtures

- **`mock_subprocess`** - Reusable subprocess mock factory
- **`mock_yaml_load`** - YAML loading mock
- **`mock_logger`** - Mock logger for log verification

## Running Tests

### Run All CLI Tests

```bash
pytest tests/unit/cli/ -v
```

### Run Specific Test File

```bash
pytest tests/unit/cli/test_cli_argument_manager.py -v
```

### Run Specific Test Class

```bash
pytest tests/unit/cli/test_cli_argument_manager.py::TestCoordinateValidation -v
```

### Run Specific Test Function

```bash
pytest tests/unit/cli/test_cli_argument_manager.py::TestCoordinateValidation::test_valid_coordinates -v
```

### Run with Coverage

```bash
pytest tests/unit/cli/ --cov=src/symfluence/cli --cov-report=term-missing
```

### Run with Markers

```bash
# Run only CLI tests
pytest -m cli

# Run quick CLI tests
pytest -m "cli and quick"

# Run unit tests excluding CLI
pytest -m "unit and not cli"
```

## Test Markers

All CLI tests are marked with:
- `@pytest.mark.unit` - Unit test marker
- `@pytest.mark.cli` - CLI component marker
- `@pytest.mark.quick` - Quick test (<5s) marker

Additional markers may be applied to specific tests as needed.

## Mocking Strategy

### File I/O Mocking

```python
from unittest.mock import patch, mock_open

@patch('builtins.open', new_callable=mock_open, read_data="config content")
@patch('yaml.safe_load')
def test_config_loading(mock_yaml, mock_file):
    mock_yaml.return_value = {'key': 'value'}
    # Test code here
```

### Subprocess Mocking

```python
from unittest.mock import patch, MagicMock

@patch('subprocess.run')
def test_command_execution(mock_subprocess):
    mock_subprocess.return_value = MagicMock(
        returncode=0,
        stdout='output',
        stderr=''
    )
    # Test code here
```

### Path Mocking

```python
from unittest.mock import patch

@patch('pathlib.Path.exists')
def test_file_existence(mock_exists):
    mock_exists.return_value = True
    # Test code here
```

## Writing New CLI Tests

### Test Template

```python
"""Unit tests for NewComponent."""

import pytest
from unittest.mock import patch, MagicMock
from symfluence.cli.new_component import NewComponent

pytestmark = [pytest.mark.unit, pytest.mark.cli, pytest.mark.quick]


class TestNewComponent:
    """Test NewComponent functionality."""

    def test_basic_functionality(self):
        """Test basic component functionality."""
        component = NewComponent()
        result = component.do_something()
        assert result is not None
```

### Best Practices

1. **Use descriptive test names** - Test names should clearly describe what is being tested
2. **Mock external dependencies** - Mock file I/O, subprocess calls, network requests
3. **Test both success and failure paths** - Include tests for error cases
4. **Use parametrize for similar scenarios** - Avoid code duplication
5. **Keep tests isolated** - Each test should be independent
6. **Use fixtures for common setup** - Define reusable fixtures in conftest.py

## Coverage Goals

**Target Coverage: >80% for all CLI modules**

Current coverage areas:
- CLIArgumentManager: Argument parsing, validation, execution plans
- BinaryManager: Installation, validation, diagnostics
- JobScheduler: SLURM script generation, job submission
- NotebookService: Notebook launching, kernel management

## Continuous Integration

CLI tests are integrated into the CI pipeline:

- **Quick CI** (`ci_quick` marker): Runs on every push to develop
- **Full CI** (`ci_full` marker): Runs on push to main and weekly schedule

All CLI unit tests are marked as `quick` and run in the quick CI validation.

## Troubleshooting

### Test Failures

If tests fail:

1. Check that all CLI components are properly imported
2. Verify that mocks are configured correctly
3. Ensure fixture dependencies are available
4. Check for changes in CLI component interfaces

### Import Errors

If you encounter import errors:

```bash
# Ensure SYMFLUENCE is installed in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/SYMFLUENCE"
```

### Fixture Errors

If fixtures are not found:

1. Ensure `conftest.py` is in the correct location
2. Verify fixture is properly defined and exported
3. Check fixture scope (session, module, function)

## Contributing

When adding new CLI functionality:

1. Write tests for the new functionality
2. Ensure tests follow existing patterns
3. Add appropriate mocking for external dependencies
4. Update this README if adding new test files
5. Run coverage analysis to verify adequate coverage
6. Ensure all tests pass before committing

## Related Documentation

- [TESTING_STRATEGY.md](../../TESTING_STRATEGY.md) - Overall testing strategy
- [pytest.ini](../../../pytest.ini) - Pytest configuration
- [pyproject.toml](../../../pyproject.toml) - Project configuration

## Contact

For questions or issues with CLI tests, please open an issue on the SYMFLUENCE GitHub repository.
