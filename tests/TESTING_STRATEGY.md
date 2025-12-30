# SYMFLUENCE Testing Strategy

## Overview

SYMFLUENCE uses a comprehensive testing strategy with multiple test levels to ensure code quality and functionality across all components.

## CI Workflows

### 1. CI - Lint Only (`.github/workflows/ci.yml`)
**Triggers:** Every push/PR to main or develop
**Duration:** ~2 minutes
**Purpose:** Fast feedback on code style

**What it does:**
- Runs `ruff` linter on all Python code
- Checks code style and common issues
- No installation of external binaries required

### 2. SYMFLUENCE - Full Install & Validate (`.github/workflows/install-validate.yml`)
**Triggers:**
- Push/PR to main or develop (runs **quick** tests)
- Manual dispatch with test level selection (smoke/quick/full)
- Weekly schedule on Sundays (runs **full** tests)
- Main branch pushes (runs **full** tests)

**What it does:**
- Full installation of SUMMA, mizuRoute, TauDEM, FUSE binaries
- Complete Python environment setup
- Comprehensive testing based on trigger

## Test Levels

### Smoke Tests (~5 minutes)
**Command:** `pytest -m "smoke"`
**When:** Manual dispatch only
**Coverage:**
- ✅ Binary validation (SUMMA, mizuRoute, TauDEM)
- ✅ Package imports
- ✅ 3-hour SUMMA workflow

### Quick Tests (~20 minutes) **[DEFAULT for develop branch]**
**Command:** Multiple pytest commands
**When:** Every push/PR to develop
**Coverage:**
- ✅ All unit tests
- ✅ Binary validation
- ✅ Package imports
- ✅ Basic integration tests
- ✅ 3-hour SUMMA workflow (quick e2e)

### Full Tests (~90 minutes) **[For main branch & weekly]**
**Command:** Multiple pytest commands
**When:** Push to main, weekly schedule
**Coverage:**
- ✅ All unit tests
- ✅ All integration tests
- ✅ Binary validation
- ✅ Package imports
- ✅ 3-hour SUMMA workflow
- ✅ 1-month SUMMA + mizuRoute workflow
- ✅ Calibration workflow

## Test Markers

Tests are organized using pytest markers defined in `pytest.ini`:

### Test Type Markers
- `@pytest.mark.unit` - Unit tests (fast, isolated functions)
- `@pytest.mark.integration` - Integration tests (module interactions)
- `@pytest.mark.e2e` - End-to-end tests (full workflows)

### Speed Markers
- `@pytest.mark.quick` - Quick tests (<5s)
- `@pytest.mark.slow` - Slow tests (>30s)

### Requirement Markers
- `@pytest.mark.requires_data` - Requires external data downloads
- `@pytest.mark.requires_cloud` - Requires cloud API credentials
- `@pytest.mark.requires_binaries` - Requires external binaries (SUMMA, etc.)

### Component Markers
- `@pytest.mark.domain` - Domain workflow tests
- `@pytest.mark.data` - Data acquisition/processing tests
- `@pytest.mark.models` - Model execution tests
- `@pytest.mark.calibration` - Calibration/optimization tests

### Model-Specific Markers
- `@pytest.mark.summa` - SUMMA-specific tests
- `@pytest.mark.fuse` - FUSE-specific tests
- `@pytest.mark.ngen` - NGEN-specific tests
- `@pytest.mark.gr` - GR-specific tests

### CI Markers
- `@pytest.mark.smoke` - Smoke tests (minimal validation)
- `@pytest.mark.ci_quick` - Quick CI validation
- `@pytest.mark.ci_full` - Full CI validation

## Test Structure

```
tests/
├── unit/                      # Fast, isolated unit tests
│   ├── test_config_loader.py
│   ├── test_multivariate_system.py
│   └── test_soil_multiplier.py
├── integration/               # Module interaction tests
│   ├── calibration/
│   ├── data/
│   ├── domain/
│   └── models/
├── e2e/                       # End-to-end workflow tests
│   └── test_install_validate.py
├── fixtures/                  # Shared test fixtures
└── utils/                     # Test utilities

```

## Writing Tests

### Unit Tests
```python
import pytest
from symfluence.module import function

pytestmark = [pytest.mark.unit, pytest.mark.quick]

def test_function_behavior():
    """Test isolated function behavior."""
    result = function(input_data)
    assert result == expected_output
```

### Integration Tests
```python
import pytest
from symfluence import SYMFLUENCE

pytestmark = [
    pytest.mark.integration,
    pytest.mark.models,
    pytest.mark.requires_data
]

def test_model_integration(symfluence_instance):
    """Test model preprocessing and execution."""
    symfluence_instance.preprocess()
    results = symfluence_instance.run_model()
    assert results.exists()
```

### E2E Tests
```python
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.ci_quick,
    pytest.mark.smoke,
    pytest.mark.requires_binaries
]

def test_complete_workflow(tmp_path):
    """Test complete workflow from setup to results."""
    # Full workflow test
    pass
```

## Running Tests Locally

### Run all unit tests (fast)
```bash
pytest -v -m "unit"
```

### Run quick tests (like CI quick mode)
```bash
pytest -v -m "unit"
pytest -v -m "ci_quick"
```

### Run integration tests
```bash
pytest -v -m "integration"
```

### Run all tests except those requiring data
```bash
pytest -v -m "not requires_data"
```

### Run smoke tests
```bash
pytest -v -m "smoke"
```

### Run specific component tests
```bash
pytest -v -m "models and summa"
pytest -v -m "calibration"
pytest -v -m "domain"
```

## Test Coverage Goals

### Current Coverage
- ✅ Binary validation
- ✅ Package imports
- ✅ Configuration loading and validation
- ✅ SUMMA quick workflows (3-hour)
- ✅ SUMMA full workflows (1-month)
- ✅ Calibration workflows

### Areas to Expand
- 🔄 Unit tests for all utility modules
- 🔄 Integration tests for data acquisition (CARRA, CERRA, ERA5, etc.)
- 🔄 Integration tests for domain workflows (distributed, lumped, regional, etc.)
- 🔄 FUSE model tests
- 🔄 mizuRoute standalone tests
- 🔄 TauDEM preprocessing tests
- 🔄 Multi-model comparison tests
- 🔄 Sensitivity analysis tests

## Best Practices

1. **Mark tests appropriately** - Use all relevant markers
2. **Use fixtures** - Share setup code via `conftest.py`
3. **Test one thing** - Each test should validate one specific behavior
4. **Use descriptive names** - Test names should describe what they validate
5. **Document test purpose** - Include docstrings explaining test intent
6. **Mock external dependencies** - Use mocks for cloud APIs, external data
7. **Clean up resources** - Use `tmp_path` fixtures and clean up test data

## Debugging Failed Tests

### View test output
```bash
pytest -v --tb=short
```

### Run specific failing test
```bash
pytest -v tests/path/to/test.py::test_name
```

### Run with full traceback
```bash
pytest -v --tb=long tests/path/to/test.py::test_name
```

### See print statements
```bash
pytest -v -s tests/path/to/test.py::test_name
```

## CI Artifacts

When tests fail, the workflow uploads artifacts:
- **test-outputs** - Simulation outputs, forcing data, settings
- **pytest-logs** - pytest cache and log files

Download these from the GitHub Actions UI to debug failures.
