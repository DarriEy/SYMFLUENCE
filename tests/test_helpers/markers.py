"""
Pytest marker utilities for SYMFLUENCE tests.

Provides helper functions for working with pytest markers and
organizing test categories.
"""

import pytest


# Marker shortcuts for common combinations
def smoke_test(func):
    """
    Decorator for smoke tests (quick, minimal validation).

    Smoke tests should:
    - Run in < 5 minutes
    - Use minimal data (3-hour simulations)
    - Test critical functionality
    - Be suitable for pre-commit checks
    """
    return pytest.mark.smoke(
        pytest.mark.ci_quick(
            pytest.mark.quick(func)
        )
    )


def integration_test(component=None):
    """
    Decorator for integration tests.

    Args:
        component: Component being tested (domain, data, models, calibration)

    Example:
        @integration_test(component="domain")
        def test_watershed_delineation():
            ...
    """
    def decorator(func):
        marks = [pytest.mark.integration]
        if component:
            marks.append(getattr(pytest.mark, component))
        return pytest.mark.slow(
            pytest.mark.requires_data(
                pytest.mark(*marks)(func)
            )
        )
    return decorator


def e2e_test(level="quick"):
    """
    Decorator for end-to-end tests.

    Args:
        level: Test level - "quick" for ci_quick, "full" for ci_full

    Example:
        @e2e_test(level="full")
        def test_full_workflow():
            ...
    """
    def decorator(func):
        marks = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.requires_data]
        if level == "quick":
            marks.append(pytest.mark.ci_quick)
        elif level == "full":
            marks.append(pytest.mark.ci_full)
        return pytest.mark(*marks)(func)
    return decorator


def model_test(model_name):
    """
    Decorator for model-specific tests.

    Args:
        model_name: Model name (summa, fuse, ngen, gr)

    Example:
        @model_test("summa")
        def test_summa_execution():
            ...
    """
    def decorator(func):
        return pytest.mark.models(
            getattr(pytest.mark, model_name.lower())(func)
        )
    return decorator


def calibration_test(func):
    """
    Decorator for calibration tests.

    Calibration tests are typically slow and require data.
    """
    return pytest.mark.calibration(
        pytest.mark.slow(
            pytest.mark.requires_data(func)
        )
    )


def requires_cloud_access(func):
    """
    Decorator for tests that require cloud API access.

    These tests need:
    - CDS API credentials for CARRA/CERRA
    - Internet connection
    - May be slow due to downloads
    """
    return pytest.mark.requires_cloud(
        pytest.mark.requires_data(
            pytest.mark.slow(func)
        )
    )


# Helper functions for skipping tests
def skip_if_no_model(model_name):
    """
    Skip test if model binary is not available.

    Checks both the system PATH and standard SYMFLUENCE_data/installs directory.

    Args:
        model_name: Model name (SUMMA, FUSE, NGEN, HYPE, etc.)

    Returns:
        pytest.mark.skipif decorator
    """
    import shutil
    import os
    from pathlib import Path

    # 1. Check PATH
    model_available = shutil.which(model_name.lower()) is not None
    if model_available:
        return pytest.mark.skipif(False, reason=f"{model_name} binary found in PATH")

    # 2. Check standard install directory
    # Try to find SYMFLUENCE_data/installs
    data_dir = os.environ.get("SYMFLUENCE_DATA")
    if not data_dir:
        # Fallback to relative path from project root
        code_dir = Path(__file__).parent.parent.parent
        data_dir = code_dir.parent / "SYMFLUENCE_data"
    else:
        data_dir = Path(data_dir)

    installs_dir = data_dir / "installs"

    # Map model names to their expected binary paths relative to installs/
    binary_mappings = {
        "SUMMA": ["summa/bin/summa.exe", "summa/bin/summa_sundials.exe"],
        "FUSE": ["fuse/bin/fuse.exe"],
        "NGEN": ["ngen/cmake_build/ngen"],
        "HYPE": ["hype/bin/hype", "hype/hype"],
        "MIZUROUTE": ["mizuroute/bin/mizuRoute.exe"],
    }

    if model_name.upper() in binary_mappings:
        for rel_path in binary_mappings[model_name.upper()]:
            if (installs_dir / rel_path).exists():
                return pytest.mark.skipif(False, reason=f"{model_name} binary found in installs")

    return pytest.mark.skipif(
        True,
        reason=f"{model_name} binary not found in PATH or {installs_dir}"
    )


def skip_if_no_cloud_credentials():
    """
    Skip test if cloud API credentials are not configured.

    Returns:
        pytest.mark.skipif decorator
    """
    from pathlib import Path
    cdsapi_rc = Path.home() / ".cdsapirc"
    return pytest.mark.skipif(
        not cdsapi_rc.exists(),
        reason="CDS API credentials not configured (~/.cdsapirc missing)"
    )
