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

    Args:
        model_name: Model name (SUMMA, FUSE, NGEN, etc.)

    Returns:
        pytest.mark.skipif decorator
    """
    import shutil
    model_available = shutil.which(model_name.lower()) is not None
    return pytest.mark.skipif(
        not model_available,
        reason=f"{model_name} binary not found in PATH"
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
