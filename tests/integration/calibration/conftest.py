"""
Calibration test fixtures.

Fixtures specific to calibration and optimization tests.
"""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def ellioaar_domain(symfluence_data_root):
    """
    Path to pre-downloaded Elliðaár domain test data.

    Returns path to example_data/ellioaar_iceland if it exists,
    otherwise returns None (test will download data if needed).

    Args:
        symfluence_data_root: Path to SYMFLUENCE_data root directory

    Returns:
        Path or None: Path to domain data if available
    """
    # Check for data in example_data bundle
    ellioaar_path = symfluence_data_root / "example_data" / "domain_ellioaar_iceland"
    if ellioaar_path.exists():
        return ellioaar_path

    # Also check alternative location
    alt_path = symfluence_data_root / "domain_ellioaar_iceland"
    if alt_path.exists():
        return alt_path

    # No pre-downloaded data available
    return None


@pytest.fixture(scope="session")
def fyris_domain(symfluence_data_root):
    """
    Path to pre-downloaded Fyris domain test data.

    Returns path to example_data/fyris_uppsala if it exists,
    otherwise returns None (test will download data if needed).

    Args:
        symfluence_data_root: Path to SYMFLUENCE_data root directory

    Returns:
        Path or None: Path to domain data if available
    """
    # Check for data in example_data bundle
    fyris_path = symfluence_data_root / "example_data" / "domain_fyris_uppsala"
    if fyris_path.exists():
        return fyris_path

    # Also check alternative location
    alt_path = symfluence_data_root / "domain_fyris_uppsala"
    if alt_path.exists():
        return alt_path

    # No pre-downloaded data available
    return None
