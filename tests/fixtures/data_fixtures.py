"""
Data Fixtures for SYMFLUENCE Tests

Provides fixtures for downloading and managing test data bundles from GitHub releases.
"""

import pytest
import requests
import zipfile
import shutil
from pathlib import Path


# Test data bundle configuration
BUNDLE_VERSION = "v0.6.0"
BUNDLE_NAME = f"example_data_{BUNDLE_VERSION}"
BUNDLE_URL = f"https://github.com/DarriEy/SYMFLUENCE/releases/download/examples-data-{BUNDLE_VERSION}/{BUNDLE_NAME}.zip"

# Fallback to v0.5.5 if v0.6.0 not available yet
FALLBACK_VERSION = "v0.5.5"
FALLBACK_NAME = f"example_data_{FALLBACK_VERSION}"
FALLBACK_URL = f"https://github.com/DarriEy/SYMFLUENCE/releases/download/examples-data-{FALLBACK_VERSION}/{FALLBACK_NAME}.zip"


@pytest.fixture(scope="session")
def example_data_bundle(symfluence_data_root):
    """
    Download and extract example data bundle from GitHub release.

    This is a session-scoped fixture that downloads the test data once per test session
    and reuses it across all tests.

    Args:
        symfluence_data_root: Path to SYMFLUENCE_data directory

    Returns:
        Path: Path to the data root containing all domains

    Yields:
        Path: Data root path during test session
    """
    data_root = symfluence_data_root
    marker_file = data_root / f".{BUNDLE_NAME}_installed"

    # Check if already downloaded
    if not marker_file.exists():
        print(f"\nDownloading example data {BUNDLE_VERSION}...")
        zip_path = data_root / f"{BUNDLE_NAME}.zip"

        try:
            # Try primary URL
            response = requests.get(BUNDLE_URL, stream=True, timeout=600)
            response.raise_for_status()
        except (requests.RequestException, requests.HTTPError):
            # Fall back to v0.5.5
            print(f"Warning: {BUNDLE_VERSION} not available, falling back to {FALLBACK_VERSION}")
            response = requests.get(FALLBACK_URL, stream=True, timeout=600)
            response.raise_for_status()
            zip_path = data_root / f"{FALLBACK_NAME}.zip"

        # Download
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Extracting example data...")
        # Extract to temp location
        extract_dir = data_root / "temp_extract"
        extract_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Move domains to data root
        extracted_bundle = extract_dir / BUNDLE_NAME
        if not extracted_bundle.exists():
            # Try fallback name
            extracted_bundle = extract_dir / FALLBACK_NAME

        if extracted_bundle.exists():
            # Move each domain to data root
            for domain_dir in extracted_bundle.iterdir():
                if domain_dir.is_dir() and domain_dir.name.startswith("domain_"):
                    dest = data_root / domain_dir.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    domain_dir.rename(dest)
                    print(f"Installed: {domain_dir.name}")

        # Cleanup
        zip_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

        # Create marker file
        marker_file.touch()
        print(f"Test data bundle ready at {data_root}")

    return data_root


@pytest.fixture(scope="session")
def ellioaar_domain(example_data_bundle):
    """
    Elliðaár Iceland test domain (CARRA forcing).

    Small 2km x 2km basin in Reykjavik, Iceland.
    Uses CARRA (Arctic Regional Reanalysis) forcing data.

    Returns:
        Path: Path to domain_ellioaar_iceland directory
    """
    domain_path = example_data_bundle / "domain_ellioaar_iceland"
    if not domain_path.exists():
        pytest.skip(f"Domain not found: {domain_path}")
    return domain_path


@pytest.fixture(scope="session")
def fyris_domain(example_data_bundle):
    """
    Fyrisån Uppsala test domain (CERRA forcing).

    Small 2km x 2km basin in Uppsala, Sweden.
    Uses CERRA (European Regional Reanalysis) forcing data.

    Returns:
        Path: Path to domain_fyris_uppsala directory
    """
    domain_path = example_data_bundle / "domain_fyris_uppsala"
    if not domain_path.exists():
        pytest.skip(f"Domain not found: {domain_path}")
    return domain_path


@pytest.fixture(scope="session")
def bow_domain(example_data_bundle):
    """
    Bow at Banff test domain (ERA5 forcing).

    Small basin in Banff, Alberta, Canada.
    Uses ERA5 (global reanalysis) forcing data.
    Includes streamflow observations.

    Returns:
        Path: Path to domain_bow_banff_minimal directory
    """
    # Try new naming convention first
    domain_path = example_data_bundle / "domain_bow_banff_minimal"
    if not domain_path.exists():
        # Fallback to old naming
        domain_path = example_data_bundle / "domain_Bow_at_Banff_lumped"
    if not domain_path.exists():
        pytest.skip(f"Bow domain not found in {example_data_bundle}")
    return domain_path


@pytest.fixture(scope="session")
def iceland_domain(example_data_bundle):
    """
    Iceland regional domain (ERA5 forcing).

    Regional domain covering part of Iceland.
    Uses ERA5 (global reanalysis) forcing data.

    Returns:
        Path: Path to domain_Iceland directory
    """
    domain_path = example_data_bundle / "domain_Iceland"
    if not domain_path.exists():
        pytest.skip(f"Domain not found: {domain_path}")
    return domain_path


@pytest.fixture(scope="session")
def paradise_domain(example_data_bundle):
    """
    Paradise SNOTEL point-scale domain (ERA5 forcing).

    Point-scale domain at Paradise, Mt. Rainier, Washington.
    Uses ERA5 (global reanalysis) forcing data.
    Includes SNOTEL observations.

    Returns:
        Path: Path to domain_paradise directory
    """
    domain_path = example_data_bundle / "domain_paradise"
    if not domain_path.exists():
        pytest.skip(f"Domain not found: {domain_path}")
    return domain_path
