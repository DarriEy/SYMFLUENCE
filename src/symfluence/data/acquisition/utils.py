"""
Data Acquisition Utility Functions.

Provides common utilities for data acquisition handlers:
- Robust HTTP session with automatic retry
- Streaming file downloads
- Atomic file operations
- Credential resolution
"""

import logging
import netrc
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# =============================================================================
# HTTP Session Utilities
# =============================================================================

def create_robust_session(
    max_retries: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: List[int] = None,
    allowed_methods: List[str] = None
) -> requests.Session:
    """
    Create a requests session with automatic retry logic for network failures.

    Uses HTTPAdapter with exponential backoff retry strategy.

    Args:
        max_retries: Maximum number of retry attempts (default: 5)
        backoff_factor: Factor for exponential backoff, e.g., 1.0 means 1s, 2s, 4s, 8s
        status_forcelist: HTTP status codes to retry on (default: [429, 500, 502, 503, 504])
        allowed_methods: HTTP methods to retry (default: ["HEAD", "GET", "OPTIONS"])

    Returns:
        Configured requests.Session object with retry adapters mounted

    Example:
        >>> session = create_robust_session(max_retries=3, backoff_factor=2.0)
        >>> response = session.get("https://api.example.com/data")
    """
    if status_forcelist is None:
        status_forcelist = [429, 500, 502, 503, 504]
    if allowed_methods is None:
        allowed_methods = ["HEAD", "GET", "OPTIONS"]

    session = requests.Session()

    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
        raise_on_status=False
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# =============================================================================
# File Download Utilities
# =============================================================================

def download_file_streaming(
    url: str,
    target_path: Path,
    session: requests.Session = None,
    chunk_size: int = 65536,
    timeout: int = 600,
    use_temp_file: bool = True,
    headers: Dict[str, str] = None,
    auth: Tuple[str, str] = None
) -> int:
    """
    Download a file using streaming with optional atomic write.

    Downloads in chunks to handle large files without memory issues.
    When use_temp_file is True, writes to a .part file first, then renames
    on success to avoid leaving partial files.

    Args:
        url: URL to download from
        target_path: Path where the file should be saved
        session: Optional requests.Session (creates one if not provided)
        chunk_size: Size of download chunks in bytes (default: 64KB)
        timeout: Request timeout in seconds (default: 600)
        use_temp_file: If True, write to .part file first (default: True)
        headers: Optional headers to include in request
        auth: Optional (username, password) tuple for basic auth

    Returns:
        Number of bytes downloaded

    Raises:
        requests.HTTPError: If the request fails
        IOError: If file writing fails
    """
    if session is None:
        session = create_robust_session()

    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Use temporary file for atomic write
    write_path = target_path.with_suffix(target_path.suffix + '.part') if use_temp_file else target_path

    try:
        with session.get(url, stream=True, timeout=timeout, headers=headers, auth=auth) as response:
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(write_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # Skip keep-alive chunks
                        f.write(chunk)
                        downloaded += len(chunk)

            # Verify complete download if size was provided
            if total_size > 0 and downloaded < total_size:
                raise IOError(f"Incomplete download: {downloaded}/{total_size} bytes")

        # Atomic rename on success
        if use_temp_file:
            write_path.replace(target_path)

        return downloaded

    except (requests.RequestException, OSError, IOError):
        # Clean up partial file on error
        if use_temp_file and write_path.exists():
            try:
                write_path.unlink()
            except OSError:
                pass
        raise


@contextmanager
def atomic_write(target_path: Path) -> Generator[Path, None, None]:
    """
    Context manager for atomic file writes using a temporary .part file.

    Writes to a .part file first, then renames to the target path on success.
    Cleans up the .part file on failure.

    Args:
        target_path: Final path where the file should be saved

    Yields:
        Path to the temporary .part file for writing

    Example:
        >>> with atomic_write(Path("output.nc")) as temp_path:
        ...     dataset.to_netcdf(temp_path)
        # File is now at output.nc
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = target_path.with_suffix(target_path.suffix + '.part')

    try:
        yield temp_path
        # Success - rename to target
        temp_path.replace(target_path)
    except BaseException:
        # Cleanup on failure (use BaseException to catch KeyboardInterrupt too)
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


# =============================================================================
# Credential Management
# =============================================================================

def resolve_credentials(
    hostname: str,
    env_prefix: str = None,
    config: Dict[str, Any] = None,
    alt_hostnames: List[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve credentials from multiple sources.

    Checks in order of preference:
    1. ~/.netrc file (most secure)
    2. Environment variables ({prefix}_USERNAME, {prefix}_PASSWORD)
    3. Config dictionary ({prefix}_USERNAME, {prefix}_PASSWORD keys)

    Args:
        hostname: Primary hostname to look up in .netrc
        env_prefix: Prefix for environment variables (e.g., "EARTHDATA")
        config: Optional config dictionary to check
        alt_hostnames: Alternative hostnames to try in .netrc

    Returns:
        Tuple of (username, password), or (None, None) if not found

    Example:
        >>> username, password = resolve_credentials(
        ...     hostname='urs.earthdata.nasa.gov',
        ...     env_prefix='EARTHDATA',
        ...     config=my_config
        ... )
    """
    # 1. Try .netrc first (preferred - more secure)
    try:
        netrc_path = Path.home() / '.netrc'
        if netrc_path.exists():
            nrc = netrc.netrc(str(netrc_path))

            # Try all hostnames
            hostnames_to_try = [hostname]
            if alt_hostnames:
                hostnames_to_try.extend(alt_hostnames)

            for host in hostnames_to_try:
                auth = nrc.authenticators(host)
                if auth:
                    logger.debug(f"Using credentials from ~/.netrc ({host})")
                    return auth[0], auth[2]
    except Exception as e:  # noqa: BLE001 — .netrc fallback is non-critical
        logger.debug(f"Could not read .netrc: {e}")

    # 2. Try environment variables
    if env_prefix:
        username = os.environ.get(f'{env_prefix}_USERNAME')
        password = os.environ.get(f'{env_prefix}_PASSWORD')
        if username and password:
            logger.debug(f"Using credentials from environment variables ({env_prefix}_*)")
            return username, password

    # 3. Try config dictionary
    if config and env_prefix:
        username = config.get(f'{env_prefix}_USERNAME')
        password = config.get(f'{env_prefix}_PASSWORD')
        if username and password:
            logger.debug(f"Using credentials from config ({env_prefix}_*)")
            return username, password

    return None, None


def get_earthdata_credentials(
    config: Dict[str, Any] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get NASA Earthdata credentials.

    Convenience wrapper around resolve_credentials for Earthdata services.

    Args:
        config: Optional config dictionary

    Returns:
        Tuple of (username, password), or (None, None) if not found
    """
    return resolve_credentials(
        hostname='urs.earthdata.nasa.gov',
        env_prefix='EARTHDATA',
        config=config,
        alt_hostnames=['earthdata.nasa.gov', 'appeears.earthdatacloud.nasa.gov']
    )


def resolve_earthdata_token(
    config: Dict[str, Any] = None
) -> Optional[str]:
    """
    Resolve a NASA Earthdata Bearer token from environment or config.

    NASA Earthdata supports token-based authentication as an alternative to
    username/password credentials.  Tokens can be generated at
    https://urs.earthdata.nasa.gov/users/<user>/user_tokens

    Checks in order:
    1. EARTHDATA_TOKEN environment variable
    2. Config dictionary (EARTHDATA_TOKEN key)

    Args:
        config: Optional config dictionary to check

    Returns:
        Token string, or None if not found
    """
    # 1. Environment variable
    token = os.environ.get('EARTHDATA_TOKEN')
    if token:
        logger.debug("Using Earthdata token from EARTHDATA_TOKEN env var")
        return token

    # 2. Config dictionary
    if config:
        token = config.get('EARTHDATA_TOKEN')
        if token:
            logger.debug("Using Earthdata token from config")
            return token

    return None


def get_cds_credentials(
    config: Dict[str, Any] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Get Copernicus Climate Data Store (CDS) credentials.

    Args:
        config: Optional config dictionary

    Returns:
        Tuple of (url, key), or (None, None) if not found
    """
    # CDS uses ~/.cdsapirc format, but we can also support env vars
    cds_url = os.environ.get('CDSAPI_URL')
    cds_key = os.environ.get('CDSAPI_KEY')

    if cds_url and cds_key:
        return cds_url, cds_key

    if config:
        cds_url = config.get('CDSAPI_URL')
        cds_key = config.get('CDSAPI_KEY')
        if cds_url and cds_key:
            return cds_url, cds_key

    return None, None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'create_robust_session',
    'download_file_streaming',
    'atomic_write',
    'resolve_credentials',
    'get_earthdata_credentials',
    'resolve_earthdata_token',
    'get_cds_credentials',
]
