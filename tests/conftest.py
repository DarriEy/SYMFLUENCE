"""
Root conftest.py - Session-scoped fixtures shared across all tests.

This file sets up the Python path and provides core fixtures for test discovery.
Additional fixtures are loaded via pytest_plugins from tests/fixtures/.
"""

# ============================================================================
# CRITICAL: HDF5/netCDF4 thread safety fix - MUST BE FIRST
# Must be set BEFORE any imports that could load HDF5 (including GDAL!).
# The HDF5 library reads this environment variable at initialization time,
# so if any import loads libhdf5 before this is set, it's too late.
# ============================================================================
import os
import sys

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'
os.environ['NETCDF_DISABLE_LOCKING'] = '1'
# Limit threading in numerical libraries to prevent HDF5 conflicts
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

# ============================================================================
# CRITICAL: Windows torch DLL conflict fix - must run BEFORE numpy imports.
# Numpy loads conda's Library\bin\uv.dll into the process.  Torch's shm.dll
# depends on a different (bundled) uv.dll.  If conda's version is already
# loaded, shm.dll fails with STATUS_ENTRYPOINT_NOT_FOUND (WinError 127).
# Importing torch first ensures torch's uv.dll is loaded and reused.
# Fixture modules (geospatial_fixtures) import numpy at module level, so
# this must come before pytest_plugins loads them.
# ============================================================================
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import torch  # noqa: F401
    except (ImportError, OSError):
        pass

import tempfile
from pathlib import Path

# ============================================================================
# GDAL exception mode configuration
# Must be set BEFORE any GDAL operations to avoid FutureWarning about
# exception mode in GDAL 4.0
# ============================================================================
try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    pass  # GDAL not installed

# Disable tqdm monitor thread to prevent segfaults with netCDF4/HDF5
# This must be done before tqdm is imported anywhere
import tqdm

tqdm.tqdm.monitor_interval = 0
# Also disable the existing monitor if already started
if tqdm.tqdm.monitor is not None:
    try:
        tqdm.tqdm.monitor.exit()
    except (AttributeError, RuntimeError):
        pass  # Monitor may already be stopped or not properly initialized
    tqdm.tqdm.monitor = None

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "symfluence_matplotlib"),
)

@pytest.fixture(scope="module", autouse=True)
def cleanup_matplotlib():
    """
    Cleanup matplotlib figures after each test module to prevent memory leaks.

    This fixture runs automatically after every test *module* (not every test)
    and closes all open matplotlib figures.  Module scope reduces overhead
    while still preventing unbounded figure accumulation across the session.
    A final cleanup also runs in pytest_sessionfinish.
    """
    yield
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        from matplotlib import _pylab_helpers
        _pylab_helpers.Gcf.destroy_all()
    except (ImportError, AttributeError):
        pass  # matplotlib not installed or API changed

# Add directories to path BEFORE importing local modules
SYMFLUENCE_CODE_DIR = Path(__file__).parent.parent.resolve()
if str(SYMFLUENCE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(SYMFLUENCE_CODE_DIR))
TESTS_DIR = Path(__file__).parent.resolve()
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# Import test utilities with robust fallback
import importlib.util


def _load_test_helpers():
    """Try to load test_helpers using various methods."""
    # Try standard import first
    try:
        from test_helpers import helpers
        from test_helpers.helpers import has_cds_credentials, load_config_template, write_config
        return load_config_template, write_config, has_cds_credentials, helpers, True
    except (ImportError, ModuleNotFoundError):
        pass

    # Try direct file load as fallback
    helpers_path = TESTS_DIR / "test_helpers" / "helpers.py"
    if helpers_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("helpers", helpers_path)
            helpers_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helpers_module)
            return (
                helpers_module.load_config_template,
                helpers_module.write_config,
                helpers_module.has_cds_credentials,
                helpers_module,
                True,
            )
        except Exception:  # noqa: BLE001
            pass

    # Fallback stubs
    def load_config_template(code_dir=None):
        raise NotImplementedError("test_helpers not available")

    def write_config(config, path):
        raise NotImplementedError("test_helpers not available")

    def has_cds_credentials():
        return False

    return load_config_template, write_config, has_cds_credentials, None, False


load_config_template, write_config, has_cds_credentials, helpers, _test_helpers_available = _load_test_helpers()

# Try to load geospatial module
try:
    from test_helpers import geospatial
except (ImportError, ModuleNotFoundError):
    geospatial = None

# Load additional fixtures from fixture modules
pytest_plugins = [
    "fixtures.data_fixtures",
    "fixtures.domain_fixtures",
    "fixtures.model_fixtures",
    "fixtures.real_data_fixtures",
    "fixtures.geospatial_fixtures",
]

def pytest_addoption(parser):
    parser.addoption(
        "--run-full",
        action="store_true",
        default=False,
        help="Run full test matrix (includes tests marked 'full')",
    )
    parser.addoption(
        "--run-cloud",
        action="store_true",
        default=False,
        help="Run tests requiring cloud access (credentials + network)",
    )
    parser.addoption(
        "--run-full-examples",
        action="store_true",
        default=False,
        help="Run full example tests (multi-year workflows with optimization)",
    )
    parser.addoption(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Clear cached data before running tests (forces re-download)",
    )
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run only fast unit tests (skip slow, integration, requires_data, e2e)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--quick"):
        # --quick: skip everything that isn't a fast unit test
        skip_reason = pytest.mark.skip(reason="Skipped in --quick mode")
        skip_markers = {"slow", "integration", "e2e", "requires_data",
                        "requires_cloud", "requires_acquisition", "full",
                        "full_examples", "ci_full"}
        for item in items:
            if skip_markers & set(item.keywords):
                item.add_marker(skip_reason)

    if not config.getoption("--run-full"):
        skip_full = pytest.mark.skip(reason="Skipped full tests (use --run-full to enable)")
        for item in items:
            if "full" in item.keywords:
                item.add_marker(skip_full)

    if not config.getoption("--run-cloud"):
        skip_cloud = pytest.mark.skip(
            reason="Skipped cloud acquisition tests (use --run-cloud to enable)"
        )
        for item in items:
            if "requires_cloud" in item.keywords or "requires_acquisition" in item.keywords:
                item.add_marker(skip_cloud)

    if not config.getoption("--run-full-examples"):
        skip_full_examples = pytest.mark.skip(
            reason="Skipped full example tests (use --run-full-examples to enable)"
        )
        for item in items:
            if "full_examples" in item.keywords:
                item.add_marker(skip_full_examples)


@pytest.fixture(scope="session")
def symfluence_code_dir():
    """Path to SYMFLUENCE code directory."""
    return SYMFLUENCE_CODE_DIR


@pytest.fixture(scope="session")
def symfluence_data_root(symfluence_code_dir):
    """Path to SYMFLUENCE_data directory (shared test data)."""
    import os
    import tempfile
    # Respect SYMFLUENCE_DATA environment variable if set (for CI)
    env_data_dir = os.environ.get("SYMFLUENCE_DATA")
    if env_data_dir:
        data_root = Path(env_data_dir)
    else:
        data_root = symfluence_code_dir.parent / "SYMFLUENCE_data"
    try:
        data_root.mkdir(parents=True, exist_ok=True)
        test_file = data_root / ".symfluence_write_test"
        test_file.touch()
        test_file.unlink()
        return data_root
    except (PermissionError, OSError):
        fallback_root = Path(tempfile.mkdtemp(prefix="symfluence_data_"))
        read_only_root = symfluence_code_dir.parent / "SYMFLUENCE_data"
        installs_src = read_only_root / "installs"
        installs_dst = fallback_root / "installs"
        if installs_src.exists() and not installs_dst.exists():
            try:
                installs_dst.symlink_to(installs_src)
            except OSError:
                # Windows without admin/Developer Mode — use junction or copy
                import shutil
                shutil.copytree(installs_src, installs_dst, dirs_exist_ok=True)
        return fallback_root


@pytest.fixture(scope="session")
def tests_dir():
    """Path to tests directory."""
    return TESTS_DIR


@pytest.fixture()
def config_template(symfluence_code_dir):
    """Load configuration template for tests."""
    return load_config_template(symfluence_code_dir)


@pytest.fixture(scope="session")
def clear_cache_flag(request):
    """
    Returns True if --clear-cache flag was passed.

    Use this fixture in tests to determine whether to clear cached data
    before running acquisition steps.
    """
    return request.config.getoption("--clear-cache")


@pytest.fixture(scope="session")
def forcing_cache_manager(symfluence_data_root):
    """
    Global forcing cache manager for all tests.

    Provides a session-scoped RawForcingCache instance that all tests can use
    to cache raw forcing data downloads. This prevents redundant API calls and
    significantly speeds up test execution.

    The cache is configured with conservative defaults:
    - Max size: 3 GB
    - TTL: 30 days
    - Checksum verification enabled

    Returns:
        RawForcingCache: Initialized cache manager instance
    """
    from symfluence.data.cache import RawForcingCache

    cache_root = symfluence_data_root / "cache" / "raw_forcing"
    cache = RawForcingCache(
        cache_root=cache_root,
        max_size_gb=3.0,
        ttl_days=30,
        enable_checksum=True
    )

    # Log cache statistics at start of session
    stats = cache.get_cache_stats()
    print("\n=== Raw Forcing Cache Statistics ===")
    print(f"Cache root: {stats['cache_root']}")
    print(f"Cache size: {stats['total_size_gb']:.2f}GB / {stats['max_size_gb']:.2f}GB")
    print(f"Cached files: {stats['file_count']}")
    print(f"Datasets: {stats['datasets']}")
    print("=" * 40)

    return cache


@pytest.fixture(scope="session", autouse=True)
def memory_monitor():
    """
    Monitor memory usage during test session.

    Logs memory statistics at the start and end of the test session
    to help identify memory leaks.
    """
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        print("\n=== Memory Usage at Start ===")
        print(f"Memory: {start_memory:.2f} MB")
        print("=" * 40)

        yield

        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        print("\n=== Memory Usage at End ===")
        print(f"Memory: {end_memory:.2f} MB")
        print(f"Delta: {end_memory - start_memory:+.2f} MB")
        print("=" * 40)
    except ImportError:
        # psutil not installed, skip monitoring
        yield


def pytest_sessionfinish(session, exitstatus):
    """
    Clean up rpy2 embedded R to prevent logging errors on shutdown.

    rpy2 tries to log "Embedded R ended." when shutting down, but by that point
    the logging streams may be closed, causing a ValueError. We suppress the
    rpy2 logger before pytest fully exits.
    """
    import logging

    # Suppress rpy2 logging to avoid "I/O operation on closed file" errors
    rpy2_logger = logging.getLogger('rpy2')
    rpy2_logger.setLevel(logging.CRITICAL + 1)  # Effectively disable all logging

    # Also suppress the embedded R interface logger
    rpy2_embedded_logger = logging.getLogger('rpy2.rinterface_lib.embedded')
    rpy2_embedded_logger.setLevel(logging.CRITICAL + 1)

    # Final cleanup of matplotlib and memory
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
        from matplotlib import _pylab_helpers
        _pylab_helpers.Gcf.destroy_all()
    except (ImportError, AttributeError):
        pass

    import gc
    gc.collect()
