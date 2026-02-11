"""
SYMFLUENCE: SYnergistic Modelling Framework for Linking and Unifying
Earth-system Nexii for Computational Exploration.

A computational environmental modeling platform that streamlines the
hydrological modeling workflow from domain setup to evaluation. Provides
an integrated framework for multi-model comparison, parameter optimization,
and automated workflow management.

Main entry points:
    SYMFLUENCE: Main workflow orchestrator class
    SymfluenceConfig: Configuration management for workflows

Example:
    >>> from symfluence import SYMFLUENCE, SymfluenceConfig
    >>> config = SymfluenceConfig.from_file('config.yaml')
    >>> workflow = SYMFLUENCE(config)
    >>> workflow.run()

For CLI usage:
    $ symfluence workflow run --config config.yaml
    $ symfluence --help
"""
# src/symfluence/__init__.py

# ============================================================================
# CRITICAL: HDF5/netCDF4 thread safety fix
# Must be set BEFORE any HDF5/netCDF4/xarray imports occur.
# The netCDF4/HDF5 libraries are not thread-safe by default, and tqdm's
# background monitor thread can cause segmentation faults when running
# concurrently with netCDF file operations (e.g., in easymore remapping).
# ============================================================================
from symfluence.core.hdf5_safety import configure_hdf5_safety
configure_hdf5_safety(disable_tqdm_monitor=True)

import logging
import os
import warnings

try:
    from .symfluence_version import __version__
except ImportError:
    try:
        from importlib.metadata import version, PackageNotFoundError

        __version__ = version("symfluence")
    except (ImportError, PackageNotFoundError):
        __version__ = "0.0.0"

# Expose core components for a cleaner API
from .core import SYMFLUENCE
from .core.config.models import SymfluenceConfig
from .core.exceptions import (
    SYMFLUENCEError,
    ConfigurationError,
    ModelExecutionError,
    DataAcquisitionError
)

__all__ = [
    "SYMFLUENCE",
    "SymfluenceConfig",
    "SYMFLUENCEError",
    "ConfigurationError",
    "ModelExecutionError",
    "DataAcquisitionError",
    "__version__"
]

# Suppress overly verbose external logging/warnings
rpy2_logger = logging.getLogger("r2.rinterface_lib.embedded")
rpy2_logger.setLevel(logging.WARNING)
rpy2_logger.addHandler(logging.NullHandler())
rpy2_logger.propagate = False

warnings.filterwarnings(
    "ignore",
    message="(?s).*Conversion of an array with ndim > 0 to a scalar is deprecated.*",
    category=DeprecationWarning,
)

os.environ.setdefault(
    "PYTHONWARNINGS",
    r"ignore:Column names longer than 10 characters will be truncated when saved to ESRI Shapefile\.:UserWarning",
)

warnings.filterwarnings(
    "ignore",
    message=r"Column names longer than 10 characters will be truncated when saved to ESRI Shapefile\.",
    category=UserWarning,
)

try:
    import pyproj

    _orig_transform = pyproj.transformer.Transformer.transform

    def _warnless_transform(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="(?s).*Conversion of an array with ndim > 0 to a scalar is deprecated.*",
                category=DeprecationWarning,
            )
            return _orig_transform(self, *args, **kwargs)

    pyproj.transformer.Transformer.transform = _warnless_transform
except ImportError:
    pass
