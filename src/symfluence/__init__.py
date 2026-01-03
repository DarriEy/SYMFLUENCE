# src/symfluence/__init__.py
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

from .core import SYMFLUENCE

__all__ = ["SYMFLUENCE", "__version__"]

# Suppress overly verbose external logging/warnings
rpy2_logger = logging.getLogger("rpy2.rinterface_lib.embedded")
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
    "ignore:Column names longer than 10 characters will be truncated when saved to ESRI Shapefile\\.:UserWarning",
)

warnings.filterwarnings(
    "ignore",
    message="Column names longer than 10 characters will be truncated when saved to ESRI Shapefile\\.",
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
