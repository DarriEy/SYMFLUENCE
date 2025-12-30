# src/symfluence/__init__.py
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