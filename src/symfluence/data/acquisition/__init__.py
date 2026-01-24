"""
Data acquisition module.

Provides cloud-based data acquisition handlers for meteorological forcing data,
remote sensing products, and observational datasets. Handlers are registered
via the AcquisitionRegistry and support various data sources including:

- ERA5/ERA5-Land (CDS API)
- AORC, HRRR, CONUS404 (AWS S3)
- RDRS (ECCC HPFX)
- MODIS products (AppEEARS API)
- GRACE/GRACE-FO (PO.DAAC)
- NEX-GDDP-CMIP6 (NASA THREDDS)
"""

import logging as _logging
from typing import Any

_logger = _logging.getLogger(__name__)

# Fail-safe imports
try:
    from .registry import AcquisitionRegistry
except ImportError as _e:
    AcquisitionRegistry: Any = None  # type: ignore
    _logger.warning("Failed to import AcquisitionRegistry: %s", _e)

try:
    from . import handlers
except ImportError as _e:
    handlers: Any = None  # type: ignore
    _logger.warning("Failed to import acquisition handlers: %s", _e)

__all__ = ["AcquisitionRegistry"]
