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

import sys as _sys

# Fail-safe imports
try:
    from .registry import AcquisitionRegistry
except ImportError as _e:
    AcquisitionRegistry = None
    print(f"WARNING: Failed to import AcquisitionRegistry: {_e}", file=_sys.stderr)

try:
    from . import handlers
except ImportError as _e:
    handlers = None
    print(f"WARNING: Failed to import acquisition handlers: {_e}", file=_sys.stderr)

__all__ = ["AcquisitionRegistry"]
