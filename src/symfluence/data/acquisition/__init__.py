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
from .registry import AcquisitionRegistry
from . import handlers

# The above 'from . import handlers' is sufficient to trigger registration
# of all handlers within the handlers directory.

__all__ = ["AcquisitionRegistry"]
