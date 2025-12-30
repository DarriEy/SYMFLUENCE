from .registry import AcquisitionRegistry
from . import handlers

# Trigger registration
try:
    from .handlers import era5, aorc, nex_gddp, em_earth, hrrr, conus404, cds_datasets, geospatial
except ImportError:
    pass

__all__ = ["AcquisitionRegistry"]
