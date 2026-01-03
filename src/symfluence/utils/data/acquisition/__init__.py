from .registry import AcquisitionRegistry
from . import handlers

# Trigger registration
try:
    from .handlers import era5, era5_cds, aorc, nex_gddp, em_earth, hrrr, conus404, cds_datasets, geospatial, grace, modis, observation_acquirers
except ImportError:
    pass

__all__ = ["AcquisitionRegistry"]
