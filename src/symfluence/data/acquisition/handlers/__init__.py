"""
Acquisition handlers for various datasets.
"""

import logging as _logging
import importlib as _importlib

_logger = _logging.getLogger(__name__)

# Import all handlers to trigger registration
# Use try/except for each to handle optional dependencies

_imported = []
_failed = []

_handler_modules = [
    'era5',
    'era5_cds',
    'era5_land',
    'aorc',
    'nex_gddp',
    'em_earth',
    'hrrr',
    'conus404',
    'cds_datasets',
    'daymet',
    'dem',
    'soilgrids',
    'landcover',
    'rdrs',
    'smap',
    'ismn',
    'esa_cci_sm',
    'fluxcom_et',
    'grace',
    'grdc',
    'glacier',
    'modis_sca',
    'modis_et',
    'modis_lai',
    'modis_lst',
    'mswep',
    'openet',
    'fluxnet',
    'gpm',
    'chirps',
    'sentinel1_sm',
    'snodas',
    'jrc_water',
    'ssebop',
    'viirs_snow',
    'ims_snow',
    'canopy_height',
]

for _module_name in _handler_modules:
    try:
        _module = _importlib.import_module(f'.{_module_name}', __name__)
        globals()[_module_name] = _module
        _imported.append(_module_name)
    except Exception as _e:
        _failed.append((_module_name, str(_e)))
        _logger.warning("Failed to import acquisition handler '%s': %s", _module_name, _e)

# Clean up
del _handler_modules, _module_name
try:
    del _module, _e
except NameError:
    pass

__all__ = _imported
