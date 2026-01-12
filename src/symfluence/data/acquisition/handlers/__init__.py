"""
Acquisition handlers for various datasets.
"""

import sys as _sys
import importlib as _importlib

# Import all handlers to trigger registration
# Use try/except for each to handle optional dependencies
# Log errors to stderr for CI debugging

_imported = []
_failed = []

_handler_modules = [
    'era5',
    'era5_cds',
    'aorc',
    'nex_gddp',
    'em_earth',
    'hrrr',
    'conus404',
    'cds_datasets',
    'geospatial',
    'rdrs',
    'observation_acquirers',
    'grace',
    'glacier',
    'modis_sca',
    'modis_et',
    'fluxnet',
]

for _module_name in _handler_modules:
    try:
        _module = _importlib.import_module(f'.{_module_name}', __name__)
        globals()[_module_name] = _module
        _imported.append(_module_name)
    except Exception as _e:
        _failed.append((_module_name, str(_e)))
        print(f"WARNING: Failed to import acquisition handler '{_module_name}': {_e}",
              file=_sys.stderr)

# Clean up
del _handler_modules, _module_name
try:
    del _module, _e
except NameError:
    pass

__all__ = _imported
