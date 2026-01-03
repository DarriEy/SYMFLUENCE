"""
Acquisition handlers for various datasets.
"""

# Import all handlers to trigger registration
# Use try/except for each to handle optional dependencies

_imported = []

try:
    from . import era5
    _imported.append('era5')
except ImportError:
    pass

try:
    from . import era5_cds
    _imported.append('era5_cds')
except ImportError:
    pass

try:
    from . import aorc
    _imported.append('aorc')
except ImportError:
    pass

try:
    from . import nex_gddp
    _imported.append('nex_gddp')
except ImportError:
    pass

try:
    from . import em_earth
    _imported.append('em_earth')
except ImportError:
    pass

try:
    from . import hrrr
    _imported.append('hrrr')
except ImportError:
    pass

try:
    from . import conus404
    _imported.append('conus404')
except ImportError:
    pass

try:
    from . import cds_datasets
    _imported.append('cds_datasets')
except ImportError:
    pass

try:
    from . import geospatial
    _imported.append('geospatial')
except ImportError:
    pass

__all__ = _imported
