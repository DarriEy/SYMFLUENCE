# Refactored: Logic moved to separate files.
# This file is kept for backward compatibility.

from symfluence.utils.data.preprocessing.forcing_resampler import ForcingResampler
from symfluence.utils.data.preprocessing.geospatial_statistics import GeospatialStatistics

# Backwards-compatible class aliases (expected by DataManager import)
forcingResampler = ForcingResampler
geospatialStatistics = GeospatialStatistics

# Also export the new PascalCase names if anyone wants to use them directly
__all__ = ['ForcingResampler', 'GeospatialStatistics', 'forcingResampler', 'geospatialStatistics']