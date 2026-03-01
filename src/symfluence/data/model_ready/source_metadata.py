# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Source metadata for model-ready data store provenance tracking.

Provides a standardized way to attach provenance information to NetCDF
variables and datasets, tracking data source, processing history,
and acquisition details.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional

# Prefix for all source metadata attributes in NetCDF files
_ATTR_PREFIX = 'source_'


@dataclass
class SourceMetadata:
    """Provenance metadata for a data variable or dataset.

    Attributes:
        source: Name of the data source (e.g. "ERA5", "USGS NWIS")
        processing: Description of processing applied
        acquisition_date: ISO 8601 date when data was acquired
        source_doi: DOI of the source dataset
        original_units: Units before any conversion
        version: Dataset version string
        url: Source URL for the data
    """
    source: str
    processing: str = ''
    acquisition_date: str = ''
    source_doi: Optional[str] = None
    original_units: Optional[str] = None
    version: Optional[str] = None
    url: Optional[str] = None

    def to_netcdf_attrs(self) -> Dict[str, str]:
        """Convert to a dict suitable for NetCDF variable/global attributes.

        Keys are prefixed with ``source_`` to avoid collisions with CF
        standard attributes. ``None`` values are omitted.
        """
        attrs: Dict[str, str] = {}
        for k, v in asdict(self).items():
            if v is not None and v != '':
                attrs[f'{_ATTR_PREFIX}{k}'] = str(v)
        return attrs

    @classmethod
    def from_netcdf_attrs(cls, attrs: Dict[str, str]) -> 'SourceMetadata':
        """Reconstruct a ``SourceMetadata`` from NetCDF attributes.

        Strips the ``source_`` prefix and ignores attributes that do not
        belong to this dataclass.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs: Dict[str, str] = {}
        for k, v in attrs.items():
            if k.startswith(_ATTR_PREFIX):
                field_name = k[len(_ATTR_PREFIX):]
                if field_name in known_fields:
                    kwargs[field_name] = v
        return cls(**kwargs) if 'source' in kwargs else cls(source='unknown')
