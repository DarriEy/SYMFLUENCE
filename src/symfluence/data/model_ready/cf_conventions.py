# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CF-1.8 convention helpers for the model-ready data store.

Provides a unified CF standard-name mapping for all variables used across
SYMFLUENCE (forcings, observations, attributes) and a builder for the
global attributes that every model-ready NetCDF file should carry.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import symfluence

# ---------------------------------------------------------------------------
# CF standard-name mapping
# ---------------------------------------------------------------------------
# Each entry maps an *internal* SYMFLUENCE variable name to a dict with
# ``standard_name``, ``units``, and ``long_name`` following CF-1.8.
# This extends the existing STANDARD_VARIABLE_ATTRIBUTES from
# data.preprocessing.dataset_handlers.base_dataset and
# SUMMA_VARIABLE_ATTRS from data.acquisition.handlers.era5_processing.

CF_STANDARD_NAMES: Dict[str, Dict[str, str]] = {
    # --- Forcing variables ---
    'surface_air_pressure':    {'standard_name': 'air_pressure',
                   'units': 'Pa',
                   'long_name': 'air pressure at measurement height'},
    'air_temperature':    {'standard_name': 'air_temperature',
                   'units': 'K',
                   'long_name': 'air temperature at measurement height'},
    'precipitation_flux':    {'standard_name': 'precipitation_flux',
                   'units': 'kg m-2 s-1',
                   'long_name': 'precipitation rate'},
    'wind_speed':    {'standard_name': 'wind_speed',
                   'units': 'm s-1',
                   'long_name': 'wind speed at measurement height'},
    'specific_humidity':    {'standard_name': 'specific_humidity',
                   'units': 'kg kg-1',
                   'long_name': 'specific humidity'},
    'relative_humidity':     {'standard_name': 'relative_humidity',
                   'units': '%',
                   'long_name': 'relative humidity'},
    'surface_downwelling_shortwave_flux':   {'standard_name': 'surface_downwelling_shortwave_flux_in_air',
                   'units': 'W m-2',
                   'long_name': 'downward shortwave radiation at the surface'},
    'surface_downwelling_longwave_flux':   {'standard_name': 'surface_downwelling_longwave_flux_in_air',
                   'units': 'W m-2',
                   'long_name': 'downward longwave radiation at the surface'},

    # --- Extended CF aliases ---
    'surface_downwelling_shortwave_flux_in_air':     {'standard_name': 'surface_downwelling_shortwave_flux_in_air',
                                                      'units': 'W m-2',
                                                      'long_name': 'downward shortwave radiation'},
    'surface_downwelling_longwave_flux_in_air':      {'standard_name': 'surface_downwelling_longwave_flux_in_air',
                                                      'units': 'W m-2',
                                                      'long_name': 'downward longwave radiation'},

    # --- Observation variables ---
    'discharge_cms':   {'standard_name': 'water_volume_transport_in_river_channel',
                        'units': 'm3 s-1',
                        'long_name': 'river discharge'},
    'swe':             {'standard_name': 'lwe_thickness_of_surface_snow_amount',
                        'units': 'kg m-2',
                        'long_name': 'snow water equivalent'},
    'sca':             {'standard_name': 'surface_snow_area_fraction',
                        'units': '1',
                        'long_name': 'snow covered area fraction'},
    'et':              {'standard_name': 'water_evapotranspiration_flux',
                        'units': 'kg m-2 s-1',
                        'long_name': 'evapotranspiration'},
    'tws_anomaly':     {'standard_name': 'liquid_water_content_of_surface_snow',
                        'units': 'mm',
                        'long_name': 'terrestrial water storage anomaly'},
    'soil_moisture':   {'standard_name': 'volume_fraction_of_condensed_water_in_soil',
                        'units': 'm3 m-3',
                        'long_name': 'volumetric soil moisture'},

    # --- Attribute variables ---
    'elev_mean':       {'standard_name': 'surface_altitude',
                        'units': 'm',
                        'long_name': 'mean elevation of hydrological response unit'},
    'hru_area':        {'standard_name': 'area',
                        'units': 'm2',
                        'long_name': 'hydrological response unit area'},
    'latitude':        {'standard_name': 'latitude',
                        'units': 'degrees_north',
                        'long_name': 'centroid latitude'},
    'longitude':       {'standard_name': 'longitude',
                        'units': 'degrees_east',
                        'long_name': 'centroid longitude'},
}


# ---------------------------------------------------------------------------
# Canonical forcing vocabulary (single source of truth)
# ---------------------------------------------------------------------------
# The model-ready store exposes forcing under these canonical names (the SUMMA
# vocabulary, which the CARRA/EASYMORE store already uses). Every accepted source
# alias maps to the canonical name + its CF standard key + canonical units, so
# the CARRA/ERA5-specific knowledge lives HERE and nowhere else. Model adapters
# must resolve through this map (see open_canonical_forcing / resolve_forcing_var)
# rather than carrying their own ``_find_variable`` candidate lists.
CANONICAL_FORCING: Dict[str, Dict[str, object]] = {
    'pptrate':  {'cf': 'precipitation_flux', 'units': 'kg m-2 s-1', 'kind': 'rate',
                 'aliases': ['pptrate', 'precipitation_flux', 'precipitation',
                             'pr', 'precip', 'tp', 'PREC', 'total_precipitation']},
    'airtemp':  {'cf': 'air_temperature', 'units': 'K', 'kind': 'state',
                 'aliases': ['airtemp', 'air_temperature', 'temperature',
                             'tas', 'temp', 't2m', 'AIR_TEMP', '2m_temperature']},
    'SWRadAtm': {'cf': 'surface_downwelling_shortwave_flux', 'units': 'W m-2', 'kind': 'state',
                 'aliases': ['SWRadAtm', 'surface_downwelling_shortwave_flux',
                             'shortwave', 'rsds', 'swdown', 'ssrd']},
    'LWRadAtm': {'cf': 'surface_downwelling_longwave_flux', 'units': 'W m-2', 'kind': 'state',
                 'aliases': ['LWRadAtm', 'surface_downwelling_longwave_flux',
                             'longwave', 'rlds', 'lwdown', 'strd']},
    'windspd':  {'cf': 'wind_speed', 'units': 'm s-1', 'kind': 'state',
                 'aliases': ['windspd', 'wind_speed', 'sfcWind', 'wind', 'ws', 'si10']},
    'spechum':  {'cf': 'specific_humidity', 'units': 'kg kg-1', 'kind': 'state',
                 'aliases': ['spechum', 'specific_humidity', 'huss', 'q', 'qair']},
    'airpres':  {'cf': 'surface_air_pressure', 'units': 'Pa', 'kind': 'state',
                 'aliases': ['airpres', 'surface_air_pressure', 'surface_pressure',
                             'ps', 'sp', 'pres']},
}

# Reverse map: any accepted alias -> the CF standard key (for metadata enrichment).
CANONICAL_FORCING_ALIASES: Dict[str, str] = {
    str(alias): str(spec['cf'])
    for spec in CANONICAL_FORCING.values()
    for alias in spec['aliases']  # type: ignore[union-attr]
}


def resolve_forcing_var(ds, canonical_name: str) -> Optional[str]:
    """Return the variable name in ``ds`` that supplies ``canonical_name``,
    trying the canonical name first then its registered aliases."""
    spec = CANONICAL_FORCING.get(canonical_name)
    if spec is None:
        return canonical_name if canonical_name in ds else None
    for alias in [canonical_name, *spec['aliases']]:  # type: ignore[misc]
        if alias in ds:
            return alias
    return None


# ---------------------------------------------------------------------------
# Global attribute builder
# ---------------------------------------------------------------------------

def build_global_attrs(
    domain_name: str,
    title: str,
    history: Optional[str] = None,
) -> Dict[str, str]:
    """Build CF-1.8 global attributes for a model-ready NetCDF file.

    Args:
        domain_name: Name of the hydrological domain.
        title: Human-readable title for the dataset.
        history: Optional processing history string.

    Returns:
        Dict of global attributes ready to write with netCDF4.
    """
    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    version = getattr(symfluence, '__version__', 'dev')

    attrs: Dict[str, str] = {
        'Conventions': 'CF-1.8',
        'title': title,
        'institution': 'SYMFLUENCE',
        'source_software': f'SYMFLUENCE v{version}',
        'creation_date': now,
        'domain_name': domain_name,
    }
    if history:
        attrs['history'] = history
    return attrs
