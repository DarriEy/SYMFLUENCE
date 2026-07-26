# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Compatibility exports for core observation path conventions."""
from __future__ import annotations

from symfluence.core.modeling.observation_paths import (
    chirps_default_observation_path,
    et_observation_candidates,
    first_existing_path,
    fluxnet_et_default_observation_path,
    groundwater_default_observation_path,
    groundwater_observation_candidates,
    modis_et_default_observation_path,
    observation_output_candidates_by_family,
    smap_default_observation_path,
    snow_cover_default_observation_path,
    snow_observation_candidates,
    soil_moisture_observation_candidates,
    streamflow_observation_candidates,
    swe_default_observation_path,
    tws_default_observation_path,
    tws_observation_candidates,
)

__all__ = [
    "chirps_default_observation_path",
    "et_observation_candidates",
    "first_existing_path",
    "fluxnet_et_default_observation_path",
    "groundwater_default_observation_path",
    "groundwater_observation_candidates",
    "modis_et_default_observation_path",
    "observation_output_candidates_by_family",
    "smap_default_observation_path",
    "snow_cover_default_observation_path",
    "snow_observation_candidates",
    "soil_moisture_observation_candidates",
    "streamflow_observation_candidates",
    "swe_default_observation_path",
    "tws_default_observation_path",
    "tws_observation_candidates",
]
