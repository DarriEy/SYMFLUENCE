# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Attribute acquisition profiles for SYMFLUENCE.

Profiles define which geospatial datasets are acquired by the
acquire_attributes workflow step beyond the core three (DEM, soil class,
land cover).  The ``ATTRIBUTE_PROFILE`` config flag selects a profile;
individual ``DOWNLOAD_*`` flags can override datasets within it.

Available profiles:

core
    Default.  No extended datasets — only DEM, soil class, and land cover
    (handled by the existing ``acquire_attributes`` logic).

camels_spat
    All 11 geospatial datasets from the CAMELS-SPAT dataset (Knoben
    et al., 2025, HESS).  Adds WorldClim climate normals, SoilGrids
    soil properties at 6 depths, Pelletier soil/regolith depth, GLHYMPS
    hydrogeology, HydroLAKES open water, MODIS LAI, forest canopy
    height, and GLCLU 2019 land cover.

full
    Everything in ``camels_spat`` plus every additional geospatial
    attribute handler available in SYMFLUENCE: glaciers (RGI), GLWD
    wetlands, depth to bedrock, aridity index, MODIS NDVI, root-zone
    storage capacity, JRC Global Surface Water, karst aquifer map
    (WOKAM), and MERIT Hydro river network attributes.

References
----------
Knoben, W. J. M., et al. (2025). Catchment Attributes and MEteorology
for Large-Sample SPATially distributed analysis (CAMELS-SPAT). Hydrol.
Earth Syst. Sci., 29, 5791-5833.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ProfileDataset:
    """A single dataset within an attribute profile.

    Parameters
    ----------
    handler_name
        Key in ``AcquisitionRegistry`` (case-insensitive).
    description
        Human-readable label for log messages.
    output_subdir
        Subdirectory under ``{project_dir}/attributes/`` where the
        handler should write its output.
    config_override_key
        Optional flat config key (e.g. ``DOWNLOAD_WORLDCLIM``).  When
        set to ``False`` in the user's config, this dataset is skipped
        even if the profile includes it.
    fatal
        If ``True``, acquisition failure raises an exception.
        If ``False`` (default), failure logs a warning and continues.
    """

    handler_name: str
    description: str
    output_subdir: str
    config_override_key: Optional[str] = None
    fatal: bool = False


_CORE_DATASETS: List[ProfileDataset] = []

_CAMELS_SPAT_DATASETS: List[ProfileDataset] = [
    ProfileDataset(
        handler_name='SOILGRIDS_PROPERTIES',
        description='SoilGrids v2.0 soil properties (6 depths)',
        output_subdir='soilclass/soilgrids',
        config_override_key='DOWNLOAD_SOILGRIDS_PROPERTIES',
    ),
    ProfileDataset(
        handler_name='PELLETIER',
        description='Pelletier soil/regolith depth',
        output_subdir='soilclass/pelletier',
        config_override_key='DOWNLOAD_PELLETIER',
    ),
    ProfileDataset(
        handler_name='GLHYMPS',
        description='GLHYMPS v2 hydrogeology (permeability, porosity)',
        output_subdir='geology/glhymps',
        config_override_key='DOWNLOAD_GLHYMPS',
    ),
    ProfileDataset(
        handler_name='HYDROLAKES',
        description='HydroLAKES v1.0 open water',
        output_subdir='lakes',
        config_override_key='DOWNLOAD_HYDROLAKES',
    ),
    ProfileDataset(
        handler_name='MODIS_LAI',
        description='MODIS MCD15A2H LAI/FPAR (8-day, 500m)',
        output_subdir='vegetation/modis_lai',
        config_override_key='DOWNLOAD_MODIS_LAI',
    ),
    ProfileDataset(
        handler_name='GLAD_TREE_HEIGHT',
        description='GLAD/GLCLUC forest canopy height',
        output_subdir='vegetation/canopy_height/glad',
        config_override_key='DOWNLOAD_FOREST_HEIGHT',
    ),
    ProfileDataset(
        handler_name='WORLDCLIM',
        description='WorldClim 2.1 monthly climate normals (30 arcsec)',
        output_subdir='climate/worldclim',
        config_override_key='DOWNLOAD_WORLDCLIM',
    ),
    ProfileDataset(
        handler_name='GLCLU_2019',
        description='GLAD GLCLU 2019 land cover/land use (30m)',
        output_subdir='landclass/glclu',
        config_override_key='DOWNLOAD_GLCLU',
    ),
]

_FULL_EXTRA_DATASETS: List[ProfileDataset] = [
    ProfileDataset(
        handler_name='GLACIER',
        description='Randolph Glacier Inventory (RGI)',
        output_subdir='glaciers',
        config_override_key='DOWNLOAD_GLACIER_DATA',
    ),
    ProfileDataset(
        handler_name='GLWD',
        description='GLWD v2 global lakes and wetlands',
        output_subdir='water/glwd',
        config_override_key='DOWNLOAD_GLWD',
    ),
    ProfileDataset(
        handler_name='BEDROCK_DEPTH',
        description='SoilGrids depth to bedrock (BDTICM)',
        output_subdir='soilclass/bedrock',
        config_override_key='DOWNLOAD_BEDROCK_DEPTH',
    ),
    ProfileDataset(
        handler_name='ARIDITY_INDEX',
        description='CGIAR Global Aridity Index and PET',
        output_subdir='climate/aridity',
        config_override_key='DOWNLOAD_ARIDITY_INDEX',
    ),
    ProfileDataset(
        handler_name='MODIS_NDVI',
        description='MODIS NDVI/EVI vegetation index',
        output_subdir='landcover/modis_ndvi',
        config_override_key='DOWNLOAD_MODIS_NDVI',
    ),
    ProfileDataset(
        handler_name='ROOT_ZONE_STORAGE',
        description='Root-zone storage capacity',
        output_subdir='soilclass/root_zone',
        config_override_key='DOWNLOAD_ROOT_ZONE_STORAGE',
    ),
    ProfileDataset(
        handler_name='JRC_WATER',
        description='JRC Global Surface Water occurrence/extent',
        output_subdir='water/jrc',
        config_override_key='DOWNLOAD_JRC_WATER',
    ),
    ProfileDataset(
        handler_name='WOKAM',
        description='World Karst Aquifer Map (WOKAM)',
        output_subdir='geology/karst',
        config_override_key='DOWNLOAD_WOKAM',
    ),
    ProfileDataset(
        handler_name='MERIT_HYDRO',
        description='MERIT Hydro river network (elv, dir, upa)',
        output_subdir='elevation/merit_hydro',
        config_override_key='DOWNLOAD_MERIT_HYDRO',
    ),
]

_FULL_DATASETS = _CAMELS_SPAT_DATASETS + _FULL_EXTRA_DATASETS

PROFILES: Dict[str, List[ProfileDataset]] = {
    'core': _CORE_DATASETS,
    'camels_spat': _CAMELS_SPAT_DATASETS,
    'full': _FULL_DATASETS,
}

# The canonical set of valid profile names lives in core so the config validator
# can check ATTRIBUTE_PROFILE without importing the data layer. Keep the two in
# sync — this mapping is the source of dataset content, that constant is the
# source of the name contract.
from symfluence.core.config.constants import VALID_ATTRIBUTE_PROFILES

assert set(PROFILES) == set(VALID_ATTRIBUTE_PROFILES), (
    "attribute-profile names drifted from core.config.constants."
    "VALID_ATTRIBUTE_PROFILES: "
    f"{set(PROFILES) ^ set(VALID_ATTRIBUTE_PROFILES)}"
)
