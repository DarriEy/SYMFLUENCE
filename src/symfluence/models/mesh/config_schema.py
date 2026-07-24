# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for MESH.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG
from symfluence.core.config.models.model_config_types import SpatialModeType


class MESHConfig(BaseModel):
    """MESH (Modélisation Environnementale-Surface Hydrology) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='MESH_INSTALL_PATH')
    exe: str = Field(default='mesh.exe', alias='MESH_EXE')
    spatial_mode: SpatialModeType = Field(default='auto', alias='MESH_SPATIAL_MODE')
    # MESH run mode: 'runrte' (WATROUTE channel routing + wf_lzs lower-zone
    # baseflow store active) or 'noroute' (streamflow taken directly from the
    # basin water balance). Left unset (None) => auto: multi-cell domains route
    # ('runrte'), a single cell has no channel network and falls back to
    # 'noroute'. Set explicitly to 'runrte' to keep the baseflow store (and its
    # FLZ/PWR/RCHARG parameters) live even on a single-cell domain.
    run_mode: Optional[str] = Field(default=None, alias='MESH_RUNMODE')
    settings_path: str = Field(default='default', alias='SETTINGS_MESH_PATH')
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MESH')
    forcing_path: str = Field(default='default', alias='MESH_FORCING_PATH')
    forcing_vars: str = Field(default='default', alias='MESH_FORCING_VARS')
    forcing_units: str = Field(default='default', alias='MESH_FORCING_UNITS')
    forcing_to_units: str = Field(default='default', alias='MESH_FORCING_TO_UNITS')
    landcover_stats_path: str = Field(default='default', alias='MESH_LANDCOVER_STATS_PATH')
    landcover_stats_dir: str = Field(default='default', alias='MESH_LANDCOVER_STATS_DIR')
    landcover_stats_file: str = Field(default='default', alias='MESH_LANDCOVER_STATS_FILE')
    main_id: str = Field(default='default', alias='MESH_MAIN_ID')
    ds_main_id: str = Field(default='default', alias='MESH_DS_MAIN_ID')
    landcover_classes: str = Field(default='default', alias='MESH_LANDCOVER_CLASSES')
    ddb_vars: str = Field(default='default', alias='MESH_DDB_VARS')
    ddb_units: str = Field(default='default', alias='MESH_DDB_UNITS')
    ddb_to_units: str = Field(default='default', alias='MESH_DDB_TO_UNITS')
    ddb_min_values: str = Field(default='default', alias='MESH_DDB_MIN_VALUES')
    gru_dim: str = Field(default='default', alias='MESH_GRU_DIM')
    hru_dim: str = Field(default='default', alias='MESH_HRU_DIM')
    outlet_value: str = Field(default='default', alias='MESH_OUTLET_VALUE')
    # Additional MESH settings
    input_file: str = Field(default='default', alias='SETTINGS_MESH_INPUT')
    params_to_calibrate: str = Field(
        default='ZSNL,MANN,RCHARG,BASEFLW,DTMINUSR',
        alias='MESH_PARAMS_TO_CALIBRATE'
    )
    spinup_days: int = Field(default=365, alias='MESH_SPINUP_DAYS')
    gru_min_total: float = Field(default=0.0, alias='MESH_GRU_MIN_TOTAL')
    # Lumped mode enforcement settings
    force_single_gru: bool = Field(default=True, alias='MESH_FORCE_SINGLE_GRU')
    apply_params_all_grus: bool = Field(default=True, alias='MESH_APPLY_PARAMS_ALL_GRUS')
    use_landcover_multipliers: bool = Field(default=True, alias='MESH_USE_LANDCOVER_MULTIPLIERS')
    enable_frozen_soil: bool = Field(default=True, alias='MESH_ENABLE_FROZEN_SOIL')
    daily_tolerance_days: int = Field(default=1, alias='MESH_DAILY_TOLERANCE_DAYS')
    param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='MESH_PARAM_BOUNDS')

    # ------------------------------------------------------------------
    # CLASS / hydrology field overrides
    # ------------------------------------------------------------------
    # meshflow hard-derives these regime-determining fields from the input
    # data (soil texture, drainage density, initial states, veg params). When
    # set, these keys OVERRIDE the meshflow-derived values so the surface-runoff
    # regime can be controlled from config alone (no shipping of hand-tuned
    # files). Left as None => keep whatever meshflow produced.
    soil_sand: Optional[List[float]] = Field(default=None, alias='MESH_SOIL_SAND')
    soil_clay: Optional[List[float]] = Field(default=None, alias='MESH_SOIL_CLAY')
    soil_orgm: Optional[List[float]] = Field(default=None, alias='MESH_SOIL_ORGM')
    drainage_density: Optional[float] = Field(default=None, alias='MESH_DD')
    mid: Optional[int] = Field(default=None, alias='MESH_MID')
    init_tbar: Optional[List[float]] = Field(default=None, alias='MESH_INIT_TBAR')
    init_thlq: Optional[List[float]] = Field(default=None, alias='MESH_INIT_THLQ')
    veg_cmas: Optional[float] = Field(default=None, alias='MESH_VEG_CMAS')
    veg_qa50: Optional[float] = Field(default=None, alias='MESH_VEG_QA50')
    veg_vpda: Optional[float] = Field(default=None, alias='MESH_VEG_VPDA')
    veg_vpdb: Optional[float] = Field(default=None, alias='MESH_VEG_VPDB')
    iwf: Optional[int] = Field(default=None, alias='MESH_IWF')


# Legacy flat config keys accepted for backward compatibility, mapped to
# this schema's field names. Core's flat->nested transform collects these
# from every schema registered in R.config_schemas (see
# core.config.legacy_aliases.iter_legacy_flat_to_nested_aliases).
MESHConfig.LEGACY_FLAT_ALIASES = {
    'SETTINGS_MESH_PATH': 'settings_path',
    'SETTINGS_MESH_INPUT': 'input_file',
}
