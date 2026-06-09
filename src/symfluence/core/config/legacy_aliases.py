# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Legacy configuration aliases and canonical key preferences.

This module isolates compatibility shims so core transformer logic can treat
canonical mappings separately from deprecated/legacy aliases.

Three categories of aliases are defined here:

1. **Normalization aliases** (``NORMALIZATION_ALIASES``): spelling, product-name,
   and shorthand normalization applied during config loading (e.g.
   ``CONFLUENCE_DATA_DIR`` → ``SYMFLUENCE_DATA_DIR``).
2. **Deprecated keys** (``DEPRECATED_KEYS``): flat keys replaced by
   canonical successors but still accepted with a deprecation warning.
3. **Legacy flat-to-nested aliases** (``LEGACY_FLAT_TO_NESTED_ALIASES``):
   flat keys that map to nested config paths for backward compatibility.
4. **Recognized flat keys** (``RECOGNIZED_FLAT_KEYS``): real keys read in flat
   form (kept in the config ``_extra`` passthrough, not a nested field) that the
   unrecognized-key validator must treat as known. Recognition only, never
   transformation.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

# Normalization aliases: spelling, product-name, and shorthand corrections
# applied during config loading by config_loader._normalize_key().
NORMALIZATION_ALIASES: Dict[str, str] = {
    "GR_SPATIAL": "GR_SPATIAL_MODE",
    "OPTIMISATION_METHODS": "OPTIMIZATION_METHODS",
    "OPTIMISATION_TARGET": "OPTIMIZATION_TARGET",
    "OPTIMIZATION_ALGORITHM": "ITERATIVE_OPTIMIZATION_ALGORITHM",
    # Legacy CONFLUENCE naming (backwards compatibility)
    "CONFLUENCE_DATA_DIR": "SYMFLUENCE_DATA_DIR",
    "CONFLUENCE_CODE_DIR": "SYMFLUENCE_CODE_DIR",
    # Legacy domain discretization naming
    "DOMAIN_DISCRETIZATION": "SUB_GRID_DISCRETIZATION",
    # Legacy optimization-metric spelling -> canonical OPTIMIZATION_METRIC
    # (maps to optimization.metric). Renaming here makes old configs both
    # recognized and functional, rather than silently inert.
    "TARGET_METRIC": "OPTIMIZATION_METRIC",
    # Legacy SUMMA decisions spelling -> canonical SUMMA_DECISION_OPTIONS.
    "SUMMA_DECISIONS": "SUMMA_DECISION_OPTIONS",
}

# Maps deprecated flat keys to their preferred replacements.
DEPRECATED_KEYS: Dict[str, str] = {
    # System legacy naming
    "MPI_PROCESSES": "NUM_PROCESSES",
    # MizuRoute legacy naming (inverted: INSTALL_PATH_MIZUROUTE -> MIZUROUTE_INSTALL_PATH)
    "INSTALL_PATH_MIZUROUTE": "MIZUROUTE_INSTALL_PATH",
    "EXE_NAME_MIZUROUTE": "MIZUROUTE_EXE",
    # NSGA-II secondary objective legacy naming
    "OPTIMIZATION_TARGET2": "NSGA2_SECONDARY_TARGET",
    "OPTIMIZATION_METRIC2": "NSGA2_SECONDARY_METRIC",
}

# Canonical + legacy key pairs used by model adapters for fallback validation.
MIZUROUTE_CANONICAL_LEGACY_KEY_PAIRS: Tuple[Tuple[str, str], ...] = (
    (DEPRECATED_KEYS["INSTALL_PATH_MIZUROUTE"], "INSTALL_PATH_MIZUROUTE"),
    (DEPRECATED_KEYS["EXE_NAME_MIZUROUTE"], "EXE_NAME_MIZUROUTE"),
)

# Canonical keys for nested paths with multiple aliases.
# When flattening nested config back to flat format, prefer these names.
CANONICAL_KEYS: Dict[Tuple[str, ...], str] = {
    ("system", "num_processes"): "NUM_PROCESSES",  # Prefer over MPI_PROCESSES
    ("optimization", "nsga2", "secondary_target"): "NSGA2_SECONDARY_TARGET",
    ("optimization", "nsga2", "secondary_metric"): "NSGA2_SECONDARY_METRIC",
    ("model", "mizuroute", "install_path"): "MIZUROUTE_INSTALL_PATH",
    ("model", "mizuroute", "exe"): "MIZUROUTE_EXE",
}

# Flat keys that remain supported for backward compatibility but are not canonical.
LEGACY_FLAT_TO_NESTED_ALIASES: Dict[str, Tuple[str, ...]] = {
    "MPI_PROCESSES": ("system", "num_processes"),
    "INSTALL_PATH_MIZUROUTE": ("model", "mizuroute", "install_path"),
    "EXE_NAME_MIZUROUTE": ("model", "mizuroute", "exe"),
    "OPTIMIZATION_TARGET2": ("optimization", "nsga2", "secondary_target"),
    "OPTIMIZATION_METRIC2": ("optimization", "nsga2", "secondary_metric"),
    # Legacy mizuRoute settings keys
    "SETTINGS_MIZU_PATH": ("model", "mizuroute", "settings_path"),
    "SETTINGS_MIZU_WITHIN_BASIN": ("model", "mizuroute", "within_basin"),
    "SETTINGS_MIZU_ROUTING_DT": ("model", "mizuroute", "routing_dt"),
    "SETTINGS_MIZU_ROUTING_UNITS": ("model", "mizuroute", "routing_units"),
    "SETTINGS_MIZU_ROUTING_VAR": ("model", "mizuroute", "routing_var"),
    "SETTINGS_MIZU_OUTPUT_FREQ": ("model", "mizuroute", "output_freq"),
    "SETTINGS_MIZU_OUTPUT_VARS": ("model", "mizuroute", "output_vars"),
    "SETTINGS_MIZU_MAKE_OUTLET": ("model", "mizuroute", "make_outlet"),
    "SETTINGS_MIZU_NEEDS_REMAP": ("model", "mizuroute", "needs_remap"),
    "SETTINGS_MIZU_TOPOLOGY": ("model", "mizuroute", "topology"),
    "SETTINGS_MIZU_PARAMETERS": ("model", "mizuroute", "parameters"),
    "SETTINGS_MIZU_CONTROL_FILE": ("model", "mizuroute", "control_file"),
    "SETTINGS_MIZU_REMAP": ("model", "mizuroute", "remap"),
    "SETTINGS_MIZU_OUTPUT_VAR": ("model", "mizuroute", "output_var"),
    "SETTINGS_MIZU_PARAMETER_FILE": ("model", "mizuroute", "parameter_file"),
    "SETTINGS_MIZU_REMAP_FILE": ("model", "mizuroute", "remap_file"),
    "SETTINGS_MIZU_TOPOLOGY_FILE": ("model", "mizuroute", "topology_file"),
    # Legacy t-route settings keys
    "SETTINGS_TROUTE_PATH": ("model", "troute", "settings_path"),
    "SETTINGS_TROUTE_TOPOLOGY": ("model", "troute", "topology_file"),
    "SETTINGS_TROUTE_CONFIG_FILE": ("model", "troute", "config_file"),
    "SETTINGS_TROUTE_DT_SECONDS": ("model", "troute", "dt_seconds"),
    # NOTE: SETTINGS_DROUTE_PATH is NOT declared here — dRoute is an external plugin package
    # (like the JAX models) and declares its own config aliases via DRouteConfig +
    # model_manifest(config_schema=...). Keeping it here would duplicate the plugin's declaration.
    "SETTINGS_FUSE_PATH": ("model", "fuse", "settings_path"),
    "SETTINGS_GR_PATH": ("model", "gr", "settings_path"),
    "SETTINGS_GSFLOW_PATH": ("model", "gsflow", "settings_path"),
    "SETTINGS_HYPE_PATH": ("model", "hype", "settings_path"),
    "SETTINGS_MESH_PATH": ("model", "mesh", "settings_path"),
    "SETTINGS_RHESSYS_PATH": ("model", "rhessys", "settings_path"),
    "SETTINGS_SUMMA_PATH": ("model", "summa", "settings_path"),
    "SETTINGS_WATFLOOD_PATH": ("model", "watflood", "settings_path"),
    # Legacy SUMMA settings keys
    "SETTINGS_SUMMA_FILEMANAGER": ("model", "summa", "filemanager"),
    "SETTINGS_SUMMA_FORCING_LIST": ("model", "summa", "forcing_list"),
    "SETTINGS_SUMMA_COLDSTATE": ("model", "summa", "coldstate"),
    "SETTINGS_SUMMA_TRIALPARAMS": ("model", "summa", "trialparams"),
    "SETTINGS_SUMMA_ATTRIBUTES": ("model", "summa", "attributes"),
    "SETTINGS_SUMMA_OUTPUT": ("model", "summa", "output"),
    "SETTINGS_SUMMA_BASIN_PARAMS_FILE": ("model", "summa", "basin_params_file"),
    "SETTINGS_SUMMA_LOCAL_PARAMS_FILE": ("model", "summa", "local_params_file"),
    "SETTINGS_SUMMA_CONNECT_HRUS": ("model", "summa", "connect_hrus"),
    "SETTINGS_SUMMA_TRIALPARAM_N": ("model", "summa", "trialparam_n"),
    "SETTINGS_SUMMA_TRIALPARAM_1": ("model", "summa", "trialparam_1"),
    "SETTINGS_SUMMA_USE_PARALLEL_SUMMA": ("model", "summa", "use_parallel"),
    "SETTINGS_SUMMA_PARALLEL_BACKEND": ("model", "summa", "parallel_backend"),
    "SETTINGS_SUMMA_CPUS_PER_TASK": ("model", "summa", "cpus_per_task"),
    "SETTINGS_SUMMA_TIME_LIMIT": ("model", "summa", "time_limit"),
    "SETTINGS_SUMMA_MEM": ("model", "summa", "mem"),
    "SETTINGS_SUMMA_GRU_COUNT": ("model", "summa", "gru_count"),
    "SETTINGS_SUMMA_GRU_PER_JOB": ("model", "summa", "gru_per_job"),
    "SETTINGS_SUMMA_PARALLEL_PATH": ("model", "summa", "parallel_path"),
    "SETTINGS_SUMMA_PARALLEL_EXE": ("model", "summa", "parallel_exe"),
    "SETTINGS_SUMMA_GLACIER_MODE": ("model", "summa", "glacier_mode"),
    "SETTINGS_SUMMA_GLACIER_ATTRIBUTES": ("model", "summa", "glacier_attributes"),
    "SETTINGS_SUMMA_GLACIER_COLDSTATE": ("model", "summa", "glacier_coldstate"),
    "SETTINGS_SUMMA_SOILPROFILE": ("model", "summa", "soilprofile"),
    "SETTINGS_FUSE_FILEMANAGER": ("model", "fuse", "filemanager"),
    "SETTINGS_FUSE_PARAMS_TO_CALIBRATE": ("model", "fuse", "params_to_calibrate"),
    "SETTINGS_GR_CONTROL": ("model", "gr", "control"),
    "SETTINGS_HYPE_INFO": ("model", "hype", "info_file"),
    "SETTINGS_MESH_INPUT": ("model", "mesh", "input_file"),
}

# Recognized flat keys that are intentionally read in flat form — via
# ``config.get('KEY')`` or ``_get_config_value(..., dict_key='KEY')`` — and
# therefore flow through to the config object's ``_extra`` passthrough rather
# than a nested Pydantic field. They are real, consumed-in-code keys that the
# unrecognized-key validator (``key_validation``) must treat as KNOWN so it does
# not false-warn on them. They are deliberately NOT in LEGACY_FLAT_TO_NESTED_ALIASES:
# giving them a nested path would relocate them out of ``_extra`` and break the
# flat readers. Adding a key here changes recognition only, never transformation.
# (Resolves the bulk of RTI review open-question Q3 / Tier 3 item 21 noise; see
# docs/adr/0006-config-unknown-keys-warn-by-default.md. Conceptual-model and
# unbacked feature families are handled separately — see that ADR's follow-on.)
RECOGNIZED_FLAT_KEYS: frozenset[str] = frozenset({
    # Multi-gauge calibration
    "MULTI_GAUGE_AGGREGATION", "MULTI_GAUGE_CALIBRATION", "MULTI_GAUGE_EXCLUDE_IDS",
    "MULTI_GAUGE_KGE_FLOOR", "MULTI_GAUGE_MAX_DISTANCE", "MULTI_GAUGE_MIN_GAUGES",
    "MULTI_GAUGE_MIN_OBS_CV", "MULTI_GAUGE_MIN_OVERLAP_DAYS", "MULTI_GAUGE_MIN_SPECIFIC_Q",
    "MULTI_GAUGE_OBS_DIR",
    # EnKF / data assimilation
    "ENKF_ASSIMILATION_INTERVAL", "ENKF_ASSIMILATION_VARIABLE", "ENKF_AUGMENT_STATE",
    "ENKF_ENFORCE_NONNEG", "ENKF_ENSEMBLE_SIZE", "ENKF_ESTIMATE_PARAMS", "ENKF_FILTER_VARIANT",
    "ENKF_FORCING_PERTURBATION", "ENKF_INFLATION_FACTOR", "ENKF_LOCALIZATION_RADIUS",
    "ENKF_OBS_ERROR_STD", "ENKF_OBS_ERROR_TYPE", "ENKF_PARAM_PERTURBATION_STD",
    "ENKF_PRECIP_PERTURBATION_STD", "ENKF_TEMP_PERTURBATION_STD",
    # ESA CCI soil moisture
    "ESA_CCI_SM_PATH", "ESA_CCI_SM_RECORD_TYPE", "ESA_CCI_SM_SENSOR",
    "ESA_CCI_SM_TIME_AGGREGATION", "ESA_CCI_SM_VARIABLE", "ESA_CCI_SM_VERSION",
    # IGNACIO fire-weather (FWI)
    "IGNACIO_CURING", "IGNACIO_DEFAULT_BUI", "IGNACIO_DEFAULT_DC", "IGNACIO_DEFAULT_DMC",
    "IGNACIO_DEFAULT_FFMC", "IGNACIO_DEFAULT_ISI", "IGNACIO_FMC", "IGNACIO_FUEL_SOURCE_TYPE",
    "IGNACIO_FWI_LATITUDE", "IGNACIO_GENERATE_PLOTS", "IGNACIO_INITIAL_RADIUS",
    "IGNACIO_MIN_ROS", "IGNACIO_NON_FUEL_CODES", "IGNACIO_N_VERTICES", "IGNACIO_OUTPUT_CRS",
    "IGNACIO_PERIMETER_FORMAT", "IGNACIO_RANDOM_SEED", "IGNACIO_SAVE_ROS_GRIDS",
    "IGNACIO_TIME_VARYING_WEATHER", "IGNACIO_WORKING_CRS",
    # HYPE process options
    "HYPE_DEEP_GROUND", "HYPE_FROZEN_SOIL_MODEL", "HYPE_INFILTRATION_MODEL",
    "HYPE_PARAM_BOUNDS", "HYPE_PET_MODEL", "HYPE_SNOW_EVAPORATION", "HYPE_SOIL_INIT_WET",
    "HYPE_SOIL_LAYER_DEPTHS", "HYPE_SURFACE_RUNOFF",
    # CanSWE snow obs
    "CANSWE_MIN_OBSERVATIONS", "CANSWE_PATH", "CANSWE_VERSION",
    # State save/load / ensemble
    "STATE_DIR", "STATE_ENSEMBLE_MEMBERS", "STATE_FILE_PATTERN", "STATE_INPUT_PATH",
    "STATE_LOAD", "STATE_MANAGEMENT_ENABLED", "STATE_OUTPUT_PATH", "STATE_SAVE",
    # Per-model parameter bounds & initial params
    "CLM_PARAM_BOUNDS", "INITIAL_PARAMETERS", "MESH_PARAM_BOUNDS", "PARAMETER_BOUNDS",
    "PARFLOW_PARAM_BOUNDS", "RHESSYS_PARAM_BOUNDS", "VIC_PARAM_BOUNDS",
    # GNN emulator
    "GNN_OUTPUT_SIZE", "GNN_USE_SNOW",
    # LSTM emulator
    "LSTM", "LSTM_PARAMETER_BOUNDS", "LSTM_PARAMS_TO_CALIBRATE",
    # GLEAM ET
    "GLEAM_ET_DOWNLOAD_URL", "GLEAM_ET_PATH",
    # Groundwater
    "GW_AUTO_ALIGN", "GW_BASE_DEPTH",
    # GRACE
    "GRACE_SUBSET",
    # Catchment shapefile
    "CATCHMENT_SHP_PATH", "CATCHMENT_SHP_SLOPE_UNITS",
    # DDS
    "DDS_STAGNATION_THRESHOLD",
    # NGEN module toggles
    "ENABLE_NOAH", "ENABLE_PET", "ENABLE_SLOTH",
    # FUSE run options
    "FUSE_RUN_MODE", "FUSE_TEMPLATE_PATH",
    # Model-error (DA)
    "MODEL_ERROR_BASE", "MODEL_ERROR_FRACTION",
    # Regionalization transfer fn
    "TRANSFER_FUNCTION_B_BOUNDS", "TRANSFER_FUNCTION_TYPE",
    # Other recognized flat keys
    "ADAM_STEPS", "CALIBRATION_NEXUS_ID", "CALIBRATION_WARMUP_DAYS", "CARRA_DOMAIN",
    "DA_METHOD", "DECISION_OPTIONS", "DOWNLOAD_CANSWE", "EM_EARTH", "ET_UNIT_CONVERSION",
    "EXPERIMENT_OUTPUT_NGEN", "FORCING_RAW_PATH", "GAUGE_SEGMENT_MAPPING", "GR_MODEL_TYPE",
    "HYDROSHEDS_LEVEL", "LIKELIHOOD_FUNCTION", "MIZUROUTE_NUM_THREADS", "MODIS_SNOW",
    "NWS_HYDROFABRIC_VERSION", "OPTIMIZATION_MAX_ITERATIONS", "SETTINGS_NGEN_REALIZATION",
    "SKIP_WARM_START", "SMAP_LAYER", "SNOTEL_STATE", "TDX_SOURCE", "USGS_GW",
})


def find_missing_canonical_keys(
    config: Mapping[str, Any],
    canonical_legacy_pairs: Iterable[Tuple[str, str]],
) -> List[str]:
    """Return canonical keys missing from config after legacy fallback checks."""
    missing: List[str] = []
    for canonical_key, legacy_key in canonical_legacy_pairs:
        value = config.get(canonical_key) or config.get(legacy_key)
        if value in (None, "", "None"):
            missing.append(canonical_key)
    return missing
