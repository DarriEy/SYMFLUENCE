"""Legacy configuration aliases and canonical key preferences.

This module isolates compatibility shims so core transformer logic can treat
canonical mappings separately from deprecated/legacy aliases.
"""

from typing import Any, Dict, Iterable, List, Mapping, Tuple

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
    "SETTINGS_DROUTE_PATH": ("model", "droute", "settings_path"),
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
    "SETTINGS_SUMMA_INIT_GRID_FILE": ("model", "summa", "init_grid_file"),
    "SETTINGS_SUMMA_ATTRIB_GRID_FILE": ("model", "summa", "attrib_grid_file"),
    "SETTINGS_FUSE_FILEMANAGER": ("model", "fuse", "filemanager"),
    "SETTINGS_FUSE_PARAMS_TO_CALIBRATE": ("model", "fuse", "params_to_calibrate"),
    "SETTINGS_GR_CONTROL": ("model", "gr", "control"),
    "SETTINGS_HYPE_INFO": ("model", "hype", "info_file"),
    "SETTINGS_MESH_INPUT": ("model", "mesh", "input_file"),
}


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
