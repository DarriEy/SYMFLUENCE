from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


ALIAS_MAP = {
    "GR_SPATIAL": "GR_SPATIAL_MODE",
    "OPTIMISATION_METHODS": "OPTIMIZATION_METHODS",
    "OPTIMISATION_TARGET": "OPTIMIZATION_TARGET",
    "OPTIMIZATION_ALGORITHM": "ITERATIVE_OPTIMIZATION_ALGORITHM",
}

REQUIRED_KEYS = {
    "SYMFLUENCE_DATA_DIR",
    "SYMFLUENCE_CODE_DIR",
    "DOMAIN_NAME",
    "EXPERIMENT_ID",
    "EXPERIMENT_TIME_START",
    "EXPERIMENT_TIME_END",
    "DOMAIN_DEFINITION_METHOD",
    "DOMAIN_DISCRETIZATION",
    "HYDROLOGICAL_MODEL",
    "FORCING_DATASET",
}

LIST_KEYS = {
    "OPTIMIZATION_METHODS",
    "NEX_MODELS",
    "NEX_SCENARIOS",
    "NEX_ENSEMBLES",
    "NEX_VARIABLES",
    "MULTI_SCALE_THRESHOLDS",
}

NUMERIC_KEYS = {
    "MPI_PROCESSES",
    "FORCING_TIME_STEP_SIZE",
    "STREAM_THRESHOLD",
    "MIN_HRU_SIZE",
    "MIN_GRU_SIZE",
    "ELEVATION_BAND_SIZE",
    "RADIATION_CLASS_NUMBER",
    "ASPECT_CLASS_NUMBER",
    "DROP_ANALYSIS_MIN_THRESHOLD",
    "DROP_ANALYSIS_MAX_THRESHOLD",
    "DROP_ANALYSIS_NUM_THRESHOLDS",
    "MOVE_OUTLETS_MAX_DISTANCE",
    "LAPSE_RATE",
    "NUMBER_OF_ITERATIONS",
    "RANDOM_SEED",
}

BOOL_STRINGS = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "1": True,
    "0": False,
}


def load_config(
    path: Path,
    overrides: Optional[Mapping[str, Any]] = None,
    *,
    validate: bool = True,
) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    if overrides:
        config.update(overrides)
    normalized = normalize_config(config)
    if validate:
        validate_config(normalized)
    return normalized


def normalize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in config.items():
        norm_key = _normalize_key(key)
        normalized[norm_key] = _coerce_value(norm_key, value)
    return normalized


def validate_config(config: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required configuration keys: {missing_list}")


def _normalize_key(key: str) -> str:
    key_upper = key.upper()
    return ALIAS_MAP.get(key_upper, key_upper)


def _coerce_value(key: str, value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        lower = stripped.lower()
        if lower in BOOL_STRINGS:
            return BOOL_STRINGS[lower]
        if lower in {"none", "null"}:
            return None
        if key in NUMERIC_KEYS:
            num = _parse_number(stripped)
            if num is not None:
                return num
        if key in LIST_KEYS:
            return _split_list(stripped)
        return stripped
    if isinstance(value, list) and key in LIST_KEYS:
        return [_coerce_list_item(v) for v in value]
    if isinstance(value, list) and key == "HYDROLOGICAL_MODEL":
        return ",".join(str(v).strip() for v in value if str(v).strip())
    return value


def _parse_number(value: str) -> Optional[float | int]:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return None


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _coerce_list_item(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        lower = stripped.lower()
        if lower in BOOL_STRINGS:
            return BOOL_STRINGS[lower]
        if lower in {"none", "null"}:
            return None
        return stripped
    return value
