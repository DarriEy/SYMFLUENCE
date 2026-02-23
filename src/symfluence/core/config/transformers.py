"""Configuration transformation utilities for SYMFLUENCE.

This module handles conversion between flat and hierarchical configuration formats:
- Flat format: Uppercase keys like {'DOMAIN_NAME': 'test', 'FORCING_DATASET': 'ERA5'}
- Nested format: Hierarchical structure like {'domain': {'name': 'test'}, 'forcing': {'dataset': 'ERA5'}}

Key functions:
- transform_flat_to_nested(): Convert flat dict to nested structure for Pydantic models
- flatten_nested_config(): Convert SymfluenceConfig instance back to flat dict for backward compatibility

Phase 2 Addition (Configuration Key Standardization):
- Standardized naming for MizuRoute keys (MIZUROUTE_INSTALL_PATH, MIZUROUTE_EXE)
- Deprecation warnings for legacy keys (INSTALL_PATH_MIZUROUTE, EXE_NAME_MIZUROUTE)
"""

import logging
import threading
import warnings
from typing import Any, Dict, Optional, Tuple

from symfluence.core.config.legacy_aliases import (
    CANONICAL_KEYS,
    DEPRECATED_KEYS,
    LEGACY_FLAT_TO_NESTED_ALIASES,
)

logger = logging.getLogger(__name__)


def _warn_deprecated_keys(flat_config: Dict[str, Any]) -> None:
    """Warn about deprecated configuration keys.

    Checks the flat configuration dictionary for any deprecated keys and
    emits deprecation warnings with guidance on the new key names.

    Args:
        flat_config: Flat configuration dictionary with uppercase keys
    """
    for old_key, new_key in DEPRECATED_KEYS.items():
        if old_key in flat_config:
            logger.warning(
                f"Configuration key '{old_key}' is deprecated, use '{new_key}' instead. "
                f"Support will be removed in v2.0."
            )
            # Also emit a Python DeprecationWarning for programmatic detection
            warnings.warn(
                f"Config key '{old_key}' is deprecated, use '{new_key}' instead. "
                f"This key will be removed in SYMFLUENCE v2.0.",
                DeprecationWarning,
                stacklevel=3,
            )


# Global cache for auto-generated mapping (thread-safe)
_AUTO_GENERATED_MAP: Optional[Dict[str, Tuple[str, ...]]] = None
_GENERATION_LOCK = threading.Lock()


def get_flat_to_nested_map() -> Dict[str, Tuple[str, ...]]:
    """Get flat-to-nested mapping via lazy auto-generation.

    Thread-safe with caching for performance.
    First call generates mapping, subsequent calls return cached version.

    Returns:
        Dictionary mapping flat keys to nested paths
    """
    global _AUTO_GENERATED_MAP

    # Fast path: return cached mapping
    if _AUTO_GENERATED_MAP is not None:
        return _AUTO_GENERATED_MAP

    # Slow path: generate mapping (thread-safe)
    with _GENERATION_LOCK:
        # Double-check after acquiring lock
        if _AUTO_GENERATED_MAP is not None:
            return _AUTO_GENERATED_MAP

        try:
            from symfluence.core.config.introspection import generate_flat_to_nested_map
            from symfluence.core.config.models import SymfluenceConfig

            _AUTO_GENERATED_MAP = generate_flat_to_nested_map(
                SymfluenceConfig,
                include_model_overrides=False,
            )

            # Merge legacy aliases (lower priority — only add missing keys)
            for key, path in LEGACY_FLAT_TO_NESTED_ALIASES.items():
                _AUTO_GENERATED_MAP.setdefault(key, path)

            logger.info(f"Auto-generated {len(_AUTO_GENERATED_MAP)} configuration mappings")

        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.debug(f"Auto-generation failed: {e}, using legacy aliases only")
            _AUTO_GENERATED_MAP = dict(LEGACY_FLAT_TO_NESTED_ALIASES)

    return _AUTO_GENERATED_MAP


# ========================================
# TRANSFORMATION FUNCTIONS
# ========================================


def _set_nested_value(d: Dict[str, Any], path: Tuple[str, ...], value: Any) -> None:
    """Helper to set value at nested path in dict.

    Args:
        d: Dictionary to modify
        path: Tuple of keys representing nested path
        value: Value to set

    Example:
        >>> d = {}
        >>> _set_nested_value(d, ('domain', 'name'), 'test')
        >>> d
        {'domain': {'name': 'test'}}
    """
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def transform_flat_to_nested(flat_config: Dict[str, Any]) -> Dict[str, Any]:
    """Transform flat configuration dict to nested structure.

    Maps uppercase keys like 'DOMAIN_NAME' to nested paths like
    {'domain': {'name': ...}}.

    Uses auto-generated mapping from Pydantic model aliases via
    get_flat_to_nested_map(), with the manual mapping as fallback.

    Emits deprecation warnings for legacy config keys
    (e.g., INSTALL_PATH_MIZUROUTE -> MIZUROUTE_INSTALL_PATH).

    Args:
        flat_config: Flat configuration dictionary with uppercase keys

    Returns:
        Nested configuration dictionary

    Example:
        >>> flat = {'DOMAIN_NAME': 'test', 'FORCING_DATASET': 'ERA5'}
        >>> nested = transform_flat_to_nested(flat)
        >>> nested
        {
            'domain': {'name': 'test'},
            'forcing': {'dataset': 'ERA5'}
        }
    """
    # Check for deprecated keys and emit warnings
    _warn_deprecated_keys(flat_config)

    nested: Dict[str, Any] = {
        'system': {},
        'domain': {},
        'data': {},
        'forcing': {},
        'model': {},
        'optimization': {},
        'evaluation': {},
        'paths': {},
        'fews': {},
    }

    # Build combined mapping: auto-generated from Pydantic aliases (with fallback)
    combined_mapping = get_flat_to_nested_map().copy()

    # Try to get model-specific transformers from ModelRegistry
    hydrological_model = flat_config.get('HYDROLOGICAL_MODEL')
    if hydrological_model:
        try:
            from symfluence.models.registries.config_registry import ConfigRegistry
            model_transformers = ConfigRegistry.get_config_transformers(hydrological_model)
            if model_transformers:
                # Model-specific transformers override base mapping
                combined_mapping.update(model_transformers)
        except (ImportError, KeyError, AttributeError):
            # If ConfigRegistry not available or model not registered, just use base mapping
            pass

    # Build reverse map: nested_path -> list of flat keys that map to it
    # This helps identify when multiple flat keys (canonical + deprecated) map to same path
    path_to_keys: Dict[Tuple[str, ...], list] = {}
    for flat_key in flat_config.keys():
        if flat_key in combined_mapping:
            path = combined_mapping[flat_key]
            path_to_keys.setdefault(path, []).append(flat_key)

    # Apply mapping, preferring canonical keys over deprecated aliases
    processed_paths: set = set()
    for flat_key, value in flat_config.items():
        if flat_key in combined_mapping:
            path = combined_mapping[flat_key]

            # If multiple keys map to this path, use the canonical key's value
            keys_for_path = path_to_keys.get(path, [flat_key])
            if len(keys_for_path) > 1 and path in CANONICAL_KEYS:
                canonical_key = CANONICAL_KEYS[path]
                if flat_key != canonical_key and canonical_key in flat_config:
                    # Skip this deprecated key, canonical key will be used
                    continue

            _set_nested_value(nested, path, value)
            processed_paths.add(path)
        else:
            # Unknown keys stored in _extra (Pydantic extra='allow' will handle)
            nested.setdefault('_extra', {})[flat_key] = value

    return nested
