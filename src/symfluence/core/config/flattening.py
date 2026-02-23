"""Flatten a SymfluenceConfig back to an uppercase flat dictionary.

Inverse of ``transform_flat_to_nested`` — used for backward compatibility
with legacy code that expects flat configs.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple

from symfluence.core.config.legacy_aliases import CANONICAL_KEYS

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


def flatten_nested_config(config: 'SymfluenceConfig') -> Dict[str, Any]:
    """Convert SymfluenceConfig instance to flat dict with uppercase keys.

    This is the inverse operation of transform_flat_to_nested, used for
    backward compatibility with legacy code expecting flat configs.

    Args:
        config: SymfluenceConfig instance

    Returns:
        Flat configuration dictionary with uppercase keys

    Example:
        >>> from symfluence.core.config.models import SymfluenceConfig
        >>> config = SymfluenceConfig.from_preset('fuse-basic')
        >>> flat = flatten_nested_config(config)
        >>> flat['DOMAIN_NAME']
        'test_basin'
    """
    flat: Dict[str, Any] = {}

    # Create reverse mapping (nested path -> flat key)
    # Use CANONICAL_KEYS for paths with multiple aliases to ensure consistent output
    from symfluence.core.config.transformers import get_flat_to_nested_map
    flat_to_nested = get_flat_to_nested_map()

    nested_to_flat: Dict[Tuple[str, ...], str] = {}
    for flat_key, nested_path in flat_to_nested.items():
        # Only set if not already set, OR if this is the canonical key for this path
        if nested_path not in nested_to_flat:
            nested_to_flat[nested_path] = flat_key
        elif nested_path in CANONICAL_KEYS and CANONICAL_KEYS[nested_path] == flat_key:
            # Override with canonical key
            nested_to_flat[nested_path] = flat_key

    # Include model-specific transformers from registered config adapters
    if hasattr(config, 'model') and config.model:
        hydrological_model = getattr(config.model, 'hydrological_model', None)
        if hydrological_model:
            try:
                from symfluence.models.registries.config_registry import ConfigRegistry
                model_transformers = ConfigRegistry.get_config_transformers(
                    str(hydrological_model)
                )
                if model_transformers:
                    for fk, nested_path in model_transformers.items():
                        if nested_path not in nested_to_flat:
                            nested_to_flat[nested_path] = fk
            except (ImportError, KeyError, AttributeError):
                pass

    def _flatten_section(section_name: str, section_obj: Any, prefix: Tuple[str, ...] = ()) -> None:
        """Recursively flatten a config section."""
        if section_obj is None:
            return

        # Get the dict representation
        # Use exclude_none=True so that .get() falls back to defaults for unset values
        if hasattr(section_obj, 'model_dump'):
            section_dict = section_obj.model_dump(by_alias=False, exclude_none=True)
        else:
            section_dict = section_obj if isinstance(section_obj, dict) else {}

        for key, value in section_dict.items():
            current_path = prefix + (key,)

            # Check if this path maps to a flat key
            if current_path in nested_to_flat:
                fk = nested_to_flat[current_path]
                # Convert Path to string for compatibility
                if isinstance(value, Path):
                    flat[fk] = str(value)
                else:
                    flat[fk] = value
            elif isinstance(value, dict) or hasattr(value, 'model_dump'):
                # Recurse into nested objects
                _flatten_section(key, value, current_path)

    # Flatten each section
    _flatten_section('system', config.system, ('system',))
    _flatten_section('domain', config.domain, ('domain',))
    _flatten_section('data', config.data, ('data',))
    _flatten_section('forcing', config.forcing, ('forcing',))
    _flatten_section('model', config.model, ('model',))
    _flatten_section('optimization', config.optimization, ('optimization',))
    _flatten_section('evaluation', config.evaluation, ('evaluation',))
    _flatten_section('paths', config.paths, ('paths',))
    if hasattr(config, 'fews') and config.fews is not None:
        _flatten_section('fews', config.fews, ('fews',))

    # Include extra fields from root config (e.g. CUSTOM_PATH in tests)
    # Extra fields can be at top-level or nested inside '_extra' dict
    if hasattr(config, 'model_extra') and config.model_extra:
        for key, value in config.model_extra.items():
            if key == '_extra' and isinstance(value, dict):
                # Handle nested _extra dict (from transform_flat_to_nested)
                for extra_key, extra_value in value.items():
                    if isinstance(extra_value, Path):
                        flat[extra_key] = str(extra_value)
                    else:
                        flat[extra_key] = extra_value
            elif isinstance(value, Path):
                flat[key] = str(value)
            else:
                flat[key] = value

    return flat
