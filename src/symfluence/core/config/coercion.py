"""Centralized config coercion to replace 60+ duplicate patterns.

This module provides a single, well-tested function for converting dict configs
to SymfluenceConfig instances. All modules should use ensure_config() instead
of duplicating the lazy import + isinstance check pattern.

Environment Variables:
    SYMFLUENCE_STRICT_CONFIG: If set to 'true', '1', or 'yes', coerce_config()
                              will raise errors instead of falling back to dict.
"""
import os
import warnings
from typing import TYPE_CHECKING, Any, Dict, Union

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


def _is_strict_mode() -> bool:
    """Check if strict config mode is enabled via environment variable."""
    value = os.environ.get('SYMFLUENCE_STRICT_CONFIG', '').lower()
    return value in ('true', '1', 'yes')


def ensure_config(config: Union[Dict[str, Any], 'SymfluenceConfig']) -> 'SymfluenceConfig':
    """
    Convert dict to SymfluenceConfig if needed.

    This function centralizes the config coercion pattern that was previously
    duplicated across 60+ files. It uses a runtime import to avoid circular
    dependency issues while providing type safety via TYPE_CHECKING.

    Args:
        config: Configuration as dict or SymfluenceConfig instance

    Returns:
        SymfluenceConfig instance

    Raises:
        TypeError: If config is neither a dict nor SymfluenceConfig

    Example:
        >>> from symfluence.core.config import ensure_config
        >>> cfg = ensure_config({'DOMAIN_NAME': 'test_domain', ...})
        >>> isinstance(cfg, SymfluenceConfig)
        True
    """
    # Runtime import to avoid circular dependency at module level
    from symfluence.core.config.models import SymfluenceConfig

    if isinstance(config, SymfluenceConfig):
        return config
    elif isinstance(config, dict):
        return SymfluenceConfig(**config)
    raise TypeError(
        f"config must be SymfluenceConfig or dict, got {type(config).__name__}. "
        "Use SymfluenceConfig.from_file() to load configuration."
    )


def coerce_config(
    config: Union[Dict[str, Any], 'SymfluenceConfig'],
    strict: bool = None,
    warn: bool = True
) -> Union['SymfluenceConfig', Dict[str, Any]]:
    """
    Convert dict to SymfluenceConfig if possible, with configurable fallback.

    Similar to ensure_config but can return the original dict if conversion fails
    (e.g., for partial configs in tests). This maintains backward compatibility
    with code that may pass incomplete configuration dictionaries.

    Args:
        config: Configuration as dict or SymfluenceConfig instance
        strict: If True, raise error instead of falling back to dict.
                If None, uses SYMFLUENCE_STRICT_CONFIG environment variable.
        warn: If True (default), emit DeprecationWarning when falling back to dict.
              Set to False for tests or cases where dict fallback is intentional.

    Returns:
        SymfluenceConfig if conversion succeeds, original dict otherwise
        (unless strict=True, which raises on failure)

    Raises:
        TypeError: If strict=True and config cannot be converted
        ValueError: If strict=True and config validation fails

    Example:
        >>> # Full config converts to SymfluenceConfig
        >>> cfg = coerce_config({'DOMAIN_NAME': 'test', ...})
        >>>
        >>> # Partial config in tests falls back to dict (with warning)
        >>> partial = coerce_config({'some_key': 'value'})
        >>> isinstance(partial, dict)
        True
        >>>
        >>> # Strict mode raises instead of falling back
        >>> coerce_config({'invalid': 'config'}, strict=True)  # Raises!

    Note:
        The fallback behavior is deprecated and will be removed in a future
        version. Use ensure_config() for strict conversion, or pass
        warn=False if you intentionally need dict fallback (e.g., in tests).
    """
    # Determine strict mode
    if strict is None:
        strict = _is_strict_mode()

    # Runtime import to avoid circular dependency at module level
    from symfluence.core.config.models import SymfluenceConfig

    if isinstance(config, SymfluenceConfig):
        return config
    elif isinstance(config, dict):
        try:
            return SymfluenceConfig(**config)
        except (TypeError, ValueError) as e:
            if strict:
                raise type(e)(
                    f"Config coercion failed (strict mode): {e}. "
                    "Ensure all required configuration fields are provided."
                ) from e

            # Emit deprecation warning for fallback behavior
            if warn:
                warnings.warn(
                    f"coerce_config() fell back to dict due to validation error: {e}. "
                    "This fallback behavior is deprecated. Use ensure_config() for "
                    "strict conversion, or fix the configuration to include all "
                    "required fields. Set SYMFLUENCE_STRICT_CONFIG=true to enforce "
                    "strict mode globally.",
                    DeprecationWarning,
                    stacklevel=2
                )
            return config
    return config
