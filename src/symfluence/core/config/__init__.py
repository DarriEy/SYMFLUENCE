from typing import Union, Dict, Any
from symfluence.core.config.config_loader import normalize_config, validate_config
from symfluence.core.config.models import SymfluenceConfig


def ensure_typed_config(config: Union[Dict[str, Any], SymfluenceConfig]) -> SymfluenceConfig:
    """
    Ensure configuration is a SymfluenceConfig instance.

    This adapter function converts dict configs to SymfluenceConfig if needed.
    Use this when interfacing with external code that may pass dict configs.

    Args:
        config: Configuration as dict or SymfluenceConfig

    Returns:
        SymfluenceConfig instance

    Example:
        >>> config = ensure_typed_config({'DOMAIN_NAME': 'test', ...})
        >>> isinstance(config, SymfluenceConfig)
        True
    """
    if isinstance(config, SymfluenceConfig):
        return config
    return SymfluenceConfig(**config)


__all__ = [
    # Hierarchical config system (recommended)
    "SymfluenceConfig",
    "ensure_typed_config",

    # Config utility functions
    "normalize_config",
    "validate_config",
]
