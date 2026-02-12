"""
Preset Registry for SYMFLUENCE initialization presets.

This module provides a registry pattern for model-specific presets,
enabling each model to register its own initialization presets without
hardcoding them in the central init_presets.py file.
"""

from typing import Dict, Any, Callable, List
import logging

logger = logging.getLogger(__name__)


class PresetRegistry:
    """
    Registry for model-specific initialization presets.

    Models register their presets using the @register_preset decorator,
    enabling dynamic discovery without hardcoding preset definitions centrally.

    Example:
        >>> @PresetRegistry.register_preset('fuse-basic')
        >>> def fuse_basic_preset():
        ...     return {
        ...         'description': 'Basic FUSE setup',
        ...         'settings': {...},
        ...         'fuse_decisions': {...},
        ...     }
    """

    _presets: Dict[str, Dict[str, Any]] = {}
    _preset_loaders: Dict[str, Callable[[], Dict[str, Any]]] = {}

    @classmethod
    def register_preset(cls, name: str) -> Callable:
        """
        Decorator to register a preset loader function.

        Args:
            name: Preset name (e.g., 'fuse-basic', 'summa-distributed')

        Returns:
            Decorator function

        Example:
            >>> @PresetRegistry.register_preset('fuse-basic')
            >>> def fuse_basic_preset():
            ...     return {'description': '...', 'settings': {...}}
        """
        def decorator(loader_func: Callable[[], Dict[str, Any]]) -> Callable:
            cls._preset_loaders[name] = loader_func
            logger.debug(f"Registered preset: {name}")
            return loader_func
        return decorator

    @classmethod
    def register_preset_dict(cls, name: str, preset: Dict[str, Any]) -> None:
        """
        Register a preset directly as a dictionary.

        Args:
            name: Preset name
            preset: Preset configuration dictionary
        """
        cls._presets[name] = preset
        logger.debug(f"Registered preset dict: {name}")

    @classmethod
    def get_preset(cls, name: str) -> Dict[str, Any]:
        """
        Get a preset by name.

        Args:
            name: Preset name

        Returns:
            Preset configuration dictionary

        Raises:
            ValueError: If preset is not registered
        """
        # Ensure presets are loaded
        cls._import_model_presets()

        # Check directly registered presets first
        if name in cls._presets:
            return cls._presets[name].copy()

        # Check loader functions
        if name in cls._preset_loaders:
            return cls._preset_loaders[name]()

        available = sorted(cls.list_presets())
        raise ValueError(
            f"Unknown preset: '{name}'. Available presets: {', '.join(available)}"
        )

    @classmethod
    def list_presets(cls) -> List[str]:
        """
        List all registered preset names.

        Returns:
            List of preset names
        """
        cls._import_model_presets()
        return sorted(set(cls._presets.keys()) | set(cls._preset_loaders.keys()))

    @classmethod
    def get_all_presets(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all presets as a dictionary.

        Returns:
            Dictionary of preset_name -> preset_config
        """
        cls._import_model_presets()
        result = {}

        # Add directly registered presets
        result.update(cls._presets)

        # Add loader-based presets
        for name, loader in cls._preset_loaders.items():
            if name not in result:
                result[name] = loader()

        return result

    @classmethod
    def _import_model_presets(cls) -> None:
        """
        Import preset modules from each model directory.

        This method attempts to import the init_preset module from
        each known model package.
        """
        import logging
        from symfluence.core.constants import SupportedModels

        for model_name in SupportedModels.WITH_PRESETS:
            try:
                __import__(
                    f'symfluence.models.{model_name}.init_preset',
                    fromlist=['init_preset']
                )
            except ImportError:
                logging.getLogger(__name__).debug(
                    f"Preset module for '{model_name}' not available"
                )
