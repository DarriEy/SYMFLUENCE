"""
Preset Registry for SYMFLUENCE initialization presets.

This module provides a registry pattern for model-specific presets,
enabling each model to register its own initialization presets without
hardcoding them in the central init_presets.py file.

Phase 4 delegation shim: resolved presets live in ``R.presets``.  The
``_preset_loaders`` dict is kept locally for lazy loader execution.
"""

import logging
import warnings
from typing import Any, Callable, Dict, List

from symfluence.core.registries import R

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

    # Keep loaders locally for lazy execution; resolved values live in
    # R.presets.
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
            warnings.warn(
                "PresetRegistry.register_preset() is deprecated; "
                "use R.presets.add() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cls._preset_loaders[name] = loader_func
            logger.debug(f"Registered preset: {name}")
            R.presets.add(name, loader_func)
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
        warnings.warn(
            "PresetRegistry.register_preset_dict() is deprecated; "
            "use R.presets.add() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.debug(f"Registered preset dict: {name}")
        R.presets.add(name, preset)

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

        # Check unified registry
        value = R.presets.get(name)
        if value is not None:
            # If stored value is a callable loader, execute it
            if callable(value) and not isinstance(value, type):
                result = value()
                R.presets.add(name, result)
                return result.copy() if isinstance(result, dict) else result
            return value.copy() if isinstance(value, dict) else value

        # Check local loaders
        if name in cls._preset_loaders:
            result = cls._preset_loaders[name]()
            R.presets.add(name, result)
            return result

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
        all_names = set(R.presets.keys())
        all_names |= set(cls._preset_loaders.keys())
        return sorted(all_names)

    @classmethod
    def get_all_presets(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all presets as a dictionary.

        Returns:
            Dictionary of preset_name -> preset_config
        """
        cls._import_model_presets()
        result = {}

        # Resolve any local loaders not yet in R.presets
        for name, loader in cls._preset_loaders.items():
            if R.presets.get(name) is None or (
                callable(R.presets.get(name))
                and not isinstance(R.presets.get(name), type)
            ):
                try:
                    resolved = loader()
                    R.presets.add(name, resolved)
                except Exception:  # noqa: BLE001
                    logger.debug(f"Failed to load preset: {name}")

        # Build result from R.presets
        for key, value in R.presets.items():
            if callable(value) and not isinstance(value, type):
                try:
                    resolved = value()
                    R.presets.add(key, resolved)
                    result[key] = resolved
                except Exception:  # noqa: BLE001
                    logger.debug(f"Failed to load preset: {key}")
            else:
                result[key] = value

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
