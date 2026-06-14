# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Preset Registry for SYMFLUENCE initialization presets.

This module provides a registry pattern for model-specific presets,
enabling each model to register its own initialization presets without
hardcoding them in the central init_presets.py file.

Lookup facade: presets register via ``R.presets.add()`` (either a resolved
dict or a lazy loader callable); lookups here read from ``R.presets`` and
resolve loaders on first access.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from symfluence.core.registries import R

logger = logging.getLogger(__name__)


class PresetRegistry:
    """
    Registry for model-specific initialization presets.

    Presets register via the unified registry (``@R.presets.add('fuse-basic')``);
    lookups here read from ``R.presets``.

    Example:
        >>> @R.presets.add('fuse-basic')
        >>> def fuse_basic_preset():
        ...     return {
        ...         'description': 'Basic FUSE setup',
        ...         'settings': {...},
        ...         'fuse_decisions': {...},
        ...     }
    """

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
        return sorted(R.presets.keys())

    @classmethod
    def get_all_presets(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all presets as a dictionary.

        Returns:
            Dictionary of preset_name -> preset_config
        """
        cls._import_model_presets()
        result = {}

        # Build result from R.presets, resolving lazy loaders
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

        from symfluence.models import model_packages_with

        for model_name in model_packages_with('init_preset'):
            try:
                __import__(
                    f'symfluence.models.{model_name}.init_preset',
                    fromlist=['init_preset']
                )
            except ImportError:
                logging.getLogger(__name__).debug(
                    f"Preset module for '{model_name}' not available"
                )
