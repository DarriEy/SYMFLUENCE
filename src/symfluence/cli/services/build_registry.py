# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Build instructions resolution for SYMFLUENCE.

Lightweight accessors over ``R.build_instructions`` for external-tool build
instructions. A model registers its build instructions with
``@R.build_instructions.add('toolname')`` on a provider function (or via
``model_manifest(build_instructions_module=...)`` for lazy module resolution);
this module resolves those entries into plain instruction dicts on access.

This module is intentionally lightweight — it does NOT import heavy
dependencies (pandas, xarray, etc.) and can be safely used by the CLI.

Example Usage:
    # In a model's build_instructions.py:
    from symfluence.core.registries import R

    @R.build_instructions.add('summa')
    def get_summa_build_instructions():
        return {
            'description': 'SUMMA model',
            'repository': 'https://github.com/...',
            ...
        }

    # In CLI code:
    from symfluence.cli.services import BuildInstructionsRegistry

    all_tools = BuildInstructionsRegistry.get_all_instructions()
    summa_config = BuildInstructionsRegistry.get_instructions('summa')
"""

import logging
from typing import Any, Dict, List, Optional

from symfluence.core.registries import R

logger = logging.getLogger(__name__)

_RESOLVE_ERRORS = (ImportError, AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError)


class BuildInstructionsRegistry:
    """Resolution helpers over ``R.build_instructions``.

    Entries are registered as either a provider callable (returns the
    instruction dict) or an already-resolved dict; this class resolves and
    caches callables on first access.
    """

    @classmethod
    def register(cls, tool_name: str):
        """Decorator registering a provider function under *tool_name*.

        The decorated function returns the build-instruction dict; it is
        resolved lazily on first access. Equivalent to
        ``@R.build_instructions.add(tool_name)``.
        """
        def decorator(provider_func):
            R.build_instructions.add(tool_name, provider_func)
            return provider_func
        return decorator

    @classmethod
    def register_instructions(cls, tool_name: str, instructions: Dict[str, Any]) -> None:
        """Directly register a resolved build-instruction dict under *tool_name*.

        Used for infrastructure tools defined centrally rather than in a model
        package. Equivalent to ``R.build_instructions.add(tool_name, instructions)``.
        """
        R.build_instructions.add(tool_name, instructions)

    @classmethod
    def get_instructions(cls, tool_name: str) -> Optional[Dict[str, Any]]:
        """Return build instructions for *tool_name*, or ``None`` if not found."""
        name = tool_name.lower()
        value = R.build_instructions.get(name)
        if value is None:
            return None
        if callable(value) and not isinstance(value, type):
            try:
                instructions = value()
                R.build_instructions.add(name, instructions)
                return instructions
            except _RESOLVE_ERRORS as e:
                logger.warning(f"Failed to load build instructions for {name}: {e}")
                return None
        return value

    @classmethod
    def get_all_instructions(cls) -> Dict[str, Dict[str, Any]]:
        """Return all registered build instructions, resolving provider callables."""
        result: Dict[str, Dict[str, Any]] = {}
        for key, value in R.build_instructions.items():
            if callable(value) and not isinstance(value, type):
                try:
                    resolved = value()
                    R.build_instructions.add(key, resolved)
                    result[key.lower()] = resolved
                except _RESOLVE_ERRORS as e:
                    logger.warning(f"Failed to load build instructions for {key}: {e}")
            else:
                result[key.lower()] = value
        return result

    @classmethod
    def list_tools(cls) -> List[str]:
        """Return a sorted list of registered tool names."""
        return sorted(k.lower() for k in R.build_instructions.keys())

    @classmethod
    def is_registered(cls, tool_name: str) -> bool:
        """Return whether *tool_name* has registered build instructions."""
        return tool_name.lower() in R.build_instructions

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (primarily for test isolation)."""
        R.build_instructions.clear()

    @classmethod
    def unregister(cls, tool_name: str) -> bool:
        """Remove *tool_name* from the registry; return whether it existed."""
        name = tool_name.lower()
        if name in R.build_instructions:
            R.build_instructions.remove(name)
            return True
        return False
