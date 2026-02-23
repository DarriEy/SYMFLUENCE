"""
Build Instructions Registry for SYMFLUENCE.

Provides a central registry for model build instructions, following the
same decorator pattern as ModelRegistry. Build instructions can be
registered from model directories or kept centralized for infrastructure tools.

This module is designed to be lightweight - it does NOT import heavy
dependencies (pandas, xarray, etc.) and can be safely used by the CLI.

Example Usage:
    # In a model's build_instructions.py:
    from symfluence.cli.services import BuildInstructionsRegistry

    @BuildInstructionsRegistry.register('summa')
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
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BuildInstructionsRegistry:
    """
    Central registry for external tool build instructions.

    Implements the Registry Pattern to enable model-specific build
    instructions to be defined in their respective model directories
    while providing a unified interface for BinaryManager.

    This follows the same pattern as ModelRegistry, ObservationRegistry,
    and EvaluationRegistry used elsewhere in SYMFLUENCE.

    Class Attributes:
        _instructions: Dict[tool_name: str] -> build_instruction_dict
        _providers: Dict[tool_name: str] -> callable that returns instruction dict
    """

    _instructions: Dict[str, Dict[str, Any]] = {}
    _providers: Dict[str, Callable[[], Dict[str, Any]]] = {}

    @classmethod
    def register(cls, tool_name: str):
        """
        Decorator to register build instructions for a tool.

        The decorated function should return a dictionary with the
        standard build instruction schema.

        Args:
            tool_name: Name of the tool (e.g., 'summa', 'fuse')

        Returns:
            Decorator function

        Example:
            @BuildInstructionsRegistry.register('summa')
            def get_summa_build_instructions():
                return {
                    'description': 'SUMMA model',
                    'repository': 'https://github.com/...',
                    ...
                }
        """
        def decorator(provider_func: Callable[[], Dict[str, Any]]):
            cls._providers[tool_name.lower()] = provider_func
            return provider_func
        return decorator

    @classmethod
    def register_instructions(cls, tool_name: str, instructions: Dict[str, Any]):
        """
        Directly register build instructions (non-decorator form).

        Useful for infrastructure tools that remain centralized in
        external_tools_config.py rather than in model directories.

        Args:
            tool_name: Name of the tool
            instructions: Build instruction dictionary
        """
        cls._instructions[tool_name.lower()] = instructions

    @classmethod
    def get_instructions(cls, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get build instructions for a tool.

        First checks direct registrations, then lazy-loads from providers.

        Args:
            tool_name: Name of the tool

        Returns:
            Build instruction dictionary, or None if not found
        """
        name = tool_name.lower()

        # Check direct registrations first
        if name in cls._instructions:
            return cls._instructions[name]

        # Check providers (lazy loading)
        if name in cls._providers:
            try:
                instructions = cls._providers[name]()
                cls._instructions[name] = instructions  # Cache it
                return instructions
            except (ImportError, AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError) as e:
                logger.warning(f"Failed to load build instructions for {name}: {e}")
                return None

        return None

    @classmethod
    def get_all_instructions(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered build instructions.

        This triggers lazy loading of all provider functions.

        Returns:
            Dictionary mapping tool names to their build instructions
        """
        # Load all provider-based instructions
        for name, provider in cls._providers.items():
            if name not in cls._instructions:
                try:
                    cls._instructions[name] = provider()
                except (ImportError, AttributeError, KeyError, OSError, TypeError, ValueError, RuntimeError) as e:
                    logger.warning(f"Failed to load build instructions for {name}: {e}")

        return dict(cls._instructions)

    @classmethod
    def list_tools(cls) -> List[str]:
        """
        List all registered tool names.

        Returns:
            Sorted list of tool names
        """
        all_tools = set(cls._instructions.keys()) | set(cls._providers.keys())
        return sorted(all_tools)

    @classmethod
    def is_registered(cls, tool_name: str) -> bool:
        """
        Check if a tool has registered build instructions.

        Args:
            tool_name: Name of the tool

        Returns:
            True if the tool is registered, False otherwise
        """
        name = tool_name.lower()
        return name in cls._instructions or name in cls._providers

    @classmethod
    def clear(cls):
        """
        Clear all registrations.

        Primarily useful for testing to ensure clean state between tests.
        """
        cls._instructions.clear()
        cls._providers.clear()

    @classmethod
    def unregister(cls, tool_name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            tool_name: Name of the tool to remove

        Returns:
            True if the tool was found and removed, False otherwise
        """
        name = tool_name.lower()
        removed = False

        if name in cls._instructions:
            del cls._instructions[name]
            removed = True

        if name in cls._providers:
            del cls._providers[name]
            removed = True

        return removed
