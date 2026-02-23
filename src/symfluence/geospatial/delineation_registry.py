"""
Delineation Strategy Registry.

Provides a registry for delineation strategies, enabling dynamic method
registration and lookup. Used by DomainDelineator to route to appropriate
delineation implementations.

Pattern:
    This implements the Registry Pattern, allowing delineators to self-register
    using decorators. Combined with the Strategy Pattern in DomainDelineator,
    this enables clean separation of concerns and easy extension.

Usage:
    # In delineator module:
    @DelineationRegistry.register('lumped')
    class LumpedWatershedDelineator(BaseGeofabricDelineator):
        ...

    # In orchestrator:
    strategy_cls = DelineationRegistry.get_strategy('lumped')
    strategy = strategy_cls(config, logger)
    result = strategy.delineate()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Type

if TYPE_CHECKING:
    pass


class DelineationRegistry:
    """
    Registry for delineation strategy classes.

    Maintains a mapping from method names (e.g., 'point', 'lumped',
    'semidistributed', 'distributed') to their implementing classes.

    Attributes:
        _strategies: Dictionary mapping method names to strategy classes.
        _aliases: Dictionary mapping alternate names to canonical names.

    Example:
        >>> @DelineationRegistry.register('lumped')
        ... class LumpedDelineator:
        ...     pass
        >>> DelineationRegistry.get_strategy('lumped')
        <class 'LumpedDelineator'>
    """

    _strategies: Dict[str, Type] = {}
    _aliases: Dict[str, str] = {
        # Backwards compatibility aliases
        'delineate': 'semidistributed',
        'distribute': 'distributed',
        'subset': 'semidistributed',  # subset is semidistributed with flag
        'discretized': 'semidistributed',  # deprecated
    }

    @classmethod
    def register(cls, method_name: str) -> Callable[[Type], Type]:
        """
        Decorator to register a delineation strategy class.

        Args:
            method_name: Canonical name for the delineation method
                (e.g., 'point', 'lumped', 'semidistributed', 'distributed')

        Returns:
            Decorator function that registers the class and returns it unchanged.

        Example:
            >>> @DelineationRegistry.register('point')
            ... class PointDelineator:
            ...     def delineate(self):
            ...         pass
        """
        def decorator(strategy_cls: Type) -> Type:
            cls._strategies[method_name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def get_strategy(cls, method_name: str) -> Optional[Type]:
        """
        Get strategy class by method name.

        Supports both canonical names and legacy aliases.

        Args:
            method_name: Delineation method name (e.g., 'lumped', 'delineate')

        Returns:
            Strategy class if found, None otherwise.

        Example:
            >>> cls = DelineationRegistry.get_strategy('lumped')
            >>> cls is not None
            True
        """
        # Normalize method name
        normalized = method_name.lower().strip()

        # Check for direct match first
        if normalized in cls._strategies:
            return cls._strategies[normalized]

        # Check aliases
        canonical = cls._aliases.get(normalized)
        if canonical and canonical in cls._strategies:
            return cls._strategies[canonical]

        return None

    @classmethod
    def list_methods(cls) -> List[str]:
        """
        List all registered delineation method names.

        Returns:
            List of registered method names (canonical names only).

        Example:
            >>> methods = DelineationRegistry.list_methods()
            >>> 'lumped' in methods
            True
        """
        return list(cls._strategies.keys())

    @classmethod
    def list_aliases(cls) -> Dict[str, str]:
        """
        List all method name aliases.

        Returns:
            Dictionary mapping aliases to canonical names.
        """
        return cls._aliases.copy()

    @classmethod
    def is_registered(cls, method_name: str) -> bool:
        """
        Check if a method name is registered (including aliases).

        Args:
            method_name: Method name to check

        Returns:
            True if method is registered or aliased.
        """
        normalized = method_name.lower().strip()
        return normalized in cls._strategies or normalized in cls._aliases

    @classmethod
    def get_canonical_name(cls, method_name: str) -> Optional[str]:
        """
        Get canonical method name from potentially aliased name.

        Args:
            method_name: Method name (canonical or alias)

        Returns:
            Canonical name if found, None otherwise.

        Example:
            >>> DelineationRegistry.get_canonical_name('delineate')
            'semidistributed'
        """
        normalized = method_name.lower().strip()

        if normalized in cls._strategies:
            return normalized

        return cls._aliases.get(normalized)

    @classmethod
    def add_alias(cls, alias: str, canonical: str) -> None:
        """
        Add a new alias for a canonical method name.

        Args:
            alias: Alias name to add
            canonical: Canonical method name this alias refers to

        Raises:
            ValueError: If canonical name is not registered.
        """
        if canonical not in cls._strategies:
            raise ValueError(
                f"Cannot add alias '{alias}' for unregistered method '{canonical}'"
            )
        cls._aliases[alias.lower().strip()] = canonical

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered strategies.

        Primarily for testing purposes.
        """
        cls._strategies.clear()
        # Reset aliases to defaults
        cls._aliases = {
            'delineate': 'semidistributed',
            'distribute': 'distributed',
            'subset': 'semidistributed',
            'discretized': 'semidistributed',
        }
