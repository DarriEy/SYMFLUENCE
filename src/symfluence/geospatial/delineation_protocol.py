"""
Delineation Strategy Protocol.

Defines the interface that all delineation strategies must implement.
Uses Python's Protocol for structural subtyping (duck typing with type hints).

This enables:
1. Type checking without requiring inheritance
2. Clear documentation of expected interface
3. Flexibility for different delineator implementations

References:
    - PEP 544: Protocols: Structural subtyping (static duck typing)
    - https://peps.python.org/pep-0544/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable


@dataclass
class DelineationResult:
    """
    Result container for delineation operations.

    Provides a standardized way to return delineation results from any
    delineation strategy, including paths to output files and metadata.

    Attributes:
        river_basins_path: Path to basin polygons shapefile.
            - Point: Single bounding box polygon
            - Lumped: Single watershed polygon
            - Semidistributed: Multiple subcatchment polygons
            - Distributed: Grid cells as polygons

        river_network_path: Path to river network shapefile (optional).
            - Point: None (no network)
            - Lumped: Simplified main stem
            - Semidistributed: Full stream network
            - Distributed: D8 flow paths

        pour_point_path: Path to outlet point shapefile (optional).

        original_basins_path: Path to original basins before aggregation (optional).
            Used for lumped + subset workflows where basins are dissolved.

        metadata: Dictionary of method-specific configuration and results.
            Examples:
                - 'grid_cell_size': '1000.0'
                - 'geofabric_type': 'merit_basins'
                - 'delineated_river_basins_path': '/path/to/delineated.shp'

        success: Whether delineation completed successfully.

        error_message: Error message if delineation failed.

    Example:
        >>> result = DelineationResult(
        ...     river_basins_path=Path('/output/basins.shp'),
        ...     river_network_path=Path('/output/network.shp'),
        ...     metadata={'grid_cell_size': '1000.0'},
        ...     success=True
        ... )
    """
    river_basins_path: Optional[Path] = None
    river_network_path: Optional[Path] = None
    pour_point_path: Optional[Path] = None
    original_basins_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None

    def to_tuple(self) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Convert to legacy tuple format for backwards compatibility.

        Returns:
            Tuple of (river_network_path, river_basins_path)
        """
        return (self.river_network_path, self.river_basins_path)

    @classmethod
    def from_tuple(
        cls,
        paths: Tuple[Optional[Path], Optional[Path]],
        method: str = 'unknown'
    ) -> 'DelineationResult':
        """
        Create from legacy tuple format.

        Args:
            paths: Tuple of (river_network_path, river_basins_path)
            method: Delineation method name for metadata

        Returns:
            DelineationResult instance
        """
        network_path, basins_path = paths
        return cls(
            river_basins_path=basins_path,
            river_network_path=network_path,
            metadata={'method': method},
            success=basins_path is not None,
        )

    def is_complete(self) -> bool:
        """
        Check if delineation produced expected outputs.

        Returns:
            True if river_basins_path is set (minimal requirement).
        """
        return self.success and self.river_basins_path is not None


@runtime_checkable
class DelineationStrategy(Protocol):
    """
    Protocol defining the interface for delineation strategies.

    All delineation strategies (point, lumped, semidistributed, distributed)
    must implement this interface. Uses @runtime_checkable to enable
    isinstance() checks.

    Required Methods:
        delineate(): Execute the delineation workflow and return results.
        get_method_name(): Return the canonical method name.

    Optional Methods (not in protocol, but commonly implemented):
        cleanup(): Clean up intermediate files.
        validate_inputs(): Validate configuration and inputs.

    Example Implementation:
        >>> class MyDelineator:
        ...     def __init__(self, config, logger):
        ...         self.config = config
        ...         self.logger = logger
        ...
        ...     def delineate(self) -> DelineationResult:
        ...         # Implementation...
        ...         return DelineationResult(
        ...             river_basins_path=Path('/output/basins.shp'),
        ...             success=True
        ...         )
        ...
        ...     def get_method_name(self) -> str:
        ...         return 'my_method'
        >>>
        >>> delineator = MyDelineator({}, None)
        >>> isinstance(delineator, DelineationStrategy)
        True
    """

    def delineate(self) -> DelineationResult:
        """
        Execute the delineation workflow.

        Returns:
            DelineationResult containing output paths and metadata.

        Raises:
            Various exceptions depending on implementation:
            - FileNotFoundError: If required input files are missing
            - ValueError: If configuration is invalid
            - RuntimeError: If external tools (TauDEM, GDAL) fail
        """
        ...

    def get_method_name(self) -> str:
        """
        Return the canonical name for this delineation method.

        Returns:
            Method name string (e.g., 'point', 'lumped', 'semidistributed', 'distributed')
        """
        ...


class DelineationStrategyAdapter:
    """
    Adapter to convert existing delineators to the DelineationStrategy protocol.

    Provides backwards compatibility for existing delineator classes that
    don't directly implement DelineationStrategy but have compatible methods.

    Example:
        >>> from symfluence.geospatial.geofabric.delineators import LumpedWatershedDelineator
        >>> delineator = LumpedWatershedDelineator(config, logger)
        >>> adapter = DelineationStrategyAdapter(delineator, 'lumped')
        >>> result = adapter.delineate()
    """

    def __init__(
        self,
        delineator: Any,
        method_name: str,
        delineate_method: str = 'delineate_lumped_watershed'
    ):
        """
        Initialize adapter with underlying delineator.

        Args:
            delineator: Underlying delineator instance
            method_name: Canonical method name
            delineate_method: Name of the delineation method to call on underlying delineator
        """
        self._delineator = delineator
        self._method_name = method_name
        self._delineate_method = delineate_method

    def delineate(self) -> DelineationResult:
        """
        Execute delineation using underlying delineator.

        Returns:
            DelineationResult containing output paths.
        """
        try:
            method = getattr(self._delineator, self._delineate_method)
            result = method()

            # Handle different return types
            if isinstance(result, tuple) and len(result) == 2:
                network_path, basins_path = result
                return DelineationResult(
                    river_basins_path=basins_path,
                    river_network_path=network_path,
                    metadata={'method': self._method_name},
                    success=basins_path is not None,
                )
            elif isinstance(result, Path):
                return DelineationResult(
                    river_basins_path=result,
                    metadata={'method': self._method_name},
                    success=True,
                )
            elif result is None:
                return DelineationResult(
                    metadata={'method': self._method_name},
                    success=False,
                    error_message="Delineation returned None",
                )
            else:
                return DelineationResult(
                    metadata={'method': self._method_name, 'raw_result': str(result)},
                    success=True,
                )
        except Exception as e:  # noqa: BLE001 — geospatial resilience
            return DelineationResult(
                metadata={'method': self._method_name},
                success=False,
                error_message=str(e),
            )

    def get_method_name(self) -> str:
        """Return the method name."""
        return self._method_name
