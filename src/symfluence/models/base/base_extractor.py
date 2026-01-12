"""
Model Result Extractor Base Interface.

Provides abstract interface for model-specific result extraction from
simulation outputs. Each model implements this interface to handle its
unique output formats, variable names, and data structures.

This separates model-specific extraction logic from generic evaluation
components, enabling the core evaluation system to remain model-agnostic.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd


class ModelResultExtractor(ABC):
    """Abstract base for model-specific result extraction.

    Each hydrological model has unique output characteristics:
    - Different file formats (NetCDF, CSV, text)
    - Different variable naming conventions
    - Different spatial dimensions (HRU, GRU, reach, segment)
    - Different unit conventions

    This adapter interface allows each model to define its own extraction
    logic while providing a consistent interface for evaluators.

    Attributes:
        model_name: Name of the model (e.g., 'SUMMA', 'NGEN')
    """

    def __init__(self, model_name: str):
        """Initialize extractor.

        Args:
            model_name: Model identifier
        """
        self.model_name = model_name.upper()

    @abstractmethod
    def get_output_file_patterns(self) -> Dict[str, List[str]]:
        """Get file patterns for locating model outputs.

        Returns dict mapping output types to glob patterns:
        {
            'streamflow': ['*_streamflow.nc', 'output/*.nc'],
            'snow': ['*_swe.nc'],
            ...
        }

        Returns:
            Dict mapping variable types to file patterns
        """
        pass

    @abstractmethod
    def extract_variable(
        self,
        output_file: Path,
        variable_type: str,
        **kwargs
    ) -> pd.Series:
        """Extract a variable time series from model output.

        Args:
            output_file: Path to model output file
            variable_type: Type of variable ('streamflow', 'snow', 'et', etc.)
            **kwargs: Additional extraction parameters (e.g., spatial_unit)

        Returns:
            Time series of extracted variable

        Raises:
            ValueError: If variable cannot be extracted
        """
        pass

    def get_variable_names(self, variable_type: str) -> List[str]:
        """Get possible variable names for a given type in model outputs.

        Override this to provide model-specific variable name mappings.

        Args:
            variable_type: Generic variable type ('streamflow', 'snow', etc.)

        Returns:
            List of possible variable names in model output files
        """
        return [variable_type]

    def requires_unit_conversion(self, variable_type: str) -> bool:
        """Check if variable requires unit conversion.

        Override this if model outputs require systematic unit conversions.

        Args:
            variable_type: Variable type to check

        Returns:
            True if unit conversion needed
        """
        return False

    def get_spatial_aggregation_method(self, variable_type: str) -> Optional[str]:
        """Get spatial aggregation method for distributed models.

        Override this for distributed models that need spatial aggregation.

        Args:
            variable_type: Variable type

        Returns:
            Aggregation method: 'mean', 'sum', 'weighted', or None
        """
        return None


class ConfigValidationError(Exception):
    """Raised when model configuration validation fails."""
    pass
