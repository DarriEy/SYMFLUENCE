"""Result Extractor Registry

Registry for model result extractors that handle model-specific output file
location and variable extraction logic. Enables standardized access to model
outputs regardless of the underlying file format or naming conventions.

Result Extractors:
    Each model's result extractor knows how to:
    - Locate output files based on model conventions
    - Extract specific variables (streamflow, SWE, soil moisture, etc.)
    - Handle time series alignment and unit conversions
    - Map model-specific variable names to standard names

Registration Pattern:
    >>> @ResultExtractorRegistry.register_result_extractor('SUMMA')
    ... class SUMMAResultExtractor(ModelResultExtractor):
    ...     def extract_variable(self, output_file, variable_type):
    ...         # SUMMA-specific extraction logic
    ...         pass
"""

import logging
from typing import Any, Callable, Dict, Optional, Type

logger = logging.getLogger(__name__)


class ResultExtractorRegistry:
    """Registry for model result extractors.

    Provides centralized storage and retrieval of result extractor classes
    for hydrological models. Each extractor handles model-specific output
    file formats and variable extraction.

    Attributes:
        _result_extractors: Dict[model_name] -> extractor_class

    Example Registration::

        @ResultExtractorRegistry.register_result_extractor('SUMMA')
        class SUMMAResultExtractor(ModelResultExtractor):
            def extract_variable(self, output_file, variable_type):
                # SUMMA-specific extraction logic
                pass

    Example Lookup::

        extractor = ResultExtractorRegistry.get_result_extractor('SUMMA')
        if extractor:
            streamflow = extractor.extract_variable(output_file, 'streamflow')
    """

    _result_extractors: Dict[str, Type] = {}

    @classmethod
    def register_result_extractor(cls, model_name: str) -> Callable[[Type], Type]:
        """Register a result extractor for a model.

        The extractor handles model-specific output file location and
        variable extraction logic.

        Args:
            model_name: Model name (e.g., 'SUMMA', 'NGEN')

        Returns:
            Decorator function that registers the extractor class

        Example:
            >>> @ResultExtractorRegistry.register_result_extractor('SUMMA')
            ... class SUMMAResultExtractor(ModelResultExtractor):
            ...     def extract_variable(self, output_file, variable_type):
            ...         # SUMMA-specific extraction logic
            ...         pass
        """
        def decorator(extractor_cls: Type) -> Type:
            cls._result_extractors[model_name.upper()] = extractor_cls
            return extractor_cls
        return decorator

    @classmethod
    def get_result_extractor(cls, model_name: str) -> Optional[Any]:
        """Get result extractor instance for a model.

        Args:
            model_name: Model name (case-insensitive via uppercase)

        Returns:
            ModelResultExtractor instance or None if not registered
        """
        extractor_cls = cls._result_extractors.get(model_name.upper())
        return extractor_cls(model_name) if extractor_cls else None

    @classmethod
    def has_result_extractor(cls, model_name: str) -> bool:
        """Check if a model has a registered result extractor.

        Args:
            model_name: Model name

        Returns:
            True if extractor is registered
        """
        return model_name.upper() in cls._result_extractors

    @classmethod
    def list_result_extractors(cls) -> list[str]:
        """List all models with registered result extractors.

        Returns:
            Sorted list of model names with result extractors
        """
        return sorted(list(cls._result_extractors.keys()))

    @classmethod
    def get_extractor_class(cls, model_name: str) -> Optional[Type]:
        """Get the result extractor class (not instance) for a model.

        Useful when you need to instantiate the extractor with custom arguments.

        Args:
            model_name: Model name (case-insensitive via uppercase)

        Returns:
            Extractor class or None if not registered
        """
        return cls._result_extractors.get(model_name.upper())
