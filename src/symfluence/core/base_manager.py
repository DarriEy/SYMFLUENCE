"""
Base Manager class for SYMFLUENCE managers.

Provides standardized initialization, configuration handling, and common
orchestration patterns for DataManager, ModelManager, and OptimizationManager.
"""

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from symfluence.core.config.coercion import ensure_config
from symfluence.core.exceptions import symfluence_error_handler
from symfluence.core.mixins import ConfigurableMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseManager(ConfigurableMixin, ABC):
    """
    Abstract base class for all SYMFLUENCE managers.

    Provides standardized:
    - Initialization with config validation and auto-conversion
    - Common service orchestration patterns
    - Consistent logging and error handling
    - Shared utility methods

    All manager classes (DataManager, ModelManager, OptimizationManager) should
    inherit from this class to ensure consistent behavior.

    Attributes:
        config: SymfluenceConfig instance (via ConfigMixin property)
        logger: Logger instance
        reporting_manager: Optional ReportingManager for visualization
        experiment_id: Current experiment identifier
    """

    def __init__(
        self,
        config: 'SymfluenceConfig',
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize the base manager.

        Args:
            config: SymfluenceConfig instance
            logger: Logger instance
            reporting_manager: Optional ReportingManager instance

        Raises:
            TypeError: If config cannot be converted to SymfluenceConfig
        """
        self._config = ensure_config(config)

        self.logger = logger
        self.reporting_manager = reporting_manager

        # Allow subclasses to initialize their services
        self._initialize_services()

    def _initialize_services(self) -> None:
        """
        Initialize manager-specific services.

        Override in subclasses to set up services needed by the manager.
        Called at the end of __init__ after all base attributes are set.

        Example:
            def _initialize_services(self):
                self.acquisition_service = AcquisitionService(self.config, self.logger)
        """
        pass

    def _execute_workflow(
        self,
        items: List[str],
        handler: Callable[[str], Any],
        operation_name: str = "workflow"
    ) -> List[Any]:
        """
        Execute a standardized workflow over a list of items.

        Provides consistent error handling and logging for batch operations.

        Args:
            items: List of items to process (e.g., model names, observation types)
            handler: Function to call for each item
            operation_name: Name of the operation for logging

        Returns:
            List of results from successful handler calls

        Raises:
            Exception: Re-raises exceptions from handler after logging
        """
        results = []
        self.logger.debug(f"Starting {operation_name} for: {items}")

        for item in items:
            with symfluence_error_handler(
                f"processing {item} in {operation_name}",
                self.logger
            ):
                self.logger.debug(f"Processing {item}")
                result = handler(item)
                if result is not None:
                    results.append(result)

        self.logger.debug(f"Completed {operation_name}: {len(results)} successful")
        return results

    def _safe_visualize(
        self,
        viz_func: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Optional[Any]:
        """
        Safely call a visualization function with error handling.

        If reporting_manager is not available or visualization fails,
        logs a warning instead of raising an exception.

        Args:
            viz_func: Visualization function to call
            *args: Positional arguments for viz_func
            **kwargs: Keyword arguments for viz_func

        Returns:
            Result of viz_func if successful, None otherwise
        """
        if not self.reporting_manager:
            self.logger.debug("Visualization skipped - reporting manager not available")
            return None

        try:
            return viz_func(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
            return None

    def _get_service(
        self,
        service_class: type,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Factory method for creating services with consistent logging.

        Args:
            service_class: Class to instantiate
            *args: Positional arguments for service constructor
            **kwargs: Keyword arguments for service constructor

        Returns:
            Instance of service_class
        """
        self.logger.debug(f"Initializing {service_class.__name__}")
        return service_class(*args, **kwargs)

    def validate_readiness(self) -> Dict[str, bool]:
        """
        Validate that this manager is ready for execution.

        Override in subclasses to check runtime prerequisites:
        file existence, directory state, external dependencies.

        Config-level validation should NOT be here — that belongs
        in Pydantic validators (core/config/models/).

        Returns:
            Dict mapping check names to pass/fail booleans.
            Empty dict means no runtime checks needed.
        """
        return {}

    def get_status(self) -> Dict[str, Any]:
        """
        Get status information for this manager.

        Override in subclasses to provide manager-specific status.

        Returns:
            Dictionary containing status information
        """
        return {
            'manager': self.__class__.__name__,
            'project_dir': str(self.project_dir),
            'experiment_id': self.experiment_id,
        }
