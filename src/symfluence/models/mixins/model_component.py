"""
Mixin for shared model component initialization.

Provides common initialization logic for BaseModelRunner, BaseModelPreProcessor,
and BaseModelPostProcessor to eliminate code duplication.
"""

import logging
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class ModelComponentMixin:
    """
    Shared initialization for model components (Runner, PreProcessor, PostProcessor).

    This mixin consolidates the common initialization pattern used across all three
    base model classes:
    - Config conversion (dict to SymfluenceConfig)
    - Logger and reporting_manager assignment
    - Required config validation
    - Base path setup (data_dir, domain_name, project_dir)
    - Model name resolution

    Usage:
        class MyBaseClass(ABC, ModelComponentMixin, PathResolverMixin):
            def __init__(self, config, logger, reporting_manager=None):
                self._init_model_component(config, logger, reporting_manager)
                # Additional class-specific initialization...
    """

    def _init_model_component(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None
    ) -> None:
        """
        Initialize common model component attributes.

        Args:
            config: SymfluenceConfig instance or dict (auto-converted)
            logger: Logger instance
            reporting_manager: Optional ReportingManager instance
        """
        # Import here to avoid circular imports at module level
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)  # type: ignore[misc]

        self.logger = logger
        self.reporting_manager = reporting_manager

        # Validate required configuration keys
        # This calls the subclass's _validate_required_config method
        self._validate_required_config()

        # Base paths - direct typed access
        self.data_dir = self.config.system.data_dir
        self.domain_name = self.config.domain.name
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"

        # Model-specific initialization
        self.model_name = self._get_model_name()
