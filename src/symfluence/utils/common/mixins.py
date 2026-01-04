"""
Core mixins for SYMFLUENCE modules.

Provides base mixins for logging, configuration, and project context that
other utilities and mixins can build upon.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union


class LoggingMixin:
    """
    Mixin providing standardized logger access.

    Ensures a logger is always available, defaulting to one named after the
    class if none is explicitly set.
    """

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        if not hasattr(self, '_logger') or self._logger is None:
            # Create a default logger if none exists
            module = self.__class__.__module__
            name = self.__class__.__name__
            self._logger = logging.getLogger(f"{module}.{name}")
        return self._logger

    @logger.setter
    def logger(self, value: logging.Logger) -> None:
        """Set the logger instance."""
        self._logger = value


class ConfigMixin:
    """
    Mixin for classes that use configuration settings.

    Provides a standardized way to access configuration regardless of whether
    it's a dictionary or a typed SymfluenceConfig object.
    """

    @property
    def config_dict(self) -> Dict[str, Any]:
        """
        Standardized access to configuration as a dictionary.

        Handles both dict-based and SymfluenceConfig-based configurations.
        """
        # Prioritize _config_dict if set directly (backward compatibility)
        if hasattr(self, '_config_dict'):
            return self._config_dict

        # Try self.typed_config or self.config
        cfg = getattr(self, 'typed_config', getattr(self, 'config', None))
        
        if cfg is not None:
            if hasattr(cfg, 'to_dict'):
                # Prioritize to_dict(flatten=True) if available
                try:
                    return cfg.to_dict(flatten=True)
                except (TypeError, AttributeError):
                    try:
                        return cfg.to_dict()
                    except (TypeError, AttributeError):
                        return {}
            if isinstance(cfg, dict):
                return cfg

        return {}

    @config_dict.setter
    def config_dict(self, value: Dict[str, Any]) -> None:
        """Set the configuration dictionary (backward compatibility)."""
        self._config_dict = value


class ProjectContextMixin(ConfigMixin):


    """


    Mixin providing standard project context attributes.





    Extracts core project parameters (data_dir, domain_name, project_dir)


    from the configuration, providing a consistent interface across modules.


    """





    @property


    def data_dir(self) -> Path:


        """Root data directory from configuration."""


        if hasattr(self, '_data_dir'):


            return self._data_dir


        return Path(self.config_dict.get('SYMFLUENCE_DATA_DIR', '.'))





    @data_dir.setter


    def data_dir(self, value: Union[str, Path]) -> None:


        """Set the data directory."""


        self._data_dir = Path(value)





    @property


    def domain_name(self) -> str:


        """Domain name from configuration."""


        if hasattr(self, '_domain_name'):


            return self._domain_name


        return self.config_dict.get('DOMAIN_NAME', 'domain')





    @domain_name.setter


    def domain_name(self, value: str) -> None:


        """Set the domain name."""


        self._domain_name = value





    @property


    def project_dir(self) -> Path:


        """


        Resolved project directory: {data_dir}/domain_{domain_name}.


        """


        if hasattr(self, '_project_dir'):


            return self._project_dir


        return self.data_dir / f"domain_{self.domain_name}"





    @project_dir.setter


    def project_dir(self, value: Union[str, Path]) -> None:


        """Set the project directory."""


        self._project_dir = Path(value)

