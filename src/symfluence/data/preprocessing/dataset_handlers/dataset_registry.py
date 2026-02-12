"""
Dataset Registry for SYMFLUENCE

Provides a central registry for dataset preprocessing handlers.
Uses standardized BaseRegistry pattern with lowercase key normalization.
"""

from typing import Dict, Type, List

from symfluence.data.base_registry import BaseRegistry


class DatasetRegistry(BaseRegistry):
    """
    Registry for dataset preprocessing handlers.

    Handlers are registered using the @register decorator and retrieved
    using get_handler(). All keys are normalized to lowercase.
    """

    _handlers: Dict[str, Type] = {}

    @classmethod
    def get_handler(
        cls,
        name: str,
        *args,
        **kwargs
    ):
        """
        Get an instance of the appropriate dataset handler.

        Args:
            name: Name of the dataset (case-insensitive)
            *args: Positional arguments (config, logger, project_dir)
            **kwargs: Additional handler arguments

        Returns:
            Handler instance

        Raises:
            ValueError: If handler not found
        """
        handler_class = cls._get_handler_class(name)

        # Dataset handlers typically expect (config, logger, project_dir)
        # and extra kwargs for forcing_timestep_seconds etc.

        # If config is provided in args or kwargs, try to inject defaults
        config = kwargs.get('config')
        if not config and len(args) > 0:
            config = args[0]

        if config:
            kwargs.setdefault("forcing_timestep_seconds", config.get("FORCING_TIME_STEP_SIZE", 3600))

        return handler_class(*args, **kwargs)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names (alias for list_handlers)."""
        return cls.list_handlers()
