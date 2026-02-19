"""
Optimizer Component Factory

Registry-based factory for creating parameter managers, workers,
and calibration targets using convention-over-configuration.
"""

import logging
from pathlib import Path
from typing import Any, TYPE_CHECKING

from symfluence.optimization.registry import OptimizerRegistry

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig
    from ..workers.base_worker import BaseWorker


class OptimizerComponentFactory:
    """Creates optimizer components via registry discovery.

    Encapsulates the convention-over-configuration logic for locating
    and instantiating parameter managers, workers, and calibration targets.

    Args:
        config: Typed SymfluenceConfig
        logger: Logger instance
        project_dir: Path to domain project directory
    """

    def __init__(
        self,
        config: 'SymfluenceConfig',
        logger: logging.Logger,
        project_dir: Path
    ):
        self.config = config
        self.logger = logger
        self.project_dir = project_dir

    def create_parameter_manager(self, model_name: str) -> Any:
        """Create parameter manager from registry.

        Args:
            model_name: Model identifier (e.g. 'SUMMA', 'MESH')

        Returns:
            ParameterManager instance

        Raises:
            RuntimeError: If no parameter manager is registered for model
        """
        param_manager_cls = OptimizerRegistry.get_parameter_manager(model_name)

        if param_manager_cls is None:
            raise RuntimeError(
                f"No parameter manager registered for model '{model_name}'. "
                f"Ensure the parameter manager is decorated with "
                f"@OptimizerRegistry.register_parameter_manager('{model_name}')"
            )

        settings_dir = self.get_settings_directory(model_name)

        self.logger.debug(
            f"Creating parameter manager: {param_manager_cls.__name__} "
            f"for {model_name} at {settings_dir}"
        )

        return param_manager_cls(self.config, self.logger, settings_dir)

    def create_worker(self, model_name: str) -> 'BaseWorker':
        """Create worker from registry.

        Args:
            model_name: Model identifier

        Returns:
            BaseWorker instance

        Raises:
            RuntimeError: If no worker is registered for model
        """
        worker_cls = OptimizerRegistry.get_worker(model_name)

        if worker_cls is None:
            raise RuntimeError(
                f"No worker registered for model '{model_name}'. "
                f"Ensure the worker is decorated with "
                f"@OptimizerRegistry.register_worker('{model_name}')"
            )

        self.logger.debug(f"Creating worker: {worker_cls.__name__} for {model_name}")
        return worker_cls(self.config, self.logger)

    def create_calibration_target(self, model_name: str, target_type: str) -> Any:
        """Create calibration target from centralized factory.

        Args:
            model_name: Model identifier
            target_type: Target type (e.g. 'streamflow', 'swe')

        Returns:
            CalibrationTarget instance
        """
        from symfluence.optimization.calibration_targets import create_calibration_target

        return create_calibration_target(
            model_name=model_name,
            target_type=target_type,
            config=self.config,
            project_dir=self.project_dir,
            logger=self.logger
        )

    def get_settings_directory(self, model_name: str) -> Path:
        """Get model settings directory using convention.

        Convention: {project_dir}/settings/{MODEL_NAME}/

        Args:
            model_name: Model identifier

        Returns:
            Path to model settings directory
        """
        return self.project_dir / 'settings' / model_name
