"""
Coupled Groundwater Model Optimizer

Generic optimizer for any land surface model coupled with MODFLOW
groundwater. The land surface model is determined by LAND_SURFACE_MODEL
config key. Copies both land surface and MODFLOW settings to parallel
directories and delegates execution to CoupledGWWorker.

When dCoupler is available, uses CouplingGraphBuilder for graph-based
coupling. Otherwise falls back to sequential file-based coupling.

No external routing is needed because MODFLOW drain discharge provides
the baseflow component directly.

Config keys:
    HYDROLOGICAL_MODEL: COUPLED_GW
    LAND_SURFACE_MODEL: SUMMA, CLM, MESH, etc.
    GROUNDWATER_MODEL: MODFLOW
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.file_utils import copy_file
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import CoupledGWWorker  # noqa: F401 - triggers worker registration


@OptimizerRegistry.register_optimizer('COUPLED_GW')
class CoupledGWModelOptimizer(BaseModelOptimizer):
    """Optimizer for coupled land-surface + MODFLOW calibration.

    Dynamically resolves the land surface model from config and sets up
    parallel directories with both land surface and MODFLOW settings.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None,
    ):
        self.config = config
        self.land_model_name = self._resolve_land_model_name(config)
        super().__init__(
            config, logger, optimization_settings_dir,
            reporting_manager=reporting_manager,
        )
        self.logger.debug(
            f"CoupledGWModelOptimizer initialized with "
            f"land_surface={self.land_model_name}"
        )

    @staticmethod
    def _resolve_land_model_name(config: Dict[str, Any]) -> str:
        """Resolve land surface model name from config."""
        if isinstance(config, dict):
            return config.get('LAND_SURFACE_MODEL', 'SUMMA').upper()
        try:
            return config.model.land_surface.upper()
        except (AttributeError, TypeError):
            return 'SUMMA'

    def _get_model_name(self) -> str:
        return 'COUPLED_GW'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to the land surface model's file manager for final evaluation."""
        # SUMMA uses fileManager.txt; other models may differ
        if self.land_model_name == 'SUMMA':
            summa_fm = self._get_config_value(
                lambda: self.config.model.summa.filemanager,
                default='fileManager.txt',
                dict_key='SETTINGS_SUMMA_FILEMANAGER',
            )
            if summa_fm == 'default':
                summa_fm = 'fileManager.txt'
            return self.project_dir / 'settings' / 'SUMMA' / summa_fm

        # For other models, return a generic path
        return self.project_dir / 'settings' / self.land_model_name

    def _create_parameter_manager(self):
        """Create joint land-surface + MODFLOW parameter manager."""
        from .parameter_manager import CoupledGWParameterManager
        settings_dir = self.project_dir / 'settings'
        return CoupledGWParameterManager(
            self.config, self.logger, settings_dir,
        )

    def _check_routing_needed(self) -> bool:
        """No external routing needed; MODFLOW provides baseflow directly."""
        return False

    def _ensure_modflow_settings(self) -> None:
        """Generate MODFLOW settings files if they don't exist."""
        source_modflow = self.project_dir / 'settings' / 'MODFLOW'
        if source_modflow.exists() and any(source_modflow.iterdir()):
            return

        self.logger.info("MODFLOW settings not found, generating from config...")
        from symfluence.models.modflow.preprocessor import MODFLOWPreProcessor
        preprocessor = MODFLOWPreProcessor(self.config, self.logger)
        # Ensure config_dict is populated for dict-based configs
        if isinstance(self.config, dict) and not preprocessor.config_dict:
            preprocessor.config_dict = self.config
        preprocessor.project_dir = self.project_dir
        preprocessor.run_preprocessing()

    def _setup_parallel_dirs(self) -> None:
        """Setup parallel directories with both land surface and MODFLOW settings."""
        self._ensure_modflow_settings()

        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            default='optimization',
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()

        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir, 'COUPLED_GW', self.experiment_id,
        )

        # Add model-specific directory aliases
        for proc_id, dirs in self.parallel_dirs.items():
            dirs['land_dir'] = dirs['sim_dir'] / self.land_model_name
            dirs['modflow_dir'] = dirs['sim_dir'] / 'MODFLOW'
            dirs['land_settings_dir'] = dirs['settings_dir'] / self.land_model_name
            dirs['modflow_settings_dir'] = dirs['settings_dir'] / 'MODFLOW'

            dirs['land_dir'].mkdir(parents=True, exist_ok=True)
            dirs['modflow_dir'].mkdir(parents=True, exist_ok=True)
            dirs['land_settings_dir'].mkdir(parents=True, exist_ok=True)
            dirs['modflow_settings_dir'].mkdir(parents=True, exist_ok=True)

        # Copy land surface model settings
        source_land = self.project_dir / 'settings' / self.land_model_name
        if source_land.exists():
            for proc_id, dirs in self.parallel_dirs.items():
                dest = dirs['land_settings_dir']
                for item in source_land.iterdir():
                    if item.is_file():
                        copy_file(item, dest / item.name)

            # Update file managers for SUMMA (other models may not need this)
            if self.land_model_name == 'SUMMA':
                fm_name = self._get_config_value(
                    lambda: self.config.model.summa.filemanager,
                    default='fileManager.txt',
                    dict_key='SETTINGS_SUMMA_FILEMANAGER',
                )
                if fm_name == 'default':
                    fm_name = 'fileManager.txt'
                # Build dirs with settings_dir and sim_dir pointing to SUMMA
                # subdirectories so update_file_managers finds the fileManager
                # and sets correct output paths
                land_dirs = {}
                for proc_id, dirs in self.parallel_dirs.items():
                    land_dirs[proc_id] = dict(dirs)
                    land_dirs[proc_id]['settings_dir'] = dirs['land_settings_dir']
                    land_dirs[proc_id]['sim_dir'] = dirs['land_dir']
                self.update_file_managers(
                    land_dirs,
                    'SUMMA',
                    self.experiment_id,
                    fm_name,
                )

        # Copy MODFLOW settings
        source_modflow = self.project_dir / 'settings' / 'MODFLOW'
        if source_modflow.exists():
            for proc_id, dirs in self.parallel_dirs.items():
                dest = dirs['modflow_settings_dir']
                for item in source_modflow.iterdir():
                    if item.is_file():
                        copy_file(item, dest / item.name)

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run coupled model for final evaluation."""
        settings_dir = self.project_dir / 'settings'
        return self.worker.run_model(
            self.config, settings_dir, output_dir,
        )

    def _update_file_manager_for_final_run(self) -> None:
        """Update file manager for final evaluation output path."""
        pass

    def _restore_file_manager_for_optimization(self) -> None:
        """Restore file manager after final evaluation."""
        pass
