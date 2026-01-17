"""
MESH Model Optimizer

MESH-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with MESH.
"""

import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.core.file_utils import safe_delete
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry
from .worker import MESHWorker  # noqa: F401 - Import to trigger worker registration


@OptimizerRegistry.register_optimizer('MESH')
class MESHModelOptimizer(BaseModelOptimizer):
    """
    MESH-specific optimizer using the unified BaseModelOptimizer framework.

    Provides access to all optimization algorithms:
    - run_dds(): Dynamically Dimensioned Search
    - run_pso(): Particle Swarm Optimization
    - run_sce(): Shuffled Complex Evolution
    - run_de(): Differential Evolution

    Example:
        optimizer = MESHModelOptimizer(config, logger)
        results_path = optimizer.run_dds()
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        """
        Initialize MESH optimizer.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            optimization_settings_dir: Optional path to optimization settings
            reporting_manager: ReportingManager instance
        """
        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("MESHModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'MESH'

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run MESH for final evaluation."""
        return self.worker.run_model(
            self.config,
            self.project_dir / 'settings' / 'MESH',
            output_dir,
            mode='run_def'
        )

    def _get_final_file_manager_path(self) -> Path:
        """Get path to MESH input file (similar to file manager)."""
        mesh_input = self._get_config_value(lambda: self.config.model.mesh.input_file, default='MESH_input_run_options.ini', dict_key='SETTINGS_MESH_INPUT')
        if mesh_input == 'default':
            mesh_input = 'MESH_input_run_options.ini'
        return self.project_dir / 'settings' / 'MESH' / mesh_input

    def _setup_parallel_dirs(self) -> None:
        """
        Setup MESH-specific parallel directories following SUMMA pattern.

        Creates:
        - simulations/run_{experiment_id}/process_N/
          - settings/MESH/
          - simulations/{experiment_id}/MESH/
          - forcing/MESH_input/  (MESH-specific)
          - output/
        """
        base_dir = self.project_dir / 'simulations' / f'run_{self.experiment_id}'

        # Create process directories using base class method
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'MESH',
            self.experiment_id
        )

        # Copy MESH settings to each process directory
        source_settings = self.project_dir / 'settings' / 'MESH'
        if source_settings.exists():
            self.copy_base_settings(source_settings, self.parallel_dirs, 'MESH')

        # MESH-SPECIFIC: Copy forcing directory to each process
        # MESH reads from forcing/MESH_input, but worker might look in settings/MESH
        source_forcing = self.project_dir / 'forcing' / 'MESH_input'
        if source_forcing.exists():
            for proc_id, dirs in self.parallel_dirs.items():
                # Create forcing directory structure: process_N/forcing/MESH_input/
                dest_forcing = dirs['root'] / 'forcing' / 'MESH_input'
                dest_forcing.parent.mkdir(parents=True, exist_ok=True)

                if dest_forcing.exists():
                    safe_delete(dest_forcing)

                shutil.copytree(source_forcing, dest_forcing, symlinks=True)
                self.logger.debug(f"Copied MESH forcing to {dest_forcing} (preserving symlinks)")

                # ALSO copy to process_N/settings/MESH because WorkerTask sets settings_dir there
                dest_settings = dirs['root'] / 'settings' / 'MESH'
                dest_settings.mkdir(parents=True, exist_ok=True)
                for f in source_forcing.glob('*.ini'):
                    shutil.copy2(f, dest_settings / f.name)
                for f in source_forcing.glob('*.txt'):
                    shutil.copy2(f, dest_settings / f.name)

                # Update parallel_dirs to include forcing path
                dirs['forcing_dir'] = dest_forcing
                dirs['settings_dir'] = dest_settings

        # Update MESH_input_run_options.ini with process-specific paths
        self._update_mesh_run_options(self.parallel_dirs)

    def _update_mesh_run_options(
        self,
        parallel_dirs: Dict[int, Dict[str, Path]]
    ) -> None:
        """
        Update MESH_input_run_options.ini with process-specific output directories.

        MESH configuration files have strict field width limits (A10 = 10 characters).
        We keep the output directory as "./" since MESH runs from the forcing directory,
        which is already process-specific in parallel mode.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
        """
        for proc_id, dirs in parallel_dirs.items():
            forcing_dir = dirs.get('forcing_dir')
            if not forcing_dir: continue

            run_options_path = forcing_dir / 'MESH_input_run_options.ini'
            if not run_options_path.exists(): continue

            try:
                # No need to update output paths - MESH runs from forcing_dir
                # and "./" resolves correctly. Absolute paths exceed MESH's 10-char limit.
                self.logger.debug(f"MESH run options for process {proc_id} using relative output paths")
            except Exception as e:
                self.logger.error(f"Failed to verify MESH run options for process {proc_id}: {e}")
