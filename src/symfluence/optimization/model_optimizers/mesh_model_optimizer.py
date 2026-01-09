"""
MESH Model Optimizer

MESH-specific optimizer inheriting from BaseModelOptimizer.
Provides unified interface for all optimization algorithms with MESH.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from ..optimizers.base_model_optimizer import BaseModelOptimizer
from ..workers.mesh_worker import MESHWorker
from ..registry import OptimizerRegistry


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

        self.logger.info(f"MESHModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'MESH'

    def _create_parameter_manager(self):
        """Create MESH parameter manager."""
        from ..parameter_managers import MESHParameterManager
        return MESHParameterManager(
            self.config,
            self.logger,
            self.optimization_settings_dir
        )

    def _create_calibration_target(self):
        """Create MESH calibration target based on configuration."""
        from ..calibration_targets import (
            StreamflowTarget, MultivariateTarget
        )

        target_type = self.config.get('OPTIMIZATION_TARGET', 'streamflow').lower()

        if target_type == 'multivariate':
            return MultivariateTarget(self.config, self.project_dir, self.logger)
        else:
            return StreamflowTarget(self.config, self.project_dir, self.logger)

    def _create_worker(self) -> MESHWorker:
        """Create MESH worker."""
        return MESHWorker(self.config, self.logger)

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
        mesh_input = self.config.get('SETTINGS_MESH_INPUT', 'MESH_input_run_options.ini')
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
        # MESH reads from forcing/MESH_input, not settings
        source_forcing = self.project_dir / 'forcing' / 'MESH_input'
        if source_forcing.exists():
            import shutil
            for proc_id, dirs in self.parallel_dirs.items():
                # Create forcing directory structure: process_N/forcing/MESH_input/
                dest_forcing = dirs['root'] / 'forcing' / 'MESH_input'
                dest_forcing.parent.mkdir(parents=True, exist_ok=True)

                if dest_forcing.exists():
                    shutil.rmtree(dest_forcing)

                shutil.copytree(source_forcing, dest_forcing, symlinks=True)
                self.logger.debug(f"Copied MESH forcing to {dest_forcing} (preserving symlinks)")

                # Update parallel_dirs to include forcing path
                dirs['forcing_dir'] = dest_forcing

        # Update MESH_input_run_options.ini with process-specific paths
        self._update_mesh_run_options(self.parallel_dirs)

    def _update_mesh_run_options(
        self,
        parallel_dirs: Dict[int, Dict[str, Path]]
    ) -> None:
        """
        Update MESH_input_run_options.ini with process-specific output directories.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
        """
        for proc_id, dirs in parallel_dirs.items():
            forcing_dir = dirs.get('forcing_dir')
            if not forcing_dir: continue

            run_options_path = forcing_dir / 'MESH_input_run_options.ini'
            if not run_options_path.exists(): continue

            try:
                with open(run_options_path, 'r') as f:
                    lines = f.readlines()

                output_path = str(dirs['output_dir']).replace('\\', '/').rstrip('/') + '/'
                updated_lines = []
                for line in lines:
                    # Look for the output directory line by comment
                    if 'Output Directory' in line:
                        comment = line.split('#', 1)[1] if '#' in line else 'Output Directory\n'
                        updated_lines.append(f"{output_path:<60} #{comment}")
                    else:
                        updated_lines.append(line)

                with open(run_options_path, 'w') as f:
                    f.writelines(updated_lines)

                self.logger.debug(f"Updated MESH run options for process {proc_id}")
            except Exception as e:
                self.logger.error(f"Failed to update MESH run options for process {proc_id}: {e}")

    def _create_calibration_tasks(
        self,
        parameter_sets: list
    ) -> list:
        """
        Create calibration tasks with MESH-specific paths.

        Overrides base class to add proc_forcing_dir for MESH workers.

        Args:
            parameter_sets: List of parameter dictionaries to evaluate

        Returns:
            List of task dictionaries
        """
        # Call parent method to create base tasks
        tasks = super()._create_calibration_tasks(parameter_sets)

        # Add MESH-specific forcing directory paths
        if self.parallel_dirs:
            for task in tasks:
                proc_id = task.get('proc_id', 0)
                if proc_id in self.parallel_dirs:
                    forcing_dir = self.parallel_dirs[proc_id].get('forcing_dir')
                    if forcing_dir:
                        task['proc_forcing_dir'] = str(forcing_dir)

        return tasks
