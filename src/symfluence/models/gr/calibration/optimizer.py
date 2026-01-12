"""
GR Model Optimizer

GR-specific optimizer inheriting from BaseModelOptimizer.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from symfluence.core.file_utils import copy_file
from symfluence.optimization.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_optimizer('GR')
class GRModelOptimizer(BaseModelOptimizer):
    """
    GR-specific optimizer using the unified BaseModelOptimizer framework.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None
    ):
        self.experiment_id = config.get('EXPERIMENT_ID')
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        
        self.gr_setup_dir = self.project_dir / 'settings' / 'GR'

        super().__init__(config, logger, optimization_settings_dir, reporting_manager=reporting_manager)

        self.logger.debug("GRModelOptimizer initialized")

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'GR'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to GR configuration (dummy for GR)."""
        # GR doesn't use a file manager in the same way as SUMMA/FUSE.
        # We return a placeholder path in the setup directory.
        return self.gr_setup_dir / 'gr_config.txt'

    def _create_parameter_manager(self):
        """Create GR parameter manager."""
        from symfluence.optimization.parameter_managers.gr_parameter_manager import GRParameterManager
        return GRParameterManager(
            self.config,
            self.logger,
            self.gr_setup_dir
        )

    def _check_routing_needed(self) -> bool:
        """Determine if routing is needed based on configuration."""
        # Use SpatialOrchestrator logic (checking if distributed mode and routing is enabled)
        routing_integration = self.config.get('GR_ROUTING_INTEGRATION', 'none')
        global_routing = self.config.get('ROUTING_MODEL', 'none')
        spatial_mode = self.config.get('GR_SPATIAL_MODE', 'auto')

        # Handle 'auto' mode - resolve from DOMAIN_DEFINITION_METHOD
        if spatial_mode in (None, 'auto', 'default'):
            domain_method = self.config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
            if domain_method == 'delineate':
                spatial_mode = 'distributed'
            else:
                spatial_mode = 'lumped'

        if spatial_mode != 'distributed':
            return False

        return (routing_integration.lower() == 'mizuroute' or
                global_routing.lower() == 'mizuroute')

    def _setup_parallel_dirs(self) -> None:
        """Setup GR-specific parallel directories."""
        algorithm = self.config.get('ITERATIVE_OPTIMIZATION_ALGORITHM', 'optimization').lower()
        base_dir = self.project_dir / 'simulations' / f'run_{algorithm}'
        self.parallel_dirs = self.setup_parallel_processing(
            base_dir,
            'GR',
            self.experiment_id
        )

        # Copy GR settings to each parallel directory
        if self.gr_setup_dir.exists():
            self.copy_base_settings(self.gr_setup_dir, self.parallel_dirs, 'GR')

        # If routing needed, also copy and configure mizuRoute settings
        if self._check_routing_needed():
            mizu_settings = self.project_dir / 'settings' / 'mizuRoute'
            if mizu_settings.exists():
                for proc_id, dirs in self.parallel_dirs.items():
                    mizu_dest = dirs['root'] / 'settings' / 'mizuRoute'
                    mizu_dest.mkdir(parents=True, exist_ok=True)
                    for item in mizu_settings.iterdir():
                        if item.is_file():
                            copy_file(item, mizu_dest / item.name)

                # Update mizuRoute control files with process-specific paths
                # GR uses mizuRoute_control_GR.txt by default
                mizu_control = self.config.get('SETTINGS_MIZU_CONTROL_FILE')
                if not mizu_control or mizu_control == 'default':
                    mizu_control = 'mizuRoute_control_GR.txt'
                    
                self.update_mizuroute_controls(
                    self.parallel_dirs,
                    'GR',
                    self.experiment_id,
                    control_file_name=mizu_control
                )

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run GR for final evaluation using best parameters."""
        # Get best parameters from results if available
        best_result = self.get_best_result()
        best_params = best_result.get('params')
        
        if not best_params:
            self.logger.warning("No best parameters found for final evaluation")
            return False
            
        # Ensure mizuRoute paths are provided for isolation (matching worker behavior)
        mizuroute_dir = output_dir / 'mizuRoute'
        mizuroute_settings_dir = output_dir / 'settings' / 'mizuRoute'
        
        return self.worker.run_model(
            self.config,
            self.gr_setup_dir,
            output_dir,
            params=best_params,
            mizuroute_dir=str(mizuroute_dir),
            mizuroute_settings_dir=str(mizuroute_settings_dir)
        )
        
    def _update_file_manager_for_final_run(self) -> None:
        """GR doesn't use a file manager."""
        pass

    def _restore_file_manager_for_optimization(self) -> None:
        """GR doesn't use a file manager."""
        pass

# Backward compatibility alias
GROptimizer = GRModelOptimizer
