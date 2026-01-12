"""
Task Builder for Optimization Evaluations

Constructs task dictionaries for parallel worker execution.
Eliminates duplicated task dict construction across evaluation methods.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    pass


def _ensure_flat_config(config: Any) -> Dict[str, Any]:
    """
    Ensure config is a flat dictionary for worker compatibility.

    Workers expect flat dictionaries with uppercase keys and .get() method.
    SymfluenceConfig objects need to be flattened for multiprocessing serialization.

    Args:
        config: Either a SymfluenceConfig instance or a flat dict

    Returns:
        Flat dictionary with uppercase keys
    """
    # Check if it's a SymfluenceConfig by checking for model_dump (Pydantic method)
    if hasattr(config, 'model_dump'):
        from symfluence.core.config.transformers import flatten_nested_config
        return flatten_nested_config(config)
    # Already a dict
    return config


class TaskBuilder:
    """
    Builds task dictionaries for parallel model evaluations.

    Centralizes task dict construction that was previously duplicated in:
    - _evaluate_population
    - _evaluate_population_objectives
    - run_de (inline)
    """

    DEFAULT_PENALTY_SCORE = -999.0

    def __init__(
        self,
        config: Dict[str, Any],
        project_dir: Path,
        domain_name: str,
        optimization_settings_dir: Path,
        default_sim_dir: Path,
        parallel_dirs: Dict[int, Dict[str, Any]],
        num_processes: int,
        target_metric: str,
        param_manager: Any,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize task builder.

        Args:
            config: Configuration dictionary
            project_dir: Project directory path
            domain_name: Domain name
            optimization_settings_dir: Settings directory path
            default_sim_dir: Default simulation directory
            parallel_dirs: Parallel processing directories by process ID
            num_processes: Number of parallel processes
            target_metric: Target optimization metric
            param_manager: Parameter manager instance
            logger: Optional logger instance
        """
        # Flatten config if it's a SymfluenceConfig to ensure workers get a plain dict
        self.config = _ensure_flat_config(config)
        self.project_dir = project_dir
        self.domain_name = domain_name
        self.optimization_settings_dir = optimization_settings_dir
        self.default_sim_dir = default_sim_dir
        self.parallel_dirs = parallel_dirs
        self.num_processes = num_processes
        self.target_metric = target_metric
        self.param_manager = param_manager
        self.logger = logger or logging.getLogger(__name__)

        # Cache paths that are frequently used
        self._summa_exe_path = None

    def set_summa_exe_path(self, path: Path) -> None:
        """Set SUMMA executable path for task construction."""
        self._summa_exe_path = path

    def build_task(
        self,
        individual_id: int,
        params: Dict[str, float],
        proc_id: int,
        evaluation_id: str,
        multiobjective: bool = False,
        objective_names: Optional[List[str]] = None,
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build a single task dictionary.

        Args:
            individual_id: Individual index in population
            params: Denormalized parameters
            proc_id: Process ID for parallel execution
            evaluation_id: Unique evaluation identifier
            multiobjective: Whether this is multi-objective evaluation
            objective_names: List of objective metric names
            random_seed: Optional random seed

        Returns:
            Task dictionary for worker execution
        """
        dirs = self.parallel_dirs.get(proc_id, {})
        settings_dir = Path(dirs.get('settings_dir', self.optimization_settings_dir))
        sim_dir = dirs.get('sim_dir', self.default_sim_dir)
        root_dir = dirs.get('root', self.project_dir)

        # HYPE uses a separate output_dir for results (configured in info.txt resultdir)
        # Other models write directly to sim_dir
        model_name = self.config.get('HYDROLOGICAL_MODEL', '').upper()
        if model_name == 'HYPE':
            output_dir = dirs.get('output_dir', sim_dir)
        else:
            output_dir = sim_dir

        # Build mizuRoute paths
        if dirs and sim_dir:
            mizuroute_dir = str(Path(sim_dir).parent / 'mizuRoute')
        else:
            mizuroute_dir = str(Path(self.default_sim_dir).parent / 'mizuRoute')

        mizuroute_settings_dir = str(root_dir / 'settings' / 'mizuRoute') if dirs else ''

        # Get file manager path
        file_manager_name = self.config.get('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')
        file_manager = str(settings_dir / file_manager_name)

        # Get original depths if available
        original_depths = None
        if hasattr(self.param_manager, 'original_depths') and self.param_manager.original_depths is not None:
            original_depths = self.param_manager.original_depths.tolist()

        task_dict = {
            'individual_id': individual_id,
            'params': params,
            'proc_id': proc_id,
            'evaluation_id': evaluation_id,
            'config': self.config,
            'target_metric': self.target_metric if not multiobjective else objective_names[0],
            'calibration_variable': self.config.get('OPTIMIZATION_TARGET', self.config.get('CALIBRATION_VARIABLE', 'streamflow')),
            'domain_name': self.domain_name,
            'project_dir': str(self.project_dir),
            'proc_settings_dir': str(settings_dir),
            'proc_output_dir': str(output_dir),
            'proc_sim_dir': str(sim_dir),
            'summa_settings_dir': str(settings_dir),
            'mizuroute_settings_dir': mizuroute_settings_dir,
            'summa_dir': str(sim_dir),
            'mizuroute_dir': mizuroute_dir,
            'file_manager': file_manager,
            'summa_exe': str(self._summa_exe_path) if self._summa_exe_path else '',
            'original_depths': original_depths,
        }

        # Add multi-objective fields if needed
        if multiobjective:
            task_dict['multiobjective'] = True
            task_dict['objective_names'] = objective_names
            
            # Add target types and metrics for multi-target optimization
            # Support both NSGA2_* keys and legacy OPTIMIZATION_TARGET* keys
            task_dict['multi_target_mode'] = True
            task_dict['primary_target_type'] = self.config.get(
                'NSGA2_PRIMARY_TARGET',
                self.config.get('OPTIMIZATION_TARGET', 'streamflow')
            )
            task_dict['secondary_target_type'] = self.config.get(
                'NSGA2_SECONDARY_TARGET',
                self.config.get('OPTIMIZATION_TARGET2', 'tws')
            )
            task_dict['primary_metric'] = self.config.get(
                'NSGA2_PRIMARY_METRIC',
                self.config.get('OPTIMIZATION_METRIC', 'KGE')
            )
            task_dict['secondary_metric'] = self.config.get(
                'NSGA2_SECONDARY_METRIC',
                self.config.get('OPTIMIZATION_METRIC2', 'KGE')
            )

        # Add random seed if provided
        if random_seed is not None:
            task_dict['random_seed'] = random_seed

        return task_dict

    def build_population_tasks(
        self,
        population: np.ndarray,
        iteration: int = 0,
        multiobjective: bool = False,
        objective_names: Optional[List[str]] = None,
        base_random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Build task dictionaries for a population of solutions.

        Args:
            population: Array of normalized parameter sets (n_individuals x n_params)
            iteration: Current iteration number
            multiobjective: Whether this is multi-objective evaluation
            objective_names: List of objective metric names
            base_random_seed: Base random seed (individual seeds derived from this)

        Returns:
            List of task dictionaries
        """
        tasks = []

        for i, params_normalized in enumerate(population):
            params = self.param_manager.denormalize_parameters(params_normalized)
            proc_id = i % self.num_processes

            # Calculate individual random seed
            random_seed = None
            if base_random_seed is not None:
                random_seed = base_random_seed + i + 1000

            task = self.build_task(
                individual_id=i,
                params=params,
                proc_id=proc_id,
                evaluation_id=f"pop_eval_{i:03d}",
                multiobjective=multiobjective,
                objective_names=objective_names,
                random_seed=random_seed
            )
            tasks.append(task)

        return tasks

    def build_trial_tasks(
        self,
        trials: List[np.ndarray],
        trial_indices: List[int],
        iteration: int,
        base_random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Build task dictionaries for trial solutions (e.g., in DE).

        Args:
            trials: List of trial parameter arrays (normalized)
            trial_indices: Corresponding indices in population
            iteration: Current iteration number
            base_random_seed: Base random seed

        Returns:
            List of task dictionaries
        """
        tasks = []

        for idx, trial in zip(trial_indices, trials):
            params = self.param_manager.denormalize_parameters(trial)
            proc_id = idx % self.num_processes

            random_seed = None
            if base_random_seed is not None:
                random_seed = base_random_seed + idx + 1000

            task = self.build_task(
                individual_id=idx,
                params=params,
                proc_id=proc_id,
                evaluation_id=f"trial_{iteration:03d}_{idx:03d}",
                random_seed=random_seed
            )
            tasks.append(task)

        return tasks
