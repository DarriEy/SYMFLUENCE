"""
Directory Manager

Manages parallel processing directories for model optimization.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict


class DirectoryManager:
    """
    Manages parallel processing directory lifecycle.

    Creates, populates, and cleans up process-specific directories
    to avoid file conflicts during parallel model evaluations.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialize directory manager.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def setup_parallel_directories(
        self,
        base_dir: Path,
        model_name: str,
        experiment_id: str,
        num_processes: int
    ) -> Dict[int, Dict[str, Path]]:
        """
        Setup parallel processing directories for each process.

        Creates process-specific directories to avoid file conflicts during
        parallel model evaluations.

        Args:
            base_dir: Base directory for parallel processing
            model_name: Name of the model (e.g., 'SUMMA', 'FUSE')
            experiment_id: Experiment identifier
            num_processes: Number of parallel processes

        Returns:
            Dictionary mapping process IDs to their directory paths
        """
        parallel_dirs = {}

        for proc_id in range(num_processes):
            proc_dir = base_dir / f'process_{proc_id}'
            sim_dir = proc_dir / 'simulations' / experiment_id / model_name
            settings_dir = proc_dir / 'settings' / model_name
            output_dir = proc_dir / 'output'

            # Create directories
            for d in [sim_dir, settings_dir, output_dir]:
                d.mkdir(parents=True, exist_ok=True)

            parallel_dirs[proc_id] = {
                'root': proc_dir,
                'sim_dir': sim_dir,
                'settings_dir': settings_dir,
                'output_dir': output_dir,
            }

            self.logger.debug(f"Created parallel directories for process {proc_id}")

        return parallel_dirs

    def copy_base_settings(
        self,
        source_settings_dir: Path,
        parallel_dirs: Dict[int, Dict[str, Path]],
        model_name: str
    ) -> None:
        """
        Copy base settings to each parallel process directory.

        Args:
            source_settings_dir: Source settings directory
            parallel_dirs: Dictionary of parallel directory paths per process
            model_name: Name of the model
        """
        for proc_id, dirs in parallel_dirs.items():
            dest_dir = dirs['settings_dir']

            if source_settings_dir.exists():
                # Copy settings files
                for item in source_settings_dir.iterdir():
                    if item.is_file():
                        dest_path = dest_dir / item.name
                        # Skip if source and destination are the same file
                        if item.resolve() != dest_path.resolve():
                            shutil.copy2(item, dest_path)
                    elif item.is_dir():
                        dest_subdir = dest_dir / item.name
                        # Skip if source and destination are the same directory
                        if item.resolve() != dest_subdir.resolve():
                            if dest_subdir.exists():
                                shutil.rmtree(dest_subdir)
                            shutil.copytree(item, dest_subdir)

                self.logger.debug(
                    f"Copied settings from {source_settings_dir} to process {proc_id}"
                )

    def cleanup(self, parallel_dirs: Dict[int, Dict[str, Path]]) -> None:
        """
        Cleanup parallel processing directories.

        Args:
            parallel_dirs: Dictionary of parallel directory paths per process
        """
        for proc_id, dirs in parallel_dirs.items():
            root_dir = dirs.get('root')
            if root_dir and root_dir.exists():
                try:
                    shutil.rmtree(root_dir)
                    self.logger.debug(f"Cleaned up parallel directory for process {proc_id}")
                except (OSError, PermissionError) as e:
                    self.logger.warning(
                        f"Failed to cleanup parallel directory for process {proc_id}: {e}"
                    )
