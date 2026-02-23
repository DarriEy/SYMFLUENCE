"""
Model Directory Conventions

Defines directory structure conventions for different hydrological models.
This allows models to specify their preferred directory layouts while
maintaining a consistent interface for the optimization framework.

Each model can register its directory conventions, and the framework
uses these conventions when setting up optimization directories.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ModelDirectoryConvention:
    """
    Directory convention for a hydrological model.

    Defines the directory structure and file locations expected by a model
    during optimization runs.

    Attributes:
        model_name: Name of the model (e.g., 'SUMMA', 'FUSE', 'NGEN')
        settings_subdir: Subdirectory name for model settings (e.g., 'SUMMA', 'fuse')
        output_subdir: Subdirectory name for model outputs (e.g., 'SUMMA', 'output')
        routing_model: Name of routing model if applicable (e.g., 'mizuRoute', 'troute')
        routing_subdir: Subdirectory name for routing outputs (e.g., 'mizuRoute')
        file_manager_name: Name of the main configuration file (e.g., 'fileManager.txt')
        requires_routing: Whether the model typically requires routing for streamflow
        additional_subdirs: Additional subdirectories needed by the model
    """
    model_name: str
    settings_subdir: str
    output_subdir: str
    routing_model: Optional[str] = None
    routing_subdir: Optional[str] = None
    file_manager_name: str = 'fileManager.txt'
    requires_routing: bool = False
    additional_subdirs: List[str] = field(default_factory=list)

    def get_settings_dir(self, base_dir: Path) -> Path:
        """Get the settings directory for this model."""
        return base_dir / 'settings' / self.settings_subdir

    def get_output_dir(self, base_dir: Path) -> Path:
        """Get the output directory for this model."""
        return base_dir / self.output_subdir

    def get_routing_output_dir(self, base_dir: Path) -> Optional[Path]:
        """Get the routing output directory if routing is configured."""
        if self.routing_subdir:
            return base_dir / self.routing_subdir
        return None

    def get_file_manager_path(self, base_dir: Path) -> Path:
        """Get the path to the file manager configuration."""
        return self.get_settings_dir(base_dir) / self.file_manager_name

    def create_directories(self, base_dir: Path) -> Dict[str, Path]:
        """
        Create all directories for this model.

        Args:
            base_dir: Base directory for optimization

        Returns:
            Dictionary mapping directory names to paths
        """
        dirs = {
            'settings': self.get_settings_dir(base_dir),
            'output': self.get_output_dir(base_dir),
        }

        if self.routing_subdir:
            dirs['routing'] = self.get_routing_output_dir(base_dir)

        for subdir in self.additional_subdirs:
            dirs[subdir] = base_dir / subdir

        # Create all directories
        for name, path in dirs.items():
            if path:
                path.mkdir(parents=True, exist_ok=True)
                # Create logs subdirectory for output dirs
                if name in ['output', 'routing']:
                    (path / 'logs').mkdir(parents=True, exist_ok=True)

        return dirs


class DirectoryConventionRegistry:
    """
    Registry for model directory conventions.

    Allows models to register their directory conventions and provides
    lookup functionality for the optimization framework.
    """

    _conventions: Dict[str, ModelDirectoryConvention] = {}

    # Default conventions for common models
    _defaults = {
        'SUMMA': ModelDirectoryConvention(
            model_name='SUMMA',
            settings_subdir='SUMMA',
            output_subdir='SUMMA',
            routing_model='mizuRoute',
            routing_subdir='mizuRoute',
            file_manager_name='fileManager.txt',
            requires_routing=True,
        ),
        'FUSE': ModelDirectoryConvention(
            model_name='FUSE',
            settings_subdir='FUSE',
            output_subdir='FUSE',
            file_manager_name='fm_fuse.txt',
            requires_routing=False,
        ),
        'NGEN': ModelDirectoryConvention(
            model_name='NGEN',
            settings_subdir='ngen',
            output_subdir='ngen',
            routing_model='troute',
            routing_subdir='troute',
            file_manager_name='realization.json',
            requires_routing=True,
        ),
        'GR': ModelDirectoryConvention(
            model_name='GR',
            settings_subdir='GR',
            output_subdir='GR',
            file_manager_name='gr_config.txt',
            requires_routing=False,
        ),
        'HYPE': ModelDirectoryConvention(
            model_name='HYPE',
            settings_subdir='HYPE',
            output_subdir='HYPE',
            file_manager_name='info.txt',
            requires_routing=False,
        ),
        'MESH': ModelDirectoryConvention(
            model_name='MESH',
            settings_subdir='MESH',
            output_subdir='MESH',
            routing_model='WATROUTE',
            routing_subdir='WATROUTE',
            file_manager_name='mesh_parameters.ini',
            requires_routing=True,
        ),
        'RHESSYS': ModelDirectoryConvention(
            model_name='RHESSYS',
            settings_subdir='RHESSys',
            output_subdir='RHESSys',
            file_manager_name='worldfile',
            requires_routing=False,
        ),
    }

    @classmethod
    def register(cls, convention: ModelDirectoryConvention) -> None:
        """
        Register a directory convention for a model.

        Args:
            convention: ModelDirectoryConvention instance
        """
        cls._conventions[convention.model_name.upper()] = convention

    @classmethod
    def get(cls, model_name: str) -> ModelDirectoryConvention:
        """
        Get the directory convention for a model.

        First checks registered conventions, then falls back to defaults.

        Args:
            model_name: Name of the model

        Returns:
            ModelDirectoryConvention for the model

        Raises:
            ValueError: If no convention found for the model
        """
        key = model_name.upper()

        # Check registered conventions first
        if key in cls._conventions:
            return cls._conventions[key]

        # Fall back to defaults
        if key in cls._defaults:
            return cls._defaults[key]

        # Return a generic convention if model not found
        return ModelDirectoryConvention(
            model_name=model_name,
            settings_subdir=model_name,
            output_subdir=model_name,
            file_manager_name='config.txt',
            requires_routing=False,
        )

    @classmethod
    def list_models(cls) -> List[str]:
        """List all models with registered or default conventions."""
        all_models = set(cls._conventions.keys()) | set(cls._defaults.keys())
        return sorted(all_models)

    @classmethod
    def has_convention(cls, model_name: str) -> bool:
        """Check if a convention exists for a model."""
        key = model_name.upper()
        return key in cls._conventions or key in cls._defaults


def get_model_directories(
    model_name: str,
    optimization_dir: Path
) -> Dict[str, Path]:
    """
    Get all directory paths for a model's optimization run.

    Convenience function that retrieves the convention for a model
    and creates all necessary directories.

    Args:
        model_name: Name of the model
        optimization_dir: Base optimization directory

    Returns:
        Dictionary mapping directory names to paths
    """
    convention = DirectoryConventionRegistry.get(model_name)
    return convention.create_directories(optimization_dir)
