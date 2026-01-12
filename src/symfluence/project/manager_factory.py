"""
Lazy loading manager factory for SYMFLUENCE.
"""

from typing import Dict, Any, Set
import logging


class LazyManagerDict:
    """
    Dictionary-like class that lazy-loads manager instances.
    
    This improves startup performance by only importing and instantiating
    managers when they are actually accessed.
    """
    def __init__(self, config: Any, logger: logging.Logger, visualize: bool = False):
        self._config = config
        self._logger = logger
        self._visualize = visualize
        self._managers: Dict[str, Any] = {}
        # Reporting manager is a shared dependency that must be instantiated early if needed
        self._reporting_manager = None
        
        # Supported manager keys
        self._keys: Set[str] = {
            'project', 'domain', 'data', 'model', 
            'analysis', 'optimization', 'reporting'
        }

    def _get_reporting_manager(self):
        """Get or create the shared reporting manager instance."""
        if not self._reporting_manager:
             from symfluence.reporting.reporting_manager import ReportingManager
             self._reporting_manager = ReportingManager(self._config, self._logger, visualize=self._visualize)
        return self._reporting_manager

    def __getitem__(self, key: str) -> Any:
        if key not in self._keys:
            raise KeyError(key)
        
        if key in self._managers:
            return self._managers[key]

        # Initialize requested manager
        manager = self._create_manager(key)
        self._managers[key] = manager
        return manager

    def _create_manager(self, key: str) -> Any:
        """Import and instantiate the specific manager."""
        self._logger.debug(f"Lazy loading manager: {key}")
        
        if key == 'reporting':
            return self._get_reporting_manager()
            
        elif key == 'project':
             from symfluence.project.project_manager import ProjectManager
             return ProjectManager(self._config, self._logger)
             
        elif key == 'domain':
             from symfluence.geospatial.domain_manager import DomainManager
             return DomainManager(self._config, self._logger, self._get_reporting_manager())
             
        elif key == 'data':
             from symfluence.data.data_manager import DataManager
             return DataManager(self._config, self._logger)
             
        elif key == 'model':
             from symfluence.models.model_manager import ModelManager
             return ModelManager(self._config, self._logger, self._get_reporting_manager())
             
        elif key == 'analysis':
             from symfluence.evaluation.analysis_manager import AnalysisManager
             return AnalysisManager(self._config, self._logger, self._get_reporting_manager())
             
        elif key == 'optimization':
             from symfluence.optimization.optimization_manager import OptimizationManager
             return OptimizationManager(self._config, self._logger)
             
        raise KeyError(f"Unknown manager key: {key}")
    
    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._keys
    
    def get(self, key: str, default: Any = None) -> Any:
        if key in self._keys:
            return self[key]
        return default
        
    def keys(self):
        return self._keys
    
    def values(self):
        # Force initialization of all managers
        return [self[k] for k in self._keys]
        
    def items(self):
        # Force initialization of all managers
        return [(k, self[k]) for k in self._keys]
