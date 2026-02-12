"""
CLI Services for SYMFLUENCE.

This package provides modular services for CLI operations:

Binary Management:
- ToolInstaller: External tool installation (clone, build)
- ToolValidator: Binary validation and testing
- SystemDiagnostics: System health checks and diagnostics

Project Management:
- InitializationService: Project initialization and scaffolding
- JobSchedulerService: SLURM/HPC job submission
- NotebookService: Jupyter notebook launching

Build Configuration:
- BuildInstructionsRegistry: Tool build configuration registry
- build_snippets: Shared shell script helpers
- build_schema: Build instruction schema and validation
"""

from .base import BaseService

# Binary management services
from .tool_installer import ToolInstaller
from .tool_validator import ToolValidator
from .system_diagnostics import SystemDiagnostics

# Project management services
from .initialization import InitializationService, InitializationManager
from .job_scheduler import JobSchedulerService, JobScheduler
from .notebook import NotebookService

# Build configuration
from .build_registry import BuildInstructionsRegistry
from .build_snippets import (
    get_common_build_environment,
    get_netcdf_detection,
    get_hdf5_detection,
    get_netcdf_lib_detection,
    get_geos_proj_detection,
    get_udunits2_detection_and_build,
    get_bison_detection_and_build,
    get_flex_detection_and_build,
    get_all_snippets,
)
from .build_schema import (
    BuildInstructionSchema,
    VerifyInstallSchema,
    validate_build_instructions,
    validate_all_instructions,
)

__all__ = [
    # Base
    'BaseService',
    # Binary management services
    'ToolInstaller',
    'ToolValidator',
    'SystemDiagnostics',
    # Project management services
    'InitializationService',
    'InitializationManager',  # Backward compatibility alias
    'JobSchedulerService',
    'JobScheduler',  # Backward compatibility alias
    'NotebookService',
    # Build registry
    'BuildInstructionsRegistry',
    # Build snippets
    'get_common_build_environment',
    'get_netcdf_detection',
    'get_hdf5_detection',
    'get_netcdf_lib_detection',
    'get_geos_proj_detection',
    'get_udunits2_detection_and_build',
    'get_bison_detection_and_build',
    'get_flex_detection_and_build',
    'get_all_snippets',
    # Build schema
    'BuildInstructionSchema',
    'VerifyInstallSchema',
    'validate_build_instructions',
    'validate_all_instructions',
]
