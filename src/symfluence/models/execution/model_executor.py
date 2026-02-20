"""Execution types: ExecutionMode, SlurmJobConfig, ExecutionResult, and helpers."""

import os
import sys
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


def augment_conda_library_paths(run_env: Dict[str, str]) -> None:
    """Prepend ``$CONDA_PREFIX/lib`` to the platform library search path in *run_env*.

    - Linux:   ``LD_LIBRARY_PATH``
    - macOS:   ``DYLD_LIBRARY_PATH``
    - Windows: ``PATH`` (conda stores DLLs in ``Library/bin``)

    No-op when ``CONDA_PREFIX`` is unset.  Idempotent (won't add duplicates).
    Mutates *run_env* in place.  Does **not** touch ``os.environ``.
    """
    conda_prefix = run_env.get('CONDA_PREFIX', '')
    if not conda_prefix:
        return

    if sys.platform == 'win32':
        conda_lib = os.path.join(conda_prefix, 'Library', 'bin')
        env_var = 'PATH'
    elif sys.platform == 'darwin':
        conda_lib = os.path.join(conda_prefix, 'lib')
        env_var = 'DYLD_LIBRARY_PATH'
    else:  # linux / other posix
        conda_lib = os.path.join(conda_prefix, 'lib')
        env_var = 'LD_LIBRARY_PATH'

    current = run_env.get(env_var, '')
    if conda_lib not in current.split(os.pathsep):
        run_env[env_var] = f"{conda_lib}{os.pathsep}{current}" if current else conda_lib


class ExecutionMode(Enum):
    """Execution mode for model runs."""
    LOCAL = "local"
    SLURM = "slurm"
    SLURM_ARRAY = "slurm_array"


@dataclass
class SlurmJobConfig:
    """Configuration for SLURM job submission.

    Attributes:
        job_name: Name of the SLURM job
        time_limit: Time limit in format HH:MM:SS
        memory: Memory per node (e.g., '4G', '16G')
        cpus_per_task: Number of CPUs per task
        partition: SLURM partition to submit to (optional)
        account: Account to charge (optional)
        array_size: For job arrays, max array index (0-based)
        output_pattern: Pattern for stdout file (supports %A, %a placeholders)
        error_pattern: Pattern for stderr file
        additional_directives: Extra #SBATCH lines as dict
    """
    job_name: str
    time_limit: str = "03:00:00"
    memory: str = "4G"
    cpus_per_task: int = 1
    partition: Optional[str] = None
    account: Optional[str] = None
    array_size: Optional[int] = None
    output_pattern: Optional[str] = None
    error_pattern: Optional[str] = None
    additional_directives: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of a model execution.

    Attributes:
        success: Whether execution completed successfully
        return_code: Process return code (0 = success)
        output_path: Path to output directory/file if applicable
        log_file: Path to log file
        duration_seconds: Execution duration in seconds
        job_id: SLURM job ID if applicable
        error_message: Error message if execution failed
        metadata: Additional execution metadata
    """
    success: bool
    return_code: int = 0
    output_path: Optional[Path] = None
    log_file: Optional[Path] = None
    duration_seconds: float = 0.0
    job_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelExecutor(ABC):
    """Deprecated shim — remove from inheritance lists.

    Execution methods now live on ``BaseModelRunner`` via
    ``SubprocessExecutionMixin`` and ``SlurmExecutionMixin``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        import warnings
        warnings.warn(
            f"{cls.__name__} inherits from ModelExecutor which is deprecated. "
            "Remove ModelExecutor from the inheritance list — execution methods "
            "are now provided by BaseModelRunner.",
            DeprecationWarning,
            stacklevel=2,
        )

    pass
