"""
System configuration model.

Contains SystemConfig for system-level settings: paths, logging, MPI, debugging.
"""

from typing import Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, ConfigDict

from .base import FROZEN_CONFIG


class SystemConfig(BaseModel):
    """System-level configuration: paths, logging, MPI, debugging"""
    model_config = FROZEN_CONFIG

    # Required paths
    data_dir: Path = Field(alias='SYMFLUENCE_DATA_DIR')
    code_dir: Path = Field(alias='SYMFLUENCE_CODE_DIR')

    # System settings
    mpi_processes: int = Field(default=1, alias='MPI_PROCESSES')
    debug_mode: bool = Field(default=False, alias='DEBUG_MODE')
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(default='INFO', alias='LOG_LEVEL')
    log_to_file: bool = Field(default=True, alias='LOG_TO_FILE')
    log_format: str = Field(default='detailed', alias='LOG_FORMAT')
    force_run_all_steps: bool = Field(default=False, alias='FORCE_RUN_ALL_STEPS')
    force_download: bool = Field(default=False, alias='FORCE_DOWNLOAD')
    use_local_scratch: bool = Field(default=False, alias='USE_LOCAL_SCRATCH')
    random_seed: Optional[int] = Field(default=None, alias='RANDOM_SEED')
    stop_on_error: bool = Field(default=True, alias='STOP_ON_ERROR')

    @field_validator('data_dir', 'code_dir')
    @classmethod
    def validate_paths(cls, v):
        """Expand and resolve paths"""
        return Path(v).expanduser().resolve()

    @field_validator('mpi_processes')
    @classmethod
    def validate_positive_integers(cls, v, info):
        """Ensure positive integers"""
        if v < 1:
            raise ValueError(f"{info.field_name} must be at least 1, got {v}")
        return v
