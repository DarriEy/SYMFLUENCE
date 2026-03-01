# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
State management and data assimilation configuration models.

Contains StateConfig for model state save/restore, EnKFConfig for
Ensemble Kalman Filter settings, and DataAssimilationConfig as the
parent container.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .base import FROZEN_CONFIG


class StateConfig(BaseModel):
    """Configuration for model state management."""
    model_config = FROZEN_CONFIG

    enabled: bool = Field(default=False, alias='STATE_MANAGEMENT_ENABLED')
    save: bool = Field(default=False, alias='STATE_SAVE')
    load: bool = Field(default=False, alias='STATE_LOAD')
    state_dir: Optional[str] = Field(default=None, alias='STATE_DIR')
    input_path: Optional[str] = Field(default=None, alias='STATE_INPUT_PATH')
    output_path: Optional[str] = Field(default=None, alias='STATE_OUTPUT_PATH')
    file_pattern: Optional[str] = Field(default=None, alias='STATE_FILE_PATTERN')
    ensemble_members: Optional[int] = Field(default=None, alias='STATE_ENSEMBLE_MEMBERS')


class EnKFConfig(BaseModel):
    """Configuration for Ensemble Kalman Filter data assimilation."""
    model_config = FROZEN_CONFIG

    ensemble_size: int = Field(default=50, alias='ENKF_ENSEMBLE_SIZE', ge=2)
    param_perturbation_std: float = Field(
        default=0.05, alias='ENKF_PARAM_PERTURBATION_STD',
        description='Std as fraction of parameter range'
    )
    forcing_perturbation: bool = Field(default=True, alias='ENKF_FORCING_PERTURBATION')
    precip_perturbation_std: float = Field(
        default=0.3, alias='ENKF_PRECIP_PERTURBATION_STD',
        description='Multiplicative std for precipitation'
    )
    temp_perturbation_std: float = Field(
        default=1.0, alias='ENKF_TEMP_PERTURBATION_STD',
        description='Additive std for temperature (K)'
    )
    obs_error_std: float = Field(
        default=0.1, alias='ENKF_OBS_ERROR_STD',
        description='Relative observation error std'
    )
    obs_error_type: Literal['relative', 'absolute'] = Field(
        default='relative', alias='ENKF_OBS_ERROR_TYPE'
    )
    filter_variant: Literal['stochastic', 'deterministic'] = Field(
        default='stochastic', alias='ENKF_FILTER_VARIANT'
    )
    inflation_factor: float = Field(
        default=1.0, alias='ENKF_INFLATION_FACTOR',
        description='Covariance inflation (1.0 = none)', ge=1.0
    )
    localization_radius: Optional[float] = Field(
        default=None, alias='ENKF_LOCALIZATION_RADIUS'
    )
    augment_state_with_predictions: bool = Field(
        default=True, alias='ENKF_AUGMENT_STATE'
    )
    assimilation_variable: str = Field(
        default='streamflow', alias='ENKF_ASSIMILATION_VARIABLE'
    )
    assimilation_interval: int = Field(
        default=1, alias='ENKF_ASSIMILATION_INTERVAL', ge=1,
        description='Assimilation interval in timesteps'
    )
    enforce_nonnegative_states: bool = Field(
        default=True, alias='ENKF_ENFORCE_NONNEG'
    )
    estimate_parameters: bool = Field(
        default=False, alias='ENKF_ESTIMATE_PARAMS',
        description='Joint state-parameter estimation'
    )


class DataAssimilationConfig(BaseModel):
    """Top-level data assimilation configuration."""
    model_config = FROZEN_CONFIG

    method: Literal['enkf'] = Field(default='enkf', alias='DA_METHOD')
    enkf: EnKFConfig = Field(default_factory=EnKFConfig)
