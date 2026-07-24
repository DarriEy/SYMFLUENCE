# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for LSTM.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG


class LSTMConfig(BaseModel):
    """LSTM neural network emulator configuration"""
    model_config = FROZEN_CONFIG

    load: bool = Field(default=False, alias='LSTM_LOAD')
    hidden_size: int = Field(default=128, alias='LSTM_HIDDEN_SIZE', ge=8, le=2048)
    num_layers: int = Field(default=3, alias='LSTM_NUM_LAYERS', ge=1, le=10)
    epochs: int = Field(default=300, alias='LSTM_EPOCHS', ge=1, le=10000)
    batch_size: int = Field(default=64, alias='LSTM_BATCH_SIZE', ge=1, le=4096)
    learning_rate: float = Field(default=0.001, alias='LSTM_LEARNING_RATE', gt=0, le=1.0)
    learning_patience: int = Field(default=30, alias='LSTM_LEARNING_PATIENCE', ge=1)
    lookback: int = Field(default=700, alias='LSTM_LOOKBACK', ge=1)
    dropout: float = Field(default=0.2, alias='LSTM_DROPOUT', ge=0, le=0.9)
    l2_regularization: float = Field(default=1e-6, alias='LSTM_L2_REGULARIZATION', ge=0)
    use_attention: bool = Field(default=True, alias='LSTM_USE_ATTENTION')
    use_snow: bool = Field(default=False, alias='LSTM_USE_SNOW')
    train_through_routing: bool = Field(default=False, alias='LSTM_TRAIN_THROUGH_ROUTING')
    params_to_calibrate: Optional[str] = Field(
        default=None, alias='LSTM_PARAMS_TO_CALIBRATE',
        description='Comma-separated parameter names for LSTM calibration'
    )
    parameter_bounds: Optional[Dict[str, List[float]]] = Field(
        default=None, alias='LSTM_PARAMETER_BOUNDS'
    )
