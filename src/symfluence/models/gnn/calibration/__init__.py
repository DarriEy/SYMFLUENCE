# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GNN Model Calibration Module.

Provides calibration infrastructure for Graph Neural Network hydrological
models, supporting hyperparameter tuning and architecture optimization.

Components:
    optimizer: GNN-specific optimizer using PyTorch training loops
    parameter_manager: Manages neural network hyperparameters and architecture
    worker: Executes GNN training runs with GPU acceleration when available

The calibration system supports:
- Message passing layer configuration
- Hidden dimension and layer depth optimization
- Learning rate and regularization tuning
- Early stopping with validation-based selection
- Optuna-based hyperparameter search
"""
