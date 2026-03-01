# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Parameter Regionalization for Distributed HBV.

Implements differentiable transfer functions (Neural Networks) to map
catchment attributes to HBV model parameters. This enables:
1. Learning parameter distributions from physical attributes
2. Transferring parameters to ungauged basins (PUB)
3. Regularizing high-dimensional distributed parameter spaces

The transfer function is implemented as a Multi-Layer Perceptron (MLP)
using pure JAX (no external NN library dependencies).
"""

from typing import Any, Dict, List, NamedTuple, Tuple

try:
    import jax
    import jax.numpy as jnp
    from jax import random
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None
    jax = None
    random = None

from .model import PARAM_BOUNDS, HBVParameters

# Parameters to regionalize (exclude smoothing_enabled)
REGIONALIZED_PARAMS = [
    'tt', 'cfmax', 'sfcf', 'cfr', 'cwh', 'fc', 'lp', 'beta',
    'k0', 'k1', 'k2', 'uzl', 'perc', 'maxbas'
]

class TransferLayer(NamedTuple):
    """Single layer of the transfer function network."""
    w: Any  # Weights [in, out]
    b: Any  # Bias [out]


class TransferFunctionConfig(NamedTuple):
    """Configuration for the transfer function."""
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: str = 'tanh'


def initialize_weights(
    key: Any,
    config: TransferFunctionConfig
) -> List[TransferLayer]:
    """
    Initialize weights for the transfer function MLP.

    Args:
        key: JAX random key
        config: Network configuration

    Returns:
        List of TransferLayer (weights and biases for each layer)
    """
    if not HAS_JAX:
        raise ImportError("JAX required for regionalization")

    layers = []
    dims = [config.input_dim] + config.hidden_dims + [config.output_dim]

    for i in range(len(dims) - 1):
        in_dim = dims[i]
        out_dim = dims[i+1]

        k1, k2 = random.split(key)
        # Xavier/Glorot initialization
        lim = jnp.sqrt(6 / (in_dim + out_dim))
        w = random.uniform(k1, (in_dim, out_dim), minval=-lim, maxval=lim)
        b = jnp.zeros(out_dim)

        layers.append(TransferLayer(w, b))
        key = k2

    return layers


def forward_transfer_function(
    weights: List[TransferLayer],
    attributes: Any,
    bounds: Dict[str, Tuple[float, float]],
    activation: str = 'tanh'
) -> HBVParameters:
    """
    Forward pass of the transfer function.

    Maps attributes to HBV parameters through the MLP.

    Args:
        weights: List of TransferLayer
        attributes: Input attributes [n_nodes, input_dim]
        bounds: Parameter bounds dictionary
        activation: Activation function ('tanh', 'relu', 'sigmoid')

    Returns:
        HBVParameters with values for each node
    """
    if not HAS_JAX:
        raise ImportError("JAX required for regionalization")

    x = attributes

    # Hidden layers
    for i in range(len(weights) - 1):
        layer = weights[i]
        x = jnp.dot(x, layer.w) + layer.b

        if activation == 'tanh':
            x = jnp.tanh(x)
        elif activation == 'relu':
            x = jnp.maximum(x, 0)
        elif activation == 'sigmoid':
            x = jax.nn.sigmoid(x)

    # Output layer (linear)
    last_layer = weights[-1]
    output = jnp.dot(x, last_layer.w) + last_layer.b

    # Map outputs to parameter ranges
    # Output has shape [n_nodes, n_params]
    # We apply sigmoid to map to [0, 1], then scale to [low, high]

    mapped_params = {}

    for i, param_name in enumerate(REGIONALIZED_PARAMS):
        # Extract raw output for this parameter
        raw_val = output[:, i]

        # Sigmoid to [0, 1]
        norm_val = jax.nn.sigmoid(raw_val)

        # Scale to bounds
        low, high = bounds.get(param_name, PARAM_BOUNDS.get(param_name, (0.0, 1.0)))
        val = low + norm_val * (high - low)

        mapped_params[param_name] = val

    # Add non-regionalized parameters (fixed defaults)
    mapped_params['smoothing'] = jnp.full(attributes.shape[0], 15.0)
    mapped_params['smoothing_enabled'] = jnp.full(attributes.shape[0], False, dtype=bool)

    return HBVParameters(**mapped_params)


def compute_regularization(
    weights: List[TransferLayer],
    l2_lambda: float = 0.01
) -> float:
    """Compute L2 regularization term for weights."""
    reg_loss = 0.0
    for layer in weights:
        reg_loss += jnp.sum(layer.w ** 2)
    return 0.5 * l2_lambda * reg_loss
