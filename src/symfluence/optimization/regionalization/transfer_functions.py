# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Transfer Function Classes for Parameter Regionalization.

Model-agnostic transfer function forms that map catchment attributes to
local parameter values.
"""

from typing import List, Tuple

import numpy as np


class TransferFunction:
    """Base class for transfer functions."""

    def __init__(self, name: str, n_coefficients: int):
        self.name = name
        self.n_coefficients = n_coefficients

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        raise NotImplementedError


class LinearTF(TransferFunction):
    """Linear: param = a + b * attr."""

    def __init__(self, a_bounds: Tuple[float, float], b_bounds: Tuple[float, float]):
        super().__init__('linear', 2)
        self.a_bounds = a_bounds
        self.b_bounds = b_bounds

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        a, b = coeffs[0], coeffs[1]
        return a + b * attr

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        return [self.a_bounds, self.b_bounds]


class PowerTF(TransferFunction):
    """Power: param = a * (attr + 0.01)^b."""

    def __init__(self, a_bounds: Tuple[float, float], b_bounds: Tuple[float, float]):
        super().__init__('power', 2)
        self.a_bounds = a_bounds
        self.b_bounds = b_bounds

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        a, b = coeffs[0], coeffs[1]
        return a * np.power(attr + 0.01, b)

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        return [self.a_bounds, self.b_bounds]


class ExponentialTF(TransferFunction):
    """Exponential: param = a * exp(b * attr)."""

    def __init__(self, a_bounds: Tuple[float, float], b_bounds: Tuple[float, float]):
        super().__init__('exponential', 2)
        self.a_bounds = a_bounds
        self.b_bounds = b_bounds

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        a, b = coeffs[0], coeffs[1]
        return a * np.exp(b * attr)

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        return [self.a_bounds, self.b_bounds]


class ConstantTF(TransferFunction):
    """Constant: param = a."""

    def __init__(self, a_bounds: Tuple[float, float]):
        super().__init__('constant', 1)
        self.a_bounds = a_bounds

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        return np.full_like(attr, coeffs[0])

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        return [self.a_bounds]


class FlexiblePowerTF(TransferFunction):
    """Flexible power-law: param = a * attr^b."""

    def __init__(
        self,
        a_bounds: Tuple[float, float],
        b_bounds: Tuple[float, float] = (-2.0, 2.0),
        calibrate_exponent: bool = True,
        fixed_exponent: float = 0.0,
    ):
        n_coeffs = 2 if calibrate_exponent else 1
        super().__init__('flexible_power', n_coeffs)
        self.a_bounds = a_bounds
        self.b_bounds = b_bounds
        self.calibrate_exponent = calibrate_exponent
        self.fixed_exponent = fixed_exponent

    def apply(self, attr: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        a = coeffs[0]
        b = coeffs[1] if self.calibrate_exponent else self.fixed_exponent
        safe_attr = np.abs(attr) + 0.01
        if np.abs(b) < 1e-10:
            return np.full_like(attr, a, dtype=float)
        return a * np.power(safe_attr, b)

    def get_coefficient_bounds(self) -> List[Tuple[float, float]]:
        if self.calibrate_exponent:
            return [self.a_bounds, self.b_bounds]
        return [self.a_bounds]
