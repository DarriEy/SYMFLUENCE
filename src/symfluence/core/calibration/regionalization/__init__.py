# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model-agnostic parameter regionalization contract."""
from __future__ import annotations

from .strategies import (
    DistributedRegionalization,
    LumpedRegionalization,
    ParameterRegionalization,
    RegionalizationFactory,
    TransferFunctionRegionalization,
    ZoneRegionalization,
    get_regionalization_info,
)
from .transfer_functions import (
    ConstantTF,
    ExponentialTF,
    FlexiblePowerTF,
    LinearTF,
    PowerTF,
    TransferFunction,
)

__all__ = [
    'ConstantTF', 'DistributedRegionalization', 'ExponentialTF',
    'FlexiblePowerTF', 'LinearTF', 'LumpedRegionalization',
    'ParameterRegionalization', 'PowerTF', 'RegionalizationFactory',
    'TransferFunction', 'TransferFunctionRegionalization',
    'ZoneRegionalization', 'get_regionalization_info',
]
