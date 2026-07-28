# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Integrity gates for scientific datasets before they become durable artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import xarray as xr


class ScientificIntegrityError(ValueError):
    """Raised when a scientific artifact violates a required invariant."""


@dataclass(frozen=True)
class BalanceCheck:
    inputs: Iterable[str]
    outputs: Iterable[str]
    storage_change: str | None = None
    relative_tolerance: float = 1e-3


def validate_dataset(
    dataset: xr.Dataset,
    *,
    require_units: bool = True,
    balance: BalanceCheck | None = None,
) -> None:
    """Validate finite values, coordinates, units, identifiers, and balance."""
    if not dataset.data_vars:
        raise ScientificIntegrityError("Dataset has no data variables")

    for name, variable in dataset.data_vars.items():
        if np.issubdtype(variable.dtype, np.number):
            values = np.asarray(variable.values)
            if not np.all(np.isfinite(values)):
                raise ScientificIntegrityError(f"Variable {name!r} contains NaN or infinite values")
        if require_units and not variable.attrs.get("units"):
            raise ScientificIntegrityError(f"Variable {name!r} is missing units metadata")

    if "time" in dataset.coords:
        index = dataset.indexes["time"]
        if not index.is_monotonic_increasing or index.has_duplicates:
            raise ScientificIntegrityError("Time coordinate must be strictly increasing and unique")

    for coordinate in ("hru", "gru", "reach", "station", "basin"):
        if coordinate in dataset.coords:
            values = np.asarray(dataset.coords[coordinate].values)
            if len(np.unique(values)) != len(values):
                raise ScientificIntegrityError(f"Coordinate {coordinate!r} contains duplicate identifiers")

    if balance is not None:
        input_total = sum(float(dataset[name].sum()) for name in balance.inputs)
        output_total = sum(float(dataset[name].sum()) for name in balance.outputs)
        storage = float(dataset[balance.storage_change].sum()) if balance.storage_change else 0.0
        residual = input_total - output_total - storage
        scale = max(abs(input_total), abs(output_total) + abs(storage), 1.0)
        if abs(residual) / scale > balance.relative_tolerance:
            raise ScientificIntegrityError(
                f"Mass-balance residual {residual:g} exceeds relative tolerance "
                f"{balance.relative_tolerance:g}"
            )
