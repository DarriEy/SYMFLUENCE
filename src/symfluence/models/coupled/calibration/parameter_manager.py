# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Generic coupled-model parameter manager.

Composes the standalone parameter managers of *whatever* models a dCoupler coupling graph wires
together -- the land-surface model plus any configured snow / groundwater / routing model -- into a
single joint parameter space, then splits a combined parameter vector back to each model and
delegates the file writes to that model's own manager. No model-specific logic lives here: each
participating model keeps its own standalone manager (e.g. SUMMA, DROUTE, MODFLOW, MIZUROUTE) and
this class only orchestrates the union/split. This generalizes the SUMMA+MODFLOW ``COUPLED_GW``
manager to an arbitrary set of coupled standalone models.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.registries import R
from symfluence.optimization.core.base_parameter_manager import BaseParameterManager

# Settings sub-directory per model (defaults to the model name when absent).
_SETTINGS_SUBDIR: Dict[str, str] = {
    "DROUTE": "dRoute",
    "MIZUROUTE": "mizuRoute",
    "SUMMA": "SUMMA",
    "MODFLOW": "MODFLOW",
    "MESH": "MESH",
    "CLM": "CLM",
}

# Config keys, in graph order, that may name a participating model.
_MODEL_KEYS: Tuple[str, ...] = (
    "HYDROLOGICAL_MODEL", "SNOW_MODULE", "GROUNDWATER_MODEL", "ROUTING_MODEL",
)
_NULL_MODELS = {"", "NONE", "N/A", "NULL"}


def settings_subdir(model: str) -> str:
    """Settings sub-directory for a model (e.g. DROUTE -> 'dRoute')."""
    return _SETTINGS_SUBDIR.get(model.upper(), model)


def coupled_component_models(config: Any, get: Any) -> List[str]:
    """Ordered, de-duplicated list of coupled models that expose a parameter manager.

    Reads HYDROLOGICAL_MODEL (land) plus any SNOW_MODULE / GROUNDWATER_MODEL / ROUTING_MODEL from
    config, keeps those that name a real model with a registered parameter manager, and (optionally)
    restricts to ``COUPLED_CALIBRATE_MODELS`` when that key is set. The land model is always first.
    """
    restrict_raw = get(config, "COUPLED_CALIBRATE_MODELS", None)
    restrict = None
    if restrict_raw and str(restrict_raw).lower() not in ("all", "default"):
        restrict = {m.strip().upper() for m in str(restrict_raw).split(",") if m.strip()}

    models: List[str] = []
    for key in _MODEL_KEYS:
        raw = get(config, key, "")
        name = str(raw or "").split(",")[0].strip().upper()
        if name in _NULL_MODELS or name in models:
            continue
        if R.parameter_managers.get(name) is None:
            continue  # model can run in the graph but is not calibratable -> not in the param space
        if restrict is not None and name not in restrict:
            continue
        models.append(name)
    return models


@R.parameter_managers.add('COUPLED')
class CoupledModelParameterManager(BaseParameterManager):
    """Joint parameter space over an arbitrary set of coupled standalone models (delegating)."""

    def __init__(self, config: Any, logger: logging.Logger, settings_dir: Path):
        super().__init__(config, logger, settings_dir)
        self._root = Path(settings_dir)
        self._models = coupled_component_models(config, self._cfg_get)
        if not self._models:
            raise ConfigurationError(
                "COUPLED calibration requires at least one coupled model with a registered "
                "parameter manager (set HYDROLOGICAL_MODEL/ROUTING_MODEL/GROUNDWATER_MODEL)."
            )
        self._pms: Dict[str, BaseParameterManager] = {m: self._make_pm(m) for m in self._models}
        # Map each combined parameter name to the model that owns it (first owner wins on collision).
        self._owner: Dict[str, str] = {}
        for m in self._models:
            for name in self._pms[m].all_param_names:
                if name in self._owner:
                    self.logger.warning(
                        f"Parameter '{name}' is provided by both {self._owner[name]} and {m}; "
                        f"keeping {self._owner[name]}")
                    continue
                self._owner[name] = m
        self.logger.info(f"COUPLED parameter space over {self._models}: "
                         f"{len(self._owner)} parameters")

    # ---- helpers ----------------------------------------------------------------------------
    def _cfg_get(self, _config, key, default):
        return self._get_config_value(lambda: None, default=default, dict_key=key)

    def _settings_for(self, model: str) -> Path:
        sub = settings_subdir(model)
        d = self._root / sub
        if d.exists():
            return d
        alt = self._root.parent / sub
        return alt if alt.exists() else d

    def _make_pm(self, model: str) -> BaseParameterManager:
        cls = R.parameter_managers.get(model)
        if cls is None:
            raise ConfigurationError(f"No parameter manager registered for '{model}'")
        return cls(self.config_dict, self.logger, self._settings_for(model))

    def _split(self, params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Partition a combined parameter dict into per-model sub-dicts by ownership."""
        out: Dict[str, Dict[str, Any]] = {m: {} for m in self._models}
        for name, val in params.items():
            owner = self._owner.get(name)
            if owner is not None:
                out[owner][name] = val
        return out

    # ---- BaseParameterManager contract ------------------------------------------------------
    def _get_parameter_names(self) -> List[str]:
        names: List[str] = []
        for m in self._models:
            names.extend(self._pms[m].all_param_names)
        return names

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        bounds: Dict[str, Dict[str, Any]] = {}
        for m in self._models:
            bounds.update(self._pms[m].param_bounds)
        return bounds

    def _format_parameter_value(self, param_name: str, value: float) -> Any:
        owner = self._owner.get(param_name)
        if owner is not None:
            return self._pms[owner]._format_parameter_value(param_name, value)
        return float(value)

    def denormalize_parameters(self, normalized_array: np.ndarray) -> Dict[str, Any]:
        params = super().denormalize_parameters(normalized_array)
        # Let each model enforce its own constraints on its slice.
        for m, sub in self._split(params).items():
            pm = self._pms[m]
            if sub and hasattr(pm, '_enforce_parameter_constraints'):
                params.update(pm._enforce_parameter_constraints(sub))
        return params

    def update_model_files(self, params: Dict[str, Any]) -> bool:
        ok = True
        for m, sub in self._split(params).items():
            if sub:
                ok = self._pms[m].update_model_files(sub) and ok
        return ok

    def get_initial_parameters(self) -> Optional[Dict[str, Any]]:
        init: Dict[str, Any] = {}
        for m in self._models:
            sub = self._pms[m].get_initial_parameters() or {}
            init.update(sub)
        return init
