# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Generic coupled-model calibration worker.

Runs an arbitrary chain of coupled standalone models (land + optional snow/groundwater/routing) and
scores one shared objective. Parameters are split + delegated to each model's own manager; the run
executes the dCoupler CouplingGraph when that backend is enabled, otherwise falls back to sequential
file-based delegation in graph order (each downstream model's standalone worker discovers its
upstream model's output). The objective is delegated to the objective model's worker (the routing
model when present, else the land model) -- e.g. multi-gauge streamflow KGE for SUMMA+dRoute.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from symfluence.core.exceptions import ConfigurationError
from symfluence.core.registries import R
from symfluence.optimization.workers.base_worker import BaseWorker

from .parameter_manager import (
    CoupledModelParameterManager,
    coupled_component_models,
    settings_subdir,
)


@R.workers.add('COUPLED')
class CoupledModelWorker(BaseWorker):
    """Runs the coupled-model chain and scores the objective model's metric."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        super().__init__(config, logger)
        cfg = config or {}
        self._models: List[str] = coupled_component_models(cfg, self._cfg_get)
        self.land_model = self._models[0] if self._models else \
            str(cfg.get('HYDROLOGICAL_MODEL', 'SUMMA')).split(',')[0].upper()
        self.objective_model = self._resolve_objective_model(cfg)
        self._workers: Dict[str, BaseWorker] = {}
        self._pm: Optional[CoupledModelParameterManager] = None

    # ---- helpers ----------------------------------------------------------------------------
    def _cfg_get(self, _config, key, default):
        return self._get_config_value(lambda: None, default=default, dict_key=key)

    def _resolve_objective_model(self, cfg: Dict[str, Any]) -> str:
        override = str(self._cfg_get(cfg, 'COUPLED_OBJECTIVE_MODEL', '') or '').upper()
        if override and override in self._models:
            return override
        # Default: the routing model owns the streamflow objective; else the last/only model.
        routing = str(self._cfg_get(cfg, 'ROUTING_MODEL', '') or '').split(',')[0].upper()
        if routing and routing in self._models:
            return routing
        return self._models[-1] if self._models else self.land_model

    def _worker(self, model: str) -> BaseWorker:
        if model not in self._workers:
            cls = R.workers.get(model)
            if cls is None:
                # Workers (unlike parameter managers/optimizers) are not auto-discovered; import
                # the model's worker module so its @R.workers.add decorator fires, then retry.
                import importlib
                try:
                    importlib.import_module(f"symfluence.models.{model.lower()}.calibration.worker")
                except ImportError as e:
                    raise ConfigurationError(f"Could not import worker for '{model}': {e}") from e
                cls = R.workers.get(model)
            if cls is None:
                raise ConfigurationError(
                    f"No worker registered for '{model}'. Available: {sorted(R.workers.keys())}")
            self._workers[model] = cls(config=self.config, logger=self.logger)
        return self._workers[model]

    def _settings_for(self, settings_dir: Path, model: str) -> Path:
        root = Path(settings_dir)
        sub = settings_subdir(model)
        if root.name in (model, sub):
            root = root.parent
        return root / sub

    def _param_manager(self, settings_dir: Path) -> CoupledModelParameterManager:
        if self._pm is None:
            self._pm = CoupledModelParameterManager(self.config or {}, self.logger, Path(settings_dir))
        return self._pm

    # ---- BaseWorker contract ----------------------------------------------------------------
    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs) -> bool:
        # Split + delegate to each participating model's standalone parameter manager.
        return self._param_manager(settings_dir).update_model_files(params)

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        if self._use_dcoupler(config):
            try:
                return self._run_dcoupler(config, settings_dir, output_dir, **kwargs)
            except (RuntimeError, ValueError, KeyError, TypeError, ImportError, OSError) as e:
                self.logger.error(f"dCoupler coupled run failed ({e}); falling back to sequential",
                                  exc_info=True)
        return self._run_sequential(config, settings_dir, output_dir, **kwargs)

    def _use_dcoupler(self, config: Dict[str, Any]) -> bool:
        backend = str(self._cfg_get(config, 'COUPLING_BACKEND', '') or
                      self._cfg_get(config, 'COUPLING_MODE', '') or '').lower()
        if backend != 'dcoupler':
            return False
        try:
            from symfluence.coupling import is_dcoupler_available
            return bool(is_dcoupler_available())
        except ImportError:
            return False

    def _run_dcoupler(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        import torch

        from symfluence.coupling.graph_builder import CouplingGraphBuilder
        graph = CouplingGraphBuilder().build(config)
        outputs = graph.forward(external_inputs=kwargs.get('external_inputs', {}),
                                n_timesteps=kwargs.get('n_timesteps', 1),
                                dt=kwargs.get('dt', 86400.0))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for comp_name, comp_outputs in outputs.items():
            cdir = Path(output_dir) / comp_name.upper(); cdir.mkdir(parents=True, exist_ok=True)
            for flux_name, tensor in comp_outputs.items():
                torch.save(tensor, cdir / f"{flux_name}.pt")
        # Graph->metrics handoff: the dCoupler graph returns raw flux tensors, but each model's
        # calculate_metrics reads its own on-disk format. Let the objective model materialize its
        # metric inputs from the graph outputs (optional hook) so calculate_metrics works unchanged.
        obj_worker = self._worker(self.objective_model)
        materialize = getattr(obj_worker, 'materialize_metric_inputs', None)
        if callable(materialize):
            obj_dir = Path(output_dir) / self.objective_model
            obj_dir.mkdir(parents=True, exist_ok=True)
            materialize(outputs, obj_dir, self._settings_for(settings_dir, self.objective_model), config)
        self.logger.info(f"Coupled run via dCoupler completed over {self._models}")
        return True

    def _run_sequential(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        """Run each model in graph order; each downstream worker discovers its upstream output."""
        run_kwargs = {k: v for k, v in kwargs.items() if k not in ('sim_dir', 'output_dir', 'settings_dir')}
        for model in self._models:
            mdir = Path(output_dir) / model
            mdir.mkdir(parents=True, exist_ok=True)
            ok = self._worker(model).run_model(config, self._settings_for(settings_dir, model), mdir, **run_kwargs)
            if not ok:
                self.logger.error(f"Coupled chain failed at model '{model}'")
                return False
        return True

    def calculate_metrics(self, output_dir: Path, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # The objective model (routing if present, else land) owns the shared objective.
        obj_dir = Path(output_dir) / self.objective_model
        if not obj_dir.exists():
            obj_dir = Path(output_dir)
        return self._worker(self.objective_model).calculate_metrics(obj_dir, config, **kwargs)
