# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""dCoupler adapter for dRoute (differentiable Saint-Venant routing + inline/subgrid lakes).

dRoute is an in-process, differentiable routing solver (CVODES + Enzyme adjoint), so it plugs into
dCoupler as a DifferentiableComponent — the routing counterpart of the in-process JAX land-model
components, but with its gradients supplied by dRoute's own adjoint rather than JAX autograd. The
component consumes a per-reach lateral-inflow series and produces the routed discharge series;
gradients w.r.t. the routing parameters flow through a torch.autograd.Function (``DRouteBridge``)
that calls dRoute's adjoint in the backward pass (mirroring how ``JAXComponent`` uses ``JAXBridge``).

This makes dRoute a coupled STANDALONE model in the graph: ``CouplingGraphBuilder`` wires
``land (SUMMA runoff) -> routing (dRoute lateral_inflow)``. Routing-parameter calibration is owned
by the standalone ``DRouteParameterManager`` (transfer functions); this component handles the
runtime coupling + the differentiable routing step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from dcoupler.core.component import (
    DifferentiableComponent,
    FluxDirection,
    FluxSpec,
    GradientMethod,
    ParameterSpec,
)
from dcoupler.wrappers.process import BMIMixin  # reuse the BMI lifecycle mixin

logger = logging.getLogger(__name__)


class _DRouteEngine:
    """Builds + runs the dRoute SV+lakes network and exposes a record/adjoint interface.

    Kept separate from the torch layer so the autograd bridge can call plain numpy in/out.
    """

    def __init__(self, network_arrays: Dict[str, Any], lakes: Dict[str, Dict],
                 dt_hours: float = 12.0, n_nodes: int = 4):
        self.na = network_arrays
        self.lakes = lakes
        self.dt = dt_hours * 3600.0
        self.sub = int(round(86400.0 / self.dt))
        self.n_nodes = n_nodes
        self.n_reaches = len(network_arrays['seg_ids'])

    def _build(self, manning: np.ndarray, record: bool):
        import droute
        from droute.lake_preprocessor import apply_lake_config_to_network
        na = self.na
        n = self.n_reaches
        oj = n
        junc_up: Dict[int, List[int]] = {i: [] for i in range(n + 1)}
        for i in range(n):
            d = int(na['downstream_idx'][i])
            junc_up[d if d >= 0 else oj].append(i)
        net = droute.Network()
        for jid in range(n + 1):
            j = droute.Junction(); j.id = jid; j.upstream_reach_ids = junc_up[jid]; net.add_junction(j)
        for i in range(n):
            r = droute.Reach(); r.id = i; r.length = float(na['lengths'][i])
            r.slope = max(float(na['slopes'][i]), 0.001); r.manning_n = float(manning[i])
            r.upstream_junction_id = i
            d = int(na['downstream_idx'][i]); r.downstream_junction_id = d if d >= 0 else oj
            net.add_reach(r)
        net.build_topology()
        apply_lake_config_to_network(net, self.lakes, na['id_to_idx'])
        cfg = droute.SaintVenantEnzymeConfig()
        cfg.dt = self.dt; cfg.n_nodes = self.n_nodes
        cfg.enable_adjoint = record; cfg.use_enzyme_adjoint = record
        return net, droute.SaintVenantEnzyme(net, cfg)

    def route(self, manning: np.ndarray, lateral: np.ndarray, record: bool):
        """Route the daily lateral-inflow series [n_days, n_reach]; return daily discharge
        [n_days, n_reach]. With record=True, keep the adjoint tape for gradients()."""
        import contextlib
        import io
        net, rt = self._build(manning, record)
        order = np.asarray(net.topological_order(), dtype=int)
        nd = lateral.shape[0]
        Q = np.zeros((nd, self.n_reaches))
        self._rt = rt
        if record:
            rt.reset_gradients(); rt.start_recording()
        with contextlib.redirect_stderr(io.StringIO()):
            for d in range(nd):
                for _ in range(self.sub):
                    for idx in order:
                        rt.set_lateral_inflow(int(idx), float(lateral[d, idx]))
                    rt.route_timestep()
                Q[d, order] = rt.get_all_discharges()
        if record:
            rt.stop_recording()
        return Q

    def manning_grad(self, gauge_reach: int, dL_dQ_daily: np.ndarray) -> np.ndarray:
        """After a recorded route(), return dL/d(manning) per reach for a loss whose gradient
        w.r.t. the gauge discharge is dL_dQ_daily (one value per recorded day)."""
        # map daily dL/dQ to the per-recorded-step series (loss at the end-of-day step)
        nrec = len(dL_dQ_daily) * self.sub
        series = np.zeros(nrec)
        for k in range(len(dL_dQ_daily)):
            series[(k + 1) * self.sub - 1] = dL_dQ_daily[k]
        self._rt.compute_gradients(int(gauge_reach), series.tolist())
        g = self._rt.get_gradients()
        return np.array([g[f"reach_{i}_manning_n"] for i in range(self.n_reaches)])


class DRouteBridge(torch.autograd.Function):
    """torch.autograd.Function bridging dRoute's Enzyme/CVODES adjoint to torch autograd.

    forward:  route the lateral-inflow series through SV+lakes with the given Manning's n.
    backward: seed dRoute's adjoint with the incoming discharge gradient and return d/d(manning)
              (and None for the non-differentiated inflow series — upstream land models are not
              torch-differentiable). Mirrors JAXBridge but uses dRoute's adjoint as the AD engine.
    """

    @staticmethod
    def forward(ctx, engine: _DRouteEngine, gauge_reach: int, manning: torch.Tensor,
                lateral: torch.Tensor) -> torch.Tensor:
        m = manning.detach().cpu().numpy().astype(float)
        lat = lateral.detach().cpu().numpy().astype(float)
        need_grad = manning.requires_grad
        Q = engine.route(m, lat, record=need_grad)
        ctx.engine = engine
        ctx.gauge_reach = int(gauge_reach)
        ctx.n_reaches = engine.n_reaches
        ctx.need_grad = need_grad
        return torch.from_numpy(Q).to(manning.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        if not ctx.need_grad:
            return None, None, None, None
        # grad_out is dL/dQ for the discharge series [n_days, n_reach]; the dRoute adjoint takes a
        # single gauge series, so contract per gauge reach. Sum manning grads over all reaches that
        # carry a nonzero output gradient (each treated as a gauge), exploiting the linear adjoint.
        go = grad_out.detach().cpu().numpy()
        gmanning = np.zeros(ctx.n_reaches)
        cols = np.where(np.abs(go).sum(axis=0) > 0)[0]
        for r in cols:
            gmanning += ctx.engine.manning_grad(int(r), go[:, r])
        return None, None, torch.from_numpy(gmanning).to(grad_out.dtype), None


class DRouteComponent(DifferentiableComponent, BMIMixin):
    """dRoute (SV + inline/subgrid lakes) as a dCoupler differentiable routing node.

    Inputs:  lateral_inflow [m^3/s] per reach (e.g. from SUMMA runoff x area).
    Outputs: discharge [m^3/s] per reach (routed streamflow).
    Gradients: supplied by dRoute's Enzyme/CVODES adjoint via DRouteBridge (gradient_method=ENZYME).
    """

    def __init__(self, name: str = "droute", config: Optional[dict] = None, **kwargs):
        self._name = name
        self._config = config or {}
        self._engine: Optional[_DRouteEngine] = None
        self._gauge_reach = 0
        self._n_reaches = 0
        self._manning: Optional[torch.nn.Parameter] = None
        self._m_lo, self._m_hi = 0.015, 0.12
        self._state: Any = None
        self._last_outputs: Dict[str, Any] = {}

    # ---- dCoupler component contract --------------------------------------------------------
    @property
    def name(self) -> str:
        return self._name

    @property
    def gradient_method(self) -> GradientMethod:
        return GradientMethod.ENZYME

    @property
    def requires_batch(self) -> bool:
        # dRoute records the whole trajectory in one CVODES/Enzyme pass; it must be driven over the
        # full lateral-inflow sequence (a stepwise graph would rebuild the solver every timestep and
        # break the batch adjoint), so the coupling graph runs it via _forward_batch.
        return True

    @property
    def input_fluxes(self) -> List[FluxSpec]:
        return [FluxSpec("lateral_inflow", "m3/s", FluxDirection.INPUT, "reach", 3600,
                         ("time", "reach"))]

    @property
    def output_fluxes(self) -> List[FluxSpec]:
        return [FluxSpec("discharge", "m3/s", FluxDirection.OUTPUT, "reach", 3600,
                         ("time", "reach"))]

    @property
    def parameters(self) -> List[ParameterSpec]:
        return [ParameterSpec("manning_n", self._m_lo, self._m_hi, spatial=True,
                              n_spatial=self._n_reaches)]

    @property
    def state_size(self) -> int:
        return 0  # dRoute carries its own internal state across the recorded route

    def initialize(self, config: dict) -> None:
        from droute.calibration.worker import DRouteWorker  # reuse input loader (canonical package)
        self._config = {**self._config, **config}
        w = DRouteWorker(self._config, logger)
        from pathlib import Path
        settings_dir = Path(self._config.get('SETTINGS_DROUTE_PATH') or
                            (Path(self._config['DOMAIN_DIR']) / 'settings' / 'dRoute'))
        inp = w._load_inputs(settings_dir)
        na = {k: inp[k] for k in ('seg_ids', 'id_to_idx', 'downstream_idx', 'lengths', 'slopes')}
        self._engine = _DRouteEngine(na, inp['lakes'],
                                     dt_hours=float(self._config.get('DROUTE_ROUTING_DT_HOURS', 12.0)))
        self._n_reaches = self._engine.n_reaches
        self._manning = torch.nn.Parameter(torch.zeros(self._n_reaches))  # 0 -> sigmoid 0.5
        gauge_seg = int(self._config.get('DROUTE_OUTLET_SEG', inp['seg_ids'][inp.get('outlet_idx', 0)]))
        self._gauge_reach = inp['id_to_idx'].get(gauge_seg, 0)

    def get_initial_state(self) -> torch.Tensor:
        return torch.zeros(0)

    def get_torch_parameters(self) -> List[torch.nn.Parameter]:
        return [self._manning] if self._manning is not None else []

    def get_physical_parameters(self) -> Dict[str, torch.Tensor]:
        m = self._m_lo + (self._m_hi - self._m_lo) * torch.sigmoid(self._manning)
        return {"manning_n": m}

    def run(self, inputs: Dict[str, torch.Tensor], state: torch.Tensor, dt: float,
            n_timesteps: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Batch route the whole lateral-inflow series (dRoute records the trajectory once)."""
        manning = self.get_physical_parameters()["manning_n"]
        lateral = inputs["lateral_inflow"]
        Q = DRouteBridge.apply(self._engine, self._gauge_reach, manning, lateral)
        return {"discharge": Q}, state

    def step(self, inputs: Dict[str, torch.Tensor], state: torch.Tensor,
             dt: float) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # dRoute is naturally a whole-series (batch) router; a single step routes one row.
        out, st = self.run({"lateral_inflow": inputs["lateral_inflow"].unsqueeze(0)}, state, dt, 1)
        return {"discharge": out["discharge"][0]}, st

    # ---- BMI lifecycle (drives the coupling graph; mirrors the generic JAXComponent BMI) -----
    def bmi_initialize(self, config: dict) -> None:
        self.initialize(config)
        self._state = self.get_initial_state()

    def bmi_set_value(self, name: str, value: Any) -> None:
        pass  # routing params are set via the torch nn.Parameter / parameter manager

    def bmi_update(self, inputs: Dict[str, Any], dt: float) -> Dict[str, Any]:
        ti = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float64))
              for k, v in inputs.items()}
        if self._state is None:
            self._state = self.get_initial_state()
        out, self._state = self.step(ti, self._state, dt)
        self._last_outputs = {k: v.detach() for k, v in out.items()}
        return {k: v.detach().numpy() for k, v in out.items()}

    def bmi_update_batch(self, inputs: Dict[str, Any], dt: float, n_timesteps: int) -> Dict[str, Any]:
        ti = {k: (v if isinstance(v, torch.Tensor) else torch.as_tensor(v, dtype=torch.float64))
              for k, v in inputs.items()}
        if self._state is None:
            self._state = self.get_initial_state()
        out, self._state = self.run(ti, self._state, dt, n_timesteps)
        self._last_outputs = {k: v.detach() for k, v in out.items()}
        return {k: v.detach().numpy() for k, v in out.items()}

    def bmi_get_value(self, name: str) -> Any:
        if name in self._last_outputs:
            return self._last_outputs[name].numpy()
        raise KeyError(f"Unknown variable '{name}'")

    def bmi_get_state(self) -> Any:
        return self._state

    def bmi_set_state(self, state: Any) -> None:
        self._state = state

    def bmi_get_input_var_names(self) -> List[str]:
        return [f.name for f in self.input_fluxes]

    def bmi_get_output_var_names(self) -> List[str]:
        return [f.name for f in self.output_fluxes]

    def bmi_finalize(self) -> None:
        self._state = None
        self._last_outputs = {}
