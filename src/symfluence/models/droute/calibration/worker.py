# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""dRoute calibration worker: route SUMMA runoff through the differentiable Saint-Venant solver
(with inline + subgrid lakes), applying transfer-function-regionalized routing parameters, and
score against the multi-gauge streamflow objective.

The worker implements the BaseWorker template (apply_parameters / run_model / calculate_metrics):
 - apply_parameters: expand the calibration coefficients to per-reach routing params (via the
   dRoute parameter manager) and write them out;
 - run_model: build the dRoute network, apply the per-reach params + lake config, route the
   (clipped) SUMMA runoff at a sub-daily step with a spin-up, and save discharge at the gauges;
 - calculate_metrics: per-gauge KGE vs WSC observations, aggregated (mean) over the gauges that
   routing can fit.

The routing recipe mirrors the validated standalone experiment (dt=12h, clip the SUMMA spin-up
runoff artifact, spin up before the evaluation window). The SV solver carries inline + subgrid
lake routing as ODE storage states, so the same network drives both forward routing and (for the
later gradient-acceleration stage) the adjoint.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from symfluence.core.registries import R
from symfluence.optimization.workers.base_worker import BaseWorker

from .parameter_manager import DRouteParameterManager

DT_DAY = 86400.0


def _kge(sim: np.ndarray, obs: np.ndarray) -> float:
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(s) < 10 or o.std() == 0:
        return np.nan
    r = np.corrcoef(s, o)[0, 1]
    return float(1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2))


def build_droute_network(seg_ids, downstream_idx, lengths, slopes, lakes, id_to_idx,
                         per_reach: Optional[Dict[str, List[float]]] = None):
    """Build a dRoute Network with Manning's n + inline/subgrid lake config, optionally overriding
    per-reach routing params from a parameter-manager expansion (``per_reach[param][reach_idx]``)."""
    import droute
    from droute.lake_preprocessor import apply_lake_config_to_network
    n = len(seg_ids)
    outlet_junc = n
    junc_up: Dict[int, List[int]] = {i: [] for i in range(n + 1)}
    for i in range(n):
        d = downstream_idx[i]
        junc_up[d if d >= 0 else outlet_junc].append(i)
    net = droute.Network()
    for jid in range(n + 1):
        j = droute.Junction(); j.id = jid; j.upstream_reach_ids = junc_up[jid]; net.add_junction(j)
    for i in range(n):
        r = droute.Reach(); r.id = i; r.length = float(lengths[i]); r.slope = max(float(slopes[i]), 0.001)
        r.manning_n = 0.035
        r.upstream_junction_id = i
        d = downstream_idx[i]; r.downstream_junction_id = d if d >= 0 else outlet_junc
        net.add_reach(r)
    net.build_topology()
    apply_lake_config_to_network(net, lakes, id_to_idx)
    # per-reach parameter overrides from the regionalized calibration
    if per_reach:
        setter = {'manning_n': 'manning_n', 'lake_q_ref': 'lake_q_ref', 'lake_exp': 'lake_exp',
                  'lake_q_min': 'lake_q_min', 'lake_spill_coef': 'lake_spill_coef',
                  'subgrid_q_ref': 'subgrid_q_ref', 'subgrid_exp': 'subgrid_exp'}
        for pname, vals in per_reach.items():
            attr = setter.get(pname)
            if attr is None:
                continue
            for i in range(n):
                setattr(net.get_reach(i), attr, float(vals[i]))
    return net


@R.workers.add('DROUTE')
class DRouteWorker(BaseWorker):
    """Worker that routes SUMMA runoff through dRoute (SV + lakes) and scores multi-gauge KGE."""

    def __init__(self, config: Any, logger: logging.Logger):
        super().__init__(config, logger)
        self._inp: Optional[Dict[str, Any]] = None  # cached inputs

    # ---- config helpers ---------------------------------------------------------------------
    def _gv(self, key, default=None):
        return self._get_config_value(lambda: None, default=default, dict_key=key)

    # ---- inputs (runoff, network arrays, gauges) --------------------------------------------
    def _load_inputs(self, settings_dir: Path) -> Dict[str, Any]:
        if self._inp is not None:
            return self._inp
        import glob

        import geopandas as gpd
        import pandas as pd
        import xarray as xr

        domain = Path(settings_dir).parent.parent
        rn_path = self._gv('RIVER_NETWORK_SHAPEFILE') or glob.glob(
            str(domain / 'shapefiles' / 'river_network' / '*.shp'))[0]
        rn = gpd.read_file(rn_path)
        seg_ids = rn['LINKNO'].astype(int).values
        id_to_idx = {int(s): i for i, s in enumerate(seg_ids)}
        downstream_idx = np.array([id_to_idx.get(int(d), -1) for d in rn['DSLINKNO'].astype(int).values])
        lengths = rn['Length'].astype(float).values
        slopes = rn['Slope'].astype(float).values

        runoff_path = self._gv('DROUTE_RUNOFF_FILE')
        if not runoff_path:
            runoff_path = glob.glob(str(domain / 'simulations' / '*' / 'SUMMA' / '*_timestep.nc'))[0]
        ds = xr.open_dataset(runoff_path)
        runoff = np.clip(ds['averageRoutedRunoff'].values, 0, None)
        gru = ds['gruId'].values.astype(int)
        time = pd.to_datetime(ds['time'].values)
        attr = xr.open_dataset(domain / 'settings' / 'SUMMA' / 'attributes.nc')
        area_by_id = {int(h): float(a) for h, a in zip(attr['hruId'].values.astype(int),
                                                       attr['HRUarea'].values.astype(float))}
        n_seg = len(seg_ids)
        seg_runoff = np.zeros((len(time), n_seg))
        for jx, gid in enumerate(gru):
            i = id_to_idx.get(int(gid))
            if i is not None:
                seg_runoff[:, i] = runoff[:, jx] * area_by_id.get(int(gid), 0.0)
        cap = float(self._gv('DROUTE_RUNOFF_CAP', 50.0))
        daily = pd.DataFrame(np.clip(seg_runoff, 0, cap), index=time).resample('D').mean()
        route_start = self._gv('DROUTE_ROUTE_START', '2010-01-01')
        eval_start = self._gv('CALIBRATION_PERIOD_START', '2011-01-01')
        eval_end = self._gv('CALIBRATION_PERIOD_END', '2012-12-31')
        daily = daily.loc[route_start:eval_end]

        import yaml
        lf = Path(settings_dir) / 'droute_lakes.yaml'
        raw = yaml.safe_load(open(lf, encoding='utf-8')) if lf.exists() else {}
        lakes = {'inline': (raw or {}).get('inline_lakes', {}) or {},
                 'subgrid': (raw or {}).get('subgrid_lakes', {}) or {}}

        # gauges from the mapping CSV (station, seg) + WSC daily obs
        obs_dir = self._gv('MULTI_GAUGE_OBS_DIR') or str(domain / 'observations' / 'streamflow')
        gmap = pd.read_csv(Path(obs_dir) / 'gauge_seg_mapping.csv')
        i_eval0 = int(np.argmax(daily.index >= pd.Timestamp(eval_start)))
        rec_dates = daily.index[i_eval0:]
        gauges = []
        for _, row in gmap.iterrows():
            seg = int(row['seg']); ridx = id_to_idx.get(seg)
            if ridx is None:
                continue
            o = pd.read_csv(Path(obs_dir) / f"wsc_{row['station']}_daily.csv",
                            parse_dates=['date']).set_index('date')['q_cms'].reindex(rec_dates).values
            if np.isfinite(o).sum() >= 10:
                gauges.append({'station': str(row['station']), 'ridx': int(ridx), 'obs': o.astype(float)})

        self._inp = dict(seg_ids=seg_ids, id_to_idx=id_to_idx, downstream_idx=downstream_idx,
                         lengths=lengths, slopes=slopes, daily=daily.values, dates=daily.index,
                         i_eval0=i_eval0, lakes=lakes, gauges=gauges, rn_path=rn_path)
        return self._inp

    # ---- BaseWorker contract ----------------------------------------------------------------
    def apply_parameters(self, params: Dict[str, float], settings_dir: Path, **kwargs) -> bool:
        pm = DRouteParameterManager(self.config, self.logger, Path(settings_dir))
        return pm.update_model_files(params)

    def run_model(self, config: Dict[str, Any], settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        import droute
        inp = self._load_inputs(Path(settings_dir))
        per_reach = None
        pf = Path(settings_dir) / 'droute_routing_params.json'
        if pf.exists():
            per_reach = json.load(open(pf))['params']
        net = build_droute_network(inp['seg_ids'], inp['downstream_idx'], inp['lengths'],
                                   inp['slopes'], inp['lakes'], inp['id_to_idx'], per_reach)
        dt_h = float(self._gv('DROUTE_ROUTING_DT_HOURS', 12.0))
        sub = int(round(DT_DAY / (dt_h * 3600.0)))
        c = droute.SaintVenantEnzymeConfig()
        c.dt = dt_h * 3600.0; c.n_nodes = int(self._gv('DROUTE_SV_NODES', 4))
        c.enable_adjoint = False; c.use_enzyme_adjoint = False
        rt = droute.SaintVenantEnzyme(net, c)
        order = np.asarray(net.topological_order(), dtype=int)
        runoff = inp['daily']; i0 = inp['i_eval0']; ndays = runoff.shape[0]
        import contextlib
        import io
        gauges = inp['gauges']
        Qd: Dict[int, List[float]] = {g['ridx']: [] for g in gauges}
        with contextlib.redirect_stderr(io.StringIO()):
            for d in range(ndays):
                for s in range(sub):
                    for idx in order:
                        rt.set_lateral_inflow(int(idx), float(runoff[d, idx]))
                    rt.route_timestep()
                if d >= i0:
                    for g in gauges:
                        Qd[g['ridx']].append(rt.get_discharge(g['ridx']))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(output_dir) / 'droute_streamflow.npz',
                 ridx=np.array([g['ridx'] for g in gauges]),
                 stations=np.array([g['station'] for g in gauges]),
                 Q=np.array([Qd[g['ridx']] for g in gauges]),
                 obs=np.array([g['obs'] for g in gauges]))
        return True

    def calculate_metrics(self, output_dir: Path, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        z = np.load(Path(output_dir) / 'droute_streamflow.npz', allow_pickle=True)
        Q, obs, stations = z['Q'], z['obs'], z['stations']
        floor = float(self._gv('MULTI_GAUGE_KGE_FLOOR', -2.0))
        per = {}
        kept = []
        for i, st in enumerate(stations):
            k = _kge(Q[i], obs[i])
            per[f'KGE_{st}'] = k
            if np.isfinite(k) and k >= floor:   # drop volume-biased gauges routing can't fit
                kept.append(k)
        mean_kge = float(np.mean(kept)) if kept else floor
        per['KGE'] = mean_kge
        per['calib_score'] = mean_kge   # maximization objective
        return per
