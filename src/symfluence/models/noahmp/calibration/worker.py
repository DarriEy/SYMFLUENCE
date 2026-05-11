# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""Noah-MP Calibration Worker."""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask


@OptimizerRegistry.register_worker('NOAHMP')
class NoahMPWorker(BaseWorker):
    _streamflow_metrics = StreamflowMetrics()

    NAMELIST_PARAMS = {'rain_snow_thresh': 'rain_snow_thresh', 'ZREF': 'ZREF', 'refkdt': 'refkdt'}
    SOILPARM_COLUMNS = {'bexp': 1, 'smcmax': 4, 'smcref': 5, 'psisat': 6, 'dksat': 7, 'smcwlt': 9}

    def apply_parameters(self, params, settings_dir: Path, **kwargs) -> bool:
        noahmp_dir = settings_dir if settings_dir.name == 'NOAHMP' else settings_dir / 'NOAHMP'
        nml = noahmp_dir / 'namelist.input'
        if not nml.exists():
            self.logger.error(f"namelist.input not found: {nml}"); return False
        nml_params = {k: v for k, v in params.items() if k in self.NAMELIST_PARAMS}
        soil_params = {k: v for k, v in params.items() if k in self.SOILPARM_COLUMNS}
        if nml_params:
            text = nml.read_text()
            for pn, val in nml_params.items():
                key = self.NAMELIST_PARAMS[pn]
                text = re.sub(rf'(\s*{key}\s*=\s*)[\d.eE+\-]+', rf'\g<1>{val:.6f}', text)
            nml.write_text(text)
        if soil_params:
            self._update_soilparm_tbl(noahmp_dir, nml, soil_params)
        return True

    def _update_soilparm_tbl(self, noahmp_dir, nml_path, params):
        text = nml_path.read_text()
        m = re.search(r'isltyp\s*=\s*(\d+)', text)
        soil_type = int(m.group(1)) if m else 4
        sp = noahmp_dir / 'parameters' / 'SOILPARM.TBL'
        if not sp.exists(): return
        lines = sp.read_text().splitlines(keepends=True)
        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            if len(parts) < 10: continue
            try: idx = int(parts[0].strip())
            except ValueError: continue
            if idx != soil_type: continue
            for pn, val in params.items():
                col = self.SOILPARM_COLUMNS.get(pn)
                if col is not None and col < len(parts):
                    parts[col] = f"  {val:.2E}" if 'E' in parts[col].strip().upper() else f"  {val:.3f}"
            lines[i] = ','.join(parts) + ('\n' if not lines[i].endswith('\n') else '')
            break
        sp.write_text(''.join(lines))

    def run_model(self, config, settings_dir: Path, output_dir: Path, **kwargs) -> bool:
        noahmp_dir = settings_dir if settings_dir.name == 'NOAHMP' else settings_dir / 'NOAHMP'
        data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
        ip = config.get('NOAHMP_INSTALL_PATH', 'default')
        exe = (data_dir / 'installs' / 'noah-owp-modular' if ip == 'default' else Path(ip)) / 'run' / config.get('NOAHMP_EXE', 'noah_owp_modular.exe')
        if not exe.exists(): self.logger.error(f"exe not found: {exe}"); return False
        try:
            r = subprocess.run([str(exe)], capture_output=True, text=True, cwd=str(noahmp_dir), timeout=int(config.get('NOAHMP_TIMEOUT', 7200)))
            if r.returncode != 0: self.logger.error(f"Noah-MP failed (rc={r.returncode})"); return False
            return True
        except (subprocess.TimeoutExpired, OSError) as e:
            self.logger.error(f"Noah-MP error: {e}"); return False

    def calculate_metrics(self, output_dir: Path, config, **kwargs):
        try:
            import xarray as xr
            domain = config.get('DOMAIN_NAME'); data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            proj = data_dir / f"domain_{domain}"
            exp = config.get('EXPERIMENT_ID', 'run_1')
            nc = proj / 'simulations' / exp / 'NOAHMP' / 'output.nc'
            if not nc.exists():
                for c in output_dir.glob('output*.nc'):
                    nc = c; break
            if not nc.exists(): return {'kge': self.penalty_score}
            ds = xr.open_dataset(nc)
            runoff = (ds['SFCRNOFF'] + ds['UGDRNOFF']).to_series() if 'SFCRNOFF' in ds and 'UGDRNOFF' in ds else ds.get('SFCRNOFF', pd.Series(dtype=float)).to_series()
            ds.close()
            obs_dir = proj / 'observations' / 'streamflow' / 'preprocessed'
            obs_files = list(obs_dir.glob('*processed*.csv'))
            if not obs_files: return {'kge': self.penalty_score}
            obs = pd.read_csv(obs_files[0], parse_dates=True, index_col=0).iloc[:, 0]
            if not isinstance(runoff.index, pd.DatetimeIndex): runoff.index = pd.to_datetime(runoff.index)
            if not isinstance(obs.index, pd.DatetimeIndex): obs.index = pd.to_datetime(obs.index)
            rd = runoff.resample('D').mean(); od = obs.resample('D').mean()
            ci = rd.index.intersection(od.index)
            if len(ci) < 10: return {'kge': self.penalty_score}
            s, o = rd.loc[ci].values, od.loc[ci].values
            mask = np.isfinite(s) & np.isfinite(o) & (o >= 0); s, o = s[mask], o[mask]
            if len(s) < 10: return {'kge': self.penalty_score}
            r = np.corrcoef(s, o)[0, 1] if np.std(s) > 0 else 0.0
            alpha = np.std(s) / np.std(o) if np.std(o) > 0 else 0.0
            beta = np.mean(s) / np.mean(o) if np.mean(o) > 0 else 0.0
            kge = 1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
            nse = 1.0 - np.sum((s-o)**2) / np.sum((o-np.mean(o))**2)
            return {'kge': float(kge), 'nse': float(nse), 'n_days': len(s)}
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.logger.error(f"Metrics error: {e}"); return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data):
        w = NoahMPWorker(config=task_data.get('config'))
        return w.evaluate(WorkerTask.from_legacy_dict(task_data)).to_legacy_dict()
