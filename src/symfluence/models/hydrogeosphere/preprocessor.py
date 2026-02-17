"""
HydroGeoSphere Preprocessor

Generates HGS input files for a lumped 1x1 horizontal grid with
vertical layering. For lumped mode the domain is a single column.

Input files generated:
    batch.pfx             — problem prefix
    <prefix>.grok         — main control file (grid, properties, BCs, solver)
    <prefix>.mprops       — material properties
    <prefix>.oprops       — overland properties
    recharge_timeseries   — recharge time series file
"""

import logging
import math
from pathlib import Path

from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("HYDROGEOSPHERE")
class HGSPreProcessor:
    """Generates HGS input files for lumped subsurface simulation."""

    def __init__(self, config, logger, **kwargs):
        self.config = config
        self.logger = logger
        self.config_dict = getattr(config, '_config_dict', {})
        if not self.config_dict and hasattr(config, 'to_dict'):
            self.config_dict = config.to_dict()

        self.project_dir = Path(
            self.config_dict.get('PROJECT_DIR', '.')
        )
        self.domain_name = self._get_cfg('DOMAIN_NAME', 'unknown')
        self.experiment_id = self._get_cfg('EXPERIMENT_ID', 'default')

    def _get_cfg(self, key, default=None):
        """Get config value with fallback."""
        val = self.config_dict.get(key)
        if val is None or val == 'default':
            try:
                if key == 'DOMAIN_NAME':
                    return self.config.domain.name
                elif key == 'EXPERIMENT_ID':
                    return self.config.domain.experiment_id
            except (AttributeError, TypeError):
                pass
            return default
        return val

    def _get_hgs_cfg(self, key, default=None):
        """Get HGS-specific config value."""
        val = self.config_dict.get(f'HGS_{key}', self.config_dict.get(key))
        if val is not None:
            return val
        try:
            hgs_cfg = self.config.model.hydrogeosphere
            if hgs_cfg:
                attr = key.lower()
                if hasattr(hgs_cfg, attr):
                    return getattr(hgs_cfg, attr)
        except (AttributeError, TypeError):
            pass
        return default

    def _get_catchment_area_m2(self) -> float:
        """Get catchment area in m2."""
        area_km2 = self.config_dict.get('CATCHMENT_AREA')
        if area_km2 is None:
            try:
                area_km2 = self.config.domain.catchment_area
            except (AttributeError, TypeError):
                pass
        if area_km2 is None:
            self.logger.warning("CATCHMENT_AREA not set, using default 2210 km2")
            area_km2 = 2210.0
        return float(area_km2) * 1e6

    def _get_time_info(self):
        """Get simulation time range."""
        import pandas as pd

        start = self.config_dict.get('EXPERIMENT_TIME_START')
        end = self.config_dict.get('EXPERIMENT_TIME_END')

        if start is None:
            try:
                start = self.config.domain.time_start
            except (AttributeError, TypeError):
                start = '2000-01-01'
        if end is None:
            try:
                end = self.config.domain.time_end
            except (AttributeError, TypeError):
                end = '2001-01-01'

        start_dt = pd.Timestamp(str(start))
        end_dt = pd.Timestamp(str(end))
        n_days = (end_dt - start_dt).days
        if n_days <= 0:
            n_days = 365

        return start_dt, end_dt, n_days

    def run_preprocessing(self):
        """Generate all HGS input files."""
        settings_dir = self.project_dir / "settings" / "HYDROGEOSPHERE"
        settings_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating HGS input files in {settings_dir}")

        # Get parameters
        area_m2 = self._get_catchment_area_m2()
        domain_width = float(self._get_hgs_cfg('DOMAIN_WIDTH', math.sqrt(area_m2)))
        k_sat = float(self._get_hgs_cfg('K_SAT', 1e-5))
        porosity = float(self._get_hgs_cfg('POROSITY', 0.4))
        vg_alpha = float(self._get_hgs_cfg('VG_ALPHA', 1.0))
        vg_n = float(self._get_hgs_cfg('VG_N', 2.0))
        vg_sres = float(self._get_hgs_cfg('VG_SRES', 0.05))
        ss = float(self._get_hgs_cfg('SS', 1e-4))
        mannings_n = float(self._get_hgs_cfg('MANNINGS_N', 0.03))
        soil_depth = float(self._get_hgs_cfg('SOIL_DEPTH', 10.0))
        solver_max_iter = int(self._get_hgs_cfg('SOLVER_MAX_ITERATIONS', 25))
        timestep = int(self._get_hgs_cfg('TIMESTEP_SECONDS', 3600))

        start_dt, end_dt, n_days = self._get_time_info()
        total_time_seconds = n_days * 86400

        prefix = "hgs_lumped"

        self.logger.info(
            f"HGS lumped grid: width={domain_width:.0f}m, "
            f"K_sat={k_sat}, porosity={porosity}, depth={soil_depth}m"
        )

        # Write all files
        self._write_batch_pfx(settings_dir, prefix)
        self._write_grok(settings_dir, prefix, domain_width, soil_depth,
                         total_time_seconds, timestep, solver_max_iter)
        self._write_mprops(settings_dir, prefix, k_sat, porosity,
                           vg_alpha, vg_n, vg_sres, ss)
        self._write_oprops(settings_dir, prefix, mannings_n)
        self._write_recharge_ts(settings_dir, start_dt, n_days)

        self.logger.info(f"HGS input files generated in {settings_dir}")

    def _write_batch_pfx(self, d: Path, prefix: str):
        """Write batch prefix file."""
        (d / "batch.pfx").write_text(f"{prefix}\n")

    def _write_grok(self, d: Path, prefix: str, width, depth,
                    total_time, timestep, max_iter):
        """Write main grok control file."""
        (d / f"{prefix}.grok").write_text(
            f"!--- HGS Control File (SYMFLUENCE lumped mode) ---\n"
            f"\n"
            f"!--- Grid definition ---\n"
            f"generate rectangles\n"
            f"  {width:.2f} {width:.2f}\n"
            f"  1 1\n"
            f"end generate rectangles\n"
            f"\n"
            f"generate layers interactive\n"
            f"  new layer\n"
            f"    layer name\n"
            f"    subsurface\n"
            f"    elevation\n"
            f"    {depth:.2f}\n"
            f"  end new layer\n"
            f"  new layer\n"
            f"    layer name\n"
            f"    surface\n"
            f"    elevation\n"
            f"    0.0\n"
            f"  end new layer\n"
            f"end generate layers interactive\n"
            f"\n"
            f"!--- Porous media properties ---\n"
            f"use domain type\n"
            f"porous media\n"
            f"\n"
            f"properties file\n"
            f"{prefix}.mprops\n"
            f"\n"
            f"!--- Overland flow ---\n"
            f"use domain type\n"
            f"surface\n"
            f"\n"
            f"properties file\n"
            f"{prefix}.oprops\n"
            f"\n"
            f"!--- Boundary conditions ---\n"
            f"!-- Recharge from file --\n"
            f"use domain type\n"
            f"porous media\n"
            f"\n"
            f"clear chosen nodes\n"
            f"choose nodes top\n"
            f"specified flux\n"
            f"  node\n"
            f"  set 1\n"
            f"  face\n"
            f"  set 1\n"
            f"  time file table\n"
            f"  recharge_timeseries\n"
            f"end specified flux\n"
            f"\n"
            f"!--- Output ---\n"
            f"output nodes\n"
            f"  make observation point\n"
            f"  Outlet\n"
            f"  0.0 0.0 0.0\n"
            f"end output nodes\n"
            f"\n"
            f"!--- Solver ---\n"
            f"initial timestep\n"
            f"{timestep:.1f}\n"
            f"\n"
            f"maximum timestep\n"
            f"86400.0\n"
            f"\n"
            f"newton maximum iterations\n"
            f"{max_iter}\n"
            f"\n"
            f"!--- Simulation time ---\n"
            f"simulation time\n"
            f"{total_time:.1f}\n"
        )

    def _write_mprops(self, d: Path, prefix: str, k_sat, porosity,
                      vg_alpha, vg_n, vg_sres, ss):
        """Write material properties file."""
        (d / f"{prefix}.mprops").write_text(
            f"!--- Material properties ---\n"
            f"subsurface\n"
            f"\n"
            f"k isotropic\n"
            f"{k_sat:.6e}\n"
            f"\n"
            f"porosity\n"
            f"{porosity:.4f}\n"
            f"\n"
            f"specific storage\n"
            f"{ss:.6e}\n"
            f"\n"
            f"unsaturated tables\n"
            f"van genuchten\n"
            f"  {vg_alpha:.4f}\n"
            f"  {vg_n:.4f}\n"
            f"  {vg_sres:.4f}\n"
            f"  0.0\n"
            f"end unsaturated tables\n"
        )

    def _write_oprops(self, d: Path, prefix: str, mannings_n):
        """Write overland flow properties file."""
        (d / f"{prefix}.oprops").write_text(
            f"!--- Overland flow properties ---\n"
            f"surface\n"
            f"\n"
            f"friction\n"
            f"manning\n"
            f"{mannings_n:.4f}\n"
            f"\n"
            f"rill storage height\n"
            f"0.001\n"
            f"\n"
            f"obstruction storage height\n"
            f"0.001\n"
        )

    def _write_recharge_ts(self, d: Path, start_dt, n_days):
        """Write placeholder recharge time series file.

        The coupler overwrites this with actual SUMMA recharge data
        for coupled runs.
        """
        lines = []
        for i in range(n_days + 1):
            t = i * 86400.0
            lines.append(f"{t:.1f} 1.0e-8")
        (d / "recharge_timeseries").write_text("\n".join(lines) + "\n")
