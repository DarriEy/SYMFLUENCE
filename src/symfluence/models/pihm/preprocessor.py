"""
PIHM Preprocessor

Generates PIHM input files for a lumped single-element mesh model.
The single triangular element represents the entire catchment.

Input files generated:
    <project>.mesh  — triangulated mesh (single triangle for lumped)
    <project>.att   — soil/land attributes
    <project>.soil  — soil layer parameters
    <project>.lc    — land cover properties
    <project>.para  — solver parameters
    <project>.calib — calibration multipliers
    <project>.init  — initial conditions
    <project>.forc  — meteorological forcing (recharge time series)
"""

import logging
import math
from pathlib import Path

from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("PIHM")
class PIHMPreProcessor:
    """Generates PIHM input files for lumped groundwater simulation."""

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

    def _get_pihm_cfg(self, key, default=None):
        """Get PIHM-specific config value."""
        val = self.config_dict.get(f'PIHM_{key}', self.config_dict.get(key))
        if val is not None:
            return val
        try:
            pihm_cfg = self.config.model.pihm
            if pihm_cfg:
                attr = key.lower()
                if hasattr(pihm_cfg, attr):
                    return getattr(pihm_cfg, attr)
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
        """Generate all PIHM input files."""
        settings_dir = self.project_dir / "settings" / "PIHM"
        settings_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating PIHM input files in {settings_dir}")

        # Get parameters
        area_m2 = self._get_catchment_area_m2()
        k_sat = float(self._get_pihm_cfg('K_SAT', 1e-5))
        porosity = float(self._get_pihm_cfg('POROSITY', 0.4))
        vg_alpha = float(self._get_pihm_cfg('VG_ALPHA', 1.0))
        vg_n = float(self._get_pihm_cfg('VG_N', 2.0))
        macropore_k = float(self._get_pihm_cfg('MACROPORE_K', 1e-4))
        macropore_depth = float(self._get_pihm_cfg('MACROPORE_DEPTH', 0.5))
        soil_depth = float(self._get_pihm_cfg('SOIL_DEPTH', 2.0))
        mannings_n = float(self._get_pihm_cfg('MANNINGS_N', 0.03))
        init_gw_depth = float(self._get_pihm_cfg('INIT_GW_DEPTH', 1.0))
        solver_reltol = float(self._get_pihm_cfg('SOLVER_RELTOL', 1e-3))
        solver_abstol = float(self._get_pihm_cfg('SOLVER_ABSTOL', 1e-4))
        timestep = int(self._get_pihm_cfg('TIMESTEP_SECONDS', 60))

        start_dt, end_dt, n_days = self._get_time_info()

        project_name = "pihm_lumped"

        self.logger.info(
            f"PIHM lumped mesh: area={area_m2/1e6:.0f} km², "
            f"K_sat={k_sat}, porosity={porosity}, soil_depth={soil_depth}m"
        )

        # Write all input files
        self._write_mesh(settings_dir, project_name, area_m2)
        self._write_att(settings_dir, project_name)
        self._write_soil(settings_dir, project_name, k_sat, porosity,
                         vg_alpha, vg_n, macropore_k, macropore_depth, soil_depth)
        self._write_lc(settings_dir, project_name, mannings_n)
        self._write_para(settings_dir, project_name, start_dt, end_dt,
                         timestep, solver_reltol, solver_abstol)
        self._write_calib(settings_dir, project_name)
        self._write_init(settings_dir, project_name, init_gw_depth)
        self._write_forc(settings_dir, project_name, start_dt, n_days)

        self.logger.info(f"PIHM input files generated in {settings_dir}")

    def _write_mesh(self, d: Path, name: str, area_m2: float):
        """Write mesh file — single triangle for lumped mode."""
        side_len = math.sqrt(4.0 * area_m2 / math.sqrt(3.0))
        (d / f"{name}.mesh").write_text(
            f"3\n"
            f"1 0.0 0.0\n"
            f"2 {side_len:.2f} 0.0\n"
            f"3 {side_len/2:.2f} {side_len*math.sqrt(3)/2:.2f}\n"
            f"1\n"
            f"1 1 2 3\n"
            f"1\n"
            f"1 1 2\n"
        )

    def _write_att(self, d: Path, name: str):
        """Write attribute file — single element, single soil/lc type."""
        (d / f"{name}.att").write_text(
            "1\n"
            "1 1 1 0 0 0\n"
        )

    def _write_soil(self, d: Path, name: str, k_sat, porosity,
                    vg_alpha, vg_n, macropore_k, macropore_depth, soil_depth):
        """Write soil parameter file."""
        (d / f"{name}.soil").write_text(
            "1\n"
            f"1 {soil_depth:.4f} {k_sat:.6e} {k_sat/10:.6e} "
            f"{macropore_k:.6e} {macropore_depth:.4f} "
            f"{porosity:.4f} {vg_alpha:.4f} {vg_n:.4f} 0.05\n"
        )

    def _write_lc(self, d: Path, name: str, mannings_n):
        """Write land cover file."""
        (d / f"{name}.lc").write_text(
            "1\n"
            f"1 0.10 {mannings_n:.4f} 0.0\n"
        )

    def _write_para(self, d: Path, name: str, start_dt, end_dt,
                    timestep, reltol, abstol):
        """Write solver parameter file."""
        start_epoch = int(start_dt.timestamp())
        end_epoch = int(end_dt.timestamp())
        (d / f"{name}.para").write_text(
            f"START {start_epoch}\n"
            f"END {end_epoch}\n"
            f"DT {timestep}\n"
            f"OUTPUT_INTERVAL 86400\n"
            f"RELTOL {reltol:.1e}\n"
            f"ABSTOL {abstol:.1e}\n"
            f"INIT_MODE 0\n"
            f"VERBOSE 0\n"
            f"DEBUG 0\n"
        )

    def _write_calib(self, d: Path, name: str):
        """Write calibration multiplier file (all 1.0 = no adjustment)."""
        (d / f"{name}.calib").write_text(
            "KSATH 1.0\n"
            "KSATV 1.0\n"
            "KMACH 1.0\n"
            "KMACV 1.0\n"
            "POROSITY 1.0\n"
            "ALPHA 1.0\n"
            "BETA 1.0\n"
            "MANNINGS 1.0\n"
            "DRIP 1.0\n"
            "RECHARGE 1.0\n"
        )

    def _write_init(self, d: Path, name: str, init_gw_depth):
        """Write initial conditions file."""
        (d / f"{name}.init").write_text(
            "1\n"
            f"1 0.0 {init_gw_depth:.4f} 0.0\n"
        )

    def _write_forc(self, d: Path, name: str, start_dt, n_days):
        """Write placeholder forcing file.

        The coupler overwrites this with actual SUMMA recharge data
        for coupled runs. For standalone runs, uses constant recharge.
        """
        lines = [f"{n_days}"]
        start_epoch = int(start_dt.timestamp())
        for i in range(n_days):
            t = start_epoch + i * 86400
            lines.append(f"{t} 0.001 0.0 0.0 0.0 0.0 0.0")
        (d / f"{name}.forc").write_text("\n".join(lines) + "\n")
