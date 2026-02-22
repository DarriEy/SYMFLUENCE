"""
MODFLOW 6 Preprocessor

Generates MODFLOW 6 text input files for a lumped single-cell groundwater
model. The cell represents the entire catchment with area-equivalent
dimensions.

Input files generated:
    mfsim.nam   — simulation name file
    gwf.nam     — GWF model name file
    gwf.tdis    — temporal discretization
    gwf.dis     — spatial discretization (1×1×1)
    gwf.ic      — initial conditions
    gwf.npf     — node property flow (hydraulic conductivity)
    gwf.sto     — storage (Sy, Ss)
    gwf.rch     — recharge package
    gwf.drn     — drain package (baseflow outlet)
    gwf.oc      — output control
    gwf.ims     — iterative model solution
"""

import logging
import math
from pathlib import Path

from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("MODFLOW")
class MODFLOWPreProcessor:
    """Generates MODFLOW 6 input files for lumped groundwater simulation."""

    def __init__(self, config, logger, **kwargs):
        self.config = config
        self.logger = logger
        self.config_dict = getattr(config, '_config_dict', {})
        if not self.config_dict and hasattr(config, 'to_dict'):
            self.config_dict = config.to_dict()

        self.domain_name = self._get_cfg('DOMAIN_NAME', 'unknown')
        self.experiment_id = self._get_cfg('EXPERIMENT_ID', 'default')

        project_dir = self.config_dict.get('PROJECT_DIR')
        if not project_dir or project_dir == '.':
            try:
                data_dir = str(config.system.data_dir)
            except (AttributeError, TypeError):
                data_dir = self._get_cfg('SYMFLUENCE_DATA_DIR', '.')
            project_dir = Path(data_dir) / f"domain_{self.domain_name}"
        self.project_dir = Path(project_dir)

    def _get_cfg(self, key, default=None):
        """Get config value with typed config first, dict fallback."""
        try:
            if key == 'DOMAIN_NAME':
                val = self.config.domain.name
                if val is not None:
                    return val
            elif key == 'EXPERIMENT_ID':
                val = self.config.domain.experiment_id
                if val is not None:
                    return val
            elif key == 'SYMFLUENCE_DATA_DIR':
                val = str(self.config.system.data_dir)
                if val is not None:
                    return val
        except (AttributeError, TypeError):
            pass
        val = self.config_dict.get(key)
        if val is not None and val != 'default':
            return val
        return default

    def _get_modflow_cfg(self, key, default=None):
        """Get MODFLOW-specific config value with typed config first."""
        try:
            mf_cfg = self.config.model.modflow
            if mf_cfg:
                attr = key.lower()
                if hasattr(mf_cfg, attr):
                    pydantic_val = getattr(mf_cfg, attr)
                    if pydantic_val is not None:
                        return pydantic_val
        except (AttributeError, TypeError):
            pass
        val = self._get_cfg(f'MODFLOW_{key}')
        if val is None:
            val = self._get_cfg(key)
        if val is not None:
            return val
        return default

    def _get_catchment_area_m2(self) -> float:
        """Get catchment area in m2.

        Resolution order:
        1. Explicit CATCHMENT_AREA config key (km2)
        2. config.domain.catchment_area
        3. Auto-detect from watershed shapefile (UTM projection)
        4. Fallback default (27 km2)
        """
        try:
            area_km2 = self.config.domain.catchment_area
        except (AttributeError, TypeError):
            area_km2 = None
        if area_km2 is None:
            area_km2 = self._get_cfg('CATCHMENT_AREA')
        if area_km2 is None:
            # Try to compute from watershed shapefile
            area_km2 = self._detect_catchment_area_km2()
        if area_km2 is None:
            self.logger.warning("CATCHMENT_AREA not set and no shapefile found, using default 27 km2")
            area_km2 = 27.0
        return float(area_km2) * 1e6

    def _detect_catchment_area_km2(self):
        """Detect catchment area from the watershed shapefile."""
        try:
            import geopandas as gpd

            domain_name = self.domain_name
            basin_path = (
                self.project_dir / 'shapefiles' / 'river_basins'
                / f"{domain_name}_riverBasins_lumped.shp"
            )
            if not basin_path.exists():
                return None

            gdf = gpd.read_file(str(basin_path))
            # Project to a metric CRS for accurate area calculation
            gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
            area_km2 = gdf_proj.geometry.area.sum() / 1e6
            self.logger.info(f"Auto-detected catchment area from shapefile: {area_km2:.1f} km2")
            return area_km2
        except Exception as e:
            self.logger.debug(f"Could not auto-detect catchment area: {e}")
            return None

    def _get_time_info(self):
        """Get simulation time range and number of stress periods."""
        import pandas as pd

        try:
            start = self.config.domain.time_start
        except (AttributeError, TypeError):
            start = None
        try:
            end = self.config.domain.time_end
        except (AttributeError, TypeError):
            end = None
        if start is None:
            start = self._get_cfg('EXPERIMENT_TIME_START', '2000-01-01')
        if end is None:
            end = self._get_cfg('EXPERIMENT_TIME_END', '2001-01-01')

        start_dt = pd.Timestamp(str(start))
        end_dt = pd.Timestamp(str(end))
        n_days = (end_dt - start_dt).days
        if n_days <= 0:
            n_days = 365

        sp_length = float(self._get_modflow_cfg('STRESS_PERIOD_LENGTH', 1.0))
        n_periods = max(1, int(math.ceil(n_days / sp_length)))

        return n_periods, sp_length, n_days

    def run_preprocessing(self):
        """Generate all MODFLOW 6 input files."""
        settings_dir = self.project_dir / "settings" / "MODFLOW"
        settings_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating MODFLOW 6 input files in {settings_dir}")

        # Get parameters
        area_m2 = self._get_catchment_area_m2()
        cell_size = float(self._get_modflow_cfg('CELL_SIZE', math.sqrt(area_m2)))
        nlay = int(self._get_modflow_cfg('NLAY', 1))
        nrow = int(self._get_modflow_cfg('NROW', 1))
        ncol = int(self._get_modflow_cfg('NCOL', 1))
        k = float(self._get_modflow_cfg('K', 5.0))
        sy = float(self._get_modflow_cfg('SY', 0.05))
        ss = float(self._get_modflow_cfg('SS', 1e-5))
        top = float(self._get_modflow_cfg('TOP', 1500.0))
        bot = float(self._get_modflow_cfg('BOT', 1400.0))
        strt = self._get_modflow_cfg('STRT')
        if strt is None:
            strt = (top + bot) / 2.0
        else:
            strt = float(strt)
        drain_elev = self._get_modflow_cfg('DRAIN_ELEVATION')
        if drain_elev is None:
            drain_elev = (top + bot) / 2.0
        else:
            drain_elev = float(drain_elev)
        # Scale drain conductance with catchment area to target a ~100-day
        # aquifer time constant (τ = SY × A / C) with the default SY.
        drain_cond_default = max(50.0, area_m2 * 0.0005)
        drain_cond = float(self._get_modflow_cfg('DRAIN_CONDUCTANCE', drain_cond_default))
        nstp = int(self._get_modflow_cfg('NSTP', 1))

        n_periods, sp_length, n_days = self._get_time_info()

        self.logger.info(
            f"MODFLOW grid: {nlay}×{nrow}×{ncol}, cell_size={cell_size:.0f}m, "
            f"K={k}, Sy={sy}, top={top}, bot={bot}"
        )
        self.logger.info(f"Stress periods: {n_periods} × {sp_length} days")

        # Write all files
        self._write_mfsim_nam(settings_dir, n_periods, sp_length, nstp)
        self._write_gwf_nam(settings_dir)
        self._write_tdis(settings_dir, n_periods, sp_length, nstp)
        self._write_dis(settings_dir, nlay, nrow, ncol, cell_size, top, bot)
        self._write_ic(settings_dir, strt)
        self._write_npf(settings_dir, k)
        self._write_sto(settings_dir, sy, ss)
        self._write_rch(settings_dir)
        self._write_drn(settings_dir, drain_elev, drain_cond)
        self._write_oc(settings_dir)
        self._write_ims(settings_dir)

        self.logger.info(f"MODFLOW 6 input files generated in {settings_dir}")

    def _write_mfsim_nam(self, d: Path, n_periods, sp_length, nstp):
        """Write simulation name file."""
        (d / "mfsim.nam").write_text(
            "BEGIN OPTIONS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN TIMING\n"
            "  TDIS6 gwf.tdis\n"
            "END TIMING\n"
            "\n"
            "BEGIN MODELS\n"
            "  GWF6 gwf.nam gwf\n"
            "END MODELS\n"
            "\n"
            "BEGIN EXCHANGES\n"
            "END EXCHANGES\n"
            "\n"
            "BEGIN SOLUTIONGROUP 1\n"
            "  IMS6 gwf.ims gwf\n"
            "END SOLUTIONGROUP 1\n"
        )

    def _write_gwf_nam(self, d: Path):
        """Write GWF model name file."""
        (d / "gwf.nam").write_text(
            "BEGIN OPTIONS\n"
            "  SAVE_FLOWS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN PACKAGES\n"
            "  DIS6 gwf.dis\n"
            "  IC6 gwf.ic\n"
            "  NPF6 gwf.npf\n"
            "  STO6 gwf.sto\n"
            "  RCH6 gwf.rch\n"
            "  DRN6 gwf.drn\n"
            "  OC6 gwf.oc\n"
            "END PACKAGES\n"
        )

    def _write_tdis(self, d: Path, n_periods, sp_length, nstp):
        """Write temporal discretization file."""
        lines = [
            "BEGIN OPTIONS\n",
            "  TIME_UNITS DAYS\n",
            "END OPTIONS\n",
            "\n",
            "BEGIN DIMENSIONS\n",
            f"  NPER {n_periods}\n",
            "END DIMENSIONS\n",
            "\n",
            "BEGIN PERIODDATA\n",
        ]
        for _ in range(n_periods):
            lines.append(f"  {sp_length} {nstp} 1.0\n")
        lines.append("END PERIODDATA\n")
        (d / "gwf.tdis").write_text("".join(lines))

    def _write_dis(self, d: Path, nlay, nrow, ncol, cell_size, top, bot):
        """Write spatial discretization file."""
        delr = cell_size  # row width
        delc = cell_size  # column width
        (d / "gwf.dis").write_text(
            "BEGIN OPTIONS\n"
            "  LENGTH_UNITS METERS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN DIMENSIONS\n"
            f"  NLAY {nlay}\n"
            f"  NROW {nrow}\n"
            f"  NCOL {ncol}\n"
            "END DIMENSIONS\n"
            "\n"
            "BEGIN GRIDDATA\n"
            "  DELR\n"
            f"    CONSTANT {delr}\n"
            "  DELC\n"
            f"    CONSTANT {delc}\n"
            "  TOP\n"
            f"    CONSTANT {top}\n"
            "  BOTM\n"
            f"    CONSTANT {bot}\n"
            "END GRIDDATA\n"
        )

    def _write_ic(self, d: Path, strt):
        """Write initial conditions file."""
        (d / "gwf.ic").write_text(
            "BEGIN OPTIONS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN GRIDDATA\n"
            "  STRT\n"
            f"    CONSTANT {strt}\n"
            "END GRIDDATA\n"
        )

    def _write_npf(self, d: Path, k):
        """Write node property flow (hydraulic conductivity) file."""
        (d / "gwf.npf").write_text(
            "BEGIN OPTIONS\n"
            "  SAVE_SPECIFIC_DISCHARGE\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN GRIDDATA\n"
            "  ICELLTYPE\n"
            "    CONSTANT 1\n"
            "  K\n"
            f"    CONSTANT {k}\n"
            "END GRIDDATA\n"
        )

    def _write_sto(self, d: Path, sy, ss):
        """Write storage file."""
        (d / "gwf.sto").write_text(
            "BEGIN OPTIONS\n"
            "  SAVE_FLOWS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN GRIDDATA\n"
            "  ICONVERT\n"
            "    CONSTANT 1\n"
            "  SS\n"
            f"    CONSTANT {ss}\n"
            "  SY\n"
            f"    CONSTANT {sy}\n"
            "END GRIDDATA\n"
            "\n"
            "BEGIN PERIOD 1\n"
            "  TRANSIENT\n"
            "END PERIOD 1\n"
        )

    def _write_rch(self, d: Path):
        """Write recharge package file.

        Uses time-series format: a separate recharge.ts file will be
        written by the coupler at runtime with actual recharge data.
        For standalone runs, a constant recharge is applied.
        """
        (d / "gwf.rch").write_text(
            "BEGIN OPTIONS\n"
            "  READASARRAYS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN PERIOD 1\n"
            "  RECHARGE\n"
            "    CONSTANT 0.001\n"
            "END PERIOD 1\n"
        )
        # Write a placeholder recharge time-array-series file.
        # The coupler overwrites gwf.rch with TAS6 reference for coupled runs.
        (d / "recharge.ts").write_text(
            "BEGIN ATTRIBUTES\n"
            "  NAME rch_array\n"
            "  METHOD LINEAR\n"
            "END ATTRIBUTES\n"
            "\n"
            "BEGIN TIME 0.0\n"
            "  CONSTANT 0.001\n"
            "END TIME 0.0\n"
        )

    def _write_drn(self, d: Path, drain_elev, drain_cond):
        """Write drain package file."""
        (d / "gwf.drn").write_text(
            "BEGIN OPTIONS\n"
            "  PRINT_INPUT\n"
            "  PRINT_FLOWS\n"
            "  SAVE_FLOWS\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN DIMENSIONS\n"
            "  MAXBOUND 1\n"
            "END DIMENSIONS\n"
            "\n"
            "BEGIN PERIOD 1\n"
            f"  1 1 1 {drain_elev} {drain_cond}\n"
            "END PERIOD 1\n"
        )

    def _write_oc(self, d: Path):
        """Write output control file."""
        (d / "gwf.oc").write_text(
            "BEGIN OPTIONS\n"
            "  HEAD FILEOUT gwf.hds\n"
            "  BUDGET FILEOUT gwf.bud\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN PERIOD 1\n"
            "  SAVE HEAD ALL\n"
            "  SAVE BUDGET ALL\n"
            "END PERIOD 1\n"
        )

    def _write_ims(self, d: Path):
        """Write iterative model solution file."""
        (d / "gwf.ims").write_text(
            "BEGIN OPTIONS\n"
            "  PRINT_OPTION SUMMARY\n"
            "END OPTIONS\n"
            "\n"
            "BEGIN NONLINEAR\n"
            "  OUTER_DVCLOSE 1.0e-6\n"
            "  OUTER_MAXIMUM 100\n"
            "END NONLINEAR\n"
            "\n"
            "BEGIN LINEAR\n"
            "  INNER_DVCLOSE 1.0e-9\n"
            "  INNER_MAXIMUM 300\n"
            "  INNER_RCLOSE 1.0e-3 STRICT\n"
            "END LINEAR\n"
        )
