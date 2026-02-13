"""
ParFlow Preprocessor

Generates ParFlow input via the pftools Python API. Creates a .pfidb
run file for a lumped single-cell (or multi-cell) variably-saturated
groundwater + overland flow model.

Output:
    settings/PARFLOW/<runname>.pfidb  -- ParFlow database file
"""

import logging
import math
from pathlib import Path

from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register_preprocessor("PARFLOW")
class ParFlowPreProcessor:
    """Generates ParFlow input files for integrated hydrologic simulation."""

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

    def _get_pf_cfg(self, key, default=None):
        """Get ParFlow-specific config value."""
        val = self.config_dict.get(f'PARFLOW_{key}', self.config_dict.get(key))
        if val is not None:
            return val
        try:
            pf_cfg = self.config.model.parflow
            if pf_cfg:
                attr = key.lower()
                if hasattr(pf_cfg, attr):
                    return getattr(pf_cfg, attr)
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
        """Get simulation time range and number of timesteps in hours."""
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
        n_hours = int((end_dt - start_dt).total_seconds() / 3600)
        if n_hours <= 0:
            n_hours = 8760  # 1 year

        timestep_hours = float(self._get_pf_cfg('TIMESTEP_HOURS', 1.0))
        n_steps = max(1, int(math.ceil(n_hours / timestep_hours)))

        return n_steps, timestep_hours, n_hours

    def run_preprocessing(self):
        """Generate ParFlow .pfidb input file.

        Uses the pftools Python API if available; falls back to writing
        a plain-text .pfidb file that ParFlow can read directly.
        """
        settings_dir = self.project_dir / "settings" / "PARFLOW"
        settings_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Generating ParFlow input files in {settings_dir}")

        # Get parameters
        nx = int(self._get_pf_cfg('NX', 1))
        ny = int(self._get_pf_cfg('NY', 1))
        nz = int(self._get_pf_cfg('NZ', 1))
        dx = float(self._get_pf_cfg('DX', 1000.0))
        dy = float(self._get_pf_cfg('DY', 1000.0))
        dz = float(self._get_pf_cfg('DZ', 100.0))
        top = float(self._get_pf_cfg('TOP', 1500.0))
        bot = float(self._get_pf_cfg('BOT', 1400.0))
        k_sat = float(self._get_pf_cfg('K_SAT', 5.0))
        porosity = float(self._get_pf_cfg('POROSITY', 0.4))
        vg_alpha = float(self._get_pf_cfg('VG_ALPHA', 1.0))
        vg_n = float(self._get_pf_cfg('VG_N', 2.0))
        s_res = float(self._get_pf_cfg('S_RES', 0.1))
        s_sat = float(self._get_pf_cfg('S_SAT', 1.0))
        ss = float(self._get_pf_cfg('SS', 1e-5))
        mannings_n = float(self._get_pf_cfg('MANNINGS_N', 0.03))
        initial_pressure = self._get_pf_cfg('INITIAL_PRESSURE')
        solver = str(self._get_pf_cfg('SOLVER', 'Richards'))

        n_steps, timestep_hours, n_hours = self._get_time_info()

        if initial_pressure is None:
            # Default: hydrostatic, water table at midpoint
            initial_pressure = (top - bot) / 2.0
        else:
            initial_pressure = float(initial_pressure)

        self.logger.info(
            f"ParFlow grid: {nx}x{ny}x{nz}, dx={dx:.0f}m, dy={dy:.0f}m, dz={dz:.0f}m, "
            f"K_sat={k_sat}, porosity={porosity}, vG(alpha={vg_alpha}, n={vg_n})"
        )
        self.logger.info(f"Timesteps: {n_steps} x {timestep_hours} hours")

        runname = self.domain_name

        # Try pftools API first
        try:
            self._write_pfidb_pftools(
                settings_dir, runname,
                nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                top=top, bot=bot, k_sat=k_sat, porosity=porosity,
                vg_alpha=vg_alpha, vg_n=vg_n, s_res=s_res, s_sat=s_sat,
                ss=ss, mannings_n=mannings_n, initial_pressure=initial_pressure,
                solver=solver, n_steps=n_steps, timestep_hours=timestep_hours,
            )
        except ImportError:
            self.logger.info(
                "pftools not available, writing plain-text .pfidb file"
            )
            self._write_pfidb_text(
                settings_dir, runname,
                nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                top=top, bot=bot, k_sat=k_sat, porosity=porosity,
                vg_alpha=vg_alpha, vg_n=vg_n, s_res=s_res, s_sat=s_sat,
                ss=ss, mannings_n=mannings_n, initial_pressure=initial_pressure,
                solver=solver, n_steps=n_steps, timestep_hours=timestep_hours,
            )

        self.logger.info(f"ParFlow input files generated in {settings_dir}")

    def _write_pfidb_pftools(self, settings_dir, runname, **params):
        """Generate .pfidb using pftools Python API."""
        from parflow import Run  # Lazy import

        run = Run(runname, __file__)

        # --- File version ---
        run.FileVersion = 4

        # --- Process topology ---
        run.Process.Topology.P = 1
        run.Process.Topology.Q = 1
        run.Process.Topology.R = 1

        # --- Computational grid ---
        run.ComputationalGrid.Lower.X = 0.0
        run.ComputationalGrid.Lower.Y = 0.0
        run.ComputationalGrid.Lower.Z = params['bot']

        run.ComputationalGrid.NX = params['nx']
        run.ComputationalGrid.NY = params['ny']
        run.ComputationalGrid.NZ = params['nz']

        run.ComputationalGrid.DX = params['dx']
        run.ComputationalGrid.DY = params['dy']
        run.ComputationalGrid.DZ = params['dz']

        # --- Domain geometry ---
        run.GeomInput.Names = 'domain_input'
        run.GeomInput.domain_input.GeomName = 'domain'
        run.GeomInput.domain_input.InputType = 'Box'

        run.Geom.domain.Lower.X = 0.0
        run.Geom.domain.Lower.Y = 0.0
        run.Geom.domain.Lower.Z = params['bot']
        run.Geom.domain.Upper.X = params['nx'] * params['dx']
        run.Geom.domain.Upper.Y = params['ny'] * params['dy']
        run.Geom.domain.Upper.Z = params['top']

        run.Geom.domain.Patches = 'x_lower x_upper y_lower y_upper z_lower z_upper'

        # --- Subsurface properties ---
        run.Geom.Perm.Names = 'domain'
        run.Geom.domain.Perm.Type = 'Constant'
        run.Geom.domain.Perm.Value = params['k_sat']

        run.Perm.TensorType = 'TensorByGeom'
        run.Geom.Perm.TensorByGeom.Names = 'domain'
        run.Geom.domain.Perm.TensorValX = 1.0
        run.Geom.domain.Perm.TensorValY = 1.0
        run.Geom.domain.Perm.TensorValZ = 1.0

        run.Geom.Porosity.GeomNames = 'domain'
        run.Geom.domain.Porosity.Type = 'Constant'
        run.Geom.domain.Porosity.Value = params['porosity']

        run.SpecificStorage.Type = 'Constant'
        run.SpecificStorage.GeomNames = 'domain'
        run.Geom.domain.SpecificStorage.Value = params['ss']

        # --- van Genuchten parameters ---
        run.Phase.RelPerm.Type = 'VanGenuchten'
        run.Phase.RelPerm.GeomNames = 'domain'
        run.Geom.domain.RelPerm.Alpha = params['vg_alpha']
        run.Geom.domain.RelPerm.N = params['vg_n']

        run.Phase.Saturation.Type = 'VanGenuchten'
        run.Phase.Saturation.GeomNames = 'domain'
        run.Geom.domain.Saturation.Alpha = params['vg_alpha']
        run.Geom.domain.Saturation.N = params['vg_n']
        run.Geom.domain.Saturation.SRes = params['s_res']
        run.Geom.domain.Saturation.SSat = params['s_sat']

        # --- Phases & contaminants ---
        run.Phase.Names = 'water'
        run.Phase.water.Density.Type = 'Constant'
        run.Phase.water.Density.Value = 1.0
        run.Phase.water.Viscosity.Type = 'Constant'
        run.Phase.water.Viscosity.Value = 1.0

        run.Contaminants.Names = ''

        # --- Gravity ---
        run.Gravity = 1.0

        # --- Solver ---
        run.Solver = params['solver']
        run.Solver.MaxIter = 25000
        run.Solver.Drop = 1e-20
        run.Solver.AbsTol = 1e-8
        run.Solver.MaxConvergenceFailures = 8

        run.Solver.Nonlinear.MaxIter = 80
        run.Solver.Nonlinear.ResidualTol = 1e-6
        run.Solver.Nonlinear.EtaChoice = 'Walker1'
        run.Solver.Nonlinear.EtaValue = 0.001
        run.Solver.Nonlinear.UseJacobian = True
        run.Solver.Nonlinear.DerivativeEpsilon = 1e-14
        run.Solver.Nonlinear.StepTol = 1e-30
        run.Solver.Nonlinear.Globalization = 'LineSearch'

        run.Solver.Linear.KrylovDimension = 80
        run.Solver.Linear.MaxRestarts = 2

        run.Solver.Linear.Preconditioner = 'MGSemi'

        # --- Timing ---
        run.TimingInfo.BaseUnit = 1.0  # hours
        run.TimingInfo.StartCount = 0
        run.TimingInfo.StartTime = 0.0
        run.TimingInfo.StopTime = float(params['n_steps'] * params['timestep_hours'])
        run.TimingInfo.DumpInterval = params['timestep_hours']

        run.TimeStep.Type = 'Constant'
        run.TimeStep.Value = params['timestep_hours']

        # --- Boundary conditions ---
        run.BCPressure.PatchNames = run.Geom.domain.Patches

        # No-flow on lateral and bottom boundaries
        for patch in ['x_lower', 'x_upper', 'y_lower', 'y_upper', 'z_lower']:
            bc = getattr(run.Patch, patch)
            bc.BCPressure.Type = 'FluxConst'
            bc.BCPressure.Cycle = 'constant'
            bc.BCPressure.alltime.Value = 0.0

        # Overland flow at top
        run.Patch.z_upper.BCPressure.Type = 'OverlandFlow'
        run.Patch.z_upper.BCPressure.Cycle = 'constant'
        run.Patch.z_upper.BCPressure.alltime.Value = 0.0

        # --- Manning's roughness for overland flow ---
        run.Mannings.Type = 'Constant'
        run.Mannings.GeomNames = 'domain'
        run.Mannings.Geom.domain.Value = params['mannings_n']

        # --- Cycle definitions ---
        run.Cycle.Names = 'constant'
        run.Cycle.constant.Names = 'alltime'
        run.Cycle.constant.alltime.Length = 1
        run.Cycle.constant.Repeat = -1

        # --- Initial conditions ---
        run.ICPressure.Type = 'HydroStaticPatch'
        run.ICPressure.GeomNames = 'domain'
        run.Geom.domain.ICPressure.Value = params['initial_pressure']
        run.Geom.domain.ICPressure.RefGeom = 'domain'
        run.Geom.domain.ICPressure.RefPatch = 'z_lower'

        # --- Domain definition ---
        run.Domain.GeomName = 'domain'

        # --- Retardation ---
        run.Geom.Retardation.GeomNames = ''

        # --- Phase sources ---
        run.PhaseSources.water.Type = 'Constant'
        run.PhaseSources.water.GeomNames = 'domain'
        run.PhaseSources.water.Geom.domain.Value = 0.0

        # --- Topographic slopes ---
        run.TopoSlopesX.Type = 'Constant'
        run.TopoSlopesX.GeomNames = 'domain'
        run.TopoSlopesX.Geom.domain.Value = 0.0

        run.TopoSlopesY.Type = 'Constant'
        run.TopoSlopesY.GeomNames = 'domain'
        run.TopoSlopesY.Geom.domain.Value = 0.0

        # --- Known solution ---
        run.KnownSolution = 'NoKnownSolution'

        # --- Wells, internal BC ---
        run.Wells.Names = ''

        # --- Output ---
        run.Solver.PrintSubsurfData = True
        run.Solver.PrintPressure = True
        run.Solver.PrintSaturation = True
        run.Solver.PrintVelocities = False

        # Write the .pfidb file
        run.write(working_directory=str(settings_dir))
        self.logger.info(f"Generated ParFlow .pfidb via pftools: {runname}")

    def _write_pfidb_text(self, settings_dir, runname, **params):
        """Write a plain-text .pfidb file without requiring pftools.

        The .pfidb format is a length-prefixed key-value database:
          <num_entries>
          <key_len>
          <key_chars>
          <val_len>
          <val_chars>
          ...
        Keys are stored WITHOUT the runname prefix (ParFlow looks up
        "ComputationalGrid.Lower.X", not "runname.ComputationalGrid.Lower.X").
        Values are always strings (ints and floats converted to str).
        """
        entries = []  # list of (key, value_str) tuples

        def add(key, value):
            # Convert Python types to ParFlow-compatible strings
            if isinstance(value, bool):
                entries.append((key, 'True' if value else 'False'))
            elif isinstance(value, float):
                # Use repr for full precision; strip trailing zeros
                entries.append((key, f'{value:g}'))
            else:
                entries.append((key, str(value)))

        # File version
        add('FileVersion', 4)

        # Process topology
        add('Process.Topology.P', 1)
        add('Process.Topology.Q', 1)
        add('Process.Topology.R', 1)

        # Computational grid
        add('ComputationalGrid.Lower.X', 0.0)
        add('ComputationalGrid.Lower.Y', 0.0)
        add('ComputationalGrid.Lower.Z', params['bot'])
        add('ComputationalGrid.NX', params['nx'])
        add('ComputationalGrid.NY', params['ny'])
        add('ComputationalGrid.NZ', params['nz'])
        add('ComputationalGrid.DX', params['dx'])
        add('ComputationalGrid.DY', params['dy'])
        add('ComputationalGrid.DZ', params['dz'])

        # Domain geometry
        add('GeomInput.Names', 'domain_input')
        add('GeomInput.domain_input.GeomName', 'domain')
        add('GeomInput.domain_input.InputType', 'Box')
        add('Geom.domain.Lower.X', 0.0)
        add('Geom.domain.Lower.Y', 0.0)
        add('Geom.domain.Lower.Z', params['bot'])
        add('Geom.domain.Upper.X', params['nx'] * params['dx'])
        add('Geom.domain.Upper.Y', params['ny'] * params['dy'])
        add('Geom.domain.Upper.Z', params['top'])
        add('Geom.domain.Patches',
            'x_lower x_upper y_lower y_upper z_lower z_upper')

        # Subsurface properties
        add('Geom.Perm.Names', 'domain')
        add('Geom.domain.Perm.Type', 'Constant')
        add('Geom.domain.Perm.Value', params['k_sat'])

        add('Perm.TensorType', 'TensorByGeom')
        add('Geom.Perm.TensorByGeom.Names', 'domain')
        add('Geom.domain.Perm.TensorValX', 1.0)
        add('Geom.domain.Perm.TensorValY', 1.0)
        add('Geom.domain.Perm.TensorValZ', 1.0)

        add('Geom.Porosity.GeomNames', 'domain')
        add('Geom.domain.Porosity.Type', 'Constant')
        add('Geom.domain.Porosity.Value', params['porosity'])

        add('SpecificStorage.Type', 'Constant')
        add('SpecificStorage.GeomNames', 'domain')
        add('Geom.domain.SpecificStorage.Value', params['ss'])

        # van Genuchten
        add('Phase.RelPerm.Type', 'VanGenuchten')
        add('Phase.RelPerm.GeomNames', 'domain')
        add('Geom.domain.RelPerm.Alpha', params['vg_alpha'])
        add('Geom.domain.RelPerm.N', params['vg_n'])

        add('Phase.Saturation.Type', 'VanGenuchten')
        add('Phase.Saturation.GeomNames', 'domain')
        add('Geom.domain.Saturation.Alpha', params['vg_alpha'])
        add('Geom.domain.Saturation.N', params['vg_n'])
        add('Geom.domain.Saturation.SRes', params['s_res'])
        add('Geom.domain.Saturation.SSat', params['s_sat'])

        # Phases
        add('Phase.Names', 'water')
        add('Phase.water.Density.Type', 'Constant')
        add('Phase.water.Density.Value', 1.0)
        add('Phase.water.Viscosity.Type', 'Constant')
        add('Phase.water.Viscosity.Value', 1.0)
        add('Contaminants.Names', '')
        add('Gravity', 1.0)

        # Solver
        add('Solver', params['solver'])
        add('Solver.MaxIter', 25000)
        add('Solver.Drop', 1e-20)
        add('Solver.AbsTol', 1e-8)
        add('Solver.MaxConvergenceFailures', 8)
        add('Solver.Nonlinear.MaxIter', 80)
        add('Solver.Nonlinear.ResidualTol', 1e-6)
        add('Solver.Nonlinear.EtaChoice', 'Walker1')
        add('Solver.Nonlinear.EtaValue', 0.001)
        add('Solver.Nonlinear.UseJacobian', True)
        add('Solver.Nonlinear.DerivativeEpsilon', 1e-14)
        add('Solver.Nonlinear.StepTol', 1e-30)
        add('Solver.Nonlinear.Globalization', 'LineSearch')
        add('Solver.Linear.KrylovDimension', 80)
        add('Solver.Linear.MaxRestarts', 2)
        add('Solver.Linear.Preconditioner', 'MGSemi')

        # Timing
        stop_time = float(params['n_steps'] * params['timestep_hours'])
        add('TimingInfo.BaseUnit', 1.0)
        add('TimingInfo.StartCount', 0)
        add('TimingInfo.StartTime', 0.0)
        add('TimingInfo.StopTime', stop_time)
        add('TimingInfo.DumpInterval', params['timestep_hours'])

        add('TimeStep.Type', 'Constant')
        add('TimeStep.Value', params['timestep_hours'])

        # Boundary conditions
        add('BCPressure.PatchNames',
            'x_lower x_upper y_lower y_upper z_lower z_upper')

        for patch in ['x_lower', 'x_upper', 'y_lower', 'y_upper', 'z_lower']:
            add(f'Patch.{patch}.BCPressure.Type', 'FluxConst')
            add(f'Patch.{patch}.BCPressure.Cycle', 'constant')
            add(f'Patch.{patch}.BCPressure.alltime.Value', 0.0)

        add('Patch.z_upper.BCPressure.Type', 'OverlandFlow')
        add('Patch.z_upper.BCPressure.Cycle', 'constant')
        add('Patch.z_upper.BCPressure.alltime.Value', 0.0)

        # Manning's roughness
        add('Mannings.Type', 'Constant')
        add('Mannings.GeomNames', 'domain')
        add('Mannings.Geom.domain.Value', params['mannings_n'])

        # Cycles
        add('Cycle.Names', 'constant')
        add('Cycle.constant.Names', 'alltime')
        add('Cycle.constant.alltime.Length', 1)
        add('Cycle.constant.Repeat', -1)

        # Initial conditions
        add('ICPressure.Type', 'HydroStaticPatch')
        add('ICPressure.GeomNames', 'domain')
        add('Geom.domain.ICPressure.Value', params['initial_pressure'])
        add('Geom.domain.ICPressure.RefGeom', 'domain')
        add('Geom.domain.ICPressure.RefPatch', 'z_lower')

        # Domain
        add('Domain.GeomName', 'domain')

        # Retardation
        add('Geom.Retardation.GeomNames', '')

        # Phase sources
        add('PhaseSources.water.Type', 'Constant')
        add('PhaseSources.water.GeomNames', 'domain')
        add('PhaseSources.water.Geom.domain.Value', 0.0)

        # Topographic slopes
        add('TopoSlopesX.Type', 'Constant')
        add('TopoSlopesX.GeomNames', 'domain')
        add('TopoSlopesX.Geom.domain.Value', 0.0)

        add('TopoSlopesY.Type', 'Constant')
        add('TopoSlopesY.GeomNames', 'domain')
        add('TopoSlopesY.Geom.domain.Value', 0.0)

        # Known solution
        add('KnownSolution', 'NoKnownSolution')

        # Wells
        add('Wells.Names', '')

        # Output
        add('Solver.PrintSubsurfData', True)
        add('Solver.PrintPressure', True)
        add('Solver.PrintSaturation', True)
        add('Solver.PrintVelocities', False)

        # Build the .pfidb binary-text format:
        # <num_entries>\n
        # <key_len>\n<key_chars>\n<val_len>\n<val_chars>\n ...
        parts = [str(len(entries))]
        for key, val in entries:
            parts.append(str(len(key)))
            parts.append(key)
            parts.append(str(len(val)))
            parts.append(val)

        pfidb_path = settings_dir / f'{runname}.pfidb'
        pfidb_path.write_text('\n'.join(parts) + '\n')

        # Also write a runname marker file
        (settings_dir / 'runname.txt').write_text(runname)

        self.logger.info(f"Generated ParFlow .pfidb (text): {pfidb_path}")
