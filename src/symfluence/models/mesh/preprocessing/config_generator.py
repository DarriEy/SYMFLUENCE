"""
MESH Configuration Generator

Generates INI configuration files for MESH model.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.mixins import ConfigMixin


class MESHConfigGenerator(ConfigMixin):
    """
    Generates MESH configuration files.

    Creates:
    - MESH_input_run_options.ini
    - MESH_parameters_CLASS.ini
    - MESH_parameters_hydrology.ini
    - MESH_input_streamflow.txt
    """

    def __init__(
        self,
        forcing_dir: Path,
        project_dir: Path,
        config: Dict[str, Any],
        logger: logging.Logger = None,
        time_window_func=None
    ):
        """
        Initialize config generator.

        Args:
            forcing_dir: Directory for MESH files
            project_dir: Project directory
            config: Configuration dictionary
            logger: Optional logger instance
            time_window_func: Function to get simulation time window
        """
        self.forcing_dir = forcing_dir
        self.project_dir = project_dir
        # Import here to avoid circular imports

        from symfluence.core.config.models import SymfluenceConfig



        # Auto-convert dict to typed config for backward compatibility

        if isinstance(config, dict):

            try:

                self._config = SymfluenceConfig(**config)

            except Exception:

                # Fallback for partial configs (e.g., in tests)

                self._config = config

        else:

            self._config = config
        self.logger = logger or logging.getLogger(__name__)
        self.get_simulation_time_window = time_window_func
        self.domain_name = config.get('DOMAIN_NAME', 'domain')

    def _get_spatial_mode(self) -> str:
        """Determine MESH spatial mode from configuration."""
        spatial_mode = self._get_config_value(lambda: self.config.model.mesh.spatial_mode, default='auto', dict_key='MESH_SPATIAL_MODE')

        if spatial_mode != 'auto':
            return spatial_mode

        domain_method = self._get_config_value(lambda: self.config.domain.definition_method, default='lumped', dict_key='DOMAIN_DEFINITION_METHOD')

        if domain_method in ['point', 'lumped']:
            return 'lumped'
        elif domain_method in ['delineate', 'semi_distributed', 'distributed']:
            return 'distributed'

        return 'lumped'

    def create_run_options(self) -> None:
        """Create MESH_input_run_options.ini file."""
        run_options_path = self.forcing_dir / "MESH_input_run_options.ini"

        spatial_mode = self._get_spatial_mode()

        # Get simulation times
        time_window = self.get_simulation_time_window() if self.get_simulation_time_window else None
        spinup_days = int(self._get_config_value(lambda: self.config.model.mesh.spinup_days, default=730, dict_key='MESH_SPINUP_DAYS'))

        if time_window:
            from datetime import timedelta
            analysis_start, end_time = time_window
            start_time = pd.Timestamp(analysis_start - timedelta(days=spinup_days))
            end_time = pd.Timestamp(end_time)
        else:
            start_time = pd.Timestamp("2004-01-01 01:00")
            end_time = pd.Timestamp("2004-01-05 23:00")

        # Use 'nc_subbasin' for both DDB and Forcing in MESH 1.5 when using dimension 'subbasin'
        shd_flag = 'nc_subbasin pad_outlets'
        forcing_flag = 'nc_subbasin'

        # Get RUNMODE from config (default to 'wf_route' for routing support)
        runmode = self._get_config_value(lambda: self.config.model.mesh.runmode, default='wf_route', dict_key='MESH_RUNMODE')

        # Set streamflow output based on routing mode
        if runmode == 'noroute':
            streamflow_flag = 'none'
        else:
            streamflow_flag = 'csv'

        content = f"""MESH input run options file                             # comment line 1                                | *
##### Control Flags #####                               # comment line 2                                | *
----#                                                   # comment line 3                                | *
   15                                                   # Number of control flags                       | I5
SHDFILEFLAG         {shd_flag}                          # Drainage database format
BASINFORCINGFLAG    {forcing_flag}                      # Forcing file format
RUNMODE             {runmode}                           # Run mode (wf_route = WATFLOOD routing, noroute = CLASS-only)
INPUTPARAMSFORMFLAG ini                                 # Parameter file format (ini = .ini files)
RESUMEFLAG          off                                 # Resume from state (off=No)
SAVERESUMEFLAG      off                                 # Save final state (off=No)
TIMESTEPFLAG        60                                  # Time step in minutes (default 60)
OUTFIELDSFLAG       default                             # Output fields (all, none, default)
BASINRUNOFFFLAG     ts                                  # Runoff output format (ts = time series)
STREAMFLOWOUTFLAG   {streamflow_flag}                   # Streamflow output (csv when routing enabled)
LOCATIONFLAG        1                                   # Centroid location
PBSMFLAG            off                                 # Blowing snow (off)
BASEFLOWFLAG        wf_lzs                              # Baseflow formulation
INTERPOLATIONFLAG   0                                   # Interpolation (0=No)
METRICSSPINUP       {spinup_days}                       # Spinup days to exclude from calibration metrics
##### Output Grid selection #####                       #15 comment line 15                             | *
----#                                                   #16 comment line 16                             | *
    0   #Maximum 5 points                               #17 Number of output grid points                | I5
---------#---------#---------#---------#---------#      #18 comment line 18                             | *
         1                                              #19 Grid number                                 | 5I10
         1                                              #20 Land class                                  | 5I10
./                                                      #21 Output directory                            | 5A10
##### Output Directory #####                            #22 comment line 22                             | *
---------#                                              #23 comment line 23                             | *
./                                                      #24 Output Directory for total-basin files      | A10
##### Simulation Run Times #####                        #25 comment line 25                             | *
---#---#---#---#                                        #26 comment line 26                             | *
{start_time.year:04d} {start_time.dayofyear:03d} {start_time.hour:3d}   0
{end_time.year:04d} {end_time.dayofyear:03d} {end_time.hour:3d}   0
"""
        with open(run_options_path, 'w') as f:
            f.write(content)
        self.logger.info(f"Created {run_options_path.name} (spatial_mode={spatial_mode})")

    def create_class_parameters(self) -> None:
        """Create MESH CLASS parameters file with defaults for all GRU types."""
        ddb_path = self.forcing_dir / "MESH_drainage_database.nc"
        if not ddb_path.exists():
            self.logger.warning("Drainage database not found, cannot create CLASS parameters")
            return

        try:
            with xr.open_dataset(ddb_path) as ds:
                ngru = ds.dims.get('NGRU', 1)
        except Exception as e:
            self.logger.warning(f"Failed to read DDB: {e}")
            ngru = 1

        self.logger.info(f"Creating CLASS parameters for {ngru} GRU classes")

        # Load defaults
        try:
            from meshflow.utility import DEFAULT_CLASS_PARAMS
            with open(DEFAULT_CLASS_PARAMS, 'r') as f:
                defaults = json.load(f)
            class_defaults = defaults.get('class_defaults', {})
        except Exception:
            class_defaults = {
                'veg': {'line5': {'fcan': 1, 'lamx': 1.45}, 'line6': {'lnz0': -1.3, 'lamn': 1.2}},
                'soil': {'line14': {'sand1': 50, 'sand2': 50, 'sand3': 50}},
            }

        class_path = self.forcing_dir / "MESH_parameters_CLASS.ini"

        with open(class_path, 'w') as f:
            f.write(";; CLASS parameter file\n")
            f.write(";; Generated by SYMFLUENCE for MESH 1.5\n\n")
            f.write("NL 1    ! number of soil layers\n")
            f.write(f"NM {ngru}    ! number of landcover classes (GRUs)\n\n")
            f.write(";; Vegetation parameters\n")

            f.write("[REF_HEIGHTS]\n")
            f.write("ZRFM 10.0  ! Reference height for wind speed\n")
            f.write("ZRFH 2.0   ! Reference height for temperature/humidity\n")
            f.write("ZBLD 50.0  ! Blending height\n\n")

            for gru in range(1, ngru + 1):
                f.write(f";; GRU {gru}\n")
                f.write(f"[GRU_{gru}]\n")
                veg = class_defaults.get('veg', {})
                f.write(f"FCAN 1.0 LAMX {veg.get('line5', {}).get('lamx', 1.45)}\n")
                f.write(f"LNZ0 {veg.get('line6', {}).get('lnz0', -1.3)} LAMN {veg.get('line6', {}).get('lamn', 1.2)}\n")
                f.write("ALVC 0.045 CMAS 4.5\n")
                f.write("ALIC 0.16 ROOT 1.09\n")
                f.write("RSMN 145 QA50 36\n")
                f.write("VPDA 0.8 VPDB 1.05\n")
                f.write("PSGA 100 PSGB 5\n")
                f.write("DRN 1 SDEP 2.5 FARE 1 DD 50\n")
                f.write("XSLP 0.03 XDRAINH 0.35 MANN 0.1 KSAT 0.05\n")
                f.write("SAND 50 50 50\n")
                f.write("CLAY 20 20 20\n")
                f.write("ORGM 0 0 0\n")
                f.write("TBAR 4 2 1 TCAN 2 TSNO 0 TPND 4\n")
                f.write("THLQ 0.25 0.15 0.04 THIC 0 0 0 ZPND 0\n")
                f.write("RCAN 0 SCAN 0 SNO 0 ALBS 0 RHOS 0 GRO 1\n\n")

        self.logger.info(f"Created CLASS parameters: {class_path}")

    def create_hydrology_parameters(self) -> None:
        """Create MESH hydrology parameters file."""
        try:
            from meshflow.utility import DEFAULT_HYDROLOGY_PARAMS
            with open(DEFAULT_HYDROLOGY_PARAMS, 'r') as f:
                json.load(f)
        except Exception as e:
            self.logger.debug(f"Could not load meshflow default hydrology params: {e}")

        hydro_path = self.forcing_dir / "MESH_parameters_hydrology.ini"

        with open(hydro_path, 'w') as f:
            f.write(";; Hydrology parameter file\n")
            f.write(";; Generated by SYMFLUENCE for MESH 1.5\n\n")
            f.write("[HYDROLOGY]\n")
            f.write("WF_R1 0.5        ! Overland flow exponent\n")
            f.write("WF_R2 1.0        ! Interflow coefficient\n")
            f.write("WF_KI 0.5        ! Interflow coefficient\n")
            f.write("WF_KC 0.5        ! Channel routing coefficient\n")
            f.write("WF_KD 0.05       ! Deep groundwater coefficient\n")
            f.write("WF_FLZS 0.1      ! Lower zone storage fraction\n")
            f.write("WF_PWR_LZS 2.0   ! Lower zone power\n\n")
            f.write("[ROUTING]\n")
            f.write("CHNL_MANN 0.035  ! Channel Manning's n\n")
            f.write("CHNL_WD 10.0     ! Channel width (m)\n")
            f.write("CHNL_DP 1.0      ! Channel depth (m)\n")

        self.logger.info(f"Created hydrology parameters: {hydro_path}")

    def create_streamflow_input(self) -> None:
        """Create MESH_input_streamflow.txt with gauge locations."""
        streamflow_path = self.forcing_dir / "MESH_input_streamflow.txt"

        time_window = self.get_simulation_time_window() if self.get_simulation_time_window else None
        spinup_days = int(self._get_config_value(lambda: self.config.model.mesh.spinup_days, default=365, dict_key='MESH_SPINUP_DAYS'))

        if time_window:
            from datetime import timedelta
            analysis_start, end_date = time_window
            sim_start = analysis_start - timedelta(days=spinup_days)
            start_year = sim_start.year
            start_month = sim_start.month
            start_day = sim_start.day
        else:
            start_year = 2001
            start_month = 1
            start_day = 1
            end_date = None

        # Find outlet subbasin
        outlet_rank, outlet_da = self._find_outlet_subbasin()
        self.logger.info(f"Setting gauge at Rank {outlet_rank} (DA={outlet_da:.1f} km²)")

        # Get observed data
        obs_data = self._load_observed_streamflow()

        gauge_id = self._get_config_value(lambda: self.config.data.streamflow_station_id, default='gauge1', dict_key='STREAMFLOW_STATION_ID')
        if gauge_id == 'default':
            gauge_id = self.domain_name

        with open(streamflow_path, 'w') as f:
            f.write(f"#{self.domain_name} streamflow gauge\n")
            f.write(f"1 0 0 24 {start_year} {start_month} {start_day}\n")
            f.write(f"1 {outlet_rank} {gauge_id}\n")

            if obs_data is not None and len(obs_data) > 0:
                for q in obs_data:
                    if np.isnan(q) or q < 0:
                        f.write("-1\t-1\n")
                    else:
                        f.write(f"{q:.3f}\t-1\n")
            else:
                if time_window:
                    from datetime import timedelta
                    n_days = (end_date - sim_start).days + 1
                else:
                    n_days = 365

                for _ in range(n_days):
                    f.write("-1\t-1\n")

        self.logger.info(f"Created streamflow input: {streamflow_path}")

    def _find_outlet_subbasin(self) -> Tuple[int, float]:
        """Find outlet subbasin rank and drainage area."""
        ddb_path = self.forcing_dir / "MESH_drainage_database.nc"

        if ddb_path.exists():
            with xr.open_dataset(ddb_path) as ds:
                next_arr = ds['Next'].values
                rank_arr = ds['Rank'].values
                da_arr = ds['DA'].values if 'DA' in ds else ds['GridArea'].values

                inside_mask = next_arr > 0
                if inside_mask.any():
                    inside_indices = np.where(inside_mask)[0]
                    max_da_idx = inside_indices[np.argmax(da_arr[inside_indices])]
                    outlet_rank = int(rank_arr[max_da_idx])
                    outlet_da = da_arr[max_da_idx] / 1e6
                else:
                    outlet_idx = np.argmax(da_arr)
                    outlet_rank = int(rank_arr[outlet_idx])
                    outlet_da = da_arr[outlet_idx] / 1e6

                return outlet_rank, outlet_da

        return 1, 0

    def _load_observed_streamflow(self) -> Optional[np.ndarray]:
        """Load observed streamflow data if available."""
        obs_dir = self.project_dir / 'observations' / 'streamflow' / 'preprocessed'
        if not obs_dir.exists():
            obs_dir = self.project_dir / 'observations' / 'streamflow' / 'raw_data'

        if not obs_dir.exists():
            return None

        csv_files = list(obs_dir.glob('*.csv'))
        if not csv_files:
            return None

        try:
            df = pd.read_csv(csv_files[0])

            q_col = None
            for col in ['discharge', 'streamflow', 'Q', 'flow', 'FLOW', 'Value']:
                if col in df.columns:
                    q_col = col
                    break

            if q_col is None:
                return None

            return np.asarray(df[q_col].values)

        except Exception as e:
            self.logger.warning(f"Failed to load observed streamflow: {e}")
            return None
