"""Groundwater and integrated surface-subsurface model configuration classes."""

from typing import Optional

from pydantic import BaseModel, Field

from .base import FROZEN_CONFIG
from .model_config_types import SpatialModeType

class MODFLOWConfig(BaseModel):
    """MODFLOW 6 (USGS modular groundwater flow model) configuration.

    MODFLOW 6 simulates three-dimensional groundwater flow using the
    finite-difference method. In SYMFLUENCE it is used as a lumped
    single-cell groundwater model coupled with land surface models
    (e.g., SUMMA) to separate baseflow from surface runoff.

    Reference:
        Langevin, C.D., et al. (2017): Documentation for the MODFLOW 6
        Groundwater Flow Model. USGS Techniques and Methods 6-A55.
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='MODFLOW_INSTALL_PATH')
    exe: str = Field(default='mf6', alias='MODFLOW_EXE')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_MODFLOW_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='MODFLOW_SPATIAL_MODE')

    # Grid discretization
    grid_type: str = Field(default='dis', alias='MODFLOW_GRID_TYPE')
    nlay: int = Field(default=1, alias='MODFLOW_NLAY', ge=1, le=100)
    nrow: int = Field(default=1, alias='MODFLOW_NROW', ge=1, le=10000)
    ncol: int = Field(default=1, alias='MODFLOW_NCOL', ge=1, le=10000)
    cell_size: Optional[float] = Field(default=None, alias='MODFLOW_CELL_SIZE', gt=0)

    # Aquifer properties
    k: float = Field(default=5.0, alias='MODFLOW_K', gt=0)
    sy: float = Field(default=0.15, alias='MODFLOW_SY', gt=0, le=0.5)
    ss: float = Field(default=1e-5, alias='MODFLOW_SS', gt=0, le=0.1)
    strt: Optional[float] = Field(default=None, alias='MODFLOW_STRT')
    top: float = Field(default=1500.0, alias='MODFLOW_TOP')
    bot: float = Field(default=1400.0, alias='MODFLOW_BOT')

    # Coupling
    coupling_source: str = Field(default='SUMMA', alias='MODFLOW_COUPLING_SOURCE')
    recharge_variable: str = Field(default='scalarSoilDrainage', alias='MODFLOW_RECHARGE_VARIABLE')

    # Drain package
    drain_elevation: Optional[float] = Field(default=None, alias='MODFLOW_DRAIN_ELEVATION')
    drain_conductance: float = Field(default=50.0, alias='MODFLOW_DRAIN_CONDUCTANCE', gt=0)

    # Stress period
    stress_period_length: float = Field(default=1.0, alias='MODFLOW_STRESS_PERIOD_LENGTH', gt=0)
    nstp: int = Field(default=1, alias='MODFLOW_NSTP', ge=1, le=1000)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_MODFLOW')

    # Calibration
    params_to_calibrate: str = Field(
        default='K,SY,DRAIN_CONDUCTANCE',
        alias='MODFLOW_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='MODFLOW_TIMEOUT', ge=60, le=86400)


class ParFlowConfig(BaseModel):
    """ParFlow integrated hydrologic model configuration.

    ParFlow solves variably-saturated flow (Richards equation) and
    overland flow. In SYMFLUENCE it is used as an alternative to MODFLOW
    for coupled land surface + groundwater simulations with full vadose
    zone support.

    Reference:
        Kollet, S.J. & Maxwell, R.M. (2006): Integrated surface-groundwater
        flow modeling. Advances in Water Resources 29(7).
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='PARFLOW_INSTALL_PATH')
    exe: str = Field(default='parflow', alias='PARFLOW_EXE')
    parflow_dir: str = Field(default='default', alias='PARFLOW_DIR')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_PARFLOW_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='PARFLOW_SPATIAL_MODE')

    # Grid discretization
    nx: int = Field(default=1, alias='PARFLOW_NX', ge=1, le=10000)
    ny: int = Field(default=1, alias='PARFLOW_NY', ge=1, le=10000)
    nz: int = Field(default=1, alias='PARFLOW_NZ', ge=1, le=100)
    dx: float = Field(default=1000.0, alias='PARFLOW_DX', gt=0)
    dy: float = Field(default=1000.0, alias='PARFLOW_DY', gt=0)
    dz: float = Field(default=100.0, alias='PARFLOW_DZ', gt=0)

    # Domain geometry
    top: float = Field(default=1500.0, alias='PARFLOW_TOP')
    bot: float = Field(default=1400.0, alias='PARFLOW_BOT')

    # Subsurface properties
    k_sat: float = Field(default=5.0, alias='PARFLOW_K_SAT', gt=0)
    porosity: float = Field(default=0.4, alias='PARFLOW_POROSITY', gt=0, le=1.0)
    vg_alpha: float = Field(default=1.0, alias='PARFLOW_VG_ALPHA', gt=0)
    vg_n: float = Field(default=2.0, alias='PARFLOW_VG_N', gt=1.0)
    s_res: float = Field(default=0.1, alias='PARFLOW_S_RES', ge=0, lt=1.0)
    s_sat: float = Field(default=1.0, alias='PARFLOW_S_SAT', gt=0, le=1.0)
    specific_storage: float = Field(default=1e-5, alias='PARFLOW_SS', gt=0)

    # Overland flow
    mannings_n: float = Field(default=0.03, alias='PARFLOW_MANNINGS_N', gt=0)

    # Initial conditions
    initial_pressure: Optional[float] = Field(default=None, alias='PARFLOW_INITIAL_PRESSURE')

    # Coupling
    coupling_source: str = Field(default='SUMMA', alias='PARFLOW_COUPLING_SOURCE')
    recharge_variable: str = Field(default='scalarSoilDrainage', alias='PARFLOW_RECHARGE_VARIABLE')

    # Solver
    solver: str = Field(default='Richards', alias='PARFLOW_SOLVER')
    timestep_hours: float = Field(default=1.0, alias='PARFLOW_TIMESTEP_HOURS', gt=0)

    # Parallel execution
    num_procs: int = Field(default=1, alias='PARFLOW_NUM_PROCS', ge=1, le=1024)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_PARFLOW')

    # Calibration
    params_to_calibrate: str = Field(
        default='K_SAT,POROSITY,VG_ALPHA,VG_N,MANNINGS_N',
        alias='PARFLOW_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='PARFLOW_TIMEOUT', ge=60, le=86400)


class CLMParFlowConfig(BaseModel):
    """ParFlow with tightly-coupled CLM (Common Land Model) configuration.

    ParFlow-CLM is built from the ParFlow source with -DPARFLOW_HAVE_CLM=ON.
    CLM is embedded as Fortran modules inside ParFlow and handles land surface
    energy balance, evapotranspiration, snow dynamics, and vegetation processes.
    This is a different binary from standalone ParFlow (-DPARFLOW_HAVE_CLM=OFF).

    Reference:
        Kollet, S.J. & Maxwell, R.M. (2008): Capturing the influence of
        groundwater dynamics on land surface processes using an integrated,
        distributed watershed model. Water Resources Research 44(2).

        Dai, Y. et al. (2003): The Common Land Model. Bulletin of the
        American Meteorological Society 84(8).
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='CLMPARFLOW_INSTALL_PATH')
    exe: str = Field(default='parflow', alias='CLMPARFLOW_EXE')
    parflow_dir: str = Field(default='default', alias='CLMPARFLOW_DIR')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_CLMPARFLOW_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='CLMPARFLOW_SPATIAL_MODE')

    # Grid discretization (same as ParFlow)
    nx: int = Field(default=3, alias='CLMPARFLOW_NX', ge=1, le=10000)
    ny: int = Field(default=1, alias='CLMPARFLOW_NY', ge=1, le=10000)
    nz: int = Field(default=1, alias='CLMPARFLOW_NZ', ge=1, le=100)
    dx: float = Field(default=1000.0, alias='CLMPARFLOW_DX', gt=0)
    dy: float = Field(default=1000.0, alias='CLMPARFLOW_DY', gt=0)
    dz: float = Field(default=2.0, alias='CLMPARFLOW_DZ', gt=0)

    # Domain geometry
    top: float = Field(default=2.0, alias='CLMPARFLOW_TOP')
    bot: float = Field(default=0.0, alias='CLMPARFLOW_BOT')

    # Subsurface properties
    k_sat: float = Field(default=5.0, alias='CLMPARFLOW_K_SAT', gt=0)
    porosity: float = Field(default=0.4, alias='CLMPARFLOW_POROSITY', gt=0, le=1.0)
    vg_alpha: float = Field(default=1.0, alias='CLMPARFLOW_VG_ALPHA', gt=0)
    vg_n: float = Field(default=2.0, alias='CLMPARFLOW_VG_N', gt=1.0)
    s_res: float = Field(default=0.1, alias='CLMPARFLOW_S_RES', ge=0, lt=1.0)
    s_sat: float = Field(default=1.0, alias='CLMPARFLOW_S_SAT', gt=0, le=1.0)
    ss: float = Field(default=1e-5, alias='CLMPARFLOW_SS', gt=0)

    # Overland flow
    mannings_n: float = Field(default=0.03, alias='CLMPARFLOW_MANNINGS_N', gt=0)
    slope_x: float = Field(default=0.01, alias='CLMPARFLOW_SLOPE_X')

    # Initial conditions
    initial_pressure: Optional[float] = Field(default=None, alias='CLMPARFLOW_INITIAL_PRESSURE')

    # CLM-specific files
    vegm_file: str = Field(default='drv_vegm.dat', alias='CLMPARFLOW_VEGM_FILE')
    vegp_file: str = Field(default='drv_vegp.dat', alias='CLMPARFLOW_VEGP_FILE')
    drv_clmin_file: str = Field(default='drv_clmin.dat', alias='CLMPARFLOW_DRV_CLMIN_FILE')

    # CLM land surface settings
    istep_start: int = Field(default=1, alias='CLMPARFLOW_ISTEP_START', ge=1)
    clm_metfile: str = Field(default='forcing.1d', alias='CLMPARFLOW_METFILE')
    clm_metpath: str = Field(default='default', alias='CLMPARFLOW_METPATH')

    # Solver
    solver: str = Field(default='Richards', alias='CLMPARFLOW_SOLVER')
    timestep_hours: float = Field(default=1.0, alias='CLMPARFLOW_TIMESTEP_HOURS', gt=0)

    # Parallel execution
    num_procs: int = Field(default=1, alias='CLMPARFLOW_NUM_PROCS', ge=1, le=1024)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_CLMPARFLOW')

    # Calibration — combined subsurface + CLM land surface + Snow-17 + routing params
    params_to_calibrate: str = Field(
        default='K_SAT,POROSITY,VG_ALPHA,VG_N,S_RES,MANNINGS_N,SNOW17_SCF,SNOW17_MFMAX,SNOW17_MFMIN,SNOW17_PXTEMP,SNOW_LAPSE_RATE,ROUTE_ALPHA,ROUTE_K_SLOW,ROUTE_BASEFLOW',
        alias='CLMPARFLOW_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=7200, alias='CLMPARFLOW_TIMEOUT', ge=60, le=86400)


class PIHMConfig(BaseModel):
    """PIHM (Penn State Integrated Hydrologic Model) configuration.

    PIHM is a finite-volume, unstructured-mesh, fully-coupled
    surface-subsurface model solving Richards equation + diffusion wave
    overland flow + 1D channel routing. Uses SUNDIALS CVODE solver.

    Reference:
        Qu, Y. & Duffy, C.J. (2007): A semidiscrete finite volume
        formulation for multiprocess watershed simulation.
        Water Resources Research 43(8).
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='PIHM_INSTALL_PATH')
    exe: str = Field(default='pihm', alias='PIHM_EXE')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_PIHM_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='PIHM_SPATIAL_MODE')

    # Subsurface properties
    k_sat: float = Field(default=1e-5, alias='PIHM_K_SAT', gt=0)
    porosity: float = Field(default=0.4, alias='PIHM_POROSITY', gt=0, le=1.0)
    vg_alpha: float = Field(default=1.0, alias='PIHM_VG_ALPHA', gt=0)
    vg_n: float = Field(default=2.0, alias='PIHM_VG_N', gt=1.0)
    macropore_k: float = Field(default=1e-4, alias='PIHM_MACROPORE_K', gt=0)
    macropore_depth: float = Field(default=0.5, alias='PIHM_MACROPORE_DEPTH', ge=0)
    soil_depth: float = Field(default=2.0, alias='PIHM_SOIL_DEPTH', gt=0)

    # Overland flow
    mannings_n: float = Field(default=0.03, alias='PIHM_MANNINGS_N', gt=0)

    # Initial conditions
    init_gw_depth: float = Field(default=1.0, alias='PIHM_INIT_GW_DEPTH', ge=0)

    # Coupling
    coupling_source: str = Field(default='SUMMA', alias='PIHM_COUPLING_SOURCE')
    recharge_variable: str = Field(default='scalarSoilDrainage', alias='PIHM_RECHARGE_VARIABLE')

    # Solver
    solver_reltol: float = Field(default=1e-3, alias='PIHM_SOLVER_RELTOL', gt=0)
    solver_abstol: float = Field(default=1e-4, alias='PIHM_SOLVER_ABSTOL', gt=0)
    timestep_seconds: int = Field(default=60, alias='PIHM_TIMESTEP_SECONDS', ge=1, le=86400)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_PIHM')

    # Calibration
    params_to_calibrate: str = Field(
        default='K_SAT,POROSITY,VG_ALPHA,VG_N,MACROPORE_K,MANNINGS_N,SOIL_DEPTH',
        alias='PIHM_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=3600, alias='PIHM_TIMEOUT', ge=60, le=86400)


class HydroGeoSphereConfig(BaseModel):
    """HydroGeoSphere (HGS) fully-coupled 3D subsurface + surface model configuration.

    HGS is a control-volume finite-element, fully-coupled 3D variably-saturated
    subsurface + 2D overland flow + 1D channel flow model. Commercial code from
    Aquanty with university/research licenses available.

    Reference:
        Therrien, R., et al. (2010): HydroGeoSphere — A Three-dimensional
        Numerical Model Describing Fully-integrated Subsurface and Surface
        Flow and Solute Transport. Groundwater Simulations Group.
    """
    model_config = FROZEN_CONFIG

    # Installation
    install_path: str = Field(default='default', alias='HGS_INSTALL_PATH')
    exe: str = Field(default='hgs', alias='HGS_EXE')
    grok_exe: str = Field(default='grok', alias='HGS_GROK_EXE')

    # Settings
    settings_path: str = Field(default='default', alias='SETTINGS_HGS_PATH')
    spatial_mode: SpatialModeType = Field(default='lumped', alias='HGS_SPATIAL_MODE')

    # Subsurface properties
    k_sat: float = Field(default=1e-5, alias='HGS_K_SAT', gt=0)
    porosity: float = Field(default=0.4, alias='HGS_POROSITY', gt=0, le=1.0)
    vg_alpha: float = Field(default=1.0, alias='HGS_VG_ALPHA', gt=0)
    vg_n: float = Field(default=2.0, alias='HGS_VG_N', gt=1.0)
    vg_sres: float = Field(default=0.05, alias='HGS_VG_SRES', ge=0, lt=1.0)
    ss: float = Field(default=1e-4, alias='HGS_SS', gt=0)

    # Overland flow
    mannings_n: float = Field(default=0.03, alias='HGS_MANNINGS_N', gt=0)

    # Domain geometry
    soil_depth: float = Field(default=10.0, alias='HGS_SOIL_DEPTH', gt=0)
    domain_width: float = Field(default=1000.0, alias='HGS_DOMAIN_WIDTH', gt=0)

    # Coupling
    coupling_source: str = Field(default='SUMMA', alias='HGS_COUPLING_SOURCE')
    recharge_variable: str = Field(default='scalarSoilDrainage', alias='HGS_RECHARGE_VARIABLE')

    # Solver
    solver_max_iterations: int = Field(default=25, alias='HGS_SOLVER_MAX_ITERATIONS', ge=1, le=1000)
    timestep_seconds: int = Field(default=3600, alias='HGS_TIMESTEP_SECONDS', ge=1, le=86400)

    # Output
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_HGS')

    # Calibration
    params_to_calibrate: str = Field(
        default='K_SAT,POROSITY,VG_ALPHA,VG_N,VG_SRES,SS,MANNINGS_N',
        alias='HGS_PARAMS_TO_CALIBRATE'
    )

    # Execution
    timeout: int = Field(default=7200, alias='HGS_TIMEOUT', ge=60, le=86400)



__all__ = [
    'MODFLOWConfig',
    'ParFlowConfig',
    'CLMParFlowConfig',
    'PIHMConfig',
    'HydroGeoSphereConfig',
]
