"""
MizuRoute Control File Writer

Unified control file generation for different source models (SUMMA, FUSE, GR).
Eliminates code duplication by using configuration-driven templates.
"""

from pathlib import Path
from typing import Dict, Any, Optional, TextIO, Union, TYPE_CHECKING
import logging

from symfluence.core.mixins import ConfigurableMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class ModelRunoffConfig:
    """Configuration for model-specific runoff settings."""

    def __init__(
        self,
        output_dir_key: str,
        output_dir_name: str,
        default_var: str,
        default_units: str,
        default_dt: str,
        output_file_pattern: str,
        hru_dim: str = 'gru',
        hru_var: str = 'gruId',
        comment_name: str = 'model'
    ):
        self.output_dir_key = output_dir_key
        self.output_dir_name = output_dir_name
        self.default_var = default_var
        self.default_units = default_units
        self.default_dt = default_dt
        self.output_file_pattern = output_file_pattern
        self.hru_dim = hru_dim
        self.hru_var = hru_var
        self.comment_name = comment_name


# Pre-defined configurations for each source model
MODEL_CONFIGS = {
    'summa': ModelRunoffConfig(
        output_dir_key='EXPERIMENT_OUTPUT_SUMMA',
        output_dir_name='SUMMA',
        default_var='averageRoutedRunoff',
        default_units='m/s',
        default_dt='3600',
        output_file_pattern='{experiment_id}_timestep.nc',
        hru_dim='hru',  # Can be overridden to 'gru' for distributed
        hru_var='hruId',
        comment_name='SUMMA'
    ),
    'fuse': ModelRunoffConfig(
        output_dir_key='EXPERIMENT_OUTPUT_FUSE',
        output_dir_name='FUSE',
        default_var='q_routed',
        default_units='m/s',
        default_dt='3600',
        output_file_pattern='{experiment_id}_timestep.nc',
        hru_dim='gru',
        hru_var='gruId',
        comment_name='FUSE'
    ),
    'gr': ModelRunoffConfig(
        output_dir_key='EXPERIMENT_OUTPUT_GR',
        output_dir_name='GR',
        default_var='q_routed',
        default_units='m/s',
        default_dt='86400',  # GR is daily
        output_file_pattern='{domain_name}_{experiment_id}_runs_def.nc',
        hru_dim='gru',
        hru_var='gruId',
        comment_name='GR4J'
    ),
}


class ControlFileWriter(ConfigurableMixin):
    """
    Unified mizuRoute control file writer.

    Handles control file generation for SUMMA, FUSE, and GR source models
    using a template-based approach with model-specific configurations.
    """

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        setup_dir: Path,
        project_dir: Path,
        experiment_id: str,
        domain_name: str,
        logger: Optional[logging.Logger] = None
    ):
        # Set up typed config via ConfigurableMixin
        from symfluence.core.config.models import SymfluenceConfig
        if isinstance(config, dict):
            self._config = SymfluenceConfig(**config)
        else:
            self._config = config
        # Backward compatibility alias
        self.config = self._config

        self.setup_dir = setup_dir
        self.project_dir = project_dir
        self.experiment_id = experiment_id
        self.domain_name = domain_name
        self.logger = logger or logging.getLogger(__name__)

        # These can be set by the preprocessor before writing
        self.summa_uses_gru_runoff = False
        self.needs_remap_lumped_distributed = False

    def write_control_file(
        self,
        model_type: str = 'summa',
        control_file_name: Optional[str] = None,
        mizu_config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Write a mizuRoute control file for the specified source model.

        Args:
            model_type: Source model type ('summa', 'fuse', or 'gr')
            control_file_name: Override control file name (default from config)
            mizu_config: MizuRoute-specific config values (topology file, remap file, etc.)

        Returns:
            Path to the generated control file
        """
        model_type = model_type.lower()
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Expected one of: {list(MODEL_CONFIGS.keys())}")

        model_config = MODEL_CONFIGS[model_type]
        mizu_config = mizu_config or {}

        # Determine control file name
        if control_file_name is None:
            default_name = 'mizuroute.control' if model_type != 'gr' else 'mizuRoute_control_GR.txt'
            control_file_name = self._get_config_value(lambda: self.config.model.mizuroute.control_file, default=default_name)

        control_path = self.setup_dir / control_file_name
        self.logger.debug(f"Creating mizuRoute control file for {model_type.upper()}: {control_path}")

        with open(control_path, 'w') as cf:
            self._write_header(cf)
            self._write_directories(cf, model_config)
            self._write_parameters(cf, mizu_config)
            self._write_simulation_controls(cf, model_type)
            self._write_topology(cf, mizu_config)
            self._write_runoff(cf, model_config)
            self._write_remapping(cf, mizu_config)
            self._write_miscellaneous(cf, mizu_config)

        self.logger.debug(f"mizuRoute control file created: {control_path}")
        return control_path

    def _write_header(self, cf: TextIO) -> None:
        """Write control file header."""
        cf.write("! mizuRoute control file generated by SUMMA public workflow scripts \n")

    def _get_model_experiment_output(self, model_config: ModelRunoffConfig) -> str:
        """Get experiment output path for a specific model type."""
        # Map model output_dir_key to typed config path
        key_to_config = {
            'EXPERIMENT_OUTPUT_SUMMA': lambda: self.config.model.summa.experiment_output,
            'EXPERIMENT_OUTPUT_FUSE': lambda: self.config.model.fuse.experiment_output,
            'EXPERIMENT_OUTPUT_GR': lambda: self.config.model.gr.experiment_output,
        }
        getter = key_to_config.get(model_config.output_dir_key)
        if getter:
            return self._get_config_value(getter, default='default')
        return 'default'

    def _write_directories(self, cf: TextIO, model_config: ModelRunoffConfig) -> None:
        """Write directory paths section."""
        # Get model-specific output directory
        model_output = self._get_model_experiment_output(model_config)
        if model_output == 'default':
            model_output = self.project_dir / f"simulations/{self.experiment_id}" / model_config.output_dir_name
        else:
            model_output = Path(model_output)

        # Get mizuRoute output directory
        mizu_output = self._get_config_value(lambda: self.config.model.mizuroute.experiment_output, default='default')
        if mizu_output == 'default' or not mizu_output:
            mizu_output = self.project_dir / f"simulations/{self.experiment_id}" / 'mizuRoute'
        else:
            mizu_output = Path(mizu_output)

        # Ensure output directory exists
        mizu_output.mkdir(parents=True, exist_ok=True)

        cf.write("!\n! --- DEFINE DIRECTORIES \n")
        cf.write(f"<ancil_dir>             {self.setup_dir}/    ! Folder that contains ancillary data (river network, remapping netCDF) \n")
        cf.write(f"<input_dir>             {model_output}/    ! Folder that contains runoff data from {model_config.comment_name} \n")
        cf.write(f"<output_dir>            {mizu_output}/    ! Folder that will contain mizuRoute simulations \n")

    def _write_parameters(self, cf: TextIO, mizu_config: Dict[str, Any]) -> None:
        """Write namelist parameters section."""
        param_file = mizu_config.get('parameters_file', self._get_config_value(lambda: self.config.model.mizuroute.parameter_file, default='mizuRoute_params.nml'))
        cf.write("!\n! --- NAMELIST FILENAME \n")
        cf.write(f"<param_nml>             {param_file}    ! Spatially constant parameter namelist (should be stored in the ancil_dir) \n")

    def _write_simulation_controls(self, cf: TextIO, model_type: str) -> None:
        """Write simulation control section with proper time handling."""
        sim_start = self._get_config_value(lambda: self.config.domain.time_start, default='')
        sim_end = self._get_config_value(lambda: self.config.domain.time_end, default='')

        # Special handling for GR: force midnight alignment for daily data
        if model_type == 'gr':
            if isinstance(sim_start, str):
                sim_start = sim_start.split(' ')[0] + " 00:00"
            if isinstance(sim_end, str):
                sim_end = sim_end.split(' ')[0] + " 00:00"
            self.logger.debug(f"Forced GR simulation period to midnight: {sim_start} to {sim_end}")

        # Ensure dates are in proper format
        if isinstance(sim_start, str) and len(sim_start) == 10:
            sim_start = f"{sim_start} 00:00"
        if isinstance(sim_end, str) and len(sim_end) == 10:
            sim_end = f"{sim_end} 23:00"

        # Get routing scheme from config (default to IRF for proper river routing)
        route_opt = self._get_config_value(
            lambda: self.config.model.mizuroute.output_vars,
            default='1',
            dict_key='SETTINGS_MIZU_OUTPUT_VARS'
        )

        # Get output file frequency from config
        output_freq = self._get_config_value(
            lambda: self.config.model.mizuroute.output_freq,
            default='yearly',
            dict_key='SETTINGS_MIZU_OUTPUT_FREQ'
        )

        cf.write("!\n! --- DEFINE SIMULATION CONTROLS \n")
        cf.write(f"<case_name>             {self.experiment_id}    ! Simulation case name \n")
        cf.write(f"<sim_start>             {sim_start}    ! Time of simulation start \n")
        cf.write(f"<sim_end>               {sim_end}    ! Time of simulation end \n")
        cf.write(f"<route_opt>             {route_opt}    ! Routing scheme. 0 -> accumRunoff, 1 -> IRF-UH, 2 -> IRF-KW, 3 -> KW-IRF, 4 -> MC-IRF \n")
        cf.write(f"<newFileFrequency>      {output_freq}    ! Output file frequency (single, yearly, monthly, daily) \n")

    def _write_topology(self, cf: TextIO, mizu_config: Dict[str, Any]) -> None:
        """Write topology file section."""
        topology_file = mizu_config.get('topology_file', self._get_config_value(lambda: self.config.model.mizuroute.topology_file, default='network_topology.nc'))

        cf.write("!\n! --- DEFINE TOPOLOGY FILE \n")
        cf.write(f"<fname_ntopOld>         {topology_file}    ! Name of input netCDF for River Network \n")
        cf.write("<dname_sseg>            seg    ! Dimension name for reach in river network netCDF \n")
        cf.write("<dname_nhru>            hru    ! Dimension name for RN_HRU in river network netCDF \n")
        cf.write("<seg_outlet>            -9999    ! Outlet reach ID at which to stop routing (i.e. use subset of full network). -9999 to use full network \n")
        cf.write("<varname_area>          area    ! Name of variable holding hru area \n")
        cf.write("<varname_length>        length    ! Name of variable holding segment length \n")
        cf.write("<varname_slope>         slope    ! Name of variable holding segment slope \n")
        cf.write("<varname_HRUid>         hruId    ! Name of variable holding HRU id \n")
        cf.write("<varname_hruSegId>      hruToSegId    ! Name of variable holding the stream segment below each HRU \n")
        cf.write("<varname_segId>         segId    ! Name of variable holding the ID of each stream segment \n")
        cf.write("<varname_downSegId>     downSegId    ! Name of variable holding the ID of the next downstream segment \n")

    def _write_runoff(self, cf: TextIO, model_config: ModelRunoffConfig) -> None:
        """Write runoff file section with model-specific settings."""
        # Get overrideable values from config
        routing_var = self._get_config_value(lambda: self.config.model.mizuroute.routing_var, default=model_config.default_var)
        if routing_var in ('default', None, ''):
            routing_var = model_config.default_var

        routing_units = self._get_config_value(lambda: self.config.model.mizuroute.routing_units, default=model_config.default_units)
        if routing_units in ('default', None, ''):
            routing_units = model_config.default_units

        routing_dt = self._get_config_value(lambda: self.config.model.mizuroute.routing_dt, default=model_config.default_dt)
        if routing_dt in ('default', None, ''):
            routing_dt = model_config.default_dt

        # Generate output file name from pattern
        output_file = model_config.output_file_pattern.format(
            experiment_id=self.experiment_id,
            domain_name=self.domain_name
        )

        # Determine HRU dimension/variable (can be overridden for distributed SUMMA)
        hru_dim = model_config.hru_dim
        hru_var = model_config.hru_var
        if self.summa_uses_gru_runoff and model_config.hru_dim == 'hru':
            hru_dim = 'gru'
            hru_var = 'gruId'

        cf.write("!\n! --- DEFINE RUNOFF FILE \n")
        cf.write(f"<fname_qsim>            {output_file}    ! netCDF name for {model_config.comment_name} runoff \n")
        cf.write(f"<vname_qsim>            {routing_var}    ! Variable name for {model_config.comment_name} runoff \n")
        cf.write(f"<units_qsim>            {routing_units}    ! Units of input runoff \n")
        cf.write(f"<dt_qsim>               {routing_dt}    ! Time interval of input runoff in seconds \n")
        cf.write("<dname_time>            time    ! Dimension name for time \n")
        cf.write("<vname_time>            time    ! Variable name for time \n")
        cf.write(f"<dname_hruid>           {hru_dim}     ! Dimension name for HM_HRU ID \n")
        cf.write(f"<vname_hruid>           {hru_var}   ! Variable name for HM_HRU ID \n")
        cf.write("<calendar>              standard    ! Calendar of the nc file \n")

    def _write_remapping(self, cf: TextIO, mizu_config: Dict[str, Any]) -> None:
        """Write runoff remapping section."""
        remap_file = mizu_config.get('remap_file', self._get_config_value(lambda: self.config.model.mizuroute.remap_file, default='runoff_remap.nc'))
        remap_flag = (
            self._get_config_value(lambda: self.config.model.mizuroute.needs_remap, default=False) or
            self.needs_remap_lumped_distributed
        )

        cf.write("!\n! --- DEFINE RUNOFF MAPPING FILE \n")
        cf.write(f"<is_remap>              {'T' if remap_flag else 'F'}    ! Logical to indicate runoff needs to be remapped to RN_HRU. T or F \n")

        if remap_flag:
            cf.write(f"<fname_remap>           {remap_file}    ! netCDF name of runoff remapping \n")
            cf.write("<vname_hruid_in_remap>  RN_hruId    ! Variable name for RN_HRUs \n")
            cf.write("<vname_weight>          weight    ! Variable name for areal weights of overlapping HM_HRUs \n")
            cf.write("<vname_qhruid>          HM_hruId    ! Variable name for HM_HRU ID \n")
            cf.write("<vname_num_qhru>        nOverlaps    ! Variable name for a numbers of overlapping HM_HRUs with RN_HRUs \n")
            cf.write("<dname_hru_remap>       hru    ! Dimension name for HM_HRU \n")
            cf.write("<dname_data_remap>      data    ! Dimension name for data \n")

    def _write_miscellaneous(self, cf: TextIO, mizu_config: Dict[str, Any]) -> None:
        """Write miscellaneous settings section."""
        within_basin = mizu_config.get('within_basin', self._get_config_value(lambda: self.config.model.mizuroute.within_basin, default=0))
        cf.write("!\n! --- MISCELLANEOUS \n")
        cf.write(f"<doesBasinRoute>        {within_basin}    ! Hillslope routing options. 0 -> no (already routed by SUMMA), 1 -> use IRF \n")
