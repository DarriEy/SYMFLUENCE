"""
WMFire Fire Definition Generator Module

Generates fire.def parameter files for RHESSys WMFire integration
with dynamic grid dimensions and configurable coefficients.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class FireDefParameters:
    """
    Parameters for fire.def file generation.

    All parameters are from RHESSys construct_fire_defaults.c
    with descriptions based on Kennedy et al. (2017).

    Reference:
        Kennedy, M.C., McKenzie, D., Tague, C., Dugger, A.L. 2017.
        Balancing uncertainty and complexity to incorporate fire spread
        in an eco-hydrological model. International Journal of Wildland Fire.
    """
    # Grid dimensions (required)
    n_rows: int = 3
    n_cols: int = 3

    # Temporal averaging
    ndays_average: float = 30.0  # Days for fuel moisture averaging

    # Fuel load coefficients (spread probability function)
    load_k1: float = 3.9    # Base spread rate coefficient
    load_k2: float = 0.07   # Fuel load sensitivity

    # Slope coefficients
    slope_k1: float = 0.91  # Base slope effect
    slope_k2: float = 1.0   # Slope sensitivity

    # Moisture coefficients (spread probability)
    moisture_k1: float = 3.8   # Base moisture effect
    moisture_k2: float = 0.27  # Moisture sensitivity

    # Wind direction coefficients
    winddir_k1: float = 0.87  # Base wind effect
    winddir_k2: float = 0.48  # Wind direction sensitivity

    # Ignition moisture coefficients
    moisture_ign_k1: float = 3.8   # Ignition moisture effect
    moisture_ign_k2: float = 0.27  # Ignition moisture sensitivity

    # Wind parameters
    windmax: float = 1.0  # Maximum wind speed factor

    # Ignition location (-1 = random)
    ignition_col: int = -1
    ignition_row: int = -1
    ignition_tmin: float = 0.0  # Min temperature for ignition (C) - lowered for mountain watersheds

    # Output options
    fire_verbose: int = 0  # Verbose output (0=off, 1=on)
    fire_write: int = 1    # Write fire grids (0=off, 1=on)
    fire_in_buffer: int = 0  # Buffer zone handling

    # Spread calculation type (9 = von Mises-Fisher based)
    spread_calc_type: int = 9

    # Wind distribution parameters (log-normal)
    mean_log_wind: float = 0.494
    sd_log_wind: float = 0.654

    # von Mises distribution parameters for fire spread direction
    mean1_rvm: float = 1.71
    mean2_rvm: float = -1.91
    kappa1_rvm: float = 2.37
    kappa2_rvm: float = 2.38
    p_rvm: float = 0.411

    # Ignition and vegetation parameters
    ign_def_mod: float = 1.0   # Ignition probability modifier
    veg_k1: float = 0.8        # Vegetation effect coefficient
    veg_k2: float = 10.0       # Vegetation sensitivity
    mean_ign: float = 1.0      # Mean ignition events

    # Random seed (0 = system time)
    ran_seed: int = 0

    # Fire effects
    calc_fire_effects: int = 0  # Calculate ecological effects
    include_wui: int = 0        # Include wildland-urban interface

    # Output file naming
    fire_size_name: int = 0

    # Wind direction shift (degrees)
    wind_shift: float = 0.0


class FireDefGenerator:
    """
    Generates fire.def parameter files for WMFire.

    Creates properly formatted fire.def files with grid dimensions
    matching the generated fire grids and optionally modified
    coefficients based on fuel and moisture statistics.
    """

    def __init__(self, config, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the FireDefGenerator.

        Args:
            config: SymfluenceConfig object with WMFire settings
            logger_instance: Optional logger for status messages
        """
        self.config = config
        self.logger = logger_instance or logger
        self._wmfire_config = self._get_wmfire_config()

    def _get_wmfire_config(self):
        """Extract WMFire configuration from config object."""
        try:
            if (hasattr(self.config, 'model') and
                hasattr(self.config.model, 'rhessys') and
                self.config.model.rhessys is not None):
                return self.config.model.rhessys.wmfire
        except AttributeError:
            pass
        return None

    def _get_config_value(self, attr_name, default=None):
        """
        Safely get a numeric config value, returning default if not a number.

        Protects against MagicMock objects in tests that would fail f-string formatting.

        Args:
            attr_name: Attribute name to retrieve
            default: Default value if attribute doesn't exist or isn't numeric

        Returns:
            Numeric value or default
        """
        if not self._wmfire_config:
            return default

        value = getattr(self._wmfire_config, attr_name, default)

        # Check if it's actually a number (not MagicMock or other object)
        if value is not None and not isinstance(value, (int, float)):
            return default

        return value

    def generate_fire_def(
        self,
        grid,
        fuel_stats: Optional[Dict[str, float]] = None,
        moisture_stats: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        """
        Generate fire.def content string.

        Args:
            grid: FireGrid object (for n_rows, n_cols)
            fuel_stats: Optional fuel statistics for coefficient adjustment
            moisture_stats: Optional moisture statistics for coefficient adjustment
            **kwargs: Additional parameter overrides

        Returns:
            Formatted fire.def content string
        """
        # Start with default parameters
        params = FireDefParameters(
            n_rows=grid.nrows,
            n_cols=grid.ncols
        )

        # Apply WMFire config overrides (using safe getter to avoid MagicMock issues)
        if self._wmfire_config:
            ndays = self._get_config_value('ndays_average')
            if ndays is not None:
                params.ndays_average = ndays

            load_k1 = self._get_config_value('load_k1')
            if load_k1 is not None:
                params.load_k1 = load_k1

            load_k2 = self._get_config_value('load_k2')
            if load_k2 is not None:
                params.load_k2 = load_k2

            moisture_k1 = self._get_config_value('moisture_k1')
            if moisture_k1 is not None:
                params.moisture_k1 = moisture_k1
                params.moisture_ign_k1 = moisture_k1

            moisture_k2 = self._get_config_value('moisture_k2')
            if moisture_k2 is not None:
                params.moisture_k2 = moisture_k2
                params.moisture_ign_k2 = moisture_k2

            ign_def_mod = self._get_config_value('ign_def_mod')
            if ign_def_mod is not None:
                params.ign_def_mod = ign_def_mod

            mean_ign = self._get_config_value('mean_ign')
            if mean_ign is not None:
                params.mean_ign = mean_ign

            windmax = self._get_config_value('windmax')
            if windmax is not None:
                params.windmax = windmax

            slope_k1 = self._get_config_value('slope_k1')
            if slope_k1 is not None:
                params.slope_k1 = slope_k1

        # Apply fuel-based coefficient adjustments
        if fuel_stats and 'load_k1' in fuel_stats:
            params.load_k1 = fuel_stats['load_k1']
        if fuel_stats and 'load_k2' in fuel_stats:
            params.load_k2 = fuel_stats['load_k2']

        # Apply moisture-based coefficient adjustments
        if moisture_stats and 'moisture_k1' in moisture_stats:
            params.moisture_k1 = moisture_stats['moisture_k1']
            params.moisture_ign_k1 = moisture_stats['moisture_k1']
        if moisture_stats and 'moisture_k2' in moisture_stats:
            params.moisture_k2 = moisture_stats['moisture_k2']
            params.moisture_ign_k2 = moisture_stats['moisture_k2']

        # Apply any direct overrides
        for key, value in kwargs.items():
            if hasattr(params, key):
                setattr(params, key, value)

        # Generate formatted content
        return self._format_fire_def(params)

    def _format_fire_def(self, params: FireDefParameters) -> str:
        """
        Format FireDefParameters into fire.def file content.

        Args:
            params: FireDefParameters object

        Returns:
            Formatted string for fire.def file
        """
        # RHESSys fire.def format: value<tab>parameter_name
        lines = [
            "1    fire_parm_ID",
            f"{params.ndays_average:.1f}    ndays_average",
            f"{params.load_k1:.2f}    load_k1",
            f"{params.load_k2:.2f}    load_k2",
            f"{params.slope_k1:.2f}    slope_k1",
            f"{params.slope_k2:.1f}    slope_k2",
            f"{params.moisture_k1:.2f}    moisture_k1",
            f"{params.moisture_k2:.2f}    moisture_k2",
            f"{params.winddir_k1:.2f}    winddir_k1",
            f"{params.winddir_k2:.2f}    winddir_k2",
            f"{params.moisture_ign_k1:.2f}    moisture_ign_k1",
            f"{params.moisture_ign_k2:.2f}    moisture_ign_k2",
            f"{params.windmax:.1f}    windmax",
            f"{params.ignition_col}    ignition_col",
            f"{params.ignition_row}    ignition_row",
            f"{params.ignition_tmin:.1f}    ignition_tmin",
            f"{params.fire_verbose}    fire_verbose",
            f"{params.fire_write}    fire_write",
            f"{params.fire_in_buffer}    fire_in_buffer",
            f"{params.n_rows}    n_rows",
            f"{params.n_cols}    n_cols",
            f"{params.spread_calc_type}    spread_calc_type",
            f"{params.mean_log_wind:.3f}    mean_log_wind",
            f"{params.sd_log_wind:.3f}    sd_log_wind",
            f"{params.mean1_rvm:.2f}    mean1_rvm",
            f"{params.mean2_rvm:.2f}    mean2_rvm",
            f"{params.kappa1_rvm:.2f}    kappa1_rvm",
            f"{params.kappa2_rvm:.2f}    kappa2_rvm",
            f"{params.p_rvm:.3f}    p_rvm",
            f"{params.ign_def_mod:.1f}    ign_def_mod",
            f"{params.veg_k1:.1f}    veg_k1",
            f"{params.veg_k2:.1f}    veg_k2",
            f"{params.mean_ign:.1f}    mean_ign",
            f"{params.ran_seed}    ran_seed",
            f"{params.calc_fire_effects}    calc_fire_effects",
            f"{params.include_wui}    include_wui",
            f"{params.fire_size_name}    fire_size_name",
            f"{params.wind_shift:.1f}    wind_shift",
        ]

        return '\n'.join(lines) + '\n'

    def write_fire_def(
        self,
        output_path: Union[str, Path],
        grid,
        fuel_stats: Optional[Dict[str, float]] = None,
        moisture_stats: Optional[Dict[str, float]] = None,
        ignition_row: int = -1,
        ignition_col: int = -1,
        **kwargs
    ) -> Path:
        """
        Write fire.def file to disk.

        Args:
            output_path: Path for output file
            grid: FireGrid object (for dimensions)
            fuel_stats: Optional fuel statistics
            moisture_stats: Optional moisture statistics
            ignition_row: Row index for ignition point (-1 for random)
            ignition_col: Column index for ignition point (-1 for random)
            **kwargs: Additional parameter overrides

        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add ignition indices to kwargs
        kwargs['ignition_row'] = ignition_row
        kwargs['ignition_col'] = ignition_col

        content = self.generate_fire_def(
            grid,
            fuel_stats=fuel_stats,
            moisture_stats=moisture_stats,
            **kwargs
        )

        output_path.write_text(content)

        ign_str = f"({ignition_row}, {ignition_col})" if ignition_row >= 0 else "random"
        self.logger.info(f"Fire defaults written: {output_path} "
                        f"(grid: {grid.nrows}x{grid.ncols}, ignition: {ign_str})")

        return output_path

    def generate_default_fire_def(
        self,
        n_rows: int,
        n_cols: int,
        output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """
        Generate fire.def with default parameters and specified dimensions.

        Convenience method for simple grid generation without
        fuel/moisture statistics.

        Args:
            n_rows: Number of grid rows
            n_cols: Number of grid columns
            output_path: Optional path to write file

        Returns:
            Fire.def content string
        """
        params = FireDefParameters(n_rows=n_rows, n_cols=n_cols)

        # Apply WMFire config if available (using safe getter)
        if self._wmfire_config:
            ndays = self._get_config_value('ndays_average')
            if ndays is not None:
                params.ndays_average = ndays

        content = self._format_fire_def(params)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content)
            self.logger.info(f"Default fire.def written: {output_path}")

        return content


def validate_fire_def(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a fire.def file and return parsed parameters.

    Args:
        file_path: Path to fire.def file

    Returns:
        Dictionary of parameter names to values

    Raises:
        ValueError: If file format is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValueError(f"Fire def file not found: {file_path}")

    params = {}
    required_params = ['n_rows', 'n_cols', 'fire_parm_ID']

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse value<whitespace>parameter_name format
            parts = line.split()
            if len(parts) >= 2:
                try:
                    value = parts[0]
                    name = parts[-1]

                    # Try to parse as number
                    if '.' in value:
                        params[name] = float(value)
                    else:
                        params[name] = int(value)
                except ValueError:
                    logger.warning(f"Could not parse line {line_num}: {line}")

    # Check required parameters
    for req in required_params:
        if req not in params:
            raise ValueError(f"Missing required parameter: {req}")

    return params
