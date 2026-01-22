"""
dRoute-specific configuration mixins.

Provides standardized access to dRoute configuration values via properties,
replacing scattered config_dict.get() calls with typed accessors.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class DRouteConfigMixin:
    """
    Mixin for dRoute configuration access.

    Provides properties for accessing dRoute-specific configuration values
    from the typed config, with sensible defaults.

    Requires the class to have:
    - self.config: SymfluenceConfig instance
    - self._get_config_value(): method from ConfigMixin
    """

    # =========================================================================
    # Core Routing Configuration
    # =========================================================================

    @property
    def droute_routing_method(self) -> str:
        """Routing method from config.model.droute.routing_method."""
        return self._get_config_value(
            lambda: self.config.model.droute.routing_method,
            default='muskingum_cunge'
        )

    @property
    def droute_routing_dt(self) -> int:
        """Routing time step in seconds from config.model.droute.routing_dt."""
        return int(self._get_config_value(
            lambda: self.config.model.droute.routing_dt,
            default=3600
        ))

    @property
    def droute_execution_mode(self) -> str:
        """Execution mode from config.model.droute.execution_mode."""
        return self._get_config_value(
            lambda: self.config.model.droute.execution_mode,
            default='python'
        )

    # =========================================================================
    # Gradient/AD Configuration
    # =========================================================================

    @property
    def droute_enable_gradients(self) -> bool:
        """Enable gradients flag from config.model.droute.enable_gradients."""
        return self._get_config_value(
            lambda: self.config.model.droute.enable_gradients,
            default=False
        )

    @property
    def droute_ad_backend(self) -> str:
        """AD backend from config.model.droute.ad_backend."""
        return self._get_config_value(
            lambda: self.config.model.droute.ad_backend,
            default='codipack'
        )

    # =========================================================================
    # File Configuration
    # =========================================================================

    @property
    def droute_topology_file(self) -> str:
        """Topology file name from config.model.droute.topology_file."""
        return self._get_config_value(
            lambda: self.config.model.droute.topology_file,
            default='topology.nc'
        )

    @property
    def droute_topology_format(self) -> str:
        """Topology format from config.model.droute.topology_format."""
        return self._get_config_value(
            lambda: self.config.model.droute.topology_format,
            default='netcdf'
        )

    @property
    def droute_config_file(self) -> str:
        """Config file name from config.model.droute.config_file."""
        return self._get_config_value(
            lambda: self.config.model.droute.config_file,
            default='droute_config.yaml'
        )

    # =========================================================================
    # Integration Configuration
    # =========================================================================

    @property
    def droute_from_model(self) -> str:
        """Source model for routing input from config.model.droute.from_model."""
        return self._get_config_value(
            lambda: self.config.model.droute.from_model,
            default='default'
        )

    # =========================================================================
    # Path Configuration
    # =========================================================================

    @property
    def droute_settings_path(self) -> Optional[str]:
        """Settings path from config.model.droute.settings_path."""
        return self._get_config_value(
            lambda: self.config.model.droute.settings_path,
            default=None
        )

    @property
    def droute_install_path(self) -> Optional[str]:
        """Install path from config.model.droute.install_path."""
        return self._get_config_value(
            lambda: self.config.model.droute.install_path,
            default=None
        )

    @property
    def droute_exe(self) -> str:
        """Executable name from config.model.droute.exe."""
        return self._get_config_value(
            lambda: self.config.model.droute.exe,
            default='droute'
        )

    @property
    def droute_experiment_output(self) -> Optional[str]:
        """Experiment output path from config.model.droute.experiment_output."""
        return self._get_config_value(
            lambda: self.config.model.droute.experiment_output,
            default=None
        )

    @property
    def droute_experiment_log(self) -> Optional[str]:
        """Experiment log path from config.model.droute.experiment_log."""
        return self._get_config_value(
            lambda: self.config.model.droute.experiment_log,
            default=None
        )

    # =========================================================================
    # Calibration Configuration
    # =========================================================================

    @property
    def droute_params_to_calibrate(self) -> str:
        """Parameters to calibrate from config.model.droute.params_to_calibrate."""
        return self._get_config_value(
            lambda: self.config.model.droute.params_to_calibrate,
            default='velocity,diffusivity'
        )


__all__ = ['DRouteConfigMixin']
