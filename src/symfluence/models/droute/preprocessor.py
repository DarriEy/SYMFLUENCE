"""
dRoute Model Preprocessor.

Handles spatial preprocessing and configuration generation for the dRoute routing model.
Follows the same pattern as MizuRoutePreProcessor to ensure consistent workflow.
"""

import os
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, Optional

from symfluence.models.registry import ModelRegistry
from symfluence.models.base import BaseModelPreProcessor
from symfluence.models.droute.mixins import DRouteConfigMixin
from symfluence.models.droute.network_adapter import DRouteNetworkAdapter


@ModelRegistry.register_preprocessor('DROUTE')
class DRoutePreProcessor(BaseModelPreProcessor, DRouteConfigMixin):
    """
    Spatial preprocessor and configuration generator for the dRoute routing model.

    This preprocessor handles all spatial setup tasks required to run dRoute:
    - Network topology loading (from mizuRoute-compatible files)
    - Conversion to dRoute format
    - Configuration file generation

    The preprocessor can reuse existing mizuRoute topology files, enabling
    seamless switching between routing models without re-preprocessing.

    Supported Source Models:
        - SUMMA: Physics-based snow hydrology
        - FUSE: Framework for Understanding Structural Errors
        - GR: Parsimonious hydrological models
        - HYPE: Semi-distributed hydrological model
        - HBV: JAX-based HBV model

    Key Methods:
        run_preprocessing(): Orchestrates all preprocessing steps
        copy_base_settings(): Copy template files
        setup_topology(): Load/convert network topology
        create_config_file(): Generate dRoute configuration YAML

    Configuration Dependencies:
        Required:
            - DOMAIN_NAME: Basin identifier
            - EXPERIMENT_ID: Experiment identifier

        Optional:
            - SETTINGS_DROUTE_PATH: Custom setup directory
            - DROUTE_TOPOLOGY_FILE: Topology file name (default: topology.nc)
            - DROUTE_TOPOLOGY_FORMAT: Topology format (netcdf/geojson/csv)
            - DROUTE_FROM_MODEL: Source model name

    Example:
        >>> config = {
        ...     'DOMAIN_NAME': 'bow_river',
        ...     'EXPERIMENT_ID': 'calibration_run',
        ...     'ROUTING_MODEL': 'DROUTE',
        ...     'droute': {'routing_method': 'muskingum_cunge'}
        ... }
        >>> preprocessor = DRoutePreProcessor(config, logger)
        >>> preprocessor.run_preprocessing()
    """

    def _get_model_name(self) -> str:
        """Return model name for directory structure."""
        return "dRoute"

    def __init__(self, config: Dict[str, Any], logger: Any):
        """
        Initialize the dRoute preprocessor.

        Args:
            config: Configuration dictionary or SymfluenceConfig object
            logger: Logger instance for status messages
        """
        super().__init__(config, logger)

        self.logger.debug(f"DRoutePreProcessor initialized. Default setup_dir: {self.setup_dir}")

        # Override setup_dir if SETTINGS_DROUTE_PATH is provided
        droute_settings_path = self.droute_settings_path
        if droute_settings_path and droute_settings_path != 'default':
            self.setup_dir: Path = Path(droute_settings_path)
            self.logger.debug(f"Using custom setup_dir: {self.setup_dir}")

        # Ensure setup directory exists
        if not self.setup_dir.exists():
            self.logger.info(f"Creating dRoute setup directory: {self.setup_dir}")
            self.setup_dir.mkdir(parents=True, exist_ok=True)

        # Initialize network adapter
        self.network_adapter = DRouteNetworkAdapter(logger)

    def run_preprocessing(self):
        """
        Run the complete dRoute preprocessing workflow.

        Steps:
        1. Copy base settings from templates
        2. Load/setup network topology (reuse mizuRoute if available)
        3. Convert topology to dRoute format
        4. Generate dRoute configuration file
        """
        self.logger.debug("Starting dRoute spatial preprocessing")

        self.copy_base_settings()
        self.setup_topology()
        self.create_config_file()

        self.logger.info("dRoute spatial preprocessing completed")

    def copy_base_settings(self, source_dir: Optional[Path] = None, file_patterns: Optional[list] = None):
        """
        Copy dRoute base settings from package resources.

        If no dRoute-specific templates exist, this is a no-op since
        dRoute generates its configuration dynamically.
        """
        if source_dir:
            return super().copy_base_settings(source_dir, file_patterns)

        self.logger.info("Setting up dRoute base configuration")
        self.setup_dir.mkdir(parents=True, exist_ok=True)

        # Check if dRoute base settings exist in resources
        try:
            from symfluence.resources import get_base_settings_dir
            base_settings_path = get_base_settings_dir('dRoute')
            if base_settings_path.exists():
                for file in os.listdir(base_settings_path):
                    copyfile(base_settings_path / file, self.setup_dir / file)
                self.logger.info("dRoute base settings copied")
            else:
                self.logger.debug("No dRoute base settings found, will generate dynamically")
        except (ImportError, FileNotFoundError):
            self.logger.debug("No dRoute base settings available, will generate dynamically")

    def setup_topology(self):
        """
        Load or create network topology for dRoute.

        First checks for existing mizuRoute topology file (enables reuse).
        Falls back to creating topology from shapefiles if needed.
        """
        self.logger.info("Setting up dRoute network topology")

        # Try to reuse existing mizuRoute topology
        topology_file = self._find_existing_topology()

        if topology_file:
            self.logger.info(f"Reusing existing topology: {topology_file}")
            self.topology_path = topology_file
        else:
            # Check for mizuRoute topology in standard location
            mizu_topology = self.project_dir / 'settings' / 'mizuRoute' / 'mizuRoute_topology.nc'
            if mizu_topology.exists():
                self.logger.info(f"Found mizuRoute topology, will reuse: {mizu_topology}")
                self.topology_path = mizu_topology
            else:
                # Need to generate topology from scratch
                self.logger.warning(
                    "No existing topology found. Please run mizuRoute preprocessing first "
                    "or provide a topology file."
                )
                self.topology_path = None
                return

        # Load and convert topology
        if self.topology_path:
            try:
                topology = self.network_adapter.load_topology(
                    self.topology_path,
                    format=self.droute_topology_format
                )

                # Validate topology
                is_valid, warnings = self.network_adapter.validate_topology(topology)
                for warning in warnings:
                    self.logger.warning(f"Topology validation: {warning}")

                if not is_valid:
                    self.logger.error("Topology validation failed with errors")

                # Convert to dRoute format
                self.droute_network = self.network_adapter.to_droute_format(
                    topology,
                    routing_method=self.droute_routing_method,
                    routing_dt=self.droute_routing_dt
                )

                self.logger.info(
                    f"Topology loaded: {self.droute_network['n_segments']} segments, "
                    f"{len(self.droute_network['outlet_indices'])} outlets"
                )

            except Exception as e:
                self.logger.error(f"Error loading topology: {e}")
                self.droute_network = None

    def _find_existing_topology(self) -> Optional[Path]:
        """
        Find existing topology file in standard locations.

        Returns:
            Path to topology file if found, None otherwise
        """
        # Check dRoute settings directory
        droute_topology = self.setup_dir / self.droute_topology_file
        if droute_topology.exists():
            return droute_topology

        # Check for specified path in config
        config_topology = self.config_dict.get('DROUTE_TOPOLOGY_PATH')
        if config_topology and config_topology != 'default':
            path = Path(config_topology)
            if path.exists():
                return path

        return None

    def create_config_file(self):
        """
        Generate dRoute configuration YAML file.

        Creates a configuration file that specifies:
        - Network topology reference
        - Routing method and parameters
        - Input/output paths
        - AD/gradient settings if enabled
        """
        self.logger.info("Creating dRoute configuration file")

        if not hasattr(self, 'droute_network') or self.droute_network is None:
            self.logger.warning("Cannot create config file: topology not loaded")
            return

        # Build configuration
        config = {
            'simulation': {
                'experiment_id': self.experiment_id,
                'domain_name': self.domain_name,
            },
            'routing': {
                'method': self.droute_routing_method,
                'dt': self.droute_routing_dt,
            },
            'paths': {
                'topology_file': str(self.topology_path) if self.topology_path else 'topology.nc',
                'input_dir': str(self._get_input_dir()),
                'output_dir': str(self._get_output_dir()),
            },
            'options': {
                'enable_gradients': self.droute_enable_gradients,
                'ad_backend': self.droute_ad_backend if self.droute_enable_gradients else None,
            },
        }

        # Write network configuration separately
        network_config_path = self.setup_dir / 'droute_network.yaml'
        self.network_adapter.write_droute_config(self.droute_network, network_config_path)

        # Write main configuration
        config_path = self.setup_dir / self.droute_config_file
        try:
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            self.logger.info(f"dRoute config written to {config_path}")
        except ImportError:
            # Fallback: write as simple text format
            self._write_config_fallback(config, config_path)

    def _write_config_fallback(self, config: Dict[str, Any], path: Path):
        """Write config as simple text if YAML not available."""
        with open(path, 'w') as f:
            f.write("# dRoute Configuration\n")
            f.write(f"# Generated by SYMFLUENCE\n\n")

            def write_section(section_name, section_data, indent=0):
                prefix = "  " * indent
                f.write(f"{prefix}{section_name}:\n")
                for key, value in section_data.items():
                    if isinstance(value, dict):
                        write_section(key, value, indent + 1)
                    else:
                        f.write(f"{prefix}  {key}: {value}\n")

            for section, data in config.items():
                write_section(section, data)

        self.logger.info(f"dRoute config written to {path} (text format)")

    def _get_input_dir(self) -> Path:
        """Get input directory for runoff data from source model."""
        from_model = self.droute_from_model.upper()
        if from_model == 'DEFAULT':
            # Try to detect from config
            hydro_model = self.config_dict.get('HYDROLOGICAL_MODEL', 'SUMMA')
            if ',' in str(hydro_model):
                from_model = str(hydro_model).split(',')[0].strip().upper()
            else:
                from_model = str(hydro_model).strip().upper()

        experiment_output = self.config_dict.get(f'EXPERIMENT_OUTPUT_{from_model}')
        if experiment_output and experiment_output != 'default':
            return Path(experiment_output)

        return self.project_dir / f"simulations/{self.experiment_id}" / from_model

    def _get_output_dir(self) -> Path:
        """Get output directory for routed streamflow."""
        droute_output = self.droute_experiment_output
        if droute_output and droute_output != 'default':
            return Path(droute_output)

        return self.project_dir / f"simulations/{self.experiment_id}" / 'dRoute'


__all__ = ['DRoutePreProcessor']
