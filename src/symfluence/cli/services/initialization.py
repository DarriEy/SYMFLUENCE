# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Initialization service for SYMFLUENCE project setup.

Handles project initialization including config generation,
template management, and project scaffolding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from symfluence.core.config.defaults import ForcingDefaults, ModelDefaults

from ..console import Console
from .base import BaseService


class InitializationService(BaseService):
    """
    Service for project initialization via --init command.

    Handles:
    - Preset listing and selection
    - Config generation from presets and CLI flags
    - Project directory scaffolding
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the InitializationService.

        Args:
            console: Console instance for output. If None, uses global console.
        """
        super().__init__(console=console)

        from ..init_presets import load_presets

        self.presets = load_presets()

        # Use centralized defaults from config.defaults via ModelRegistry
        # ModelDefaults.get_defaults_for_model() retrieves defaults from Pydantic schemas
        # We merge with routing defaults since ROUTING_MODEL is workflow-level, not model-specific
        self.model_defaults = {
            "FUSE": {
                **ModelDefaults.get_defaults_for_model("FUSE"),
                "ROUTING_MODEL": "none",  # FUSE default: no routing
            },
            "SUMMA": {
                **ModelDefaults.get_defaults_for_model("SUMMA"),
                "ROUTING_MODEL": "mizuRoute",  # SUMMA default: mizuRoute
            },
            "GR": {
                **ModelDefaults.get_defaults_for_model("GR"),
                "ROUTING_MODEL": "none",  # GR default: no routing
            },
            "HYPE": {
                **ModelDefaults.get_defaults_for_model("HYPE"),
                "ROUTING_MODEL": "none",  # HYPE default: no routing
            },
        }

        self.forcing_defaults = {
            "ERA5": ForcingDefaults.ERA5,
            "CONUS404": ForcingDefaults.CONUS404,
            "RDRS": ForcingDefaults.RDRS,
            "NLDAS": ForcingDefaults.NLDAS,
        }

    def list_presets(self) -> List[Dict[str, Any]]:
        """Return all available presets with descriptions and print to stdout."""
        self._console.newline()
        self._console.info("Available Presets:")
        self._console.rule()

        preset_list = []
        for i, (name, preset) in enumerate(self.presets.items(), 1):
            desc = preset.get("description", "No description")
            self._console.info(f" {i}. {name:20s} - {desc}")
            preset_list.append({"name": name, "description": desc})
        self._console.rule()
        self._console.info(f"Total: {len(preset_list)} presets")
        return preset_list

    def show_preset(self, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Display detailed information about a preset.

        Args:
            preset_name: Name of preset to show

        Returns:
            Dict: Preset configuration if found, None otherwise
        """
        from ..init_presets import get_preset

        try:
            preset = get_preset(preset_name)
        except ValueError as e:
            self._console.error(str(e))
            return None

        self._console.newline()
        self._console.info(f"Preset: {preset_name}")
        self._console.info(f"Description: {preset['description']}")
        self._console.newline()

        settings = preset["settings"]

        # Show key settings
        self._console.info("Key Settings:")
        if "DOMAIN_NAME" in settings:
            self._console.indent(f"Domain: {settings['DOMAIN_NAME']}")
        if "HYDROLOGICAL_MODEL" in settings:
            model = settings["HYDROLOGICAL_MODEL"]
            spatial_mode = settings.get(
                "FUSE_SPATIAL_MODE", settings.get("SUB_GRID_DISCRETIZATION", "N/A")
            )
            self._console.indent(f"Model: {model} ({spatial_mode})")
        if "FORCING_DATASET" in settings:
            self._console.indent(f"Forcing: {settings['FORCING_DATASET']}")
        if "EXPERIMENT_TIME_START" in settings and "EXPERIMENT_TIME_END" in settings:
            self._console.indent(
                f"Period: {settings['EXPERIMENT_TIME_START']} to {settings['EXPERIMENT_TIME_END']}"
            )
        if "DOMAIN_DEFINITION_METHOD" in settings:
            self._console.indent(f"Domain Definition: {settings['DOMAIN_DEFINITION_METHOD']}")
        if "SUB_GRID_DISCRETIZATION" in settings:
            self._console.indent(f"Discretization: {settings['SUB_GRID_DISCRETIZATION']}")

        # Show calibration info
        if "SETTINGS_FUSE_PARAMS_TO_CALIBRATE" in settings:
            params = settings["SETTINGS_FUSE_PARAMS_TO_CALIBRATE"].split(",")
            self._console.indent(f"Calibration: {len(params)} FUSE parameters")
        elif "PARAMS_TO_CALIBRATE" in settings:
            params = settings["PARAMS_TO_CALIBRATE"].split(",")
            self._console.indent(f"Calibration: {len(params)} SUMMA parameters")

        # Show model decisions if present
        if "fuse_decisions" in preset:
            self._console.newline()
            self._console.info("FUSE Model Decisions:")
            for category, options in preset["fuse_decisions"].items():
                self._console.indent(f"{category}: {options[0]}")

        if "summa_decisions" in preset:
            self._console.newline()
            self._console.info("SUMMA Model Decisions:")
            for category, options in preset["summa_decisions"].items():
                self._console.indent(f"{category}: {options[0]}")

        self._console.newline()
        self._console.info(f"To use: symfluence project init {preset_name}")
        self._console.newline()

        return preset

    def generate_config(
        self,
        preset_name: Optional[str],
        cli_overrides: Dict[str, Any],
        minimal: bool = False,
        comprehensive: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate config dict from preset and CLI flags.

        Args:
            preset_name: Name of preset to use (or None)
            cli_overrides: Dict of CLI flag overrides
            minimal: Create minimal config (required fields only)
            comprehensive: Create comprehensive config (all fields)

        Returns:
            Dict containing complete config

        Raises:
            ValueError: If preset is invalid or required fields missing
        """
        # Step 0: Validate that either preset or required CLI overrides are provided
        if not preset_name:
            # When no preset is specified, user must provide at least domain and forcing
            required_cli_fields = ['domain', 'forcing']
            missing_fields = [f for f in required_cli_fields if f not in cli_overrides or not cli_overrides[f]]
            if missing_fields:
                missing_config_names = []
                if 'domain' in missing_fields:
                    missing_config_names.append('DOMAIN_NAME')
                if 'forcing' in missing_fields:
                    missing_config_names.append('FORCING_DATASET')
                raise ValueError(
                    "Config validation failed:\n" +
                    "\n".join(f"  - Missing required field: {f}" for f in missing_config_names)
                )

        # Step 1: Load base template
        if minimal:
            config = self._create_minimal_config()
        else:
            from symfluence.resources import get_config_template

            template_path = get_config_template("config_template_comprehensive.yaml")
            config = self._load_yaml(template_path)

        # Step 2: Apply preset if specified
        if preset_name:
            from ..init_presets import get_preset

            preset = get_preset(preset_name)

            # Merge preset settings
            config.update(preset["settings"])

            # Apply model-specific decisions
            if "fuse_decisions" in preset:
                config["FUSE_DECISION_OPTIONS"] = preset["fuse_decisions"]
            if "summa_decisions" in preset:
                config["DECISION_OPTIONS"] = preset["summa_decisions"]

        # Step 3: Apply CLI overrides
        cli_config = self._parse_cli_overrides(cli_overrides)
        config.update(cli_config)

        # Step 4: Apply smart defaults based on model/forcing
        self._apply_smart_defaults(config)

        # Step 5: Auto-set paths
        self._auto_set_paths(config)

        # Step 6: Validate config
        self._validate_config(config)

        return config

    def write_config(self, config: Dict[str, Any], output_path: Path) -> Path:
        """
        Write config dict to YAML file.

        Args:
            config: Configuration dictionary
            output_path: Path to write config file

        Returns:
            Path: Path to written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            # Write header comment
            f.write(
                "### ============================================= SYMFLUENCE Configuration File ===========================================\n"
            )
            f.write(
                "# This configuration file was generated using the --init command.\n"
            )
            f.write(
                "# Please review and customize the settings below for your specific use case.\n"
            )
            f.write("#\n")
            f.write("# Configuration sections:\n")
            f.write("# 1. Global settings - paths, experiment IDs, time periods\n")
            f.write("# 2. Geospatial settings - domain definition and discretization\n")
            f.write("# 3. Model agnostic settings - forcing data and preprocessing\n")
            f.write("# 4. Model specific settings - model configuration\n")
            f.write("# 5. Evaluation settings - metrics and observation data\n")
            f.write("# 6. Optimization settings - calibration algorithms\n")
            f.write(
                "###============================================================================================================================\n\n"
            )

            # Write YAML content
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, width=120)

        return output_path

    def create_scaffold(self, config: Dict[str, Any], force: bool = False) -> Path:
        """
        Create project directory structure.

        Args:
            config: Configuration dictionary
            force: Whether to overwrite existing directories

        Returns:
            Path: Path to created domain directory
        """
        # Get domain name
        domain_name = config.get("DOMAIN_NAME", "unnamed_domain")

        # Get data directory
        data_dir = config.get(
            "SYMFLUENCE_DATA_DIR", str(Path.home() / "symfluence_data")
        )
        data_dir = Path(data_dir)

        # Create domain directory
        domain_dir = data_dir / f"domain_{domain_name}"

        if domain_dir.exists() and not force:
            raise ValueError(
                f"Domain directory already exists: {domain_dir}\n"
                f"Use --force to overwrite or choose a different domain name"
            )

        # Create directory structure
        dirs_to_create = [
            domain_dir / "shapefiles" / "pour_point",
            domain_dir / "shapefiles" / "catchment",
            domain_dir / "shapefiles" / "river_network",
            domain_dir / "shapefiles" / "river_basins",
            domain_dir / "data" / "attributes",
            domain_dir / "data" / "forcing" / "raw_data",
            domain_dir / "data" / "forcing" / "processed",
            domain_dir / "settings",
            domain_dir / "simulations" / config.get("EXPERIMENT_ID", "run_1"),
            domain_dir / "data" / "observations" / "streamflow",
            domain_dir / "data" / "model_ready" / "forcings",
            domain_dir / "data" / "model_ready" / "observations",
            domain_dir / "data" / "model_ready" / "attributes",
        ]

        # Add model-specific directories
        model = config.get("HYDROLOGICAL_MODEL")
        if model == "FUSE":
            dirs_to_create.append(domain_dir / "settings" / "FUSE")
        elif model == "SUMMA":
            dirs_to_create.append(domain_dir / "settings" / "SUMMA")
            if config.get("ROUTING_MODEL") == "mizuRoute":
                dirs_to_create.append(domain_dir / "settings" / "mizuRoute")
        elif model == "HYPE":
            dirs_to_create.append(domain_dir / "settings" / "HYPE")

        # Create all directories
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)

        return domain_dir

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file into dictionary."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _create_minimal_config(self) -> Dict[str, Any]:
        """Create minimal config with only required fields."""
        return {
            "SYMFLUENCE_DATA_DIR": "default",
            "SYMFLUENCE_CODE_DIR": "default",
            "DOMAIN_NAME": "unnamed_domain",
            "EXPERIMENT_ID": "run_1",
            "EXPERIMENT_TIME_START": "2010-01-01 00:00",
            "EXPERIMENT_TIME_END": "2020-12-31 23:00",
            "DOMAIN_DEFINITION_METHOD": "lumped",
            "SUB_GRID_DISCRETIZATION": "lumped",
            "HYDROLOGICAL_MODEL": "FUSE",
            "FORCING_DATASET": "ERA5",
            "NUM_PROCESSES": 1,
        }

    def _parse_cli_overrides(self, cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse CLI overrides into config format.

        Args:
            cli_overrides: Dict from CLI flags

        Returns:
            Dict in config format
        """
        config = {}

        # Map CLI flags to config keys
        if cli_overrides.get("domain"):
            config["DOMAIN_NAME"] = cli_overrides["domain"]

        if cli_overrides.get("model"):
            config["HYDROLOGICAL_MODEL"] = cli_overrides["model"]

        if cli_overrides.get("start_date"):
            # Format: YYYY-MM-DD -> "YYYY-MM-DD 00:00"
            config["EXPERIMENT_TIME_START"] = f"{cli_overrides['start_date']} 00:00"

        if cli_overrides.get("end_date"):
            # Format: YYYY-MM-DD -> "YYYY-MM-DD 23:00"
            config["EXPERIMENT_TIME_END"] = f"{cli_overrides['end_date']} 23:00"

        if cli_overrides.get("forcing"):
            config["FORCING_DATASET"] = cli_overrides["forcing"]

        if cli_overrides.get("discretization"):
            config["SUB_GRID_DISCRETIZATION"] = cli_overrides["discretization"]

        if cli_overrides.get("definition_method"):
            config["DOMAIN_DEFINITION_METHOD"] = cli_overrides["definition_method"]

        return config

    def _apply_smart_defaults(self, config: Dict[str, Any]) -> None:
        """
        Apply smart defaults based on model and forcing.

        Args:
            config: Config dict to modify in-place
        """
        # Apply model-specific defaults
        model = config.get("HYDROLOGICAL_MODEL")
        if model and model in self.model_defaults:
            for key, value in self.model_defaults[model].items():
                if key not in config:
                    config[key] = value

        # Apply forcing-specific defaults
        forcing = config.get("FORCING_DATASET")
        if forcing and forcing in self.forcing_defaults:
            for key, value in self.forcing_defaults[forcing].items():
                if key not in config:
                    config[key] = value

        # Set other common defaults
        if "NUM_PROCESSES" not in config:
            config["NUM_PROCESSES"] = 1

        if "FORCE_RUN_ALL_STEPS" not in config:
            config["FORCE_RUN_ALL_STEPS"] = False

        if "DATA_ACCESS" not in config:
            config["DATA_ACCESS"] = "cloud"

    def _auto_set_paths(self, config: Dict[str, Any]) -> None:
        """
        Auto-set default paths.

        Args:
            config: Config dict to modify in-place
        """
        # Set data directory if not specified or set to default
        if (
            not config.get("SYMFLUENCE_DATA_DIR")
            or config.get("SYMFLUENCE_DATA_DIR") == "default"
        ):
            config["SYMFLUENCE_DATA_DIR"] = str(Path.home() / "symfluence_data")

        # Set code directory if not specified or set to default
        if (
            not config.get("SYMFLUENCE_CODE_DIR")
            or config.get("SYMFLUENCE_CODE_DIR") == "default"
        ):
            # Try to detect from current installation
            config["SYMFLUENCE_CODE_DIR"] = self._detect_code_dir()

    def _detect_code_dir(self) -> str:
        """Detect SYMFLUENCE code directory."""
        from symfluence.core.config.factories import _resolve_default_code_dir
        return _resolve_default_code_dir()

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration.

        Args:
            config: Config to validate

        Raises:
            ValueError: If validation fails
        """
        errors = []

        # Check required fields
        required_fields = [
            "SYMFLUENCE_DATA_DIR",
            "SYMFLUENCE_CODE_DIR",
            "DOMAIN_NAME",
            "EXPERIMENT_ID",
            "EXPERIMENT_TIME_START",
            "EXPERIMENT_TIME_END",
            "DOMAIN_DEFINITION_METHOD",
            "HYDROLOGICAL_MODEL",
            "FORCING_DATASET",
        ]

        for field in required_fields:
            if field not in config or not config[field]:
                errors.append(f"Missing required field: {field}")

        # Validate dates
        if "EXPERIMENT_TIME_START" in config and "EXPERIMENT_TIME_END" in config:
            start = config["EXPERIMENT_TIME_START"]
            end = config["EXPERIMENT_TIME_END"]
            # Basic check that end comes after start (string comparison works for ISO format)
            # Only validate if both are non-None
            if start is not None and end is not None and start >= end:
                errors.append(f"End date must be after start date: {start} >= {end}")

        # Validate model
        valid_models = [
            "SUMMA",
            "FUSE",
            "GR",
            "HYPE",
            "LSTM",
            "MESH",
            "RHESSys",
            "NGEN",
            "TEST",
        ]
        model = config.get("HYDROLOGICAL_MODEL")
        if model and model not in valid_models:
            errors.append(f"Invalid model: {model}. Must be one of {valid_models}")

        if errors:
            raise ValueError(
                "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )


# Backward compatibility alias
InitializationManager = InitializationService
