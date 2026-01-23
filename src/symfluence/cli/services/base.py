"""
Base service class for SYMFLUENCE CLI services.

Provides shared functionality for all CLI services including:
- Console injection for output
- Configuration loading
- Data directory resolution
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..console import Console, get_console


class BaseService:
    """
    Base class for CLI services with shared functionality.

    All CLI services inherit from this class to get:
    - Console injection for formatted output
    - Configuration loading from SYMFLUENCE instance or template
    - Data directory resolution

    Args:
        console: Optional Console instance for output. Uses global console if not provided.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize the base service.

        Args:
            console: Console instance for output. If None, uses global console.
        """
        self._console = console or get_console()

    def _load_config(self, symfluence_instance=None) -> Dict[str, Any]:
        """
        Load configuration from SYMFLUENCE instance or fall back to template.

        Args:
            symfluence_instance: Optional SYMFLUENCE instance with config attribute.

        Returns:
            Configuration dictionary.
        """
        if symfluence_instance and hasattr(symfluence_instance, "config"):
            return symfluence_instance.config
        if symfluence_instance and hasattr(symfluence_instance, "workflow_orchestrator"):
            return symfluence_instance.workflow_orchestrator.config

        try:
            from symfluence.resources import get_config_template

            config_path = get_config_template()
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return self._ensure_valid_config_paths(config, config_path)
        except (ImportError, FileNotFoundError, yaml.YAMLError) as e:
            self._console.debug(f"Could not load config: {e}")
            return {}

    def _get_data_dir(self, config: Dict[str, Any]) -> Path:
        """
        Get SYMFLUENCE data directory from environment or config.

        Args:
            config: Configuration dictionary.

        Returns:
            Path to data directory.
        """
        data_dir = os.getenv("SYMFLUENCE_DATA") or config.get("SYMFLUENCE_DATA_DIR", ".")
        return Path(data_dir)

    def _ensure_valid_config_paths(
        self, config: Dict[str, Any], config_path: Path
    ) -> Dict[str, Any]:
        """
        Ensure SYMFLUENCE_DATA_DIR and SYMFLUENCE_CODE_DIR paths exist and are valid.

        Args:
            config: Configuration dictionary to validate.
            config_path: Path to the configuration file.

        Returns:
            Updated configuration dictionary with valid paths.
        """
        data_dir = config.get("SYMFLUENCE_DATA_DIR")
        code_dir = config.get("SYMFLUENCE_CODE_DIR")

        data_dir_valid = False
        code_dir_valid = False

        if data_dir:
            try:
                data_path = Path(data_dir)
                if data_path.exists():
                    test_file = data_path / ".symfluence_test"
                    try:
                        test_file.touch()
                        test_file.unlink()
                        data_dir_valid = True
                    except (PermissionError, OSError):
                        pass
                else:
                    try:
                        data_path.mkdir(parents=True, exist_ok=True)
                        data_dir_valid = True
                    except (PermissionError, OSError):
                        pass
            except (ValueError, OSError):
                pass

        if code_dir:
            try:
                code_path = Path(code_dir)
                if code_path.exists() and os.access(code_path, os.R_OK):
                    code_dir_valid = True
            except (ValueError, OSError):
                pass

        if not data_dir_valid or not code_dir_valid:
            self._console.warning(
                "Detected invalid or inaccessible paths in config template:"
            )

            if not code_dir_valid:
                new_code_dir = Path.cwd().resolve()
                config["SYMFLUENCE_CODE_DIR"] = str(new_code_dir)
                self._console.success(f"SYMFLUENCE_CODE_DIR set to: {new_code_dir}")

            if not data_dir_valid:
                new_data_dir = (Path.cwd().parent / "SYMFLUENCE_data").resolve()
                config["SYMFLUENCE_DATA_DIR"] = str(new_data_dir)
                try:
                    new_data_dir.mkdir(parents=True, exist_ok=True)
                    self._console.success(f"SYMFLUENCE_DATA_DIR set to: {new_data_dir}")
                except OSError:
                    pass

            try:
                backup_path = config_path.with_name(
                    f"{config_path.stem}_backup{config_path.suffix}"
                )
                if config_path.exists():
                    shutil.copy2(config_path, backup_path)
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            except OSError:
                pass

        return config
