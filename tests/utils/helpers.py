"""
Helper functions for SYMFLUENCE tests.

Configuration management utilities for loading and writing test configs.
"""

from pathlib import Path
import os
import yaml


def load_config_template(symfluence_code_dir):
    """
    Load the configuration template from the config files directory.

    Args:
        symfluence_code_dir: Path to SYMFLUENCE code directory

    Returns:
        dict: Configuration dictionary loaded from template

    Raises:
        FileNotFoundError: If config template doesn't exist

    Note:
        symfluence_code_dir parameter is deprecated, templates are now loaded from package data
    """
    from symfluence.resources import get_config_template

    template_path = get_config_template()
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def write_config(config, output_path):
    """
    Write a configuration dictionary to a YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path where config should be written

    Creates parent directories if they don't exist.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def has_cds_credentials():
    """Return True when CDS credentials are configured via env or ~/.cdsapirc."""
    env_url = os.environ.get("CDSAPI_URL")
    env_key = os.environ.get("CDSAPI_KEY")
    if env_url and env_key:
        return True

    cdsapirc = Path.home() / ".cdsapirc"
    if not cdsapirc.exists():
        return False

    try:
        content = cdsapirc.read_text(encoding="utf-8")
    except OSError:
        return False

    has_url = False
    has_key = False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("url:"):
            has_url = bool(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("key:"):
            has_key = bool(stripped.split(":", 1)[1].strip())

    return has_url and has_key
