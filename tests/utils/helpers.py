"""
Helper functions for SYMFLUENCE tests.

Configuration management utilities for loading and writing test configs.
"""

from pathlib import Path
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
    """
    template_path = Path(symfluence_code_dir) / "0_config_files" / "config_template.yaml"

    if not template_path.exists():
        raise FileNotFoundError(f"Config template not found at {template_path}")

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
