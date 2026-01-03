"""
Helper functions for SYMFLUENCE tests.
"""

from pathlib import Path
import yaml


def load_config_template(symfluence_code_dir=None):
    """Load the configuration template from package data.

    Args:
        symfluence_code_dir: Deprecated, kept for backward compatibility
    """
    from symfluence.utils.resources import get_config_template

    template_path = get_config_template()
    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def write_config(config, output_path):
    """Write a configuration dictionary to a YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
