"""
Helper functions for SYMFLUENCE tests.
"""

from pathlib import Path
import yaml


def load_config_template(symfluence_code_dir):
    """Load the configuration template from the config files directory."""
    template_path = Path(symfluence_code_dir) / "0_config_files" / "config_template.yaml"

    if not template_path.exists():
        raise FileNotFoundError(f"Config template not found at {template_path}")

    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def write_config(config, output_path):
    """Write a configuration dictionary to a YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
