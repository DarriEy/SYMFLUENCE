from pathlib import Path

import yaml


def load_config_template(symfluence_code_dir: Path):
    template_path = symfluence_code_dir / "0_config_files" / "config_template.yaml"
    with open(template_path, "r") as f:
        return yaml.safe_load(f)


def write_config(config, cfg_path: Path):
    with open(cfg_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
