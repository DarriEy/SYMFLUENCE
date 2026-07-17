"""Shared defaults used by CLI parsing and command execution."""
from __future__ import annotations

import os

DEFAULT_CONFIG_PATH = os.environ.get(
    'SYMFLUENCE_DEFAULT_CONFIG',
    './0_config_files/config_template.yaml',
)
