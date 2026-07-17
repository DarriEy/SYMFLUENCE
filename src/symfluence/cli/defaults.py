"""Shared defaults used by CLI parsing and command execution."""
from __future__ import annotations

import os

# Reference templates ship with the package under symfluence/resources/config_templates/.
DEFAULT_CONFIG_PATH = os.environ.get(
    'SYMFLUENCE_DEFAULT_CONFIG',
    './config.yaml',
)
