# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Resource loading utilities for SYMFLUENCE package data."""

from .manager import (
    copy_base_settings_to_project,
    copy_config_template_to_project,
    get_base_settings_dir,
    get_config_template,
    get_system_deps_registry_path,
    list_config_templates,
)

__all__ = [
    'get_base_settings_dir',
    'get_config_template',
    'get_system_deps_registry_path',
    'list_config_templates',
    'copy_base_settings_to_project',
    'copy_config_template_to_project',
]
