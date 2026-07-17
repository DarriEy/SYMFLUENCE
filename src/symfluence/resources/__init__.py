# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Resource loading utilities for SYMFLUENCE package data."""
from __future__ import annotations

from .manager import (
    agent_cache_root,
    copy_base_settings_to_project,
    copy_config_template_to_project,
    get_agents_dir,
    get_base_settings_dir,
    get_config_template,
    get_skills_dir,
    get_system_deps_registry_path,
    list_config_templates,
    prepare_agent_context,
)

__all__ = [
    'agent_cache_root',
    'get_agents_dir',
    'get_base_settings_dir',
    'get_config_template',
    'get_skills_dir',
    'get_system_deps_registry_path',
    'list_config_templates',
    'prepare_agent_context',
    'copy_base_settings_to_project',
    'copy_config_template_to_project',
]
