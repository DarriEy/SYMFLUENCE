# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Base-settings resolution for the model-adapter contract tier.

A model's *base settings* are the template input files (decision files,
parameter tables, control files) a preprocessor copies into a project before a
run. Where they live is a **contract** question, not a packaging one:

* a model package declares the anchor package whose ``base_settings/`` data
  directory ships its templates (``R.base_settings.add("SUMMA",
  "symfluence.models.summa")``), and
* the framework-bundled ``symfluence.resources.base_settings/`` directory is
  the fallback for models that never registered an anchor (plus framework-owned
  fixtures such as ``TEST``).

This resolution used to live in :mod:`symfluence.resources`, which made the
bundled-asset package import ``symfluence.core.registries`` — a cycle, since
core's preprocessor base imported the resolver straight back. Owning it here
breaks that cycle and gives model packages a single core-level entry point, so
the models distribution depends on ``symfluence.core`` and nothing else in the
framework. ``symfluence.resources`` keeps deprecated shims for both names.
"""
from __future__ import annotations

import shutil
from importlib.resources import files
from pathlib import Path

__all__ = ['copy_base_settings_to_project', 'get_base_settings_dir']


def get_base_settings_dir(model_name: str) -> Path:
    """
    Get path to base settings directory for a specific model.

    Resolution order is registry-first, bundled-fallback second. Works in both
    development and installed modes by using importlib.resources to locate
    package data.

    Args:
        model_name: Model name (e.g. 'FUSE', 'SUMMA', 'mizuRoute', 'troute',
            'NOAH'). Registry lookup is case/separator-insensitive; the bundled
            fallback is a literal directory name.

    Returns:
        Path to base settings directory for the model

    Raises:
        FileNotFoundError: If model base settings don't exist

    Examples:
        >>> fuse_dir = get_base_settings_dir('FUSE')
        >>> summa_dir = get_base_settings_dir('SUMMA')
    """
    # Registry first: each model package registers the anchor package whose
    # ``base_settings/`` data directory ships its template settings (in-tree
    # packages and external plugins use the same path).
    try:
        from symfluence.core.registries import R

        anchor = R.base_settings.get(model_name)
    except Exception:  # noqa: BLE001 — registry unavailable in stripped contexts
        anchor = None
    if anchor is not None:
        try:
            model_settings = files(anchor) / 'base_settings'
            path = Path(model_settings) if hasattr(model_settings, '__fspath__') \
                else Path(str(model_settings))
            if path.exists():
                return path
        except (ModuleNotFoundError, AttributeError):
            pass

    try:
        # Central fallback (framework-owned fixtures such as TEST, and any
        # model whose package has not registered a base_settings anchor). The
        # import is deferred so core stays importable when the bundled-asset
        # package is not installed -- a missing ``symfluence.resources`` then
        # reads as "no bundled settings", the same as a missing directory.
        from symfluence.resources import get_bundled_base_settings_dir

        return get_bundled_base_settings_dir(model_name)

    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
        raise FileNotFoundError(
            f"Base settings for model '{model_name}' not found. It is served "
            f"from the model package registered in R.base_settings (is the "
            f"model package installed and registered?), with "
            f"symfluence.resources.base_settings as the legacy fallback."
        ) from e


def copy_base_settings_to_project(model_name: str, destination: Path) -> None:
    """
    Copy base settings files from package data to a project directory.

    This is used during project initialization to copy template files
    from the package to the user's project workspace.

    Args:
        model_name: Model name (e.g., 'FUSE', 'SUMMA')
        destination: Destination directory path

    Raises:
        FileNotFoundError: If model base settings don't exist
        PermissionError: If destination is not writable

    Examples:
        >>> from pathlib import Path
        >>> dest = Path('./my_project/settings/FUSE')
        >>> copy_base_settings_to_project('FUSE', dest)
    """
    source_dir = get_base_settings_dir(model_name)

    # Create destination directory
    destination.mkdir(parents=True, exist_ok=True)

    # Copy all files from source to destination
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Base settings directory not found: {source_dir}")

    # Recursively copy all files and subdirectories
    for item in source_dir.rglob('*'):
        if item.is_file():
            # Compute relative path from source_dir
            rel_path = item.relative_to(source_dir)
            dest_file = destination / rel_path

            # Create parent directories if needed
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            shutil.copy2(item, dest_file)
