# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Package data resource access for SYMFLUENCE.

Handles loading base_settings and config_templates from package data
in both development (editable install) and production (site-packages) modes.
"""

import shutil
import sys
from pathlib import Path

# Python 3.9+ importlib.resources
if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    # Fallback for older Python versions (though we require 3.9+)
    from importlib_resources import files


def get_base_settings_dir(model_name: str) -> Path:
    """
    Get path to base settings directory for a specific model.

    Works in both development and installed modes by using importlib.resources
    to locate package data.

    Args:
        model_name: Model name (e.g., 'FUSE', 'SUMMA', 'mizuRoute', 'troute', 'NOAH')

    Returns:
        Path to base settings directory for the model

    Raises:
        FileNotFoundError: If model base settings don't exist

    Examples:
        >>> fuse_dir = get_base_settings_dir('FUSE')
        >>> summa_dir = get_base_settings_dir('SUMMA')
    """
    try:
        # Get the package data directory using importlib.resources
        base_settings_root = files('symfluence.resources.base_settings')
        model_settings = base_settings_root / model_name

        # Convert Traversable to Path
        # In editable mode, this is already a Path
        # In installed mode, this is a Traversable that we convert
        if hasattr(model_settings, '__fspath__'):
            path = Path(model_settings)
        else:
            # For Traversable objects, convert to string then Path
            path = Path(str(model_settings))

        # Verify the directory exists
        if not path.exists():
            raise FileNotFoundError(
                f"Base settings directory for model '{model_name}' not found at: {path}"
            )

        return path

    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
        raise FileNotFoundError(
            f"Base settings for model '{model_name}' not found. "
            f"Expected at: symfluence.resources.base_settings.{model_name}\n"
            f"Available models: CLM, FUSE, MESH, NOAH, SUMMA, mizuRoute, troute"
        ) from e


def get_config_template(template_name: str = 'config_template.yaml') -> Path:
    """
    Get path to a configuration template file.

    Args:
        template_name: Name of template file (default: 'config_template.yaml')
                      Available templates:
                      - config_template.yaml
                      - config_template_comprehensive.yaml
                      - fluxnet_template.yaml
                      - camelsspat_template.yaml
                      - norswe_template.yaml

    Returns:
        Path to the template file

    Raises:
        FileNotFoundError: If template doesn't exist

    Examples:
        >>> template = get_config_template()
        >>> comprehensive = get_config_template('config_template_comprehensive.yaml')
    """
    try:
        # Get the templates directory
        templates_root = files('symfluence.resources.config_templates')
        template_file = templates_root / template_name

        # Convert to Path
        if hasattr(template_file, '__fspath__'):
            path = Path(template_file)
        else:
            path = Path(str(template_file))

        # Verify file exists
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Template '{template_name}' not found at: {path}")

        return path

    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
        # Provide helpful error message with available templates
        available = ['config_template.yaml', 'config_template_comprehensive.yaml',
                    'fluxnet_template.yaml', 'camelsspat_template.yaml', 'norswe_template.yaml']
        raise FileNotFoundError(
            f"Config template '{template_name}' not found.\n"
            f"Available templates: {', '.join(available)}"
        ) from e


def list_config_templates() -> list[Path]:
    """
    List all available configuration templates.

    Returns:
        List of Paths to template files (sorted alphabetically)

    Examples:
        >>> templates = list_config_templates()
        >>> for t in templates:
        ...     print(t.name)
    """
    try:
        templates_root = files('symfluence.resources.config_templates')

        # Handle both installed and editable modes
        if hasattr(templates_root, '__fspath__'):
            # Editable mode - can use pathlib
            root_path = Path(templates_root)
            templates = [f for f in root_path.glob('*.yaml') if f.is_file()]
        else:
            # Installed mode - use Traversable API
            templates = []
            try:
                for item in templates_root.iterdir():
                    if item.name.endswith('.yaml') and not item.name.startswith('__'):
                        # Convert Traversable to Path
                        templates.append(Path(str(item)))
            except AttributeError:
                # Fallback: manually construct known templates
                known_templates = [
                    'config_template.yaml',
                    'config_template_comprehensive.yaml',
                    'fluxnet_template.yaml',
                    'camelsspat_template.yaml',
                    'norswe_template.yaml'
                ]
                for name in known_templates:
                    try:
                        path = get_config_template(name)
                        templates.append(path)
                    except FileNotFoundError:
                        pass

        return sorted(templates, key=lambda p: p.name)

    except (FileNotFoundError, ModuleNotFoundError):
        return []


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


def get_system_deps_registry_path() -> Path:
    """
    Get path to the system dependencies YAML registry.

    Returns:
        Path to system_deps.yml

    Raises:
        FileNotFoundError: If registry file doesn't exist
    """
    try:
        registry_file = files('symfluence.resources') / 'system_deps.yml'

        if hasattr(registry_file, '__fspath__'):
            path = Path(registry_file)
        else:
            path = Path(str(registry_file))

        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"System deps registry not found at: {path}")

        return path

    except (FileNotFoundError, ModuleNotFoundError, AttributeError) as e:
        raise FileNotFoundError(
            "System dependency registry (system_deps.yml) not found in package resources."
        ) from e


def copy_config_template_to_project(destination: Path,
                                    template_name: str = 'config_template.yaml',
                                    output_name: str = None) -> Path:
    """
    Copy a config template from package data to a project directory.

    Args:
        destination: Destination directory path
        template_name: Name of template to copy (default: 'config_template.yaml')
        output_name: Output filename (default: same as template_name)

    Returns:
        Path to the copied config file

    Raises:
        FileNotFoundError: If template doesn't exist
        PermissionError: If destination is not writable

    Examples:
        >>> dest = Path('./my_project')
        >>> config_path = copy_config_template_to_project(dest, output_name='my_config.yaml')
    """
    template_path = get_config_template(template_name)

    # Create destination directory
    destination.mkdir(parents=True, exist_ok=True)

    # Determine output filename
    if output_name is None:
        output_name = template_name

    dest_file = destination / output_name

    # Copy template
    shutil.copy2(template_path, dest_file)

    return dest_file
