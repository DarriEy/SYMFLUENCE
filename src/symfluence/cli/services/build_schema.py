# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Build instruction schema and validation for SYMFLUENCE external tools.

Defines the standard structure for tool build instructions and provides
validation utilities. This module is lightweight and has no heavy dependencies.
"""

from typing import Any, Dict, List, Literal, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class VerifyInstallSchema(TypedDict, total=False):
    """Schema for installation verification."""
    file_paths: List[str]
    check_type: Literal['exists', 'exists_any', 'exists_all']


class BuildInstructionSchema(TypedDict, total=False):
    """
    Standard schema for external tool build instructions.

    Required fields:
        description: Human-readable description of the tool
        order: Installation order (lower = earlier)

    Optional fields (for tools requiring build):
        repository: Git repository URL
        branch: Git branch to checkout
        install_dir: Directory name for installation
        build_commands: List of shell command strings
        requires: List of tool dependencies (other tools)
        dependencies: List of system dependencies (binaries in PATH)

    Validation fields:
        verify_install: Verification criteria
        test_command: Command argument for testing (e.g., '--version')

    Config integration fields:
        config_path_key: Key in config file for installation path
        config_exe_key: Key in config file for executable name
        default_path_suffix: Default relative path for installation
        default_exe: Default executable/library filename
    """
    # Required
    description: str
    order: int

    # Build configuration
    repository: Optional[str]
    branch: Optional[str]
    install_dir: str
    build_commands: List[str]
    requires: List[str]
    dependencies: List[str]

    # Verification
    verify_install: VerifyInstallSchema
    test_command: Optional[str]

    # Config integration
    config_path_key: str
    config_exe_key: str
    default_path_suffix: str
    default_exe: str


def validate_build_instructions(instructions: Dict[str, Any]) -> List[str]:
    """
    Validate build instructions against the schema.

    Args:
        instructions: Build instruction dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Required fields
    if 'description' not in instructions:
        errors.append("Missing required field: 'description'")
    if 'order' not in instructions:
        errors.append("Missing required field: 'order'")

    # Type validations
    if 'build_commands' in instructions:
        if not isinstance(instructions['build_commands'], list):
            errors.append("'build_commands' must be a list")
        else:
            for i, cmd in enumerate(instructions['build_commands']):
                if not isinstance(cmd, str):
                    errors.append(f"'build_commands[{i}]' must be a string")

    if 'verify_install' in instructions:
        verify = instructions['verify_install']
        if not isinstance(verify, dict):
            errors.append("'verify_install' must be a dictionary")
        elif 'file_paths' not in verify:
            errors.append("'verify_install' must contain 'file_paths'")
        elif not isinstance(verify['file_paths'], list):
            errors.append("'verify_install.file_paths' must be a list")
        else:
            check_type = verify.get('check_type', 'exists')
            if check_type not in ('exists', 'exists_any', 'exists_all'):
                errors.append(
                    f"'verify_install.check_type' must be one of: "
                    f"'exists', 'exists_any', 'exists_all' (got '{check_type}')"
                )

    if 'requires' in instructions:
        if not isinstance(instructions['requires'], list):
            errors.append("'requires' must be a list")

    if 'dependencies' in instructions:
        if not isinstance(instructions['dependencies'], list):
            errors.append("'dependencies' must be a list")

    if 'order' in instructions:
        if not isinstance(instructions['order'], int):
            errors.append("'order' must be an integer")
        elif instructions['order'] < 1:
            errors.append("'order' must be >= 1")

    return errors


def validate_all_instructions(
    all_instructions: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    Validate all build instructions.

    Args:
        all_instructions: Dictionary of tool_name -> instructions

    Returns:
        Dictionary of tool_name -> list of errors (empty lists for valid tools)
    """
    results = {}
    for tool_name, instructions in all_instructions.items():
        errors = validate_build_instructions(instructions)
        if errors:
            results[tool_name] = errors
    return results
