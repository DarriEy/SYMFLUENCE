# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Auto-generation utilities for config mapping.

This module provides functions to automatically generate the flat-to-nested
mapping by introspecting Pydantic model aliases. This eliminates the need
for manual maintenance of the 497-line FLAT_TO_NESTED_MAP.

The auto-generated mapping is validated against the manual mapping during
tests to ensure backward compatibility.
"""

import logging
from typing import Any, Dict, Optional, Tuple, Type, Union, get_args, get_origin

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def generate_flat_to_nested_map(
    root_model: Type[BaseModel],
    section_field_names: Optional[Dict[str, str]] = None,
    include_model_overrides: bool = True
) -> Dict[str, Tuple[str, ...]]:
    """
    Auto-generate flat-to-nested mapping from Pydantic model structure.

    Walks all config models recursively and extracts:
    - Field aliases (becomes flat key, e.g., 'DOMAIN_NAME')
    - Model path in hierarchy (becomes nested path, e.g., ('domain', 'name'))

    Args:
        root_model: The root Pydantic model class (e.g., SymfluenceConfig)
        section_field_names: Optional mapping of section names to their field names
                           in the root model. If None, uses default discovery.
        include_model_overrides: If True, merge model-specific transformers from ModelRegistry.
                               This allows models to override base mappings for custom needs.

    Returns:
        Dictionary mapping flat keys to nested paths.

    Example:
        >>> from symfluence.core.config.models import SymfluenceConfig
        >>> mapping = generate_flat_to_nested_map(SymfluenceConfig)
        >>> mapping['DOMAIN_NAME']
        ('domain', 'name')
    """
    mapping: Dict[str, Tuple[str, ...]] = {}

    # Default section mapping for SymfluenceConfig
    if section_field_names is None:
        section_field_names = {
            'system': 'system',
            'domain': 'domain',
            'data': 'data',
            'forcing': 'forcing',
            'model': 'model',
            'optimization': 'optimization',
            'evaluation': 'evaluation',
            'paths': 'paths',
            'fews': 'fews',
        }

    def get_base_type(field_type: Any) -> Optional[Type]:
        """Extract base type from Optional/Union types."""
        origin = get_origin(field_type)

        # Handle Optional[X] which is Union[X, None]
        if origin is Union:
            args = get_args(field_type)
            # Filter out NoneType
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                return non_none_args[0]
        return field_type

    # For duplicate aliases across sections, prefer the first match (lower section
    # order wins).  This matches the manual CANONICAL_FLAT_TO_NESTED_MAP where each
    # key appeared exactly once — e.g., STREAMFLOW_DATA_PROVIDER was placed in
    # 'data', not 'evaluation', even though both Pydantic models define it.
    section_priority = {'system': 9, 'domain': 8, 'data': 7, 'forcing': 6,
                       'model': 5, 'optimization': 4, 'evaluation': 3, 'paths': 2,
                       'fews': 1}

    def walk_model(
        model_class: Type[BaseModel],
        prefix: Tuple[str, ...] = (),
        visited: Optional[set] = None
    ) -> None:
        """Recursively walk model fields and collect aliases."""
        if visited is None:
            visited = set()

        # Prevent infinite recursion on circular references
        model_id = id(model_class)
        if model_id in visited:
            return
        visited.add(model_id)

        if not hasattr(model_class, 'model_fields'):
            return

        for field_name, field_info in model_class.model_fields.items():
            # Collect all uppercase aliases from both alias and validation_alias
            aliases_to_register: list = []

            # Get flat key from alias
            alias = field_info.alias
            if alias and alias.isupper():
                aliases_to_register.append(alias)

            # Also check validation_alias (e.g., AliasChoices)
            val_alias = field_info.validation_alias
            if val_alias is not None and val_alias != alias:
                try:
                    from pydantic import AliasChoices, AliasPath
                    if isinstance(val_alias, AliasChoices):
                        for choice in val_alias.choices:
                            if isinstance(choice, str) and choice.isupper():
                                aliases_to_register.append(choice)
                            elif isinstance(choice, AliasPath):
                                # AliasPath elements — skip complex paths
                                pass
                    elif isinstance(val_alias, str) and val_alias.isupper():
                        aliases_to_register.append(val_alias)
                except ImportError:
                    pass

            # Register all discovered aliases
            for discovered_alias in aliases_to_register:
                current_path = prefix + (field_name,)

                # Handle duplicates by preferring higher priority (last wins)
                if discovered_alias in mapping:
                    existing_section = mapping[discovered_alias][0] if mapping[discovered_alias] else ''
                    new_section = prefix[0] if prefix else ''
                    existing_priority = section_priority.get(existing_section, 0)
                    new_priority = section_priority.get(new_section, 0)

                    # Higher priority number wins (mimics dict behavior where later entry overwrites)
                    if new_priority > existing_priority:
                        mapping[discovered_alias] = current_path
                else:
                    mapping[discovered_alias] = current_path

            # Get the field type
            field_type = field_info.annotation
            base_type = get_base_type(field_type)

            # Recurse into nested Pydantic models
            if (base_type is not None and
                isinstance(base_type, type) and
                issubclass(base_type, BaseModel)):
                walk_model(base_type, prefix + (field_name,), visited.copy())

    # Walk the root model starting from each section
    if hasattr(root_model, 'model_fields'):
        for section_name, field_name in section_field_names.items():
            if field_name in root_model.model_fields:
                field_info = root_model.model_fields[field_name]
                field_type = field_info.annotation
                base_type = get_base_type(field_type)

                if (base_type is not None and
                    isinstance(base_type, type) and
                    issubclass(base_type, BaseModel)):
                    walk_model(base_type, (section_name,))

    # Add model-specific transformer overrides if requested
    if include_model_overrides:
        try:
            from symfluence.models.registries.config_registry import ConfigRegistry

            # Get all registered model names from config adapters
            model_names = list(ConfigRegistry._config_adapters.keys())

            for model_name in model_names:
                try:
                    # Get model-specific transformers via adapter fallback
                    model_transformers = ConfigRegistry.get_config_transformers(model_name)

                    if model_transformers:
                        # Override base mappings with model-specific ones
                        mapping.update(model_transformers)
                        logger.debug(
                            f"Added {len(model_transformers)} transformers for {model_name}"
                        )
                except (ImportError, KeyError, AttributeError, TypeError, ValueError, RuntimeError) as e:
                    # Log but don't fail - model might not have custom transformers
                    logger.debug(f"No custom transformers for {model_name}: {e}")

        except ImportError as e:
            # ConfigRegistry not available - this is fine during early initialization
            logger.debug(f"ConfigRegistry not available for override merging: {e}")

    return mapping


def validate_mapping_equivalence(
    auto_generated: Dict[str, Tuple[str, ...]],
    manual_mapping: Dict[str, Tuple[str, ...]]
) -> Dict[str, Any]:
    """
    Validate that auto-generated mapping is equivalent to manual mapping.

    This function is used during testing to ensure the auto-generation
    produces the same results as the manual mapping before the manual
    mapping is removed.

    Args:
        auto_generated: Mapping generated by generate_flat_to_nested_map
        manual_mapping: The existing FLAT_TO_NESTED_MAP

    Returns:
        Dictionary with validation results:
        - 'equivalent': bool - True if mappings are equivalent
        - 'missing_in_auto': list - Keys in manual but not in auto
        - 'extra_in_auto': list - Keys in auto but not in manual
        - 'mismatched': dict - Keys with different paths
    """
    missing_in_auto = []
    extra_in_auto = []
    mismatched = {}

    # Find keys missing in auto-generated
    for key, path in manual_mapping.items():
        if key not in auto_generated:
            missing_in_auto.append(key)
        elif auto_generated[key] != path:
            mismatched[key] = {
                'manual': path,
                'auto': auto_generated[key]
            }

    # Find extra keys in auto-generated
    for key in auto_generated:
        if key not in manual_mapping:
            extra_in_auto.append(key)

    equivalent = (
        len(missing_in_auto) == 0 and
        len(extra_in_auto) == 0 and
        len(mismatched) == 0
    )

    return {
        'equivalent': equivalent,
        'missing_in_auto': missing_in_auto,
        'extra_in_auto': extra_in_auto,
        'mismatched': mismatched,
        'auto_count': len(auto_generated),
        'manual_count': len(manual_mapping)
    }


def get_all_aliased_fields(root_model: Type[BaseModel]) -> Dict[str, str]:
    """
    Get all fields with aliases from the Pydantic model hierarchy.

    Useful for generating documentation or config templates.

    Args:
        root_model: The root Pydantic model class

    Returns:
        Dictionary mapping aliases to their dotted paths (e.g., 'domain.name')
    """
    mapping = generate_flat_to_nested_map(root_model)
    return {alias: '.'.join(path) for alias, path in mapping.items()}


def check_template_coverage(
    root_model: Type[BaseModel],
    template_keys: set
) -> Dict[str, Any]:
    """
    Check which Pydantic aliases are missing from a config template.

    Args:
        root_model: The root Pydantic model class
        template_keys: Set of keys present in the template

    Returns:
        Dictionary with coverage analysis:
        - 'missing_from_template': Keys in Pydantic but not in template
        - 'extra_in_template': Keys in template but not in Pydantic
        - 'coverage_ratio': Percentage of Pydantic keys covered
    """
    all_aliases = set(generate_flat_to_nested_map(root_model).keys())

    missing_from_template = all_aliases - template_keys
    extra_in_template = template_keys - all_aliases

    coverage_ratio = len(all_aliases & template_keys) / len(all_aliases) if all_aliases else 1.0

    return {
        'missing_from_template': sorted(missing_from_template),
        'extra_in_template': sorted(extra_in_template),
        'coverage_ratio': coverage_ratio,
        'total_aliases': len(all_aliases),
        'template_keys': len(template_keys)
    }
