# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Mapping validation utility for development and testing.

Compares the manual FLAT_TO_NESTED_MAP against the auto-generated mapping
derived from Pydantic model aliases.  Useful in CI or during model changes
to ensure the two stay in sync.
"""

from typing import Any, Dict

from symfluence.core.config.canonical_mappings import FLAT_TO_NESTED_MAP


def validate_mapping_against_pydantic() -> Dict[str, Any]:
    """Validate that FLAT_TO_NESTED_MAP matches auto-generated mapping from Pydantic models.

    This function can be used in tests to ensure the manual mapping stays in sync
    with Pydantic model aliases. In the future, this manual mapping can be replaced
    entirely with auto-generation.

    Returns:
        Dictionary with validation results:
        - 'equivalent': bool - True if mappings match
        - 'missing_in_manual': list - Keys in Pydantic but not in manual
        - 'extra_in_manual': list - Keys in manual but not in Pydantic
        - 'mismatched': dict - Keys with different paths
    """
    from symfluence.core.config.introspection import (
        generate_flat_to_nested_map,
        validate_mapping_equivalence,
    )
    from symfluence.core.config.models import SymfluenceConfig

    auto_mapping = generate_flat_to_nested_map(SymfluenceConfig)
    result = validate_mapping_equivalence(auto_mapping, FLAT_TO_NESTED_MAP)

    # Rename keys for clarity from manual mapping perspective
    return {
        'equivalent': result['equivalent'],
        'missing_in_manual': result['extra_in_auto'],   # In Pydantic but not manual
        'extra_in_manual': result['missing_in_auto'],    # In manual but not Pydantic
        'mismatched': result['mismatched'],
        'manual_count': result['manual_count'],
        'pydantic_count': result['auto_count'],
    }
