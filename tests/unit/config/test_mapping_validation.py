# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Cover the flat<->nested mapping consistency utility.

``validate_mapping_against_pydantic()`` compares the hand-maintained
FLAT_TO_NESTED_MAP against the mapping auto-derived from Pydantic aliases. It is
a development/CI guard; this test exercises it and characterizes its current
output so a *regression* (e.g. the report shape breaking, or the two mappings
diverging much further) surfaces.
"""

from __future__ import annotations

import pytest

from symfluence.core.config.mapping_validation import validate_mapping_against_pydantic

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def test_validate_mapping_returns_wellformed_report():
    report = validate_mapping_against_pydantic()

    expected_keys = {
        "equivalent",
        "missing_in_manual",
        "extra_in_manual",
        "mismatched",
        "manual_count",
        "pydantic_count",
    }
    assert expected_keys <= set(report)

    assert isinstance(report["equivalent"], bool)
    assert isinstance(report["missing_in_manual"], (list, set, tuple))
    assert isinstance(report["extra_in_manual"], (list, set, tuple))
    assert isinstance(report["mismatched"], dict)
    # Both mappings are substantial — a collapse to near-zero means the
    # introspection or the manual map broke.
    assert report["manual_count"] > 500
    assert report["pydantic_count"] > 500
