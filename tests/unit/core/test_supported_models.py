# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""SupportedModels.is_valid is registry-derived, with no hardcoded model list.

RTI review item 18: the SupportedModels.ALL / WITH_* tuples were hardcoded and
had drifted from reality. Validity is now computed from the component registry —
registration is the whitelist.
"""

from __future__ import annotations

import symfluence  # noqa: F401 - import triggers bootstrap entry-point registration
from symfluence.core.constants import SupportedModels
from symfluence.core.registries import R


def test_registered_model_is_valid_case_insensitive():
    assert SupportedModels.is_valid("summa")
    assert SupportedModels.is_valid("SUMMA")


def test_unregistered_model_is_invalid():
    assert not SupportedModels.is_valid("definitely_not_a_model")


def test_is_valid_tracks_the_registry():
    """Every registered runner is valid — derived, not hardcoded."""
    for name in R.runners.keys():
        assert SupportedModels.is_valid(name), f"{name} has a runner but is_valid is False"


def test_schema_only_model_is_valid():
    """A model registered via config schema (no runner), e.g. GNN, is still valid."""
    assert SupportedModels.is_valid("gnn")


def test_hardcoded_model_lists_are_removed():
    """The drifted ALL / WITH_* tuples are gone (item 18)."""
    for attr in ("ALL", "WITH_FORCING_ADAPTER", "WITH_PLOTTERS", "WITH_PRESETS"):
        assert not hasattr(SupportedModels, attr), f"SupportedModels.{attr} should be removed"
