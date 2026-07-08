# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Back-compat shim: spatial mode expressed in the discretization key.

Older flat configs (pre-DOMAIN_DEFINITION_METHOD) set the spatial mode via
``SUB_GRID_DISCRETIZATION``/``DOMAIN_DISCRETIZATION: semi_distributed`` (etc.).
``SymfluenceConfig`` migrates that to ``DOMAIN_DEFINITION_METHOD`` with a
DeprecationWarning, without disturbing configs that use the keys correctly.
"""
from __future__ import annotations

import warnings

import pytest

from symfluence.core.config.models import SymfluenceConfig

pytestmark = [pytest.mark.unit, pytest.mark.quick]

def _base(tmp_path):
    return {
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "t",
        "EXPERIMENT_ID": "e",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-01-02 00:00",
        "HYDROLOGICAL_MODEL": "SUMMA",
        "FORCING_DATASET": "ERA5",
    }


def _build(tmp_path, **overrides):
    cfg = _base(tmp_path)
    cfg.update(overrides)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        config = SymfluenceConfig(**cfg)
    migrated = any(
        issubclass(w.category, DeprecationWarning) and "legacy spatial-mode" in str(w.message)
        for w in caught
    )
    return config, migrated


@pytest.mark.parametrize("key", ["SUB_GRID_DISCRETIZATION", "DOMAIN_DISCRETIZATION"])
@pytest.mark.parametrize(
    "token,expected_method",
    [
        ("semi_distributed", "semidistributed"),
        ("semi-distributed", "semidistributed"),
        ("semidistributed", "semidistributed"),
        ("distributed", "distributed"),
    ],
)
def test_legacy_spatial_token_migrates(tmp_path, key, token, expected_method):
    """A spatial token in the discretization key, with no DOMAIN_DEFINITION_METHOD,
    migrates to the canonical method and resets the discretization to GRUs."""
    config, migrated = _build(tmp_path, **{key: token})
    assert config.domain.definition_method == expected_method
    assert config.domain.discretization == "GRUs"
    assert migrated


def test_explicit_method_is_never_overridden(tmp_path):
    """An explicit DOMAIN_DEFINITION_METHOD wins even if the discretization key
    happens to hold a spatial-mode token — no migration, no warning."""
    config, migrated = _build(
        tmp_path, DOMAIN_DEFINITION_METHOD="lumped", SUB_GRID_DISCRETIZATION="semi_distributed"
    )
    assert config.domain.definition_method == "lumped"
    assert config.domain.discretization == "semi_distributed"
    assert not migrated


@pytest.mark.parametrize(
    "method,disc",
    [
        ("semidistributed", "elevation"),
        ("lumped", "lumped"),  # 'lumped' is a valid discretization value
        ("distributed", "GRUs"),
        ("lumped", "landclass"),
    ],
)
def test_valid_configs_untouched(tmp_path, method, disc):
    """Correct modern configs (and valid 'lumped'/'GRUs'/'elevation' discretization
    values) are passed through unchanged with no deprecation warning."""
    config, migrated = _build(tmp_path, DOMAIN_DEFINITION_METHOD=method, SUB_GRID_DISCRETIZATION=disc)
    assert config.domain.definition_method == method
    assert config.domain.discretization == disc
    assert not migrated


def test_migration_preserves_real_subgrid_in_other_key(tmp_path):
    """A spatial token in SUB_GRID_DISCRETIZATION alongside a genuine sub-grid
    method in the legacy DOMAIN_DISCRETIZATION alias must migrate the mode but
    keep the real discretization rather than discarding it."""
    config, migrated = _build(
        tmp_path,
        SUB_GRID_DISCRETIZATION="semi_distributed",
        DOMAIN_DISCRETIZATION="elevation",
    )
    assert config.domain.definition_method == "semidistributed"
    assert config.domain.discretization == "elevation"  # not silently dropped
    assert migrated
