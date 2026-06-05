# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared fixtures for config unit tests (item 17 coverage)."""

from __future__ import annotations

import pytest

from symfluence.core.config.flattening import flatten_nested_config
from symfluence.core.config.models import SymfluenceConfig


def build_minimal_config(**overrides) -> SymfluenceConfig:
    """A valid minimal SymfluenceConfig via the documented from_minimal factory."""
    params = dict(
        domain_name="test_basin",
        model="SUMMA",
        EXPERIMENT_TIME_START="2020-01-01 00:00",
        EXPERIMENT_TIME_END="2020-12-31 23:00",
    )
    params.update(overrides)
    return SymfluenceConfig.from_minimal(**params)


@pytest.fixture
def minimal_config() -> SymfluenceConfig:
    """A valid minimal typed config."""
    return build_minimal_config()


@pytest.fixture
def minimal_flat_config(minimal_config) -> dict:
    """The minimal config as a flat dict (round-trippable into SymfluenceConfig)."""
    return flatten_nested_config(minimal_config)
