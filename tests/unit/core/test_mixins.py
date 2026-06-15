# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for core mixin composition (review item 17).

Covers ConfigMixin._get_config_value resolution order, ProjectContextMixin path
derivation, LoggingMixin/TimingMixin, and the ConfigurableMixin composition — the
shared base behind most managers, previously untested except ShapefileAccessMixin.
"""

from __future__ import annotations

import logging
from pathlib import Path

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.mixins.config import ConfigMixin
from symfluence.core.mixins.configurable import ConfigurableMixin
from symfluence.core.mixins.project import resolve_data_subdir, resolve_forcing_basin_path


class _Configurable(ConfigurableMixin):
    """Minimal concrete composition of all core mixins."""


def _minimal() -> SymfluenceConfig:
    return SymfluenceConfig.from_minimal(
        domain_name="test_basin", model="SUMMA",
        EXPERIMENT_TIME_START="2020-01-01 00:00", EXPERIMENT_TIME_END="2020-12-31 23:00",
    )


# ---- ConfigMixin._get_config_value resolution order ---------------------


def test_typed_accessor_returns_value():
    obj = ConfigMixin()
    obj.config = _minimal()
    assert obj._get_config_value(lambda: obj.config.domain.name) == "test_basin"


def test_override_dict_takes_precedence():
    obj = ConfigMixin()
    obj.config = _minimal()
    obj._config_dict_override = {"DOMAIN_NAME": "overridden"}
    # Even though the typed accessor would return 'test_basin', the override wins.
    assert obj._get_config_value(
        lambda: obj.config.domain.name, dict_key="DOMAIN_NAME"
    ) == "overridden"


def test_dict_fallback_when_typed_accessor_fails():
    obj = ConfigMixin()
    obj.config = {"MY_KEY": "from_dict"}  # plain-dict config
    value = obj._get_config_value(
        lambda: obj.config.some.typed.path,  # raises AttributeError on a dict
        dict_key="MY_KEY",
    )
    assert value == "from_dict"


def test_default_when_nothing_resolves():
    obj = ConfigMixin()
    obj.config = _minimal()
    assert obj._get_config_value(lambda: None, default="fallback") == "fallback"


# ---- ProjectContextMixin path derivation --------------------------------


def test_project_dir_derivation(tmp_path):
    obj = _Configurable()
    obj.data_dir = tmp_path / "symfluence-data"
    obj.domain_name = "bow"
    assert obj.project_dir == tmp_path / "symfluence-data" / "domain_bow"
    assert obj.project_forcing_dir == obj.project_dir / "data" / "forcing"


def test_resolve_data_subdir_prefers_new_then_legacy(tmp_path):
    project = tmp_path / "domain_x"
    # Neither exists -> returns the new-style path.
    assert resolve_data_subdir(project, "shapefiles") == project / "data" / "shapefiles"
    # Legacy layout present -> returns it.
    (project / "shapefiles").mkdir(parents=True)
    assert resolve_data_subdir(project, "shapefiles") == project / "shapefiles"
    # New layout present -> preferred over legacy.
    (project / "data" / "shapefiles").mkdir(parents=True)
    assert resolve_data_subdir(project, "shapefiles") == project / "data" / "shapefiles"


def test_resolve_forcing_basin_path_prefers_store_then_legacy(tmp_path):
    project = tmp_path / "domain_x"
    # Establish the legacy forcing layout so resolve_data_subdir resolves to it.
    legacy = project / "forcing" / "basin_averaged_data"
    legacy.mkdir(parents=True)
    store = project / "data" / "model_ready" / "forcings"

    # No store -> legacy basin_averaged_data path.
    assert resolve_forcing_basin_path(project) == legacy

    # Store dir exists but is empty -> still legacy (must hold NetCDF files).
    store.mkdir(parents=True)
    assert resolve_forcing_basin_path(project) == legacy

    # Store populated with a NetCDF -> store wins.
    (store / "forcing_2020.nc").write_bytes(b"")
    assert resolve_forcing_basin_path(project) == store


def test_resolve_forcing_basin_path_respects_data_layout(tmp_path):
    """When no store, falls back through resolve_data_subdir (data/forcing)."""
    project = tmp_path / "domain_y"
    (project / "data" / "forcing").mkdir(parents=True)
    assert resolve_forcing_basin_path(project) == project / "data" / "forcing" / "basin_averaged_data"


# ---- LoggingMixin / TimingMixin -----------------------------------------


def test_logger_is_lazily_created_and_settable():
    obj = _Configurable()
    assert isinstance(obj.logger, logging.Logger)
    custom = logging.getLogger("custom.test.logger")
    obj.logger = custom
    assert obj.logger is custom


def test_time_limit_context_manager_runs():
    obj = _Configurable()
    ran = False
    with obj.time_limit("unit-test-task"):
        ran = True
    assert ran


# ---- ConfigurableMixin composition --------------------------------------


def test_composition_exposes_all_mixin_behaviors():
    obj = _Configurable()
    obj.config = _minimal()
    # ConfigMixin, ProjectContextMixin, LoggingMixin, FileUtilsMixin, TimingMixin
    for attr in ("config", "data_dir", "project_dir", "logger", "ensure_dir", "time_limit"):
        assert hasattr(obj, attr), f"composed object missing {attr}"
