# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The model-metadata extension seam (service-decomposition prep).

Core used to tabulate, per model, the runoff artifact routing consumes
(``runoff_loader.MODEL_CONFIGS``) and the config keys the routing decision
reads (``RoutingDecider.SPATIAL_MODE_KEYS`` / ``ROUTING_INTEGRATION_KEYS``).
Those tables are gone: a model declares all of it on its registered
``ModelConfigSchema``, so an external model package becomes routable by
registering a schema and editing nothing in core.

Value parity across that move is pinned by ``test_model_knowledge_parity``;
this module pins the *seam* — that registration, and only registration, is
what makes core see a model.
"""
from __future__ import annotations

import pytest

from symfluence.core.modeling import config_schema as cs
from symfluence.core.modeling.config_schema import (
    ExecutionConfig,
    InputConfig,
    InstallationConfig,
    ModelConfigSchema,
    OutputConfig,
    RunoffConfig,
    register_model_schema,
)
from symfluence.core.modeling.utilities.routing_decider import RoutingDecider
from symfluence.core.modeling.utilities.runoff_loader import get_model_config

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep registrations test-local."""
    saved = dict(cs.REGISTERED_SCHEMAS)
    yield
    cs.REGISTERED_SCHEMAS.clear()
    cs.REGISTERED_SCHEMAS.update(saved)


def _schema(**overrides) -> ModelConfigSchema:
    kwargs = dict(
        model_name="ExtModel",
        installation=InstallationConfig("EXT_INSTALL_PATH", "installs/ext"),
        execution=ExecutionConfig(),
        input=InputConfig("FORCING_EXT_PATH", "forcing/EXT_input"),
        output=OutputConfig("EXPERIMENT_OUTPUT_EXT", "simulations/{experiment_id}/EXT"),
    )
    kwargs.update(overrides)
    return ModelConfigSchema(**kwargs)


_EXT_RUNOFF = RunoffConfig(
    output_dir_key="EXPERIMENT_OUTPUT_EXT",
    output_dir_name="EXTMODEL",
    default_var="q_ext",
    default_units="m3/s",
    default_dt="900",
    output_file_pattern="{domain_name}_{experiment_id}_ext.nc",
    hru_dim="subbasin",
    hru_var="subbasinId",
    comment_name="ExtModel",
    aliases=("EXT1", "EXT2"),
)


def test_external_package_contributes_runoff_metadata():
    register_model_schema("extmodel", _schema(runoff=_EXT_RUNOFF))

    cfg = get_model_config("EXTMODEL")
    assert cfg is _EXT_RUNOFF
    assert cfg.output_file_pattern == "{domain_name}_{experiment_id}_ext.nc"
    assert (cfg.hru_dim, cfg.hru_var, cfg.default_dt) == ("subbasin", "subbasinId", "900")


def test_model_owned_aliases_drive_source_normalization():
    """Variant spellings (GR's GR4J/GR5J/GR6J) are declared by the model."""
    register_model_schema("extmodel", _schema(runoff=_EXT_RUNOFF))

    assert get_model_config("EXT1") is _EXT_RUNOFF
    assert get_model_config("  ext2  ") is _EXT_RUNOFF


def test_external_package_contributes_routing_keys():
    register_model_schema(
        "extmodel",
        _schema(
            spatial_mode_key="EXT_SPATIAL_MODE",
            routing_integration_key="EXT_ROUTING_INTEGRATION",
        ),
    )

    assert RoutingDecider.SPATIAL_MODE_KEYS["EXTMODEL"] == "EXT_SPATIAL_MODE"
    assert RoutingDecider.ROUTING_INTEGRATION_KEYS["EXTMODEL"] == "EXT_ROUTING_INTEGRATION"
    # ...and the decision actually consults them.
    decider = RoutingDecider()
    assert decider.needs_routing(
        {"EXT_ROUTING_INTEGRATION": "mizuRoute"}, "EXTMODEL"
    ) is True
    assert decider.needs_routing({"EXT_SPATIAL_MODE": "distributed"}, "EXTMODEL") is True
    assert decider.needs_routing({"EXT_SPATIAL_MODE": "lumped"}, "EXTMODEL") is False


def test_undeclared_keys_leave_the_model_out_of_the_tables():
    """Absence is a declaration too: no key means core does not consult one."""
    register_model_schema("extmodel", _schema())

    assert "EXTMODEL" not in RoutingDecider.SPATIAL_MODE_KEYS
    assert "EXTMODEL" not in RoutingDecider.ROUTING_INTEGRATION_KEYS


def test_routing_key_is_not_the_routing_integration_key():
    """``routing_key`` is the model's own descriptive key, not core's input.

    SUMMA declares ``ROUTING_DELINEATION`` there — a different question — so
    the routing decision reads only the explicit ``routing_integration_key``.
    """
    register_model_schema("extmodel", _schema(routing_key="EXT_ROUTING_INTEGRATION"))

    assert "EXTMODEL" not in RoutingDecider.ROUTING_INTEGRATION_KEYS


def test_tables_are_live_not_snapshotted_at_import():
    before = dict(RoutingDecider.SPATIAL_MODE_KEYS)
    register_model_schema("latemodel", _schema(spatial_mode_key="LATE_SPATIAL_MODE"))

    assert "LATEMODEL" not in before
    assert RoutingDecider.SPATIAL_MODE_KEYS["LATEMODEL"] == "LATE_SPATIAL_MODE"


def test_models_without_runoff_declaration_are_absent_from_the_view():
    from symfluence.core.modeling.utilities.runoff_loader import MODEL_CONFIGS

    register_model_schema("extmodel", _schema())
    assert "extmodel" not in MODEL_CONFIGS

    register_model_schema("extmodel", _schema(runoff=_EXT_RUNOFF))
    assert MODEL_CONFIGS["extmodel"] is _EXT_RUNOFF
