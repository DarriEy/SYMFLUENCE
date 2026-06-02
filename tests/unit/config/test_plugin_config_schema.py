# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Integration test for the typed-config plugin path (RTI review item 18).

A plugin ships a runner *and its own typed Pydantic config* via
``model_manifest(config_schema=...)``; the core config pipeline must then
validate that config into the tree and expose it at ``config.model.<plugin>``
without any edit to ``core/config/``. Before the registry migration Phase B,
``R.config_schemas`` was never read by ``core/config/`` and this path was
notional.
"""

import pytest
from pydantic import BaseModel, ConfigDict, Field

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.config.models.model_configs import ModelConfig
from symfluence.core.registries import R
from symfluence.core.registry import model_manifest


class _FakePluginConfig(BaseModel):
    """Minimal plugin-provided typed config (mirrors in-tree *Config classes)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True, frozen=True)

    exe: str = Field(default="fakeplugin.exe", alias="FAKEPLUGIN_EXE")
    iterations: int = Field(default=7, alias="FAKEPLUGIN_ITERATIONS")


@pytest.fixture
def fake_plugin():
    """Register a fake plugin's typed config via the real model_manifest path."""
    name = "FAKEPLUGIN"
    model_manifest(name, config_schema=_FakePluginConfig)
    try:
        yield name
    finally:
        R.config_schemas.remove(name)


class TestPluginConfigSchema:
    def test_schema_registered_via_model_manifest(self, fake_plugin):
        assert R.config_schemas.get("FAKEPLUGIN") is _FakePluginConfig

    def test_selected_plugin_config_is_validated_into_tree(self, fake_plugin):
        cfg = ModelConfig(HYDROLOGICAL_MODEL="FAKEPLUGIN")
        # Accessible via the same attribute path in-tree models use.
        assert isinstance(cfg.fakeplugin, _FakePluginConfig)
        assert cfg.fakeplugin.exe == "fakeplugin.exe"
        assert cfg.fakeplugin.iterations == 7
        assert "fakeplugin" in cfg.model_specific

    def test_plugin_payload_is_validated(self, fake_plugin):
        cfg = ModelConfig(
            HYDROLOGICAL_MODEL="FAKEPLUGIN",
            fakeplugin={"FAKEPLUGIN_ITERATIONS": 42},
        )
        assert cfg.fakeplugin.iterations == 42  # supplied value
        assert cfg.fakeplugin.exe == "fakeplugin.exe"  # default preserved

    def test_unselected_plugin_resolves_to_none(self, fake_plugin):
        cfg = ModelConfig(HYDROLOGICAL_MODEL="SUMMA")
        # Registered-but-unselected -> None (legacy ``Optional[*Config]`` semantics).
        assert cfg.fakeplugin is None

    def test_plugin_aliases_participate_in_flat_transform(self, fake_plugin):
        from symfluence.core.config.introspection import generate_flat_to_nested_map

        mapping = generate_flat_to_nested_map(SymfluenceConfig)
        assert mapping.get("FAKEPLUGIN_EXE") == ("model", "fakeplugin", "exe")
        assert mapping.get("FAKEPLUGIN_ITERATIONS") == ("model", "fakeplugin", "iterations")

    def test_plugin_config_round_trips_through_dump(self, fake_plugin):
        cfg = ModelConfig(HYDROLOGICAL_MODEL="FAKEPLUGIN", fakeplugin={"FAKEPLUGIN_ITERATIONS": 99})
        dumped = cfg.model_dump(by_alias=False, exclude_none=True)
        # Serializer hoists model_specific back to the flat shape.
        assert "fakeplugin" in dumped and "model_specific" not in dumped
        restored = ModelConfig.model_validate(dumped)
        assert restored.fakeplugin.iterations == 99
