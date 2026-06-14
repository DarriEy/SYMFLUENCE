# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Flat STATE_* / ENKF_* / DA_METHOD keys must populate the typed sections.

Regression tests for the introspection walker gap where the ``state`` and
``data_assimilation`` root sections were not walked: flat configs silently
left StateConfig / EnKFConfig / DataAssimilationConfig at their defaults
(e.g. the DA manager ran with ensemble_size=50 regardless of the user's
ENKF_ENSEMBLE_SIZE). The keys rode in ``_extra`` via RECOGNIZED_FLAT_KEYS
instead of transforming into their typed fields.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def base_flat_config(tmp_path):
    return {
        "DOMAIN_NAME": "state_da_test",
        "HYDROLOGICAL_MODEL": "SUMMA",
        "EXPERIMENT_ID": "run1",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "FORCING_DATASET": "ERA5",
        "SYMFLUENCE_DATA_DIR": str(tmp_path / "data"),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "EXPERIMENT_TIME_START": "2010-01-01 00:00",
        "EXPERIMENT_TIME_END": "2010-12-31 23:00",
    }


def _build(flat):
    from symfluence.core.config.models import SymfluenceConfig
    from symfluence.core.config.transformers import transform_flat_to_nested

    return SymfluenceConfig(**transform_flat_to_nested(flat))


class TestIntrospectionCoversStateAndDA:
    def test_state_and_da_keys_in_transform_map(self):
        from symfluence.core.config.transformers import build_combined_flat_to_nested_map

        mapping = build_combined_flat_to_nested_map(None)
        assert mapping.get("STATE_SAVE") == ("state", "save")
        assert mapping.get("STATE_DIR") == ("state", "state_dir")
        assert mapping.get("DA_METHOD") == ("data_assimilation", "method")
        assert mapping.get("ENKF_ENSEMBLE_SIZE") == (
            "data_assimilation", "enkf", "ensemble_size",
        )

    def test_all_root_sections_are_walked(self):
        """Every nested-model section on the root config must be reachable
        by the introspection walker, so new sections cannot silently detach
        their flat-key aliases again."""
        from pydantic import BaseModel

        from symfluence.core.config.introspection import generate_flat_to_nested_map
        from symfluence.core.config.models import SymfluenceConfig

        mapping = generate_flat_to_nested_map(SymfluenceConfig)
        walked_sections = {path[0] for path in mapping.values()}
        for field_name, field_info in SymfluenceConfig.model_fields.items():
            annotation = field_info.annotation
            # Unwrap Optional[X]
            args = getattr(annotation, "__args__", None)
            if args:
                non_none = [a for a in args if a is not type(None)]
                annotation = non_none[0] if non_none else annotation
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                assert field_name in walked_sections, (
                    f"root section '{field_name}' has a Pydantic model but its "
                    "aliases are not discovered by generate_flat_to_nested_map — "
                    "add it to section_field_names in introspection.py"
                )


class TestFlatKeysPopulateTypedSections:
    def test_enkf_flat_keys_reach_typed_fields(self, base_flat_config):
        cfg = _build({
            **base_flat_config,
            "ENKF_ENSEMBLE_SIZE": 10,
            "ENKF_INFLATION_FACTOR": 1.5,
            "ENKF_FILTER_VARIANT": "deterministic",
        })
        assert cfg.data_assimilation.enkf.ensemble_size == 10
        assert cfg.data_assimilation.enkf.inflation_factor == 1.5
        assert cfg.data_assimilation.enkf.filter_variant == "deterministic"

    def test_state_flat_keys_reach_typed_fields(self, base_flat_config):
        cfg = _build({
            **base_flat_config,
            "STATE_SAVE": True,
            "STATE_DIR": "/x/states",
            "STATE_ENSEMBLE_MEMBERS": 3,
        })
        assert cfg.state.save is True
        assert cfg.state.state_dir == "/x/states"
        assert cfg.state.ensemble_members == 3

    def test_flat_view_round_trips(self, base_flat_config):
        """config.get('KEY') must keep working after promotion out of _extra."""
        cfg = _build({
            **base_flat_config,
            "ENKF_ENSEMBLE_SIZE": 10,
            "STATE_SAVE": True,
        })
        assert cfg.get("ENKF_ENSEMBLE_SIZE") == 10
        assert cfg.get("STATE_SAVE") is True
        # defaults of the typed sections are visible in the flat view too
        assert cfg.get("DA_METHOD") == "enkf"

    def test_defaults_when_keys_absent(self, base_flat_config):
        cfg = _build(base_flat_config)
        # data_assimilation stays None when no DA keys are set (the DA manager
        # falls back to DataAssimilationConfig() itself); state has a factory.
        assert cfg.data_assimilation is None
        assert cfg.state.save is False
        assert cfg.state.enabled is False

    def test_partial_enkf_keys_fill_remaining_defaults(self, base_flat_config):
        cfg = _build({**base_flat_config, "ENKF_ENSEMBLE_SIZE": 10})
        assert cfg.data_assimilation.enkf.ensemble_size == 10
        # untouched fields get their EnKFConfig defaults
        assert cfg.data_assimilation.enkf.inflation_factor == 1.0
        assert cfg.data_assimilation.enkf.filter_variant == "stochastic"


class TestValidatorRecognition:
    def test_keys_known_without_recognized_flat_keys_entry(self, base_flat_config):
        """The validator allowlist must cover these keys via the transform map
        (they were removed from RECOGNIZED_FLAT_KEYS when the sections were wired)."""
        from symfluence.core.config.key_validation import find_unknown_keys
        from symfluence.core.config.legacy_aliases import RECOGNIZED_FLAT_KEYS
        from symfluence.core.config.transformers import build_combined_flat_to_nested_map

        flat = {
            **base_flat_config,
            "ENKF_ENSEMBLE_SIZE": 10,
            "STATE_SAVE": True,
            "DA_METHOD": "enkf",
        }
        known = set(build_combined_flat_to_nested_map("SUMMA"))
        assert find_unknown_keys(flat, known) == []
        # and the legacy recognition entries are gone (no double bookkeeping)
        for key in ("ENKF_ENSEMBLE_SIZE", "STATE_SAVE", "DA_METHOD", "STATE_DIR"):
            assert key not in RECOGNIZED_FLAT_KEYS
