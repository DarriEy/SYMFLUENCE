# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for tiered unrecognized-key validation (RTI review item 21 / Q3).

Covers the mechanism only: warn-by-default, strict opt-in (config key and
``SYMFLUENCE_STRICT_CONFIG`` env var), the ``ALLOW_UNKNOWN_KEYS`` escape hatch,
"did you mean?" suggestions, and that core / legacy / plugin keys are accepted.
"""
from __future__ import annotations

import logging

import pytest
import yaml

from symfluence.core.config.key_validation import (
    RESERVED_CONTROL_KEYS,
    find_unknown_keys,
    is_strict_config_mode,
    validate_known_flat_keys,
)
from symfluence.core.exceptions import ConfigValidationError

KNOWN = {"DOMAIN_NAME", "HYDROLOGICAL_MODEL", "FORCING_DATASET", "EXPERIMENT_ID"}


class TestFindUnknownKeys:
    def test_all_known(self):
        assert find_unknown_keys({"DOMAIN_NAME": "x"}, KNOWN) == []

    def test_unknown_detected_case_insensitive(self):
        assert find_unknown_keys({"domain_name": "x", "BOGUS_KEY": 1}, KNOWN) == ["BOGUS_KEY"]

    def test_reserved_control_keys_never_flagged(self):
        cfg = {k: True for k in RESERVED_CONTROL_KEYS}
        assert find_unknown_keys(cfg, KNOWN) == []

    def test_explicit_allow_unknown_arg(self):
        assert find_unknown_keys({"BOGUS": 1}, KNOWN, allow_unknown=["BOGUS"]) == []

    def test_allow_unknown_keys_from_config_list(self):
        cfg = {"BOGUS": 1, "ALLOW_UNKNOWN_KEYS": ["BOGUS"]}
        assert find_unknown_keys(cfg, KNOWN) == []

    def test_allow_unknown_keys_from_config_csv(self):
        cfg = {"BOGUS": 1, "OTHER": 2, "ALLOW_UNKNOWN_KEYS": "BOGUS, OTHER"}
        assert find_unknown_keys(cfg, KNOWN) == []

    def test_non_string_keys_ignored(self):
        assert find_unknown_keys({1: "x", "DOMAIN_NAME": "y"}, KNOWN) == []


class TestStrictResolution:
    def test_default_lenient(self, monkeypatch):
        monkeypatch.delenv("SYMFLUENCE_STRICT_CONFIG", raising=False)
        assert is_strict_config_mode({}) is False

    @pytest.mark.parametrize("value", ["true", "1", "yes", "TRUE", "Yes"])
    def test_env_var_enables_strict(self, monkeypatch, value):
        monkeypatch.setenv("SYMFLUENCE_STRICT_CONFIG", value)
        assert is_strict_config_mode({}) is True

    def test_config_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("SYMFLUENCE_STRICT_CONFIG", "true")
        assert is_strict_config_mode({"STRICT_CONFIG": False}) is False

    def test_config_key_bool_and_str(self, monkeypatch):
        monkeypatch.delenv("SYMFLUENCE_STRICT_CONFIG", raising=False)
        assert is_strict_config_mode({"STRICT_CONFIG": True}) is True
        assert is_strict_config_mode({"STRICT_CONFIG": "yes"}) is True
        assert is_strict_config_mode({"STRICT_CONFIG": "no"}) is False


class TestValidateKnownFlatKeys:
    def test_clean_config_returns_empty(self):
        assert validate_known_flat_keys({"DOMAIN_NAME": "x"}, KNOWN) == []

    def test_warns_by_default(self, caplog, monkeypatch):
        monkeypatch.delenv("SYMFLUENCE_STRICT_CONFIG", raising=False)
        with caplog.at_level(logging.WARNING):
            unknown = validate_known_flat_keys({"BOGUS_KEY": 1}, KNOWN)
        assert unknown == ["BOGUS_KEY"]
        assert "BOGUS_KEY" in caplog.text
        assert "Unrecognized configuration key" in caplog.text

    def test_strict_raises(self):
        with pytest.raises(ConfigValidationError) as exc:
            validate_known_flat_keys({"BOGUS_KEY": 1}, KNOWN, strict=True)
        assert "BOGUS_KEY" in str(exc.value)

    def test_strict_from_env(self, monkeypatch):
        monkeypatch.setenv("SYMFLUENCE_STRICT_CONFIG", "1")
        with pytest.raises(ConfigValidationError):
            validate_known_flat_keys({"BOGUS_KEY": 1}, KNOWN)

    def test_strict_from_config_key(self, monkeypatch):
        monkeypatch.delenv("SYMFLUENCE_STRICT_CONFIG", raising=False)
        with pytest.raises(ConfigValidationError):
            validate_known_flat_keys({"BOGUS_KEY": 1, "STRICT_CONFIG": True}, KNOWN)

    def test_did_you_mean_suggestion(self):
        with pytest.raises(ConfigValidationError) as exc:
            # Close typo of a known key -> difflib should suggest it.
            validate_known_flat_keys({"DOMAIN_NAMW": "x"}, KNOWN, strict=True)
        assert "did you mean 'DOMAIN_NAME'" in str(exc.value)

    def test_source_in_message(self):
        with pytest.raises(ConfigValidationError) as exc:
            validate_known_flat_keys({"BOGUS": 1}, KNOWN, strict=True, source="my.yaml")
        assert "my.yaml" in str(exc.value)

    def test_escape_hatch_suppresses(self, caplog):
        with caplog.at_level(logging.WARNING):
            unknown = validate_known_flat_keys(
                {"BOGUS": 1, "ALLOW_UNKNOWN_KEYS": ["BOGUS"]}, KNOWN
            )
        assert unknown == []
        assert "BOGUS" not in caplog.text


def _minimal_yaml(tmp_path, extra: dict | None = None):
    cfg = {
        "DOMAIN_NAME": "test_basin",
        "EXPERIMENT_ID": "run_1",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-12-31 23:00",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "grus",
        "HYDROLOGICAL_MODEL": "SUMMA",
        "FORCING_DATASET": "ERA5",
    }
    if extra:
        cfg.update(extra)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


class TestFromFileIntegration:
    """The gate is wired into SymfluenceConfig.from_file's flat branch."""

    def test_clean_config_no_warning(self, tmp_path, caplog, monkeypatch):
        from symfluence.core.config.models import SymfluenceConfig

        monkeypatch.delenv("SYMFLUENCE_STRICT_CONFIG", raising=False)
        path = _minimal_yaml(tmp_path)
        with caplog.at_level(logging.WARNING):
            SymfluenceConfig.from_file(path, use_env=False)
        assert "Unrecognized configuration key" not in caplog.text

    def test_typo_warns_but_loads(self, tmp_path, caplog, monkeypatch):
        from symfluence.core.config.models import SymfluenceConfig

        monkeypatch.delenv("SYMFLUENCE_STRICT_CONFIG", raising=False)
        path = _minimal_yaml(tmp_path, {"DOMAIN_NAMW": "oops"})
        with caplog.at_level(logging.WARNING):
            cfg = SymfluenceConfig.from_file(path, use_env=False)
        assert cfg is not None
        assert "DOMAIN_NAMW" in caplog.text

    def test_strict_env_raises_on_typo(self, tmp_path, monkeypatch):
        from symfluence.core.config.models import SymfluenceConfig

        monkeypatch.setenv("SYMFLUENCE_STRICT_CONFIG", "true")
        path = _minimal_yaml(tmp_path, {"DOMAIN_NAMW": "oops"})
        with pytest.raises(ConfigValidationError):
            SymfluenceConfig.from_file(path, use_env=False)

    def test_strict_config_key_raises_on_typo(self, tmp_path, monkeypatch):
        from symfluence.core.config.models import SymfluenceConfig

        monkeypatch.delenv("SYMFLUENCE_STRICT_CONFIG", raising=False)
        path = _minimal_yaml(tmp_path, {"DOMAIN_NAMW": "oops", "STRICT_CONFIG": True})
        with pytest.raises(ConfigValidationError):
            SymfluenceConfig.from_file(path, use_env=False)

    def test_escape_hatch_silences_in_strict(self, tmp_path, monkeypatch):
        from symfluence.core.config.models import SymfluenceConfig

        monkeypatch.setenv("SYMFLUENCE_STRICT_CONFIG", "true")
        path = _minimal_yaml(
            tmp_path, {"MY_CUSTOM_KEY": 1, "ALLOW_UNKNOWN_KEYS": ["MY_CUSTOM_KEY"]}
        )
        # Should not raise — explicitly allowlisted.
        cfg = SymfluenceConfig.from_file(path, use_env=False)
        assert cfg is not None


class TestShippedTemplatesClean:
    """Strict-in-CI anchor: shipped example tutorials carry no unrecognized keys.

    Validates the *key layer* only (not full Pydantic value validation), so it
    isolates the unrecognized-key contract (RTI review item 21) from unrelated
    field-value issues. Guards against new typos / cruft re-entering templates.
    """

    def _example_templates(self):
        import glob
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        pattern = str(root / "src/symfluence/resources/config_templates/examples/*.yaml")
        paths = sorted(glob.glob(pattern))
        assert paths, f"no example templates found at {pattern}"
        return paths

    def test_no_unrecognized_keys_in_example_templates(self):
        from symfluence.core.config.config_loader import _normalize_key
        from symfluence.core.config.key_validation import find_unknown_keys
        from symfluence.core.config.transformers import build_combined_flat_to_nested_map

        offenders = {}
        for path in self._example_templates():
            raw = yaml.safe_load(open(path)) or {}
            flat = {_normalize_key(k): v for k, v in raw.items()}
            known = build_combined_flat_to_nested_map(flat.get("HYDROLOGICAL_MODEL"))
            unknown = find_unknown_keys(flat, known)
            if unknown:
                offenders[path.split("/")[-1]] = unknown
        assert not offenders, f"Unrecognized keys in shipped templates: {offenders}"


class TestPluginKeysAccepted:
    """A plugin-registered schema's keys must be recognized, not flagged."""

    def test_plugin_alias_is_known(self):
        from pydantic import BaseModel, ConfigDict, Field

        from symfluence.core.config import transformers
        from symfluence.core.registries import R
        from symfluence.core.registry import model_manifest

        class _FakePluginConfig(BaseModel):
            model_config = ConfigDict(extra="allow", populate_by_name=True, frozen=True)
            exe: str = Field(default="x", alias="FAKEVALPLUGIN_EXE")

        model_manifest("FAKEVALPLUGIN", config_schema=_FakePluginConfig)
        # The flat<->nested map is cached globally; invalidate it so the newly
        # registered plugin participates (mirrors bootstrap registering plugins
        # before the first config load builds the cache).
        transformers._AUTO_GENERATED_MAP = None
        try:
            known = transformers.build_combined_flat_to_nested_map("FAKEVALPLUGIN")
            assert "FAKEVALPLUGIN_EXE" in known
            assert find_unknown_keys({"FAKEVALPLUGIN_EXE": "y"}, known) == []
        finally:
            R.config_schemas.remove("FAKEVALPLUGIN")
            transformers._AUTO_GENERATED_MAP = None
