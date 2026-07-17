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
    find_unknown_nested_keys,
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
    """Strict-in-CI anchor: shipped flat-style templates carry no unrecognized keys.

    Validates the *key layer* only (not full Pydantic value validation), so it
    isolates the unrecognized-key contract (RTI review item 21) from unrelated
    field-value issues. Guards against new typos / cruft re-entering templates.
    """

    # Flat UPPERCASE-key templates; the *_nested.yaml variants use the nested
    # schema and are covered by full-config validation tests instead.
    FLAT_TEMPLATES = ["config_template.yaml", "config_template_comprehensive.yaml"]

    def _shipped_templates(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[3]
        templates_dir = root / "src/symfluence/resources/config_templates"
        paths = [templates_dir / name for name in self.FLAT_TEMPLATES]
        missing = [str(p) for p in paths if not p.exists()]
        assert not missing, f"shipped templates not found: {missing}"
        return [str(p) for p in paths]

    def test_no_unrecognized_keys_in_shipped_templates(self):
        from symfluence.core.config.config_loader import _normalize_key
        from symfluence.core.config.key_validation import find_unknown_keys
        from symfluence.core.config.transformers import build_combined_flat_to_nested_map

        offenders = {}
        for path in self._shipped_templates():
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


class TestRecognizedFlatKeysAndLegacySpellings:
    """RTI Q3 follow-on (T1+T2): real flat-read keys are recognized, and the
    TARGET_METRIC legacy spelling normalizes to the canonical OPTIMIZATION_METRIC.
    """

    def test_recognized_flat_keys_are_known_by_the_validator(self):
        from symfluence.core.config.legacy_aliases import RECOGNIZED_FLAT_KEYS
        from symfluence.core.config.transformers import build_combined_flat_to_nested_map

        known = set(build_combined_flat_to_nested_map("SUMMA")) | RECOGNIZED_FLAT_KEYS
        sample = {
            "MULTI_GAUGE_OBS_DIR": "/obs",
            "ENKF_ENSEMBLE_SIZE": 20,
            "ESA_CCI_SM_PATH": "/sm",
            "IGNACIO_FWI_LATITUDE": 64.0,
            "HYPE_PET_MODEL": "1",
            "STATE_SAVE": True,
        }
        assert find_unknown_keys(sample, known) == []

    def test_recognized_keys_not_added_to_transform_map(self):
        # Recognition only: these keys must NOT acquire a nested path, else they
        # would be relocated out of the _extra passthrough that flat readers use.
        from symfluence.core.config.legacy_aliases import RECOGNIZED_FLAT_KEYS
        from symfluence.core.config.transformers import build_combined_flat_to_nested_map

        transform_map = build_combined_flat_to_nested_map("SUMMA")
        assert not (RECOGNIZED_FLAT_KEYS & set(transform_map))

    def test_target_metric_normalizes_to_optimization_metric(self):
        from symfluence.core.config.config_loader import _normalize_key
        from symfluence.core.config.transformers import build_combined_flat_to_nested_map

        assert _normalize_key("TARGET_METRIC") == "OPTIMIZATION_METRIC"
        # ...and the canonical spelling maps to the real nested field.
        assert build_combined_flat_to_nested_map("SUMMA")["OPTIMIZATION_METRIC"] == (
            "optimization",
            "metric",
        )


class TestFindUnknownNestedKeys:
    """Unknown-key detection for nested (lowercase-section) configs.

    Motivating bug: ``evaluation: snotel: station_id:`` — SNOTELConfig's field
    is ``station`` (alias SNOTEL_STATION), so the typo validated fine under
    ``extra='allow'`` and was silently ignored at runtime.
    """

    def test_known_nested_key_passes(self):
        assert find_unknown_nested_keys({"evaluation": {"snotel": {"station": "679"}}}) == []

    def test_unknown_nested_key_flagged_with_dotted_path(self):
        result = find_unknown_nested_keys({"evaluation": {"snotel": {"station_id": "679"}}})
        assert result == ["evaluation.snotel.station_id"]

    def test_alias_spelling_inside_section_is_known(self):
        # Aliases live at the SAME level as the field they populate: with
        # populate_by_name=True, SNOTEL_STATION is a legal input spelling for
        # SNOTELConfig.station inside evaluation.snotel.
        assert find_unknown_nested_keys(
            {"evaluation": {"snotel": {"SNOTEL_STATION": "679"}}}
        ) == []

    def test_flat_uppercase_top_level_checked_against_flat_universe(self):
        data = {
            "DOMAIN_NAME": "x",  # recognized flat key
            "HYDROLOGICAL_MDOEL": "SUMMA",  # the classic typo
            "domain": {"name": "x"},
        }
        assert find_unknown_nested_keys(data) == ["HYDROLOGICAL_MDOEL"]

    def test_unregistered_model_subsection_skipped(self):
        # A model.<name> subsection whose schema is not registered (e.g. an
        # external plugin absent from this environment) must not raise false
        # positives — its contents are unknowable here.
        data = {"model": {"hydrological_model": "SUMMA",
                          "notarealmodelxyz": {"whatever_key": 1}}}
        assert find_unknown_nested_keys(data) == []

    def test_registered_model_subsection_validated(self):
        data = {"model": {"hydrological_model": "SUMMA",
                          "summa": {"definitely_not_a_summa_field": 1}}}
        assert find_unknown_nested_keys(data) == ["model.summa.definitely_not_a_summa_field"]

    def test_dict_valued_fields_are_opaque(self):
        # parameter_bounds is Dict[str, Any]: its contents are user data, not
        # config keys, and must never be walked.
        data = {"optimization": {"parameter_bounds": {"albedoMax": [0.7, 0.95]}}}
        assert find_unknown_nested_keys(data) == []

    def test_reserved_extra_key_skipped(self):
        assert find_unknown_nested_keys({"_extra": {"anything": 1},
                                         "domain": {"_extra": 1}}) == []

    def test_explicit_model_root(self):
        from symfluence.core.config.models.evaluation import SNOTELConfig

        assert find_unknown_nested_keys({"station": "679"}, model=SNOTELConfig) == []
        assert find_unknown_nested_keys({"station_id": "679"}, model=SNOTELConfig) == ["station_id"]
