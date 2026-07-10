"""
Unit tests for configuration normalization and validation.

Tests the config_loader module's ability to handle aliases, type coercion,
and validation of configuration dictionaries.
"""
from __future__ import annotations

import tempfile

import pytest
import yaml
from pydantic import BaseModel, Field, ValidationError

from symfluence.core.config.config_loader import (
    _coerce_value,
    _format_validation_error,
    _load_env_overrides,
    _normalize_key,
    normalize_config,
    validate_config,
)

pytestmark = [pytest.mark.unit, pytest.mark.quick]


# Minimal config that satisfies validate_config's 10 mandatory keys.
_TMP = tempfile.gettempdir()
VALID_CONFIG = {
    "SYMFLUENCE_DATA_DIR": f"{_TMP}/symfluence_data",
    "SYMFLUENCE_CODE_DIR": f"{_TMP}/symfluence_code",
    "DOMAIN_NAME": "Bow",
    "EXPERIMENT_ID": "run1",
    "EXPERIMENT_TIME_START": "2015-01-01 00:00",
    "EXPERIMENT_TIME_END": "2015-12-31 23:00",
    "DOMAIN_DEFINITION_METHOD": "lumped",
    "SUB_GRID_DISCRETIZATION": "lumped",
    "HYDROLOGICAL_MODEL": "SUMMA",
    "FORCING_DATASET": "ERA5",
}


def test_normalize_config_aliases_and_case():
    raw = {
        "GR_spatial": "lumped",
        "optimisation_target": "streamflow",
        "domain_name": "Bow",
    }
    normalized = normalize_config(raw)
    assert "GR_SPATIAL_MODE" in normalized
    assert normalized["GR_SPATIAL_MODE"] == "lumped"
    assert "GR_spatial" not in normalized
    assert normalized["OPTIMIZATION_TARGET"] == "streamflow"
    assert normalized["DOMAIN_NAME"] == "Bow"


def test_normalize_config_type_coercion():
    raw = {
        "DOWNLOAD_SNOTEL": "true",
        "FORCE_RUN_ALL_STEPS": "False",
        "NUM_PROCESSES": "4",
        "LAPSE_RATE": "0.0065",
        "NEX_MODELS": "ACCESS-CM2,GFDL-ESM4",
        "MULTI_SCALE_THRESHOLDS": "10000,5000,2500",
        "RANDOM_SEED": "None",
    }
    normalized = normalize_config(raw)
    assert normalized["DOWNLOAD_SNOTEL"] is True
    assert normalized["FORCE_RUN_ALL_STEPS"] is False
    assert normalized["NUM_PROCESSES"] == 4
    assert normalized["LAPSE_RATE"] == 0.0065
    assert normalized["NEX_MODELS"] == ["ACCESS-CM2", "GFDL-ESM4"]
    assert normalized["MULTI_SCALE_THRESHOLDS"] == ["10000", "5000", "2500"]
    assert normalized["RANDOM_SEED"] is None


def test_normalize_config_from_yaml():
    text = """
DOMAIN_NAME: Paradise
DOWNLOAD_SNOTEL: "false"
OPTIMISATION_METHODS: [iteration, emulation]
"""
    raw = yaml.safe_load(text)
    normalized = normalize_config(raw)
    assert normalized["DOMAIN_NAME"] == "Paradise"
    assert normalized["DOWNLOAD_SNOTEL"] is False
    assert normalized["OPTIMIZATION_METHODS"] == ["iteration", "emulation"]


def test_validate_config_missing_required():
    with pytest.raises(ValueError) as exc:
        validate_config({"DOMAIN_NAME": "test"})
    assert "Missing required configuration keys" in str(exc.value)


# ----------------------------------------------------------------------
# _coerce_value — every documented branch
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True), ("yes", True), ("1", True),
        ("false", False), ("no", False), ("0", False),
        ("none", None), ("null", None), ("", None),
        ("  True  ", True),          # surrounding whitespace stripped
        ("42", 42), ("-7", -7),
        ("3.14", 3.14), ("1.0", 1.0),
        ("a,b,c", ["a", "b", "c"]),
        ("x, y , z", ["x", "y", "z"]),  # inner whitespace trimmed
        ("plain", "plain"),
        ("v1.2.3", "v1.2.3"),        # not a number, no comma -> passthrough
    ],
)
def test_coerce_value_branches(value, expected):
    assert _coerce_value(value) == expected


def test_coerce_value_non_string_passthrough():
    obj = {"already": "typed"}
    assert _coerce_value(obj) is obj
    assert _coerce_value(5) == 5
    assert _coerce_value(True) is True


def test_normalize_key_alias_and_plain():
    assert _normalize_key("confluence_data_dir") == "SYMFLUENCE_DATA_DIR"
    assert _normalize_key("domain_name") == "DOMAIN_NAME"  # no alias -> just upper


# ----------------------------------------------------------------------
# _load_env_overrides — prefix stripping, normalization, coercion
# ----------------------------------------------------------------------


def test_load_env_overrides(monkeypatch):
    monkeypatch.setenv("SYMFLUENCE_DOMAIN_NAME", "EnvBasin")
    monkeypatch.setenv("SYMFLUENCE_NUM_PROCESSES", "8")
    monkeypatch.setenv("SYMFLUENCE_CONFLUENCE_DATA_DIR", "/env/data")  # legacy alias
    monkeypatch.setenv("UNRELATED_VAR", "ignored")

    overrides = _load_env_overrides()

    assert overrides["DOMAIN_NAME"] == "EnvBasin"
    assert overrides["NUM_PROCESSES"] == 8           # coerced to int
    assert overrides["SYMFLUENCE_DATA_DIR"] == "/env/data"  # alias-normalized key
    assert "UNRELATED_VAR" not in overrides
    assert not any(k.startswith("UNRELATED") for k in overrides)


def test_load_env_overrides_full_prefix_keys(monkeypatch):
    """SYMFLUENCE_DATA_DIR / SYMFLUENCE_CODE_DIR keep their full names.

    Stripping the SYMFLUENCE_ prefix would yield DATA_DIR / CODE_DIR, which
    match no config alias and were previously dropped silently.
    """
    monkeypatch.setenv("SYMFLUENCE_DATA_DIR", "/env/data_dir")
    monkeypatch.setenv("SYMFLUENCE_CODE_DIR", "/env/code_dir")

    overrides = _load_env_overrides()

    assert overrides["SYMFLUENCE_DATA_DIR"] == "/env/data_dir"
    assert overrides["SYMFLUENCE_CODE_DIR"] == "/env/code_dir"
    assert "DATA_DIR" not in overrides
    assert "CODE_DIR" not in overrides


# ----------------------------------------------------------------------
# validate_config — success path returns a typed, dumped config
# ----------------------------------------------------------------------


def test_validate_config_success_returns_dict():
    result = validate_config(dict(VALID_CONFIG))
    assert isinstance(result, dict) and result
    # model_dump yields the nested structure with the standard sections present.
    assert {"domain", "model", "forcing"} <= set(result)


# ----------------------------------------------------------------------
# _format_validation_error — missing / invalid / other-error branches
# ----------------------------------------------------------------------


def _make_validation_error() -> ValidationError:
    """A ValidationError carrying a 'missing', a '*_type', and an 'other' error.

    The formatter buckets errors by ``err['type']``: 'missing' -> missing fields,
    a type containing 'type'/'literal' -> invalid values, everything else ->
    other errors. ``name=123`` yields ``string_type`` (invalid) and ``count=-1``
    yields ``greater_than`` (other).
    """

    class _M(BaseModel):
        DOMAIN_NAME: str            # omitted -> 'missing'
        name: str                   # 123 -> 'string_type' -> invalid values
        count: int = Field(gt=0)    # -1 -> 'greater_than' -> other errors

    try:
        _M(name=123, count=-1)
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")  # pragma: no cover


def test_format_validation_error_sections():
    err = _make_validation_error()
    msg = _format_validation_error(err, {"NAME": 123, "COUNT": -1})

    assert "Configuration Validation Failed" in msg
    assert "Missing Required Fields" in msg and "DOMAIN_NAME" in msg
    assert "Invalid Field Values" in msg          # the string_type error
    assert "Validation Errors" in msg             # the greater_than 'other' error
    # the footer with actionable help is always appended
    assert "symfluence config list" in msg


def test_format_validation_error_suggests_typos():
    """Unknown keys near a real flat key get a 'Did you mean?' suggestion."""
    err = _make_validation_error()
    # DOMAINNAME / FORCING_DATSET are near-misses of real flat config keys.
    msg = _format_validation_error(err, {"DOMAINNAME": "x", "FORCING_DATSET": "ERA5"})

    assert "Did you mean" in msg
    assert "DOMAIN_NAME" in msg
    assert "FORCING_DATASET" in msg


def test_format_validation_error_no_typos_for_known_keys():
    """Recognized flat keys are not flagged as typos."""
    err = _make_validation_error()
    msg = _format_validation_error(err, {"DOMAIN_NAME": "Bow", "FORCING_DATASET": "ERA5"})
    assert "Did you mean" not in msg
