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


def test_load_env_overrides_natural_path_spelling(monkeypatch):
    """The canonical flat keys for the two paths already carry a SYMFLUENCE_
    prefix, so the natural env spelling must survive the loader's prefix strip
    (previously it decayed to an unrecognized bare DATA_DIR / CODE_DIR)."""
    monkeypatch.setenv("SYMFLUENCE_DATA_DIR", "/env/data")
    monkeypatch.setenv("SYMFLUENCE_CODE_DIR", "/env/code")

    overrides = _load_env_overrides()

    assert overrides["SYMFLUENCE_DATA_DIR"] == "/env/data"
    assert overrides["SYMFLUENCE_CODE_DIR"] == "/env/code"
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


# ----------------------------------------------------------------------
# Nested-config path resolution — `default` sentinel and env precedence
# ----------------------------------------------------------------------


def _write_nested_config(tmp_path, data_dir="default", code_dir="default"):
    cfg = {
        "system": {"data_dir": data_dir, "code_dir": code_dir},
        "domain": {
            "name": "Bow",
            "time_start": "2015-01-01 00:00",
            "time_end": "2015-12-31 23:00",
            "definition_method": "lumped",
            "discretization": "lumped",
            "experiment_id": "run1",
        },
        "model": {"hydrological_model": "SUMMA"},
        "forcing": {"dataset": "ERA5"},
    }
    path = tmp_path / "nested.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_nested_default_sentinel_resolves_not_literal_default(tmp_path, monkeypatch):
    """A nested `system.data_dir: default` must resolve to the computed default
    (a SYMFLUENCE_data sibling of the code dir), never the literal relative
    path ./default."""
    from symfluence.core.config.models import SymfluenceConfig

    # Clear every env var that feeds default path resolution. SYMFLUENCE_DATA
    # is the secondary fallback consulted by _resolve_default_data_dir, and CI
    # (install-validate) sets it to a lowercase `symfluence_data` workspace
    # path — leaving it set makes the resolver return that path instead of the
    # computed sibling.
    for var in ("SYMFLUENCE_DATA_DIR", "SYMFLUENCE_DATA", "SYMFLUENCE_CODE_DIR"):
        monkeypatch.delenv(var, raising=False)

    cfg = SymfluenceConfig.from_file(_write_nested_config(tmp_path))

    # The literal `default` sentinel must not survive as a relative path, and
    # the data dir must be the computed SYMFLUENCE_data sibling of the code dir.
    # Compare the name case-insensitively so a case-preserving-but-insensitive
    # filesystem (macOS) that canonicalises an existing dir can't flake this.
    assert cfg.system.data_dir.name.lower() == "symfluence_data"
    assert cfg.system.data_dir.parent == cfg.system.code_dir.parent
    assert cfg.system.code_dir.name != "default"


def test_nested_env_override_beats_sentinel(tmp_path, monkeypatch):
    """The natural SYMFLUENCE_DATA_DIR env spelling overrides the file sentinel
    for nested configs (regression: it was silently dropped)."""
    from symfluence.core.config.models import SymfluenceConfig

    monkeypatch.setenv("SYMFLUENCE_DATA_DIR", str(tmp_path / "envdata"))

    cfg = SymfluenceConfig.from_file(_write_nested_config(tmp_path))

    assert cfg.system.data_dir == (tmp_path / "envdata").resolve()
