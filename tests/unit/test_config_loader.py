"""
Unit tests for configuration normalization and validation.

Tests the config_loader module's ability to handle aliases, type coercion,
and validation of configuration dictionaries.
"""

import pytest
import yaml

from symfluence.utils.config.config_loader import normalize_config, validate_config


pytestmark = [pytest.mark.unit, pytest.mark.quick]


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
        "MPI_PROCESSES": "4",
        "LAPSE_RATE": "0.0065",
        "NEX_MODELS": "ACCESS-CM2,GFDL-ESM4",
        "MULTI_SCALE_THRESHOLDS": "10000,5000,2500",
        "RANDOM_SEED": "None",
    }
    normalized = normalize_config(raw)
    assert normalized["DOWNLOAD_SNOTEL"] is True
    assert normalized["FORCE_RUN_ALL_STEPS"] is False
    assert normalized["MPI_PROCESSES"] == 4
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
