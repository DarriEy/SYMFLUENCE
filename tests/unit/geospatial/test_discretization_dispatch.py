import logging
from pathlib import Path

import pytest

from symfluence.geospatial.discretization import DomainDiscretizer
from symfluence.geospatial.discretization.attributes import combined, elevation
from symfluence.core.config.models import SymfluenceConfig


def _base_config(tmp_path, discretization):
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "test_domain",
        "DEM_NAME": "default",
        "DEM_PATH": "default",
        "DOMAIN_DEFINITION_METHOD": "delineate",
        "CATCHMENT_PATH": "default",
        "CATCHMENT_SHP_NAME": "default",
        "CATCHMENT_SHP_GRUID": "GRU_ID",
        "CATCHMENT_SHP_HRUID": "HRU_ID",
        "SUB_GRID_DISCRETIZATION": discretization,
        "EXPERIMENT_ID": "test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-01-02 00:00",
        "FORCING_DATASET": "ERA5",
        "HYDROLOGICAL_MODEL": "SUMMA",
    }
    return SymfluenceConfig(**config_dict)


def test_discretize_domain_dispatches_single_attribute(tmp_path, monkeypatch):
    config = _base_config(tmp_path, "elevation")
    logger = logging.getLogger("test_discretize_domain_dispatches_single_attribute")

    called = {}

    def fake_discretize(self):
        called["method"] = "elevation"

    monkeypatch.setattr(elevation, "discretize", fake_discretize)

    expected = Path(tmp_path / "sorted.shp")
    monkeypatch.setattr(
        DomainDiscretizer, "sort_catchment_shape", lambda self: expected
    )

    discretizer = DomainDiscretizer(config, logger)
    result = discretizer.discretize_domain()

    assert called["method"] == "elevation"
    assert result == expected


def test_discretize_domain_dispatches_combined_attributes(tmp_path, monkeypatch):
    config = _base_config(tmp_path, "elevation, landclass")
    logger = logging.getLogger("test_discretize_domain_dispatches_combined_attributes")

    captured = {}

    def fake_combined(self, attrs):
        captured["attrs"] = attrs

    monkeypatch.setattr(combined, "discretize", fake_combined)

    expected = Path(tmp_path / "sorted_combined.shp")
    monkeypatch.setattr(
        DomainDiscretizer, "sort_catchment_shape", lambda self: expected
    )

    discretizer = DomainDiscretizer(config, logger)
    result = discretizer.discretize_domain()

    assert captured["attrs"] == ["elevation", "landclass"]
    assert result == expected


def test_discretize_domain_rejects_unknown_method(tmp_path):
    config = _base_config(tmp_path, "not_a_method")
    logger = logging.getLogger("test_discretize_domain_rejects_unknown_method")

    discretizer = DomainDiscretizer(config, logger)

    with pytest.raises(ValueError):
        discretizer.discretize_domain()
