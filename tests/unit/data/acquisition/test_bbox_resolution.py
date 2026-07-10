# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Unit tests for AcquisitionService._resolve_bounding_box.

NB reported that point-domain configs (e.g. config_paradise_point.yaml)
fail data acquisition with 'BOUNDING_BOX_COORDS is required ...' even
though POUR_POINT_COORDS is set. The resolver auto-derives a small
square bbox around the pour point for point domains so these configs
work out of the box.
"""
from __future__ import annotations

import logging

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.data.acquisition.acquisition_service import AcquisitionService

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _service(tmp_path, **overrides):
    """Build an AcquisitionService bound to a minimal config under tmp_path."""
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "bbox_test",
        "EXPERIMENT_ID": "test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-01-02 00:00",
        "FORCING_DATASET": "ERA5",
        "HYDROLOGICAL_MODEL": "SUMMA",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
    }
    config_dict.update(overrides)
    config = SymfluenceConfig(**config_dict)
    return AcquisitionService(
        config, logging.getLogger("test_bbox_resolution")
    )


def test_explicit_bbox_wins(tmp_path):
    """Explicit BOUNDING_BOX_COORDS must always take precedence."""
    svc = _service(
        tmp_path,
        DOMAIN_DEFINITION_METHOD="point",
        POUR_POINT_COORDS="46.78/-121.75",
        BOUNDING_BOX_COORDS="44.5/-87.9/44.2/-87.5",
    )
    assert svc._resolve_bounding_box("attributes") == "44.5/-87.9/44.2/-87.5"


def test_point_domain_auto_derives_bbox_with_default_buffer(tmp_path):
    """Point domain + pour_point_coords + no bbox + no buffer → 0.01° square."""
    svc = _service(
        tmp_path,
        DOMAIN_DEFINITION_METHOD="point",
        POUR_POINT_COORDS="46.78/-121.75",
    )
    bbox = svc._resolve_bounding_box("attributes")
    parts = [float(p) for p in bbox.split("/")]
    # Order is north/west/south/east
    assert parts[0] == pytest.approx(46.79)   # lat + 0.01
    assert parts[1] == pytest.approx(-121.76)  # lon - 0.01
    assert parts[2] == pytest.approx(46.77)   # lat - 0.01
    assert parts[3] == pytest.approx(-121.74)  # lon + 0.01


def test_point_domain_auto_derives_with_custom_buffer(tmp_path):
    """POINT_BUFFER_DISTANCE must be honored when set."""
    svc = _service(
        tmp_path,
        DOMAIN_DEFINITION_METHOD="point",
        POUR_POINT_COORDS="50.0/-120.0",
        POINT_BUFFER_DISTANCE=0.05,
    )
    bbox = svc._resolve_bounding_box("forcing")
    parts = [float(p) for p in bbox.split("/")]
    assert parts[0] == pytest.approx(50.05)
    assert parts[1] == pytest.approx(-120.05)
    assert parts[2] == pytest.approx(49.95)
    assert parts[3] == pytest.approx(-119.95)


def test_lumped_domain_without_bbox_raises_actionable_error(tmp_path):
    """Non-point domains still require an explicit bbox; the error must
    name the call site (purpose) and point at how to fix it."""
    svc = _service(
        tmp_path,
        DOMAIN_DEFINITION_METHOD="lumped",
        POUR_POINT_COORDS="50.0/-120.0",
    )
    with pytest.raises(ValueError) as exc:
        svc._resolve_bounding_box("attributes")
    msg = str(exc.value)
    assert "BOUNDING_BOX_COORDS" in msg
    assert "attributes" in msg
    assert "POUR_POINT_COORDS" in msg  # mentions the point-domain shortcut


def test_point_domain_without_pour_point_raises(tmp_path):
    """A point domain config that forgets POUR_POINT_COORDS must fail
    cleanly rather than silently auto-deriving from nothing."""
    svc = _service(
        tmp_path,
        DOMAIN_DEFINITION_METHOD="point",
    )
    with pytest.raises(ValueError) as exc:
        svc._resolve_bounding_box("forcing")
    assert "BOUNDING_BOX_COORDS" in str(exc.value)


def test_paradise_point_config_resolves_bbox(tmp_path):
    """End-to-end: load 01_domain_definition/config_paradise_point.yaml
    (which intentionally omits BOUNDING_BOX_COORDS) and confirm the resolver
    produces a bbox without raising."""
    from pathlib import Path

    import yaml

    repo_root = Path(__file__).resolve().parents[4]
    cfg_path = (
        repo_root
        / "examples/paper_case_studies/configs"
        / "01_domain_definition/config_paradise_point.yaml"
    )
    with cfg_path.open() as fh:
        data = yaml.safe_load(fh)
    # Override data_dir to keep the test hermetic
    data.setdefault("system", {})["data_dir"] = str(tmp_path)
    config = SymfluenceConfig.model_validate(data)
    assert config.domain.bounding_box_coords is None  # confirms the gap exists

    svc = AcquisitionService(config, logging.getLogger("test_paradise_point"))
    bbox = svc._resolve_bounding_box("attributes")
    parts = [float(p) for p in bbox.split("/")]
    # pour_point_coords: 46.78/-121.75 + default buffer 0.01°
    assert parts[0] == pytest.approx(46.79)
    assert parts[2] == pytest.approx(46.77)


# ---------------------------------------------------------------------------
# Shared ConfigMixin._resolve_point_bbox helper — single source of truth used
# by AcquisitionService, BaseAcquisitionHandler, and PointDelineator.
# ---------------------------------------------------------------------------

def test_resolve_point_bbox_point_default_buffer(tmp_path):
    """Point domain with only a pour point → 0.01° square (north/west/south/east)."""
    svc = _service(tmp_path, DOMAIN_DEFINITION_METHOD="point", POUR_POINT_COORDS="46.78/-121.75")
    assert svc._resolve_point_bbox() == "46.79/-121.76/46.77/-121.74"


def test_resolve_point_bbox_custom_buffer(tmp_path):
    """POINT_BUFFER_DISTANCE is honored via the canonical domain field."""
    svc = _service(
        tmp_path, DOMAIN_DEFINITION_METHOD="point",
        POUR_POINT_COORDS="50.0/-120.0", POINT_BUFFER_DISTANCE=0.05,
    )
    assert svc._resolve_point_bbox() == "50.05/-120.05/49.95/-119.95"


def test_resolve_point_bbox_non_point_returns_none(tmp_path):
    """Non-point domains do not auto-derive (caller decides how to handle)."""
    svc = _service(tmp_path, DOMAIN_DEFINITION_METHOD="lumped", POUR_POINT_COORDS="50.0/-120.0")
    assert svc._resolve_point_bbox() is None


def test_resolve_point_bbox_missing_pour_point_returns_none(tmp_path):
    """Point domain without a pour point yields None rather than guessing."""
    svc = _service(tmp_path, DOMAIN_DEFINITION_METHOD="point")
    assert svc._resolve_point_bbox() is None


def test_malformed_pour_point_rejected_at_config_validation(tmp_path):
    """A malformed pour point is caught at config validation, never reaching the
    helper — so the helper only ever sees well-formed 'lat/lon' strings."""
    from symfluence.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="POUR_POINT_COORDS"):
        _service(tmp_path, DOMAIN_DEFINITION_METHOD="point", POUR_POINT_COORDS="not-a-coordinate")


def test_resolve_point_bbox_shared_across_call_sites():
    """The helper has a single definition on ConfigMixin (no per-class copies)."""
    from symfluence.core.mixins.config import ConfigMixin
    from symfluence.data.acquisition.base import BaseAcquisitionHandler
    from symfluence.geospatial.geofabric.delineators.point_delineator import PointDelineator

    for cls in (AcquisitionService, BaseAcquisitionHandler, PointDelineator):
        assert cls._resolve_point_bbox is ConfigMixin._resolve_point_bbox
