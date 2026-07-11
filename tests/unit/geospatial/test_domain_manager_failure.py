# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression tests for DomainManager.define_domain failure handling.

When the underlying delineator produces no domain (e.g. TauDEM is not installed
so lumped/semi-distributed watershed delineation cannot run), define_domain must
fail loudly rather than reporting the step complete and letting the downstream
discretize_domain step die with a confusing "no such file" error.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from symfluence.core.exceptions import GeospatialError
from symfluence.geospatial.domain_manager import DomainManager


def _manager_with_delineator_result(result, artifacts):
    """Build a DomainManager without running its heavy __init__."""
    mgr = DomainManager.__new__(DomainManager)
    mgr._config = MagicMock()  # config is a property over _config
    mgr.logger = MagicMock()
    mgr.reporting_manager = None
    mgr.domain_delineator = MagicMock()
    mgr.domain_delineator.define_domain.return_value = (result, artifacts)
    return mgr


class TestDefineDomainFailure:
    def test_raises_when_delineation_produces_no_domain(self):
        """A falsy delineator result must raise a clear, actionable error."""
        artifacts = MagicMock()
        artifacts.river_basins_path = None
        mgr = _manager_with_delineator_result(None, artifacts)

        with pytest.raises(GeospatialError) as exc_info:
            mgr.define_domain()

        msg = str(exc_info.value).lower()
        # The message should point the user at the most common cause + fix.
        assert "taudem" in msg
        assert "domain definition failed" in msg

    def test_succeeds_when_delineation_returns_a_result(self):
        """A truthy result returns normally (no raise) and reports success."""
        artifacts = MagicMock()
        artifacts.river_basins_path = None  # skip diagnostic-plot branch
        result = "shapefiles/river_basins/domain_riverBasins_lumped.shp"
        mgr = _manager_with_delineator_result(result, artifacts)

        out_result, out_artifacts = mgr.define_domain()

        assert out_result == result
        assert out_artifacts is artifacts
