# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Self-test for scripts/check_manifest_consistency.py (review item 16 guard)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_GUARD_PATH = Path(__file__).resolve().parents[3] / "scripts" / "check_manifest_consistency.py"
_spec = importlib.util.spec_from_file_location("check_manifest_consistency", _GUARD_PATH)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


class TestNormalize:
    def test_case_and_underscore(self):
        assert guard.normalize("netCDF4") == "netcdf4"
        assert guard.normalize("pint_xarray") == "pint-xarray"
        assert guard.normalize("SALib") == "salib"

    def test_strips_channel_prefix(self):
        assert guard.normalize("pytorch::pytorch") == "pytorch"

    def test_strips_version_and_extras(self):
        assert guard.normalize("numpy>=2.0.0,<3.0.0") == "numpy"
        assert guard.normalize("gdal[extra]>=3.0") == "gdal"
        assert guard.normalize("requests ; python_version>'3'") == "requests"


class TestParseBounds:
    def test_lower_and_upper(self):
        assert guard.parse_bounds(">=0.7.0,<1.0.0") == {">=": (0, 7, 0), "<": (1, 0, 0)}

    def test_star_has_no_bounds(self):
        assert guard.parse_bounds("*") == {}

    def test_conda_style(self):
        assert guard.parse_bounds(">=1.6,<2") == {">=": (1, 6), "<": (2,)}


class TestBoundsDisagree:
    def test_equal_after_normalization(self):
        # 2.0.0 == 2.0 ; <3.0.0 == <3
        a = guard.parse_bounds(">=2.0.0,<3.0.0")
        b = guard.parse_bounds(">=2.0,<3")
        assert guard.bounds_disagree(a, b) == []

    def test_real_cdsapi_style_mismatch(self):
        a = guard.parse_bounds(">=0.7.0,<1.0.0")
        b = guard.parse_bounds(">=0.6,<1")
        diffs = guard.bounds_disagree(a, b)
        assert any(">=" in d for d in diffs)

    def test_only_compares_shared_operators(self):
        # pixi pins only an upper bound; pyproject pins both -> no disagreement
        a = guard.parse_bounds(">=1.0,<2.0")
        b = guard.parse_bounds("<2")
        assert guard.bounds_disagree(a, b) == []


def test_repo_manifests_are_consistent():
    """Regression guard: the three real manifests agree on every core package."""
    issues = guard.check_consistency()
    assert issues == [], "Manifest drift detected:\n" + "\n".join(issues)
