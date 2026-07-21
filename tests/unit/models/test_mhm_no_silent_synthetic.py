# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""mHM must not silently fabricate forcing when the real forcing won't load.

A stray, incompatibly-shaped forcing file once made mHM fall back to synthetic
sine-wave weather and calibrate 1500 iterations against it, reporting a
plausible KGE of -0.17 with no sign the input was fake. A real run must raise;
synthetic forcing is reachable only behind an explicit opt-in.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from symfluence.models.mhm.preprocessor import MHMPreProcessor


class _Harness(MHMPreProcessor):
    """Bypass __init__; exercise only the forcing-fallback decision."""

    def __init__(self):
        self.forcing_basin_path = "/nonexistent/forcing"

    def _get_simulation_dates(self):
        return datetime(2002, 1, 1), datetime(2003, 1, 1)

    def _load_forcing_data(self):
        raise ValueError("conflicting dimension sizes: {1, 12}")  # the real error

    def _generate_synthetic_forcing(self, start_date, end_date):
        self.synthetic_called = True


def test_real_run_raises_instead_of_synthesizing(monkeypatch):
    """Default path: the read error propagates; no synthetic fallback."""
    monkeypatch.delenv("SYMFLUENCE_MHM_ALLOW_SYNTHETIC_FORCING", raising=False)
    h = _Harness()
    with pytest.raises(ValueError, match=r"conflicting dimension sizes: \{1, 12\}"):
        h._generate_forcing_files()
    assert not getattr(h, "synthetic_called", False)


def test_opt_in_allows_synthetic(monkeypatch):
    monkeypatch.setenv("SYMFLUENCE_MHM_ALLOW_SYNTHETIC_FORCING", "1")
    h = _Harness()
    h._generate_forcing_files()  # must not raise
    assert h.synthetic_called is True
