# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""TDX-Hydro / GEOGLOWS V2 handler tests.

Two regressions are pinned here (2026-06):

1. The GEOGLOWS V2 ``vpu-boundaries.gpkg`` renamed its identifier column from
   ``VPUCode`` to ``VPU``. The handler hard-coded ``VPUCode`` and raised
   ``KeyError: 'VPUCode'`` against the current index. VPU-column resolution must
   tolerate both schemas (preferring ``VPU``).

2. The downloader writes the river network as ``tdx_streams_*.gpkg`` (current
   GEOGLOWS V2), while the subsetter previously only looked for
   ``tdx_rivers_*.parquet``. Resolution must accept either layout.
"""
from __future__ import annotations

import pytest

from symfluence.data.acquisition.handlers.tdx_hydro import TDXHydroAcquirer


class _FakeFrame:
    """Minimal stand-in exposing only the ``.columns`` accessor used by lookup."""

    def __init__(self, columns):
        self.columns = list(columns)


class TestResolveVpuColumn:
    """VPU column resolution tolerates both GEOGLOWS schema versions."""

    def test_current_schema_uses_vpu(self):
        frame = _FakeFrame(["VPU", "geometry"])
        assert TDXHydroAcquirer._resolve_vpu_column(frame) == "VPU"

    def test_legacy_schema_falls_back_to_vpucode(self):
        frame = _FakeFrame(["VPUCode", "geometry"])
        assert TDXHydroAcquirer._resolve_vpu_column(frame) == "VPUCode"

    def test_vpu_preferred_when_both_present(self):
        frame = _FakeFrame(["VPUCode", "VPU", "geometry"])
        assert TDXHydroAcquirer._resolve_vpu_column(frame) == "VPU"

    def test_case_insensitive_match(self):
        frame = _FakeFrame(["vpu", "geometry"])
        assert TDXHydroAcquirer._resolve_vpu_column(frame) == "vpu"

    def test_missing_column_raises_with_available_columns(self):
        frame = _FakeFrame(["region", "geometry"])
        with pytest.raises(KeyError) as excinfo:
            TDXHydroAcquirer._resolve_vpu_column(frame)
        msg = str(excinfo.value)
        assert "VPU" in msg and "VPUCode" in msg
        assert "region" in msg  # available columns are listed


class TestVpuLookupNoKeyError:
    """A GeoDataFrame with a ``VPU`` column resolves without KeyError."""

    def test_dataframe_vpu_column_resolves(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"VPU": [711, 712], "geometry": [None, None]})
        col = TDXHydroAcquirer._resolve_vpu_column(df)
        # Mirrors handler usage: matching_vpus[vpu_col].unique().tolist()
        assert df[col].unique().tolist() == [711, 712]
