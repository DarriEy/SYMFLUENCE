# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Logging-protocol tests for the ERA5 ARCO chunk pipeline.

Per docs/adr/0005-logging-level-policy.md, per-chunk progress belongs at
DEBUG; INFO carries only the one-line summaries ("ERA5 chunks: X/N cached,
downloading Y" / "Y chunks downloaded, Z failed").
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from symfluence.data.acquisition.handlers import era5 as era5_module


class TestPerChunkLoggingIsDebug:
    """The per-chunk save line must not be emitted at INFO."""

    def test_saved_chunk_logged_at_debug(self, monkeypatch, tmp_path, caplog):
        logger = logging.getLogger("test_era5_chunk_logging")
        logger.setLevel(logging.DEBUG)

        sentinel_ds = object()
        monkeypatch.setattr(
            era5_module, "_materialize_era5_chunk", lambda *a, **kw: sentinel_ds
        )
        written = {}

        def _fake_to_netcdf(ds, path, encoding=None, logger=None):
            written["path"] = Path(path)

        monkeypatch.setattr(era5_module, "_safe_to_netcdf", _fake_to_netcdf)

        class _FakeChunk:
            sizes = {"time": 4, "latitude": 2, "longitude": 2}
            data_vars = ["airtemp"]

        monkeypatch.setattr(
            era5_module, "era5_to_summa_schema", lambda ds, source, logger=None: _FakeChunk()
        )

        start = pd.Timestamp("2020-01-01")
        end = pd.Timestamp("2020-01-31 23:00")
        with caplog.at_level(logging.DEBUG, logger="test_era5_chunk_logging"):
            idx, chunk_file, status = era5_module._process_era5_chunk_threadsafe(
                1, (start, end), None, 1, tmp_path, "testdom", 3, logger
            )

        assert status == "success"
        assert chunk_file == written["path"]
        saved_records = [
            r for r in caplog.records if "Saved ERA5 chunk" in r.getMessage()
        ]
        assert saved_records, "expected a per-chunk saved record"
        assert all(r.levelno == logging.DEBUG for r in saved_records)
        assert not any(
            r.levelno == logging.INFO and "chunk" in r.getMessage().lower()
            for r in caplog.records
        )
