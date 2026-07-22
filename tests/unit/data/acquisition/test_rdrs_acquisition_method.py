# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
from __future__ import annotations

"""
Tests for pinning the RDRS/CaSR acquisition pathway.

The default resilient behaviour tries OPeNDAP and falls back to the tiled HTTP
archive on error, so which endpoint a run uses depends on runtime network
conditions — two machines can land on different endpoints, which return
slightly different forcing (a large OPeNDAP request can silently truncate).
RDRS_ACQUISITION_METHOD pins the pathway so acquisition is reproducible.
"""

from pathlib import Path

from symfluence.data.acquisition.handlers.rdrs import RDRSAcquirer


def _acquirer(method, opendap_raises=False):
    a = RDRSAcquirer.__new__(RDRSAcquirer)
    a.domain_name = "Test"
    a.start_date = _Stamp(2015)
    a.end_date = _Stamp(2020)
    a.calls = []

    def cfg(_lam, default=None, dict_key=None):
        if dict_key == "RDRS_ACQUISITION_METHOD":
            return method
        if dict_key == "FORCE_DOWNLOAD":
            return True  # always attempt, never short-circuit on an existing file
        return default

    a._get_config_value = cfg

    def opendap(final_file):
        a.calls.append("opendap")
        if opendap_raises:
            raise RuntimeError("simulated OPeNDAP failure")
        return Path("opendap.nc")

    def http(output_dir, final_file):
        a.calls.append("http")
        return Path("http.nc")

    a._download_opendap = opendap
    a._download_http = http
    return a


class _Stamp:
    def __init__(self, year):
        self.year = year


def test_auto_prefers_opendap_then_falls_back(tmp_path):
    a = _acquirer("auto", opendap_raises=True)
    a.download(tmp_path)
    assert a.calls == ["opendap", "http"]


def test_auto_uses_opendap_when_it_succeeds(tmp_path):
    a = _acquirer("auto", opendap_raises=False)
    a.download(tmp_path)
    assert a.calls == ["opendap"]


def test_pin_opendap_never_falls_back(tmp_path):
    a = _acquirer("opendap", opendap_raises=True)
    try:
        a.download(tmp_path)
    except RuntimeError:
        pass
    # OPeNDAP was attempted and the error propagated — no silent tiled fallback.
    assert a.calls == ["opendap"]


def test_pin_tiled_skips_opendap(tmp_path):
    a = _acquirer("tiled", opendap_raises=False)
    a.download(tmp_path)
    assert a.calls == ["http"]


def test_pin_is_case_insensitive(tmp_path):
    a = _acquirer("OPeNDAP", opendap_raises=False)
    a.download(tmp_path)
    assert a.calls == ["opendap"]
