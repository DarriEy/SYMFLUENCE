# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Output parsing for the WATFLOOD and GSFLOW standalone postprocessors.

These two models ran and calibrated but had no registered postprocessor
(documented in the INCOMPLETE_MODELS baseline) until the implementations
that already lived in their packages were wired into model_manifest().
The parsers only need a logger, so they are exercised unbound.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def fake_self():
    return SimpleNamespace(logger=__import__("logging").getLogger("test"))


class TestRegistration:
    def test_both_postprocessors_registered(self):
        from symfluence.core.registries import R

        assert R.postprocessors.get("WATFLOOD") is not None
        assert R.postprocessors.get("GSFLOW") is not None

    def test_validate_model_reports_complete(self):
        from symfluence.core.registries import R

        for model in ("WATFLOOD", "GSFLOW"):
            report = R.validate_model(model)
            assert report.get("postprocessor") or report.get("valid"), report


class TestWATFLOODTb0Parsing:
    def test_parses_tb0_timeseries(self, tmp_path, fake_self):
        from symfluence.models.watflood.postprocessor import WATFLOODPostProcessor

        tb0 = tmp_path / "spl_test.tb0"
        tb0.write_text(
            ":FileType tb0  ASCII\n"
            ":Name streamflow\n"
            "#\n"
            "2020 01 01 00 1.25\n"
            "2020 01 02 00 2.50\n"
            "2020 01 03 00 3.75\n"
        )
        series = WATFLOODPostProcessor._parse_tb0(fake_self, tb0)
        assert series is not None
        assert len(series) == 3
        assert series.iloc[1] == 2.50
        assert str(series.index[0].date()) == "2020-01-01"

    def test_returns_none_on_garbage(self, tmp_path, fake_self):
        from symfluence.models.watflood.postprocessor import WATFLOODPostProcessor

        bad = tmp_path / "spl_bad.tb0"
        bad.write_text(":Header only\n# no data lines\n")
        assert WATFLOODPostProcessor._parse_tb0(fake_self, bad) is None


class TestGSFLOWStatvarParsing:
    def test_parses_statvar_with_cfs_conversion(self, tmp_path, fake_self):
        from symfluence.models.gsflow.postprocessor import GSFLOWPostProcessor

        statvar = tmp_path / "statvar.dat"
        # statvar data rows: idx year month day hour min sec val1 val2 ...
        # the parser reads the date from parts[1:4] and flow (cfs) from parts[7]
        statvar.write_text(
            "1 2020 1 1 0 0 0 100.0 5.0\n"
            "2 2020 1 2 0 0 0 200.0 6.0\n"
        )
        series = GSFLOWPostProcessor._extract_from_file(fake_self, statvar)
        assert series is not None
        assert len(series) == 2
        # 100 cfs * 0.0283168 = 2.83168 cms
        assert series.iloc[0] == pytest.approx(2.83168)

    def test_csv_seg_outflow_passthrough(self, tmp_path, fake_self):
        import pandas as pd

        from symfluence.models.gsflow.postprocessor import GSFLOWPostProcessor

        csv = tmp_path / "gsflow_out.csv"
        pd.DataFrame(
            {"date": ["2020-01-01", "2020-01-02"], "seg_outflow": [1.0, 2.0]}
        ).to_csv(csv, index=False)
        series = GSFLOWPostProcessor._extract_from_file(fake_self, csv)
        assert series is not None
        # seg_outflow is already cms — no conversion
        assert list(series) == [1.0, 2.0]
