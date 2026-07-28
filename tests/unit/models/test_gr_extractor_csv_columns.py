# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""GRResultExtractor resolves the lumped CSV column by name, not by position.

``GRRunner`` writes ``GR_results.csv`` as ``data.frame(datetime=...,
q_sim = OutputsModel$Qsim)``, so the streamflow column is ``q_sim``. The
extractor's candidate list began at ``Qsim`` -- the airGR R *object* field
name, which never appears as a column header -- so every real GR run matched
nothing and fell through to "first column, whatever it is named". That worked
only because ``q_sim`` happens to be first today; any extra column written
ahead of it would have silently returned the wrong series.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from symfluence.models.gr.extractor import GRResultExtractor

pytestmark = [pytest.mark.unit]


def _write_csv(path: Path, columns: dict) -> Path:
    out = path / 'GR_results.csv'
    pd.DataFrame(
        {'datetime': pd.date_range('2010-01-01', periods=4, freq='D'), **columns}
    ).to_csv(out, index=False)
    return out


def test_extracts_the_column_gr_actually_writes(tmp_path):
    """The real GR_results.csv layout resolves to q_sim."""
    csv = _write_csv(tmp_path, {'q_sim': [1.0, 2.0, 3.0, 4.0]})

    series = GRResultExtractor('GR').extract_variable(csv, 'streamflow')

    assert list(series) == [1.0, 2.0, 3.0, 4.0]
    assert series.name == 'q_sim'


def test_named_lookup_beats_column_order(tmp_path):
    """A column ahead of q_sim must not be mistaken for streamflow.

    This is the case the positional fallback got wrong: with the old candidate
    list nothing matched by name, so the first column won regardless of meaning.
    """
    csv = _write_csv(
        tmp_path,
        {'precip': [9.0, 9.0, 9.0, 9.0], 'q_sim': [1.0, 2.0, 3.0, 4.0]},
    )

    series = GRResultExtractor('GR').extract_variable(csv, 'streamflow')

    assert series.name == 'q_sim'
    assert list(series) == [1.0, 2.0, 3.0, 4.0]


def test_q_sim_is_the_first_streamflow_candidate():
    """Ordering is the fix: the name GR writes must be tried first."""
    names = GRResultExtractor('GR').get_variable_names('streamflow')

    assert names[0] == 'q_sim'
    # Tolerant fallbacks for hand-edited / external output are kept.
    assert 'Qsim' in names


def test_q_routed_stays_out_of_the_shared_candidate_list():
    """The list is shared with the NetCDF path, which cannot aggregate GRUs.

    ``_extract_from_netcdf`` reduces a multi-GRU variable with ``isel(gru=0)``
    while ``GRPostProcessor`` sums over ``gru``. Listing ``q_routed`` here would
    trade an explicit error for a silently wrong single-GRU hydrograph.
    """
    assert 'q_routed' not in GRResultExtractor('GR').get_variable_names('streamflow')
