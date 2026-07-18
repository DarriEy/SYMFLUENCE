# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Regression tests for degenerate-forcing detection in the CaSR/RDRS acquirer.

The tiled CaSR archive once returned a downwelling-longwave field that had the
right name, units and shape but was entirely zero. Acquisition accepted it, and
the failure only surfaced much later in SUMMA's preprocessing guard — or not at
all for models without one. These tests pin the behaviour that such a variable
is rejected at acquisition, while incidental all-zero fields (freezing rain in
summer, land masks) are left alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.data.acquisition.handlers.rdrs import (
    DEFAULT_CASR_V32_VARIABLES,
    RDRSAcquirer,
)


def _acquirer():
    """An RDRSAcquirer with only the fields the validator needs."""
    probe = RDRSAcquirer.__new__(RDRSAcquirer)
    probe.start_date = pd.Timestamp('2002-01-01')
    probe.end_date = pd.Timestamp('2009-12-31')
    probe._casr_variables = lambda: [f'CaSR_v3.2_{n}' for n in DEFAULT_CASR_V32_VARIABLES]
    return probe


def _dataset(**variables):
    time = pd.date_range('2005-06-01', periods=24, freq='h')
    return xr.Dataset(
        {
            name: (('time',), np.asarray(values, dtype='float64'), attrs)
            for name, (values, attrs) in variables.items()
        },
        coords={'time': time},
    )


def _plausible(n=24):
    return np.linspace(220.0, 340.0, n)


def test_all_zero_requested_variable_is_rejected():
    ds = _dataset(rlds=(np.zeros(24), {'original_variable': 'P_FI_SFC'}))

    with pytest.raises(RuntimeError, match='rlds'):
        _acquirer()._validate_acquired_variables(ds, source='test')


def test_all_nan_requested_variable_is_rejected():
    ds = _dataset(rlds=(np.full(24, np.nan), {'original_variable': 'P_FI_SFC'}))

    with pytest.raises(RuntimeError, match='missing/NaN'):
        _acquirer()._validate_acquired_variables(ds, source='test')


def test_raw_archive_names_are_matched_without_cf_rename():
    """Tiled files may keep the raw CaSR name instead of the CF alias."""
    ds = _dataset(CaSR_v3_2_P_FI_SFC=(np.zeros(24), {}))
    ds = ds.rename({'CaSR_v3_2_P_FI_SFC': 'CaSR_v3.2_P_FI_SFC'})

    with pytest.raises(RuntimeError, match='P_FI_SFC'):
        _acquirer()._validate_acquired_variables(ds, source='test')


def test_populated_variable_passes():
    ds = _dataset(rlds=(_plausible(), {'original_variable': 'P_FI_SFC'}))

    _acquirer()._validate_acquired_variables(ds, source='test')


def test_unrequested_all_zero_variable_is_ignored():
    """Freezing-rain rate is legitimately zero outside freezing-rain events."""
    ds = _dataset(
        rlds=(_plausible(), {'original_variable': 'P_FI_SFC'}),
        prrpmod=(np.zeros(24), {}),
    )

    _acquirer()._validate_acquired_variables(ds, source='test')


def test_static_field_without_time_dimension_is_ignored():
    ds = _dataset(rlds=(_plausible(), {'original_variable': 'P_FI_SFC'}))
    ds['sftlkf'] = ((), 0.0)

    _acquirer()._validate_acquired_variables(ds, source='test')


def test_error_names_every_degenerate_variable():
    ds = _dataset(
        rlds=(np.zeros(24), {'original_variable': 'P_FI_SFC'}),
        rsds=(np.zeros(24), {'original_variable': 'P_FB_SFC'}),
    )

    with pytest.raises(RuntimeError) as excinfo:
        _acquirer()._validate_acquired_variables(ds, source='test')

    assert 'rlds' in str(excinfo.value)
    assert 'rsds' in str(excinfo.value)
