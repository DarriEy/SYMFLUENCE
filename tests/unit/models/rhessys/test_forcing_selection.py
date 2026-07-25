# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""RHESSys forcing selection tolerates a shared, multi-remap forcing store.

The basin-averaged store is domain-shared and can hold more than one remap for
the same period. The ``--patched`` RHESSys binary in particular builds a
distributed landscape and drops a multi-HRU remap into the same
``{domain}_{forcing}_remapped_*`` namespace as the lumped 1-HRU basin average.
The reader ``open_canonical_forcing`` aligns everything it is handed, so an
incompatible file crashes on the ``hru {1, 12}`` dimension mismatch. RHESSys
basin-averages every HRU away regardless, so it must select the single-HRU
lumped remap and leave the shared store untouched for other models.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from symfluence.core.exceptions import FileOperationError
from symfluence.models.rhessys.climate_generator import RHESSysClimateGenerator


def _gen(tmp_path, config=None):
    return RHESSysClimateGenerator(config or {}, tmp_path / "domain_test", "test")


def _write_remap(gen, suffix, n_hru, n=120):
    """Write a canonically-named remap with ``n_hru`` spatial points."""
    fdir = gen.forcing_basin_path
    fdir.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2005-01-01", periods=n, freq="h")
    path = fdir / f"test_ERA5_remapped_{suffix}.nc"
    xr.Dataset(
        {
            "precipitation_flux": (("time", "hru"), np.full((n, n_hru), 1e-4)),
            "air_temperature": (("time", "hru"), np.full((n, n_hru), 283.15)),
        },
        coords={"time": times, "hru": list(range(n_hru))},
    ).to_netcdf(path)
    return path


def test_lumped_selection_drops_distributed_remap(tmp_path):
    """A lumped store with a stray 12-HRU remap resolves to the 1-HRU file."""
    gen = _gen(tmp_path)
    lumped = _write_remap(gen, "CDS_2002_2009", n_hru=1)
    _write_remap(gen, "4ae454551262b9b7", n_hru=12)  # patched-distributed stray

    selected = gen._select_forcing_files(list(gen.forcing_basin_path.glob("*.nc")))
    assert selected == [lumped]

    # And the full load path must not raise the hru {1, 12} AlignmentError.
    ds = gen._load_forcing_data(datetime(2005, 1, 1), datetime(2005, 1, 3))
    assert int(ds.sizes.get("hru", 1)) == 1


def test_same_shape_time_chunks_are_kept(tmp_path):
    """Legitimate multi-file forcing of one shape is not discarded."""
    gen = _gen(tmp_path)
    a = _write_remap(gen, "2002_2005", n_hru=1)
    b = _write_remap(gen, "2006_2009", n_hru=1)
    selected = gen._select_forcing_files([b, a])
    assert set(selected) == {a, b}


def test_ambiguous_incompatible_shapes_raise(tmp_path):
    """No single-HRU candidate + mixed shapes must fail loudly, not merge."""
    gen = _gen(tmp_path)
    _write_remap(gen, "grid6", n_hru=6)
    _write_remap(gen, "grid12", n_hru=12)
    with pytest.raises(FileOperationError, match="incompatible spatial shapes"):
        gen._select_forcing_files(list(gen.forcing_basin_path.glob("*.nc")))


def test_single_file_store_is_returned_unchanged(tmp_path):
    """The common single-file case is a byte-for-byte passthrough."""
    gen = _gen(tmp_path)
    only = _write_remap(gen, "CDS_2002_2009", n_hru=1)
    assert gen._select_forcing_files([only]) == [only]
