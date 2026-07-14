# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Access-pattern contract for the ARCO-ERA5 acquisition path.

The ARCO store is chunked (1 hour, 721 lat, 1440 lon): every Zarr chunk read is a
full global field.  The transfer therefore scales with the number of
(variable, hour) chunks the dask graph touches, and *not* with domain size.  Two
properties keep that number minimal, and both are pinned here:

* the store is opened once per acquisition, and the variable + spatial subset is
  applied to that single lazy view before any time slicing;
* each (variable, hour) chunk is fetched exactly once — the subset is
  materialised a single time, so de-accumulation and the longwave sanity check
  cannot trigger a second pass over the network.
"""
from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest
import xarray as xr

from symfluence.data.acquisition.handlers import era5 as era5_mod

ARCO_VARS = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
]
# Variables the ARCO store carries but this acquisition never asks for.  They
# stand in for the ~265 unused arrays on the real store.
DECOY_VARS = ["sea_surface_temperature", "snow_depth"]

LATS = np.array([47.25, 47.0, 46.75, 46.5])   # descending, ERA5 convention
LONS = np.array([237.75, 238.0, 238.25, 238.5])  # 0-360 convention


class _ChunkCountingSource:
    """numpy-array facade that records every chunk read dask performs."""

    def __init__(self, values: np.ndarray, name: str, reads: list):
        self._values = values
        self._name = name
        self._reads = reads
        self.shape = values.shape
        self.dtype = values.dtype
        self.ndim = values.ndim

    def __getitem__(self, key):
        # dask hands us one slice tuple per chunk it materialises.  Zero-size
        # reads are dask's dtype/meta probes, not data transfer.
        block = self._values[key]
        if block.size:
            self._reads.append((self._name, key))
        return block


def _make_fake_arco_store(reads: list) -> xr.Dataset:
    """Synthetic ARCO-shaped store: hourly, globe-per-chunk, dask-backed."""
    import dask.array as dsa

    times = xr.date_range("2015-12-31T23:00", "2016-02-29T23:00", freq="h",
                          use_cftime=False)
    shape = (len(times), len(LATS), len(LONS))
    rng = np.random.default_rng(0)

    data_vars = {}
    for name in ARCO_VARS + DECOY_VARS:
        if "temperature" in name:
            values = rng.uniform(260, 290, shape)
        elif name == "surface_pressure":
            values = rng.uniform(85000, 101000, shape)
        elif name == "surface_thermal_radiation_downwards":
            # accumulated J/m2; must de-accumulate to a sane LW flux
            values = np.cumsum(rng.uniform(250 * 3600, 350 * 3600, shape), axis=0)
        elif "radiation" in name:
            values = np.cumsum(rng.uniform(0, 400 * 3600, shape), axis=0)
        elif name == "total_precipitation":
            values = np.cumsum(rng.uniform(0, 5e-4, shape), axis=0)
        else:
            values = rng.uniform(-8, 8, shape)
        values = values.astype("float32")
        source = _ChunkCountingSource(values, name, reads)
        # Chunk exactly like ARCO: one hour x the whole grid.
        arr = dsa.from_array(source, chunks=(1, len(LATS), len(LONS)), name=name,
                             asarray=False)
        data_vars[name] = (("time", "latitude", "longitude"), arr)

    return xr.Dataset(
        data_vars,
        coords={"time": times, "latitude": LATS, "longitude": LONS},
    )


@pytest.fixture()
def arco_probe(monkeypatch):
    """Install a fake gcsfs + a counting fake ARCO store; report what was read."""
    reads: list = []
    opens: list = []

    fake_gcsfs = types.ModuleType("gcsfs")

    class _FS:
        def __init__(self, *a, **kw):
            pass

        def get_mapper(self, path):
            return {"__store__": path}

    fake_gcsfs.GCSFileSystem = _FS
    monkeypatch.setitem(sys.modules, "gcsfs", fake_gcsfs)

    def _fake_open_zarr(mapper, **kwargs):
        opens.append(mapper)
        return _make_fake_arco_store(reads)

    monkeypatch.setattr(era5_mod.xr, "open_zarr", _fake_open_zarr)
    return types.SimpleNamespace(reads=reads, opens=opens)


def _config(tmp_path, end="2016-02-29 23:00"):
    return {
        "DOMAIN_NAME": "paradise_point",
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "SYMFLUENCE_CODE_DIR": str(tmp_path),
        "FORCING_DATASET": "ERA5",
        # one ERA5 cell (46.75 / 238.25 in 0-360 == -121.75)
        "BOUNDING_BOX_COORDS": "46.80/-121.80/46.72/-121.70",
        "EXPERIMENT_TIME_START": "2016-01-01 01:00",
        "EXPERIMENT_TIME_END": end,
    }


def _run(tmp_path):
    acq = era5_mod.ERA5ARCOAcquirer(_config(tmp_path), logging.getLogger("test"))
    return acq.download(tmp_path / "forcing")


def test_store_opened_once_for_the_whole_acquisition(arco_probe, tmp_path):
    """Two monthly chunks must not re-open (and re-index) the Zarr store."""
    _run(tmp_path)
    assert len(arco_probe.opens) == 1


def test_unrequested_variables_are_never_fetched(arco_probe, tmp_path):
    """Variable selection happens before materialisation, so decoys stay untouched."""
    _run(tmp_path)
    touched = {name for name, _ in arco_probe.reads}
    assert touched.isdisjoint(DECOY_VARS)
    assert touched == set(ARCO_VARS)


def test_subset_applied_before_materialisation(arco_probe, tmp_path, monkeypatch):
    """Schema conversion must receive an in-memory, already-narrowed dataset.

    If the dataset reaching era5_to_summa_schema is still dask-backed, the
    de-accumulation and the longwave sanity mean each trigger their own pass over
    the store; if it is not yet spatially narrowed, the whole globe is in hand.
    """
    seen = []
    orig = era5_mod.era5_to_summa_schema

    def _spy(ds, **kwargs):
        seen.append(ds)
        return orig(ds, **kwargs)

    monkeypatch.setattr(era5_mod, "era5_to_summa_schema", _spy)
    _run(tmp_path)

    assert seen, "expected schema conversion to run"
    for ds in seen:
        assert ds.sizes["latitude"] == 1 and ds.sizes["longitude"] == 1
        assert set(ds.data_vars) == set(ARCO_VARS)
        assert all(v.chunks is None for v in ds.data_vars.values()), "still lazy"


def test_each_chunk_fetched_exactly_once(arco_probe, tmp_path):
    """No Zarr chunk may be pulled twice within one output chunk.

    De-accumulation, the longwave sanity mean and the NetCDF write must all be
    served from the single materialisation, not from three passes over the store.
    """
    acq = era5_mod.ERA5ARCOAcquirer(_config(tmp_path, end="2016-01-31 23:00"),
                                    logging.getLogger("test"))
    acq.download(tmp_path / "forcing")

    seen: dict = {}
    for name, key in arco_probe.reads:
        seen[(name, str(key))] = seen.get((name, str(key)), 0) + 1
    duplicated = {k: v for k, v in seen.items() if v > 1}
    assert not duplicated, f"chunks fetched more than once: {sorted(duplicated)[:5]}"
    # 744 hourly chunks x 8 variables, and nothing more.
    assert len(arco_probe.reads) == 744 * len(ARCO_VARS)


def test_outputs_keep_their_filenames_and_schema(arco_probe, tmp_path):
    """The fix must not move the goalposts: same files, same variables."""
    _run(tmp_path)
    out = sorted((tmp_path / "forcing").glob("*.nc"))
    assert [p.name for p in out] == [
        "domain_paradise_point_ERA5_merged_201601.nc",
        "domain_paradise_point_ERA5_merged_201602.nc",
    ]
    with xr.open_dataset(out[0]) as ds:
        assert set(ds.data_vars) == {
            "air_temperature",
            "surface_air_pressure",
            "wind_speed",
            "specific_humidity",
            "precipitation_flux",
            "surface_downwelling_shortwave_flux",
            "surface_downwelling_longwave_flux",
        }
        assert ds.sizes["latitude"] == 1
        assert ds.sizes["longitude"] == 1
        # January's 744 hours minus the leading step consumed by de-accumulation
        assert ds.sizes["time"] == 743


def test_existing_chunk_files_are_skipped(arco_probe, tmp_path):
    """skip-if-exists still short-circuits the download."""
    _run(tmp_path)
    arco_probe.reads.clear()
    _run(tmp_path)
    assert arco_probe.reads == []
