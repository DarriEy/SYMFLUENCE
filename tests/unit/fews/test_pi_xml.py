"""Tests for FEWS PI-XML timeseries reader and writer."""

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.exceptions import PIXMLError
from symfluence.fews.pi_xml import read_pi_xml_timeseries, write_pi_xml_timeseries


class TestReadPIXML:
    def test_read_basic(self, sample_pi_xml):
        ds = read_pi_xml_timeseries(sample_pi_xml)
        assert "P.obs" in ds.data_vars
        assert "T.obs" in ds.data_vars
        assert "time" in ds.coords

    def test_read_timesteps(self, sample_pi_xml):
        ds = read_pi_xml_timeseries(sample_pi_xml)
        assert len(ds.time) == 4

    def test_missing_values_become_nan(self, sample_pi_xml):
        ds = read_pi_xml_timeseries(sample_pi_xml, missing_value=-999.0)
        # Third event is -999.0 -> should be NaN
        assert np.isnan(ds["P.obs"].values[2])

    def test_location_id_in_attrs(self, sample_pi_xml):
        ds = read_pi_xml_timeseries(sample_pi_xml)
        assert ds["P.obs"].attrs["location_id"] == "station_01"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(PIXMLError, match="not found"):
            read_pi_xml_timeseries(tmp_path / "nonexistent.xml")

    def test_malformed_xml(self, tmp_path):
        path = tmp_path / "bad.xml"
        path.write_text("<broken>")
        with pytest.raises(PIXMLError, match="Malformed"):
            read_pi_xml_timeseries(path)

    def test_no_series(self, tmp_path):
        path = tmp_path / "empty.xml"
        path.write_text('<?xml version="1.0"?><TimeSeries></TimeSeries>')
        with pytest.raises(PIXMLError, match="No <series>"):
            read_pi_xml_timeseries(path)


class TestWritePIXML:
    def test_write_basic(self, tmp_path):
        times = np.arange("2023-01-01", "2023-01-01T04:00:00", dtype="datetime64[h]")
        ds = xr.Dataset(
            {"precip": ("time", [1.0, 2.0, 3.0, 4.0])},
            coords={"time": times},
        )
        path = tmp_path / "output.xml"
        write_pi_xml_timeseries(ds, path)
        assert path.exists()

    def test_round_trip(self, tmp_path):
        times = np.arange("2023-06-01", "2023-06-01T03:00:00", dtype="datetime64[h]")
        ds = xr.Dataset(
            {
                "Q": ("time", [10.0, 20.0, 30.0]),
                "H": ("time", [1.5, 2.0, 2.5]),
            },
            coords={"time": times},
        )
        path = tmp_path / "roundtrip.xml"
        write_pi_xml_timeseries(ds, path, location_id="station_99")

        ds2 = read_pi_xml_timeseries(path)
        assert "Q" in ds2.data_vars
        assert "H" in ds2.data_vars
        np.testing.assert_allclose(ds2["Q"].values, [10.0, 20.0, 30.0], rtol=1e-5)

    def test_nan_becomes_missing(self, tmp_path):
        times = np.arange("2023-01-01", "2023-01-01T02:00:00", dtype="datetime64[h]")
        ds = xr.Dataset(
            {"Q": ("time", [1.0, np.nan])},
            coords={"time": times},
        )
        path = tmp_path / "nan.xml"
        write_pi_xml_timeseries(ds, path, missing_value=-999.0)

        ds2 = read_pi_xml_timeseries(path, missing_value=-999.0)
        assert np.isnan(ds2["Q"].values[1])
