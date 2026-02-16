"""Tests for FEWS pre-adapter."""

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.config import FEWSConfig
from symfluence.fews.exceptions import FEWSAdapterError
from symfluence.fews.pi_diagnostics import DiagnosticsCollector
from symfluence.fews.pre_adapter import FEWSPreAdapter


class TestFEWSPreAdapter:
    def test_pre_adapter_netcdf(self, sample_run_info_xml, sample_netcdf, tmp_work_dir):
        adapter = FEWSPreAdapter(
            run_info_path=sample_run_info_xml,
            data_format="netcdf-cf",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        config, run_info = adapter.run(diag=diag)

        assert run_info.start_time.year == 2023
        assert not diag.has_fatal
        # Forcing should be written to input dir
        assert (tmp_work_dir / "toModel" / "forcing.nc").exists()

    def test_pre_adapter_pi_xml(self, sample_run_info_xml, sample_pi_xml, tmp_work_dir):
        adapter = FEWSPreAdapter(
            run_info_path=sample_run_info_xml,
            data_format="pi-xml",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        config, run_info = adapter.run(diag=diag)

        assert not diag.has_fatal

    def test_pre_adapter_no_forcing(self, sample_run_info_xml, tmp_work_dir):
        # Remove all files from toModel
        for f in (tmp_work_dir / "toModel").iterdir():
            f.unlink()

        adapter = FEWSPreAdapter(
            run_info_path=sample_run_info_xml,
            data_format="netcdf-cf",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        with pytest.raises(FEWSAdapterError, match="No NetCDF files"):
            adapter.run(diag=diag)

    def test_diagnostics_on_failure(self, sample_run_info_xml, tmp_work_dir):
        # Remove all files from toModel
        for f in (tmp_work_dir / "toModel").iterdir():
            f.unlink()

        adapter = FEWSPreAdapter(
            run_info_path=sample_run_info_xml,
            data_format="netcdf-cf",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        try:
            adapter.run(diag=diag)
        except FEWSAdapterError:
            pass

        # Diagnostics should have been populated
        assert len(diag.messages) > 0

    def test_pre_adapter_with_custom_config(self, sample_run_info_xml, sample_netcdf, tmp_work_dir):
        fews_cfg = FEWSConfig(
            work_dir=str(tmp_work_dir),
            data_format="netcdf-cf",
            auto_id_map=True,
        )
        adapter = FEWSPreAdapter(
            run_info_path=sample_run_info_xml,
            fews_config=fews_cfg,
        )
        config, run_info = adapter.run()
        assert run_info is not None
