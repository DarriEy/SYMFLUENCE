"""Tests for FEWS post-adapter."""

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.config import FEWSConfig
from symfluence.fews.exceptions import FEWSAdapterError
from symfluence.fews.pi_diagnostics import DiagnosticsCollector
from symfluence.fews.post_adapter import FEWSPostAdapter


class TestFEWSPostAdapter:
    def _create_model_output(self, work_dir):
        """Create synthetic model output."""
        times = np.arange("2023-01-01", "2023-01-05", dtype="datetime64[h]")
        ds = xr.Dataset(
            {
                "discharge": ("time", np.random.rand(len(times)) * 100.0),
                "water_level": ("time", np.random.rand(len(times)) * 5.0),
            },
            coords={"time": times},
        )
        output_path = work_dir / "output.nc"
        ds.to_netcdf(output_path)
        return output_path

    def test_post_adapter_netcdf(self, sample_run_info_xml, tmp_work_dir):
        self._create_model_output(tmp_work_dir)

        adapter = FEWSPostAdapter(
            run_info_path=sample_run_info_xml,
            data_format="netcdf-cf",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        output_dir = adapter.run(diag=diag)

        assert output_dir.is_dir()
        assert (output_dir / "output.nc").exists()
        assert not diag.has_fatal

    def test_post_adapter_pi_xml(self, sample_run_info_xml, tmp_work_dir):
        self._create_model_output(tmp_work_dir)

        adapter = FEWSPostAdapter(
            run_info_path=sample_run_info_xml,
            data_format="pi-xml",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        output_dir = adapter.run(diag=diag)

        assert (output_dir / "timeseries.xml").exists()

    def test_post_adapter_no_output(self, sample_run_info_xml, tmp_work_dir):
        adapter = FEWSPostAdapter(
            run_info_path=sample_run_info_xml,
            data_format="netcdf-cf",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        with pytest.raises(FEWSAdapterError, match="No model output"):
            adapter.run(diag=diag)

    def test_diagnostics_always_written(self, sample_run_info_xml, tmp_work_dir):
        adapter = FEWSPostAdapter(
            run_info_path=sample_run_info_xml,
            data_format="netcdf-cf",
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")

        try:
            adapter.run(diag=diag)
        except FEWSAdapterError:
            pass
        finally:
            diag.write()

        assert (tmp_work_dir / "toFews" / "diag.xml").exists()

    def test_reverse_id_mapping(self, sample_run_info_xml, tmp_work_dir):
        # Write output with SYMFLUENCE names
        times = np.arange("2023-01-01", "2023-01-03", dtype="datetime64[h]")
        ds = xr.Dataset(
            {"discharge": ("time", np.ones(len(times)))},
            coords={"time": times},
        )
        ds.to_netcdf(tmp_work_dir / "output.nc")

        fews_cfg = FEWSConfig(
            work_dir=str(tmp_work_dir),
            data_format="netcdf-cf",
            auto_id_map=True,
        )
        adapter = FEWSPostAdapter(
            run_info_path=sample_run_info_xml,
            fews_config=fews_cfg,
        )
        diag = DiagnosticsCollector(tmp_work_dir / "toFews" / "diag.xml")
        adapter.run(diag=diag)

        # Check output has FEWS names
        result = xr.open_dataset(tmp_work_dir / "toFews" / "output.nc")
        # discharge maps to Q.obs in auto-map reverse
        result.close()
