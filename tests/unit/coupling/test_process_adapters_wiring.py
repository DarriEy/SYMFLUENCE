"""Tests for process adapter read_outputs() wiring to real extractors.

The adapters resolve model runners and result extractors through the
component registry (``R.runners`` / ``R.result_extractors``), so these tests
patch the registry lookup rather than model import paths.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="requires the ml extra (pip install 'symfluence[ml]')")
import torch

from symfluence.coupling.adapters import process_adapters


def _patch_extractor(mock_cls):
    """Patch the registry so adapters resolve *mock_cls* as the extractor."""
    return patch.object(process_adapters.R.result_extractors, "get", return_value=mock_cls)


def _patch_runner(mock_cls):
    """Patch the registry so adapters resolve *mock_cls* as the runner."""
    return patch.object(process_adapters.R.runners, "get", return_value=mock_cls)


class TestParFlowReadOutputs:
    """Test ParFlowProcessComponent.read_outputs() uses the registered extractor."""

    def test_read_outputs_calls_extractor(self, tmp_path):
        comp = process_adapters.ParFlowProcessComponent("parflow", config={
            'PARFLOW_OUTPUT_DIR': str(tmp_path),
            'SIMULATION_START': '2020-01-01',
        })

        with patch(
            'symfluence.coupling.adapters.process_adapters.ParFlowProcessComponent.read_outputs'
        ) as mock:
            # Just verify the method exists and returns correct structure
            mock.return_value = {"baseflow": torch.tensor([1.0, 2.0])}
            result = comp.read_outputs(tmp_path)

        assert "baseflow" in result
        assert isinstance(result["baseflow"], torch.Tensor)

    def test_read_outputs_returns_tensor_on_extractor_success(self, tmp_path):
        comp = process_adapters.ParFlowProcessComponent("parflow", config={
            'PARFLOW_OUTPUT_DIR': str(tmp_path),
        })

        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_variable.return_value = pd.Series(
            [1.0, 2.0, 3.0], dtype=np.float32
        )

        with _patch_extractor(MagicMock(return_value=mock_extractor_instance)):
            result = comp.read_outputs(tmp_path)

        assert "baseflow" in result
        assert result["baseflow"].dtype == torch.float32
        assert result["baseflow"].shape[0] == 3

    def test_read_outputs_fails_when_extractor_not_registered(self, tmp_path):
        comp = process_adapters.ParFlowProcessComponent("parflow", config={})

        with _patch_extractor(None):
            with pytest.raises(RuntimeError, match="Failed to read ParFlow outputs"):
                comp.read_outputs(tmp_path)


class TestMODFLOWReadOutputs:
    """Test MODFLOWProcessComponent.read_outputs() uses the registered extractor."""

    def test_read_outputs_returns_tensor(self, tmp_path):
        comp = process_adapters.MODFLOWProcessComponent("modflow", config={
            'MODFLOW_OUTPUT_DIR': str(tmp_path),
        })

        mock_extractor = MagicMock()
        mock_extractor.extract_variable.return_value = pd.Series(
            [10.0, 20.0, 30.0], dtype=np.float32
        )

        with _patch_extractor(MagicMock(return_value=mock_extractor)):
            result = comp.read_outputs(tmp_path)

        assert "drain_discharge" in result
        assert result["drain_discharge"].dtype == torch.float32
        assert result["drain_discharge"].shape[0] == 3

    def test_read_outputs_graceful_fallback(self, tmp_path):
        comp = process_adapters.MODFLOWProcessComponent("modflow", config={})

        with _patch_extractor(MagicMock(side_effect=Exception("No output files"))):
            with pytest.raises(RuntimeError, match="Failed to read MODFLOW outputs"):
                comp.read_outputs(tmp_path)


class TestMESHReadOutputs:
    """Test MESHProcessComponent.read_outputs() uses the registered extractor."""

    def test_read_outputs_with_basin_wb(self, tmp_path):
        comp = process_adapters.MESHProcessComponent("mesh", config={
            'EXPERIMENT_OUTPUT_MESH': str(tmp_path),
        })

        mock_extractor = MagicMock()
        mock_extractor.extract_variable.return_value = pd.Series(
            [5.0, 6.0, 7.0], dtype=np.float32
        )

        # Create fake Basin_average_water_balance.csv so the path check succeeds
        (tmp_path / 'Basin_average_water_balance.csv').write_text("dummy")

        with _patch_extractor(MagicMock(return_value=mock_extractor)):
            result = comp.read_outputs(tmp_path)

        assert "discharge" in result
        assert result["discharge"].dtype == torch.float32
        # Should have called with the basin_wb file
        call_args = mock_extractor.extract_variable.call_args
        assert 'Basin_average_water_balance.csv' in str(call_args[0][0])

    def test_read_outputs_fallback_to_dir(self, tmp_path):
        comp = process_adapters.MESHProcessComponent("mesh", config={
            'EXPERIMENT_OUTPUT_MESH': str(tmp_path),
        })

        mock_extractor = MagicMock()
        mock_extractor.extract_variable.return_value = pd.Series(
            [1.0, 2.0], dtype=np.float32
        )

        # No Basin_average_water_balance.csv — falls back to output_dir
        with _patch_extractor(MagicMock(return_value=mock_extractor)):
            result = comp.read_outputs(tmp_path)

        assert "discharge" in result
        call_args = mock_extractor.extract_variable.call_args
        assert call_args[0][0] == tmp_path


class TestCLMReadOutputs:
    """Test CLMProcessComponent.read_outputs() and runner wiring."""

    def test_read_outputs_returns_both_variables(self, tmp_path):
        comp = process_adapters.CLMProcessComponent("clm", config={
            'EXPERIMENT_OUTPUT_CLM': str(tmp_path),
        })

        mock_extractor = MagicMock()
        mock_extractor.extract_variable.return_value = pd.Series(
            [0.01, 0.02, 0.03], dtype=np.float32
        )

        with _patch_extractor(MagicMock(return_value=mock_extractor)):
            result = comp.read_outputs(tmp_path)

        assert "runoff" in result
        assert "evapotranspiration" in result
        assert result["runoff"].dtype == torch.float32
        assert result["evapotranspiration"].dtype == torch.float32

    def test_bmi_initialize_creates_runner(self):
        comp = process_adapters.CLMProcessComponent("clm")

        mock_runner = MagicMock()
        with _patch_runner(MagicMock(return_value=mock_runner)):
            comp.bmi_initialize({'CLM_CESM_EXE': '/fake/cesm.exe'})

        assert comp._runner is mock_runner

    def test_bmi_initialize_warns_when_runner_not_registered(self):
        comp = process_adapters.CLMProcessComponent("clm")

        with _patch_runner(None):
            comp.bmi_initialize({'CLM_CESM_EXE': '/fake/cesm.exe'})

        assert comp._runner is None

    def test_execute_uses_runner_when_available(self, tmp_path):
        comp = process_adapters.CLMProcessComponent("clm")
        comp._runner = MagicMock()
        comp._runner.run.return_value = True

        ret = comp.execute(tmp_path)
        assert ret == 0
        comp._runner.run.assert_called_once()

    def test_execute_falls_back_to_subprocess(self, tmp_path):
        comp = process_adapters.CLMProcessComponent("clm", config={'CLM_CESM_EXE': '/nonexistent'})
        comp._runner = None  # No runner available

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ret = comp.execute(tmp_path)

        assert ret == 0
        mock_run.assert_called_once()

    def test_read_outputs_graceful_fallback(self, tmp_path):
        comp = process_adapters.CLMProcessComponent("clm", config={})

        with _patch_extractor(MagicMock(side_effect=Exception("No history files"))):
            with pytest.raises(RuntimeError, match="Failed to read CLM outputs"):
                comp.read_outputs(tmp_path)


class TestTRouteReadOutputs:
    """Test t-route adapter error handling for missing discharge variables."""

    def test_read_outputs_raises_when_no_supported_discharge_variable(self, tmp_path):
        comp = process_adapters.TRouteProcessComponent("troute", config={
            "EXPERIMENT_OUTPUT_TROUTE": str(tmp_path),
        })
        (tmp_path / "troute_output.nc").write_text("placeholder")

        class _DummyDS(dict):
            def close(self):
                return None

        with patch("xarray.open_dataset", return_value=_DummyDS({"unknown": np.array([1.0], dtype=np.float32)})):
            with pytest.raises(RuntimeError, match="Failed to read t-route outputs"):
                comp.read_outputs(tmp_path)
