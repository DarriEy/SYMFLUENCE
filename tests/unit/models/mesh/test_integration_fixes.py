# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>
"""
Regression tests for MESH distributed-domain integration fixes.

Covers three defects that blocked a multi-cell (routed) MESH setup:
  * the meshflow ``extract_rank_next`` monkey-patch rebound only the defining
    module, not the ``meshflow.utility`` package re-export that meshflow.core
    actually calls (routing-topology build crashed on every multi-cell domain);
  * ``copy_settings_to_forcing`` copied a shipped ``MESH_input_streamflow.txt``
    template over the per-domain gauge file generated with the correct outlet
    Rank (so MESH produced no routed streamflow output);
  * the runner accepted only a per-gauge streamflow CSV in routed mode, not the
    always-produced basin-average water balance.
"""
from __future__ import annotations

import logging

import pytest


class TestMeshflowExtractRankNextPatch:
    """The runtime patch must rebind BOTH meshflow bindings of the function."""

    def test_both_bindings_patched(self):
        meshflow_network = pytest.importorskip("meshflow.utility.network")
        import meshflow.utility as meshflow_utility

        # Importing the manager applies the patch at import time.
        import symfluence.models.mesh.preprocessing.meshflow_manager  # noqa: F401

        patched = meshflow_network.extract_rank_next
        # The package-level re-export (what meshflow.core calls) must be the
        # same patched object, not the stale original from ``import *``.
        assert meshflow_utility.extract_rank_next is patched


class TestStreamflowFileNotClobbered:
    """copy_settings_to_forcing must not overwrite the generated gauge file."""

    def test_streamflow_txt_is_skipped(self, tmp_path):
        from symfluence.models.mesh.preprocessing.data_preprocessor import (
            MESHDataPreprocessor,
        )

        setup_dir = tmp_path / "settings"
        forcing_dir = tmp_path / "forcing"
        setup_dir.mkdir()
        forcing_dir.mkdir()

        # Shipped template in settings (placeholder gauge coords).
        (setup_dir / "MESH_input_streamflow.txt").write_text(
            "#template\n2 0 0 24 1980 1 0\n3070.3 -6934.3 05BB001\n"
        )
        # A plain settings file that SHOULD be copied.
        (setup_dir / "MESH_input_soil_levels.txt").write_text("3\n")
        # The correct per-domain gauge file already generated in forcing.
        good = "#domain gauge\n1 0 0 24 2004 1 1\n1 49 05BB001\n"
        (forcing_dir / "MESH_input_streamflow.txt").write_text(good)

        pre = MESHDataPreprocessor(
            forcing_dir=forcing_dir, setup_dir=setup_dir,
            config={"HYDROLOGICAL_MODEL": "MESH"}, logger=logging.getLogger("t"),
        )
        pre.copy_settings_to_forcing()

        # Generated gauge file survived (not clobbered by the template).
        assert (forcing_dir / "MESH_input_streamflow.txt").read_text() == good
        # Other settings files were still copied through.
        assert (forcing_dir / "MESH_input_soil_levels.txt").exists()


class TestRunnerAcceptsBasinAverageWhenRouted:
    """In routed (multi-cell) mode the runner must accept the basin-average
    water balance, not only a per-gauge streamflow CSV (which MESH may not
    write on a subbasin domain)."""

    def test_basin_average_accepted_in_routing_mode(
        self, mesh_config, mock_logger, setup_mesh_directories
    ):
        from unittest.mock import patch

        from symfluence.models.mesh.runner import MESHRunner

        runner = MESHRunner(mesh_config, mock_logger)
        runner.output_dir = setup_mesh_directories['simulations_dir']
        runner.forcing_mesh_path = setup_mesh_directories['forcing_dir']

        # Only the basin-average water balance exists (no per-gauge CSV).
        wb = runner.forcing_mesh_path / 'Basin_average_water_balance.csv'
        wb.write_text(
            "YEAR,JDAY,RFF,LKG\n" + "".join(f"2004,{d},1.0,0.5\n" for d in range(1, 20))
        )

        # Force routed (non-lumped) mode and skip the day-coverage check.
        with patch.object(runner, '_is_lumped_mode', return_value=False), \
             patch.object(runner, '_get_expected_days', return_value=None):
            assert runner._verify_outputs() is True
