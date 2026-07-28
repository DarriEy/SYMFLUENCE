# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
TRoute Model Preprocessor.

Handles spatial preprocessing and configuration generation for the t-route routing model.
Supports all domain types: distributed, grid-based, point-scale, and lumped-to-distributed.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Any, Dict, List, Optional

import yaml

from symfluence.core.geometry_utils import GeospatialUtilsMixin
from symfluence.core.modeling.base import BaseModelPreProcessor
from symfluence.core.registries import R
from symfluence.models.troute.mixins import TRouteConfigMixin


@R.preprocessors.add('TROUTE')
class TRoutePreProcessor(BaseModelPreProcessor, GeospatialUtilsMixin, TRouteConfigMixin):  # type: ignore[misc]
    """
    A standalone preprocessor for t-route within the SYMFLUENCE framework.

    Supports all SYMFLUENCE domain types (distributed, grid, point, lumped-to-distributed)
    via the shared BaseTopologyGenerator infrastructure with cycle detection/fix.
    """

    MODEL_NAME = "troute"

    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        super().__init__(config, logger)
        self.summa_uses_gru_runoff = False
        self.needs_remap_lumped_distributed = False
        self.subcatchment_weights = None
        self.subcatchment_gru_ids = None

    def run_preprocessing(self):
        """Main entry point for running all t-route preprocessing steps."""
        self.logger.info("--- Starting t-route Preprocessing ---")
        self.copy_base_settings()
        self.create_troute_topology_file()
        self.create_troute_yaml_config()
        self.logger.info("--- t-route Preprocessing Completed Successfully ---")

    def copy_base_settings(self, source_dir: Optional[Path] = None, file_patterns: Optional[List[str]] = None):
        """Copies base settings for t-route from package data."""
        self.logger.info("Copying t-route base settings...")
        from symfluence.core.modeling.base_settings import get_base_settings_dir

        if source_dir:
            return super().copy_base_settings(source_dir, file_patterns)

        try:
            base_settings_path = get_base_settings_dir('troute')
        except FileNotFoundError:
            self.logger.warning("Base settings for t-route not found in package. Skipping copy.")
            return

        self.setup_dir.mkdir(parents=True, exist_ok=True)

        for file in os.listdir(base_settings_path):
            copyfile(base_settings_path / file, self.setup_dir / file)
        self.logger.info("t-route base settings copied.")

    def create_troute_topology_file(self):
        """
        Creates the NetCDF network topology file using t-route's expected NWM variable names.

        Uses the shared BaseTopologyGenerator infrastructure for cycle detection/fix,
        headwater basin handling, and multi-domain-type support.
        """
        from symfluence.models.troute.topology_generator import TRouteTopologyGenerator

        self.logger.info("Creating t-route network topology file...")
        generator = TRouteTopologyGenerator(self)
        topology_data = generator.write_topology_with_shapefiles()

        # Store state flags for config file generation
        self.summa_uses_gru_runoff = topology_data.summa_uses_gru_runoff
        self.needs_remap_lumped_distributed = topology_data.needs_remap_lumped_distributed
        if topology_data.subcatchment_weights is not None:
            self.subcatchment_weights = topology_data.subcatchment_weights
            self.subcatchment_gru_ids = topology_data.subcatchment_gru_ids

    def create_troute_yaml_config(self):
        """Creates the t-route YAML configuration file from SYMFLUENCE config settings."""
        from symfluence.core.modeling.utilities.runoff_loader import get_model_config

        self.logger.info("Creating t-route YAML configuration file...")

        source_model = self.troute_from_model.upper()
        model_cfg = get_model_config(source_model)

        input_dir = self.project_dir / f"simulations/{self.experiment_id}" / model_cfg.output_dir_name
        output_dir = self.project_dir / f"simulations/{self.experiment_id}" / 'troute'
        topology_name = self.troute_topology_file

        # Calculate nts (Number of Timesteps)
        start_dt = datetime.fromisoformat(self.time_start)
        end_dt = datetime.fromisoformat(self.time_end)
        time_step_seconds = self.troute_dt_seconds
        total_seconds = (end_dt - start_dt).total_seconds() + time_step_seconds
        nts = int(total_seconds / time_step_seconds)

        # Determine file pattern from source model config
        file_pattern = model_cfg.output_file_pattern.format(
            experiment_id=self.experiment_id,
            domain_name=self.domain_name,
        )

        config_dict = {
            'log_parameters': {'showtiming': True, 'log_level': 'DEBUG'},
            'network_topology_parameters': {
                'supernetwork_parameters': {
                    'geo_file_path': str(self.setup_dir / topology_name),
                }
            },
            'compute_parameters': {
                'restart_parameters': {'start_datetime': self.time_start},
                'forcing_parameters': {
                    'nts': nts,
                    'qlat_input_folder': str(input_dir),
                    'qlat_file_pattern_filter': file_pattern,
                },
            },
            'output_parameters': {
                'stream_output': {'stream_output_directory': str(output_dir)},
            },
        }

        yaml_filename = self.troute_config_file
        yaml_filepath = self.setup_dir / yaml_filename
        with open(yaml_filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)
        self.logger.info(f"t-route YAML config written to {yaml_filepath}")
