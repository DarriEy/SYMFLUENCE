# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
IGNACIO Fire Model Configuration for SYMFLUENCE

The full field set lives in the registered core schema
(:class:`symfluence.core.config.models.model_configs_ml_fire.IGNACIOConfig`)
so flat ``IGNACIO_*`` keys transform into typed ``config.model.ignacio.*``
fields. This module subclasses it to add the conversion to IGNACIO's own
YAML configuration format.
"""
from __future__ import annotations

from typing import Any, Dict

from symfluence.core.config.models.model_configs_ml_fire import (
    IGNACIOConfig as CoreIGNACIOConfig,
)


class IGNACIOConfig(CoreIGNACIOConfig):
    """IGNACIO fire spread model configuration with YAML generation.

    Maps SYMFLUENCE configuration parameters to IGNACIO's config format.
    IGNACIO uses YAML configuration files; this class helps generate them
    from SYMFLUENCE's unified configuration system.
    """

    def to_ignacio_config(self) -> Dict[str, Any]:
        """
        Convert to IGNACIO YAML config format.

        Returns:
            Dictionary that can be written as IGNACIO's YAML config.
        """
        config = {
            'project': {
                'name': self.project_name,
                'output_dir': self.output_dir,
                'random_seed': self.random_seed,
            },
            'crs': {
                'working_crs': self.working_crs,
                'output_crs': self.output_crs or self.working_crs,
            },
            'terrain': {
                'dem_path': self.dem_path,
                'slope_path': self.slope_path,
                'aspect_path': self.aspect_path,
            },
            'fuel': {
                'source_type': self.fuel_source_type,
                'path': self.fuel_path,
                'non_fuel_codes': self.non_fuel_codes,
            },
            'ignition': {
                'source_type': 'shapefile',
                'point_path': self.ignition_shapefile,
                'cause': self.ignition_cause,
                'n_iterations': self.n_iterations,
            },
            'weather': {
                'station_path': self.station_path,
                'calculate_fwi': self.calculate_fwi,
                'fwi_latitude': self.fwi_latitude,
            },
            'fbp': {
                'defaults': {
                    'ffmc': self.default_ffmc,
                    'dmc': self.default_dmc,
                    'dc': self.default_dc,
                    'isi': self.default_isi,
                    'bui': self.default_bui,
                },
                'fmc': self.fmc,
                'curing': self.curing,
            },
            'simulation': {
                'dt': self.dt,
                'max_duration': self.max_duration,
                'n_vertices': self.n_vertices,
                'initial_radius': self.initial_radius,
                'min_ros': self.min_ros,
                'time_varying_weather': self.time_varying_weather,
                'start_datetime': self.ignition_date,
            },
            'output': {
                'save_perimeters': self.save_perimeters,
                'save_ros_grids': self.save_ros_grids,
                'perimeter_format': self.perimeter_format,
                'generate_plots': self.generate_plots,
            },
        }
        return config
