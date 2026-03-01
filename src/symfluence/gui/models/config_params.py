# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Parameterized wrapper exposing the most common SYMFLUENCE config fields
as typed, widget-ready params for the Panel config editor sidebar.
"""

import param


class BasicConfigParams(param.Parameterized):
    """Exposes ~20 key config fields as typed params for auto-generated widgets."""

    # Domain identification
    domain_name = param.String(default='unnamed_domain', doc="Domain / basin name")
    experiment_id = param.String(default='run_1', doc="Experiment identifier")

    # Timing
    time_start = param.String(default='2010-01-01 00:00', doc="Experiment start (YYYY-MM-DD HH:MM)")
    time_end = param.String(default='2020-12-31 23:00', doc="Experiment end (YYYY-MM-DD HH:MM)")
    calibration_period = param.String(
        default='', allow_None=True,
        doc="Calibration period (YYYY-MM-DD, YYYY-MM-DD)"
    )
    evaluation_period = param.String(
        default='', allow_None=True,
        doc="Evaluation period (YYYY-MM-DD, YYYY-MM-DD)"
    )

    # Spatial definition
    definition_method = param.Selector(
        default='lumped',
        objects=['point', 'lumped', 'semidistributed', 'distributed'],
        doc="Domain definition method"
    )
    discretization = param.String(default='lumped', doc="Sub-grid discretization method")
    pour_point_coords = param.String(default='', doc="Pour point as lat/lon (e.g. 51.17/-115.57)")
    bounding_box_coords = param.String(
        default='', allow_None=True,
        doc="Bounding box as north/west/south/east"
    )
    stream_threshold = param.Number(default=5000.0, bounds=(0, None), doc="Stream delineation threshold")

    # Forcing
    forcing_dataset = param.Selector(
        default='ERA5',
        objects=['ERA5', 'RDRS', 'CASR', 'CARRA', 'CERRA', 'MSWEP', 'AORC',
                 'CONUS404', 'HRRR', 'DAYMET', 'NLDAS', 'NLDAS2', 'NEX-GDDP', 'EM-EARTH', 'local'],
        doc="Meteorological forcing dataset"
    )

    # Model
    hydrological_model = param.Selector(
        default='SUMMA',
        objects=['SUMMA', 'FUSE', 'GR', 'HYPE', 'MESH', 'RHESSys', 'NGEN', 'LSTM'],
        doc="Hydrological model"
    )

    # Optimization
    optimization_algorithm = param.Selector(
        default='PSO',
        objects=['PSO', 'DE', 'DDS', 'ASYNC-DDS', 'SCE-UA', 'NSGA-II',
                 'ADAM', 'LBFGS', 'CMA-ES', 'DREAM', 'GLUE',
                 'BASIN-HOPPING', 'NELDER-MEAD', 'GA',
                 'BAYESIAN-OPT', 'MOEAD',
                 'SIMULATED-ANNEALING', 'ABC'],
        doc="Optimization algorithm"
    )
    optimization_metric = param.Selector(
        default='KGE',
        objects=['KGE', 'KGEP', 'NSE', 'RMSE', 'MAE', 'PBIAS', 'R2', 'CORRELATION'],
        doc="Optimization objective metric"
    )
    iterations = param.Integer(default=1000, bounds=(1, None), doc="Number of optimization iterations")
    population_size = param.Integer(default=50, bounds=(2, 10000), doc="Population size")
