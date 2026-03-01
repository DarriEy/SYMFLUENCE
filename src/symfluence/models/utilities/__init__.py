# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Shared utilities for model preprocessors.

Provides common functionality for time window management, forcing data
processing, data quality handling, and dataset alignment that is used
across multiple model preprocessors (SUMMA, FUSE, NGEN, GR, MESH).

Also includes RoutingDecider for unified routing decision logic across models.
"""

from symfluence.data.preprocessing.dataset_alignment_manager import DatasetAlignmentManager, align_forcing_datasets
from symfluence.data.preprocessing.time_window_manager import TimeWindowManager

from .base_forcing_processor import BaseForcingProcessor
from .data_quality_handler import DataQualityHandler
from .forcing_data_processor import ForcingDataProcessor
from .routing_decider import RoutingDecider

__all__ = [
    'TimeWindowManager',
    'ForcingDataProcessor',
    'DataQualityHandler',
    'DatasetAlignmentManager',
    'align_forcing_datasets',
    'BaseForcingProcessor',
    'RoutingDecider',
]
