# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Multi-gauge calibration metrics for distributed FUSE + mizuRoute.

This module re-exports :class:`MultiGaugeMetrics` and
:func:`create_multi_gauge_config` from their canonical shared location
at :mod:`symfluence.optimization.multi_gauge.metrics` for backward
compatibility. New code should import directly from the shared module.

The FUSE-specific :func:`ensure_gauge_segment_mapping` helper remains
defined here.
"""

import logging
from pathlib import Path
from typing import Optional

# Re-export shared multi-gauge utilities for backward compatibility
from symfluence.optimization.multi_gauge.metrics import (  # noqa: F401
    MultiGaugeMetrics,
    create_multi_gauge_config,
)


def ensure_gauge_segment_mapping(
    project_dir: Path,
    lamah_root: Path,
    domain_name: str,
    logger: logging.Logger,
) -> Optional[Path]:
    """Ensure gauge_segment_mapping.csv exists at the canonical project path.

    Generates the mapping by spatial-joining the LaMAH-Ice gauge points
    against the domain's mizuRoute river-basin polygons. Returns the canonical
    path on success; None if either input shapefile is unavailable.

    Delegates to :func:`symfluence.optimization.multi_gauge.gauge_mapping.ensure_gauge_mapping`.
    """
    from symfluence.optimization.multi_gauge.gauge_mapping import ensure_gauge_mapping

    return ensure_gauge_mapping(
        project_dir,
        lamah_root,
        domain_name,
        output_subdir='mizuRoute',
        output_filename='gauge_segment_mapping.csv',
        logger=logger,
        prefer_coastal_basins=False,
    )
