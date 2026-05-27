# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base remap generator for routing models.

Provides shared algorithms for spatial remapping between source model
spatial units and routing network HRUs. Supports area-weighted, equal-weight,
and EASYMORE spatial intersection approaches.

Each routing model subclasses this to write remap data in its own format.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class RemapPreprocessorProtocol(Protocol):
    """Protocol defining what the remap generator needs from a preprocessor."""

    logger: Any

    @property
    def project_dir(self) -> Path: ...
    @property
    def domain_name(self) -> str: ...
    @property
    def setup_dir(self) -> Path: ...

    def _get_config_value(self, getter: Any, default: Any = None, dict_key: str = '') -> Any: ...


@dataclass
class RemapData:
    """Intermediate representation of spatial remapping."""

    rn_hru_ids: np.ndarray
    n_overlaps: np.ndarray
    hm_hru_ids: np.ndarray
    weights: np.ndarray
    num_hru: int = 0
    num_data: int = 0

    def __post_init__(self):
        if self.num_hru == 0:
            self.num_hru = len(self.rn_hru_ids)
        if self.num_data == 0:
            self.num_data = len(self.hm_hru_ids)


class BaseRemapGenerator(ABC):
    """
    Base class for routing model remap generators.

    Provides shared algorithms for computing spatial remapping weights.
    Subclasses implement write_remap_file() to produce model-specific output.
    """

    def __init__(self, preprocessor: RemapPreprocessorProtocol):
        self.pp = preprocessor

    # =========================================================================
    # Abstract methods
    # =========================================================================

    @abstractmethod
    def write_remap_file(self, remap_data: RemapData, output_path: Path) -> None:
        """Write remap data in the model-specific format."""

    # =========================================================================
    # Remapping strategies
    # =========================================================================

    def create_area_weighted_remap(
        self,
        hru_ids: np.ndarray,
        weights: np.ndarray,
        source_gru_id: int = 1,
    ) -> RemapData:
        """
        Create area-weighted remapping from delineated catchment weights.

        Each routing HRU receives runoff from a single source GRU,
        distributed according to areal weights (fractional subcatchment areas).
        """
        n_hrus = len(hru_ids)
        return RemapData(
            rn_hru_ids=hru_ids.astype(int),
            n_overlaps=np.ones(n_hrus, dtype=int),
            hm_hru_ids=np.full(n_hrus, source_gru_id, dtype=int),
            weights=weights.astype(float),
        )

    def create_equal_weight_remap(
        self,
        hru_ids: np.ndarray,
        source_gru_id: int = 1,
    ) -> RemapData:
        """
        Create equal-weight remapping for all routing HRUs.

        Distributes runoff uniformly from a single source GRU to all
        routing HRUs. Weight = 1/n_hrus for each.
        """
        n_hrus = len(hru_ids)
        equal_weight = 1.0 / n_hrus
        return RemapData(
            rn_hru_ids=hru_ids.astype(int),
            n_overlaps=np.ones(n_hrus, dtype=int),
            hm_hru_ids=np.full(n_hrus, source_gru_id, dtype=int),
            weights=np.full(n_hrus, equal_weight),
        )

    def create_spatial_intersection_remap(
        self,
        hm_shape_path: Path,
        rm_shape_path: Path,
        hm_hru_col: str = 'GRU_ID',
        rm_hru_col: str = 'GRU_ID',
    ) -> RemapData:
        """
        Create remapping via EASYMORE spatial intersection.

        Performs area-weighted spatial intersection between source model (HM)
        catchments and routing model (RM) basins.
        """
        import geopandas as gpd
        import pandas as pd

        hm_shape = gpd.read_file(hm_shape_path)
        rm_shape = gpd.read_file(rm_shape_path)

        # Reproject to equal-area for accurate intersection
        hm_shape = hm_shape.to_crs('EPSG:6933')
        rm_shape = rm_shape.to_crs('EPSG:6933')

        esmr_obj = _create_easymore_instance()
        intersected = esmr_obj.intersection_shp(rm_shape, hm_shape)

        # Process intersection results
        int_rm_id = f"S_1_{rm_hru_col}"
        int_hm_id = f"S_2_{hm_hru_col}"
        int_weight = 'AP1N'

        intersected = intersected.sort_values(by=[int_rm_id, int_hm_id])

        rn_hru_ids = intersected.groupby(int_rm_id).agg({int_rm_id: pd.unique}).values.astype(int).flatten()
        n_overlaps = intersected.groupby(int_rm_id).agg({int_hm_id: 'count'}).values.astype(int).flatten()

        nested_hm_ids = intersected.groupby(int_rm_id).agg({int_hm_id: list}).values.tolist()
        hm_hru_ids = np.array([item for sublist in nested_hm_ids for item in sublist[0]], dtype=int)

        nested_weights = intersected.groupby(int_rm_id).agg({int_weight: list}).values.tolist()
        weights = np.array([item for sublist in nested_weights for item in sublist[0]], dtype=float)

        return RemapData(
            rn_hru_ids=rn_hru_ids,
            n_overlaps=n_overlaps,
            hm_hru_ids=hm_hru_ids,
            weights=weights,
        )


def _create_easymore_instance():
    """Create an EASYMORE instance handling different module structures."""
    import easymore

    if hasattr(easymore, "Easymore"):
        return easymore.Easymore()
    if hasattr(easymore, "easymore"):
        return easymore.easymore()
    raise AttributeError("easymore module does not expose an Easymore class")
