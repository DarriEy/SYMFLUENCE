# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""FUSE ParameterTransferManager.

Transfer function base classes have moved to:
    symfluence.optimization.regionalization.transfer_functions
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.models.fuse.calibration.parameter_regionalization import (
    FUSE_DEFAULT_PARAM_CONFIG,
)
from symfluence.optimization.regionalization.transfer_functions import (
    FlexiblePowerTF,
    TransferFunction,
)


class ParameterTransferManager:
    """Manages transfer functions for all FUSE parameters."""

    DEFAULT_PARAM_CONFIG = FUSE_DEFAULT_PARAM_CONFIG

    def __init__(
        self,
        attributes_path: Path,
        param_bounds: Dict[str, Tuple[float, float]],
        param_config: Optional[Dict[str, Dict]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.param_bounds = param_bounds
        self.param_config = param_config or self.DEFAULT_PARAM_CONFIG
        self.attributes = pd.read_csv(attributes_path)
        self.n_subcatchments = len(self.attributes)
        self.logger.info(f"Loaded attributes for {self.n_subcatchments} subcatchments")
        self._normalize_attributes()
        self.transfer_functions: Dict[str, Tuple[TransferFunction, str]] = {}
        self.coefficient_map: Dict[str, List[str]] = {}
        self._build_transfer_functions()

    def _normalize_attributes(self):
        self.attr_stats: Dict[str, Dict] = {}
        for col in ['elev_m', 'precip_mm_yr', 'temp_C', 'aridity', 'snow_frac', 'temp_range_C']:
            if col in self.attributes.columns:
                min_val = self.attributes[col].min()
                max_val = self.attributes[col].max()
                self.attr_stats[col] = {'min': min_val, 'max': max_val}
                range_val = max_val - min_val
                if range_val > 0:
                    self.attributes[f'{col}_norm'] = (self.attributes[col] - min_val) / range_val
                else:
                    self.attributes[f'{col}_norm'] = 0.5

    def _build_transfer_functions(self):
        self.all_coefficients: List[str] = []
        self.coeff_bounds: List[Tuple[float, float]] = []
        for param_name, config in self.param_config.items():
            if param_name not in self.param_bounds:
                continue
            p_min, p_max = self.param_bounds[param_name]
            attr_name = config.get('attribute', 'precip_mm_yr')
            calibrate_b = config.get('calibrate_b', False)
            norm_attr_name = f'{attr_name}_norm'
            if norm_attr_name in self.attributes.columns:
                attr_name = norm_attr_name
            tf = FlexiblePowerTF(
                a_bounds=(p_min, p_max), b_bounds=(-1.5, 1.5),
                calibrate_exponent=calibrate_b, fixed_exponent=0.0,
            )
            coeff_names = [f'{param_name}_a']
            if calibrate_b:
                coeff_names.append(f'{param_name}_b')
            self.transfer_functions[param_name] = (tf, attr_name)
            self.coefficient_map[param_name] = coeff_names
            for name in coeff_names:
                self.all_coefficients.append(name)
            self.coeff_bounds.extend(tf.get_coefficient_bounds())
        n_varying = sum(1 for c in self.param_config.values() if c.get('calibrate_b', False))
        n_uniform = len(self.transfer_functions) - n_varying
        self.logger.info(
            f"Built {len(self.transfer_functions)} transfer functions: "
            f"{n_varying} spatially varying, {n_uniform} uniform, "
            f"{len(self.all_coefficients)} total coefficients"
        )

    def get_calibration_parameters(self) -> Dict[str, Tuple[float, float]]:
        return dict(zip(self.all_coefficients, self.coeff_bounds))

    def coefficients_to_parameters(
        self, coefficients: Dict[str, float],
    ) -> Tuple[np.ndarray, List[str]]:
        n_params = len(self.transfer_functions)
        param_array = np.zeros((self.n_subcatchments, n_params))
        param_names: List[str] = []
        for i, (param_name, (tf, attr_name)) in enumerate(self.transfer_functions.items()):
            param_names.append(param_name)
            if attr_name == 'constant' or attr_name not in self.attributes.columns:
                attr_values = np.ones(self.n_subcatchments)
            else:
                attr_values = self.attributes[attr_name].values
            coeff_names = self.coefficient_map[param_name]
            coeffs = np.array([coefficients[cn] for cn in coeff_names])
            param_values = tf.apply(attr_values, coeffs)
            p_min, p_max = self.param_bounds[param_name]
            param_values = np.clip(param_values, p_min, p_max)
            param_array[:, i] = param_values
        return param_array, param_names

    def create_distributed_para_def(
        self, coefficients: Dict[str, float], template_path: Path, output_path: Path,
    ) -> bool:
        try:
            import shutil

            import netCDF4 as nc
            param_array, param_names = self.coefficients_to_parameters(coefficients)
            shutil.copy(template_path, output_path)
            with nc.Dataset(output_path, 'r+') as ds:
                if 'par' in ds.dimensions:
                    current_size = ds.dimensions['par'].size
                    if current_size != self.n_subcatchments:
                        self.logger.warning(
                            f"para_def has par={current_size}, need {self.n_subcatchments}."
                        )
                        for i, param_name in enumerate(param_names):
                            if param_name in ds.variables:
                                ds.variables[param_name][0] = float(np.mean(param_array[:, i]))
                        return True
                for i, param_name in enumerate(param_names):
                    if param_name in ds.variables:
                        ds.variables[param_name][:] = param_array[:, i]
                ds.sync()
            self.logger.info(f"Created distributed para_def with {len(param_names)} parameters")
            return True
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Error creating distributed para_def: {e}")
            return False

    def summarize_spatial_variation(self, coefficients: Dict[str, float]) -> pd.DataFrame:
        param_array, param_names = self.coefficients_to_parameters(coefficients)
        summary = []
        for i, param_name in enumerate(param_names):
            values = param_array[:, i]
            config = self.param_config.get(param_name, {})
            summary.append({
                'parameter': param_name,
                'attribute': config.get('attribute', 'constant'),
                'min': values.min(), 'max': values.max(),
                'mean': values.mean(), 'std': values.std(),
                'cv': values.std() / values.mean() if values.mean() > 0 else 0,
            })
        return pd.DataFrame(summary)
