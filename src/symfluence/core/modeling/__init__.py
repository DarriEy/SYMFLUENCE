# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""The model-adapter contract tier (``models`` contract family, ADR-0009).

Everything a model package subclasses or builds against at adapter level:
runner/preprocessor/postprocessor/extractor bases (``.base``), adapter mixins
(``.mixins``), execution infrastructure (``.execution``), model state
management (``.state``), scaffolding templates (``.templates``), shared
adapter utilities (``.utilities``), spatial-mode validation
(``.spatial_modes``), and the ConfigKey schema machinery
(``.config_schema``). Historical ``symfluence.models.*`` import paths remain
as shims.
"""
from __future__ import annotations

from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.modeling.config_schema import (
    ModelConfigSchema,
    get_model_schema,
    register_model_schema,
)
from symfluence.core.modeling.coupling import INSTALL_SUGGESTION, build_coupling_graph, is_dcoupler_available
from symfluence.core.modeling.evaluation_registry import EvaluationRegistry
from symfluence.core.modeling.netcdf_utils import create_minimal_encoding, create_netcdf_encoding
from symfluence.core.modeling.structure_ensemble import BaseStructureEnsembleAnalyzer
from symfluence.core.modeling.variable_utils import VariableHandler, VariableStandardizer

__all__ = [
    'BaseModelRunner',
    'BaseStructureEnsembleAnalyzer',
    'EvaluationRegistry',
    'INSTALL_SUGGESTION',
    'ModelConfigSchema',
    'get_model_schema',
    'register_model_schema',
    'create_minimal_encoding',
    'create_netcdf_encoding',
    'build_coupling_graph',
    'is_dcoupler_available',
    'VariableHandler',
    'VariableStandardizer',
]
