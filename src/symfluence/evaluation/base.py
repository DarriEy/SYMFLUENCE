# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base Evaluator for SYMFLUENCE

Note: This module is maintained for backward compatibility.
New evaluators should inherit from symfluence.evaluation.evaluators.ModelEvaluator.
"""

from .evaluators import ModelEvaluator as BaseEvaluator

__all__ = ["BaseEvaluator"]
