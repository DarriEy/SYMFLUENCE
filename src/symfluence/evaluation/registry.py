# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Evaluation Registry for SYMFLUENCE

Provides a central registry for performance evaluation handlers.
"""
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.registries import R


class EvaluationRegistry:

    @classmethod
    def register(cls, variable_type: str):
        """Decorator to register an evaluation handler."""
        def decorator(handler_class):
            warnings.warn(
                "EvaluationRegistry.register() is deprecated; "
                "use R.evaluators.add() or model_manifest() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            R.evaluators.add(variable_type, handler_class)
            return handler_class
        return decorator

    @classmethod
    def get_evaluator(
        cls,
        variable_type: str,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        project_dir: Optional[Path] = None,
        **kwargs
    ):
        """Get an instance of the appropriate evaluation handler."""
        handler_class = R.evaluators.get(variable_type.upper())
        if handler_class is None:
            return None

        handler_logger = logger or logging.getLogger(handler_class.__name__)
        handler_project_dir = project_dir or Path(".")
        return handler_class(config, handler_project_dir, handler_logger, **kwargs)

    @classmethod
    def list_evaluators(cls) -> list:
        return R.evaluators.keys()
