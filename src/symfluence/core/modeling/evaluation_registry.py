# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Evaluation Registry for SYMFLUENCE

Provides a central registry for performance evaluation handlers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.registries import R


class EvaluationRegistry:
    """Lookup facade over ``R.evaluators``.

    Registration goes through ``R.evaluators.add()`` or ``model_manifest()``.
    """

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
