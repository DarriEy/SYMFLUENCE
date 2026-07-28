# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Spatial Mode Detection Mixin.

Provides consistent spatial mode detection across all model preprocessors and runners.
Consolidates the 3-step detection pattern that was previously duplicated across HBV,
GR, FUSE, LSTM, and other model components.
"""
from __future__ import annotations

import logging
from typing import Optional

from symfluence.core.modeling.spatial_modes import (
    SpatialMode,
    get_model_capabilities,
    validate_spatial_mode,
)


class SpatialModeDetectionMixin:
    """
    Mixin for consistent spatial mode detection across models.

    Provides a standardized 3-step approach to determining spatial mode:
    1. Check explicit configuration (e.g., model.hbv.spatial_mode)
    2. If 'auto' or None, infer from domain_definition_method
    3. Validate against model capabilities and warn if routing required

    This mixin expects the following attributes to be available on the class:
    - config: SymfluenceConfig object (typed configuration)
    - config_dict: Dictionary configuration
    - domain_definition_method: String indicating domain method (from BaseModel)
    - logger: Logger instance
    - routing_model: Optional routing model configured (from BaseModel)

    Usage:
        class HBVPreProcessor(BaseModelPreProcessor, SpatialModeDetectionMixin):
            def __init__(self, config, logger):
                super().__init__(config, logger)
                self.spatial_mode = self.detect_spatial_mode('HBV')

    Supported Models:
        Any model with a ``model.<name>`` config section — the section is
        looked up by lowercased model name, so a model (in-tree or external)
        needs no edit here to have its configured spatial mode honoured. A
        model without a section resolves ``None`` and falls through to
        domain-based inference.

        Capability validation (step 3) is likewise per-model and declared
        through :func:`~symfluence.core.modeling.spatial_modes.register_model_spatial_capability`;
        a model that declares nothing is permitted any mode.
    """

    def detect_spatial_mode(
        self,
        model_name: str,
        configured_mode: Optional[str] = None,
        log_detection: bool = True
    ) -> SpatialMode:
        """
        Detect spatial mode using the standard 3-step logic.

        Step 1: Check for explicitly configured mode
        Step 2: If 'auto'/None, infer from domain_definition_method
        Step 3: Validate against model capabilities and warn if needed

        Args:
            model_name: Name of the model (e.g., 'HBV', 'GR', 'FUSE')
            configured_mode: Explicitly configured mode from config, or None
                            to auto-detect from typed config
            log_detection: Whether to log the detection result

        Returns:
            Detected SpatialMode enum value
        """
        logger = self._get_logger()
        model_name_upper = model_name.upper()

        # Step 1: Get configured mode (from param or typed config)
        if configured_mode is None:
            configured_mode = self._get_configured_spatial_mode(model_name_upper)

        # Step 2: If 'auto', 'default', or None, infer from domain definition
        if configured_mode in (None, 'auto', 'default'):
            spatial_mode = self._infer_spatial_mode_from_domain()
            if log_detection:
                domain_method = self._get_domain_definition_method()
                logger.debug(
                    f"{model_name} spatial mode auto-detected as '{spatial_mode}' "
                    f"(DOMAIN_DEFINITION_METHOD: {domain_method})"
                )
        else:
            spatial_mode = SpatialMode.from_string(configured_mode)
            if log_detection:
                logger.debug(f"{model_name} spatial mode set to '{spatial_mode}' from configuration")

        # Step 3: Validate and warn about routing if needed
        self._validate_spatial_mode_for_model(model_name_upper, spatial_mode)

        return spatial_mode

    def _get_configured_spatial_mode(self, model_name: str) -> Optional[str]:
        """
        Extract configured spatial mode from typed config.

        Safely accesses the model-specific spatial_mode attribute from the
        typed configuration object, returning None if not available. The
        config section is resolved purely by lowercasing the model name, so
        every model that declares a ``spatial_mode`` field is served — no
        per-model registration in this file.

        Args:
            model_name: Model name in any case (e.g., 'HBV', 'GR', 'FUSE')

        Returns:
            Configured spatial mode string, or None when the model has no
            config section or the section declares no spatial mode.
        """
        config = getattr(self, 'config', None)
        if config is None or not hasattr(config, 'model'):
            return None

        model_config = config.model
        if model_config is None:
            return None

        # Config sections are named after the model in lowercase, without
        # exception. A hand-maintained allow-list used to sit here whose every
        # entry was just ``model_name.lower()``, so the ~20 models absent from
        # it silently resolved None for a spatial mode they had configured.
        # ``getattr(..., None)`` already yields None for a model with no
        # section, which is the only thing the allow-list ever contributed.
        model_specific_config = getattr(model_config, model_name.lower(), None)
        if model_specific_config is None:
            return None

        return getattr(model_specific_config, 'spatial_mode', None)

    def _infer_spatial_mode_from_domain(self) -> SpatialMode:
        """
        Infer spatial mode from domain definition method.

        Uses the domain_definition_method to determine the appropriate
        spatial mode:
        - 'delineate' -> DISTRIBUTED
        - 'subset', 'semi_distributed' -> SEMI_DISTRIBUTED
        - 'point', 'lumped' or other -> LUMPED

        Returns:
            Inferred SpatialMode enum value
        """
        domain_method = self._get_domain_definition_method()

        if domain_method in ('delineate', 'distributed'):
            return SpatialMode.DISTRIBUTED
        elif domain_method in ('subset', 'semi_distributed', 'semidistributed'):
            return SpatialMode.SEMI_DISTRIBUTED
        else:
            return SpatialMode.LUMPED

    def _validate_spatial_mode_for_model(
        self,
        model_name: str,
        spatial_mode
    ) -> None:
        """
        Validate spatial mode against model capabilities and warn if needed.

        Checks if the detected spatial mode is supported by the model and
        emits appropriate warnings for:
        - Unsupported spatial modes
        - Distributed modes that require routing but none is configured
        - Model-specific recommendations (e.g., LSTM works best in lumped mode)

        Args:
            model_name: Uppercase model name
            spatial_mode: SpatialMode enum or string
        """
        logger = self._get_logger()

        # Convert string to SpatialMode enum for validation
        if isinstance(spatial_mode, SpatialMode):
            spatial_mode_enum = spatial_mode
        else:
            try:
                spatial_mode_enum = SpatialMode(spatial_mode)
            except ValueError:
                logger.warning(f"Unknown spatial mode '{spatial_mode}', defaulting to lumped")
                return

        # Check if routing is configured
        routing_model = getattr(self, 'routing_model', None)
        has_routing = routing_model is not None and routing_model.lower() != 'none'

        # Validate using the centralized validation function
        is_valid, warning_msg = validate_spatial_mode(
            model_name,
            spatial_mode_enum,
            has_routing_configured=has_routing
        )

        if not is_valid:
            logger.error(warning_msg)
            raise ValueError(warning_msg)

        if warning_msg:
            logger.warning(warning_msg)

    def _get_domain_definition_method(self) -> str:
        """
        Get the domain definition method from config.

        Safely retrieves domain_definition_method from the object,
        falling back to typed config if the attribute isn't directly available.

        Returns:
            Domain definition method string
        """
        # Try attribute first (set by base class)
        if hasattr(self, 'domain_definition_method'):
            return getattr(self, 'domain_definition_method', 'lumped')

        # Fall back to typed config
        if hasattr(self, '_get_config_value'):
            return self._get_config_value(
                lambda: self.config.domain.definition_method,
                default='lumped',
            )
        return 'lumped'

    def _get_logger(self) -> logging.Logger:
        """
        Get the logger instance from the object.

        Returns:
            Logger instance, or module-level logger if not available
        """
        if hasattr(self, 'logger') and self.logger is not None:
            return self.logger
        return logging.getLogger(__name__)

    def is_distributed_mode(self, spatial_mode=None) -> bool:
        """
        Check if the current or specified spatial mode is distributed.

        Args:
            spatial_mode: Optional SpatialMode or string to check. If None, uses self.spatial_mode

        Returns:
            True if mode is distributed or semi_distributed
        """
        mode = spatial_mode or getattr(self, 'spatial_mode', SpatialMode.LUMPED)
        return mode in (SpatialMode.DISTRIBUTED, SpatialMode.SEMI_DISTRIBUTED)

    def requires_routing_for_mode(self, model_name: str, spatial_mode=None) -> bool:
        """
        Check if the current spatial mode requires routing for the given model.

        Args:
            model_name: Model name to check
            spatial_mode: Optional SpatialMode or string to check. If None, uses self.spatial_mode

        Returns:
            True if the mode/model combination requires routing
        """
        mode = spatial_mode or getattr(self, 'spatial_mode', SpatialMode.LUMPED)

        capability = get_model_capabilities(model_name)
        if capability is None:
            # No declaration -> core makes no routing claim for this model.
            return False

        if isinstance(mode, SpatialMode):
            mode_enum = mode
        else:
            try:
                mode_enum = SpatialMode(mode)
            except ValueError:
                return False
        return capability.requires_routing.get(mode_enum, False)
