"""
Routing Decider

Unified routing decision logic for all hydrological models.
Consolidates duplicate needs_routing() implementations from workers.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class RoutingDecider:
    """
    Unified routing decision logic for all models.

    Determines whether mizuRoute routing is needed based on:
    - Calibration variable (streamflow only)
    - Explicit routing model configuration
    - Model-specific routing integration settings
    - Spatial mode configuration
    - Domain definition method
    - Routing delineation settings
    - Existence of mizuRoute control files
    """

    # Model-specific config keys for spatial mode
    SPATIAL_MODE_KEYS: Dict[str, str] = {
        'SUMMA': 'DOMAIN_DEFINITION_METHOD',
        'FUSE': 'FUSE_SPATIAL_MODE',
        'HYPE': 'HYPE_SPATIAL_MODE',
        'GR': 'GR_SPATIAL_MODE',
        'MESH': 'MESH_SPATIAL_MODE',
        'NGEN': 'NGEN_SPATIAL_MODE',
    }

    # Model-specific routing integration config keys
    ROUTING_INTEGRATION_KEYS: Dict[str, str] = {
        'FUSE': 'FUSE_ROUTING_INTEGRATION',
    }

    def needs_routing(
        self,
        config: Dict[str, Any],
        model: str,
        settings_dir: Optional[Path] = None
    ) -> bool:
        """
        Determine if routing (mizuRoute) is needed for a model run.

        Decision hierarchy:
        1. If CALIBRATION_VARIABLE != 'streamflow': False
        2. If ROUTING_MODEL == 'mizuRoute' or 'default': True
        3. If model-specific routing integration requests mizuRoute: True
        4. If spatial_mode in ['semi_distributed', 'distributed']: True
        5. If domain_method not in ['point', 'lumped']: True
        6. If lumped but ROUTING_DELINEATION == 'river_network': True
        7. If mizuRoute control file exists in settings_dir: True
        8. Otherwise: False

        Args:
            config: Configuration dictionary
            model: Model name (e.g., 'SUMMA', 'FUSE', 'HYPE')
            settings_dir: Optional settings directory to check for mizuRoute control files

        Returns:
            True if routing is needed
        """
        result, _ = self._evaluate_routing(config, model, settings_dir)
        return result

    def needs_routing_verbose(
        self,
        config: Dict[str, Any],
        model: str,
        settings_dir: Optional[Path] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if routing is needed with diagnostic information.

        Args:
            config: Configuration dictionary
            model: Model name
            settings_dir: Optional settings directory

        Returns:
            Tuple of (needs_routing, diagnostics_dict)
            diagnostics_dict contains 'reason' and 'checks' keys
        """
        return self._evaluate_routing(config, model, settings_dir)

    def _evaluate_routing(
        self,
        config: Dict[str, Any],
        model: str,
        settings_dir: Optional[Path] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Internal method that evaluates routing need with diagnostics.

        Returns:
            Tuple of (needs_routing, diagnostics)
        """
        model = model.upper()
        diagnostics: Dict[str, Any] = {
            'model': model,
            'reason': None,
            'checks': {}
        }

        # 1. Calibration variable check
        calibration_var = config.get('CALIBRATION_VARIABLE', 'streamflow')
        diagnostics['checks']['calibration_variable'] = calibration_var
        if calibration_var != 'streamflow':
            diagnostics['reason'] = 'calibration_variable_not_streamflow'
            return False, diagnostics

        # 2. Explicit routing model check
        routing_model = config.get('ROUTING_MODEL', 'none')
        if routing_model == 'default':
            routing_model = 'mizuRoute'
        diagnostics['checks']['routing_model'] = routing_model
        if routing_model == 'mizuRoute':
            diagnostics['reason'] = 'explicit_routing_model_mizuroute'
            return True, diagnostics

        # 3. Model-specific routing integration (e.g., FUSE)
        if model in self.ROUTING_INTEGRATION_KEYS:
            integration_key = self.ROUTING_INTEGRATION_KEYS[model]
            integration = config.get(integration_key, 'none')
            diagnostics['checks']['routing_integration'] = integration

            if integration == 'mizuRoute':
                diagnostics['reason'] = f'{model.lower()}_routing_integration_mizuroute'
                return True, diagnostics

            # If integration is 'default', inherit from ROUTING_MODEL
            if integration == 'default':
                base_routing = config.get('ROUTING_MODEL', 'none')
                if base_routing == 'mizuRoute':
                    diagnostics['reason'] = f'{model.lower()}_routing_integration_default_to_mizuroute'
                    return True, diagnostics
                # If integration is 'default' but ROUTING_MODEL is not mizuRoute,
                # don't proceed with routing for this model
                if base_routing not in ['mizuRoute', 'default']:
                    diagnostics['reason'] = 'routing_integration_default_but_routing_model_not_mizuroute'
                    return False, diagnostics

        # 4. Spatial mode check
        spatial_key = self.SPATIAL_MODE_KEYS.get(model, 'DOMAIN_DEFINITION_METHOD')
        spatial_mode = config.get(spatial_key, 'lumped')
        diagnostics['checks']['spatial_mode'] = spatial_mode
        diagnostics['checks']['spatial_mode_key'] = spatial_key
        if spatial_mode in ['semi_distributed', 'distributed']:
            diagnostics['reason'] = 'spatial_mode_distributed'
            return True, diagnostics

        # 5. Domain definition method check (especially for SUMMA, HYPE)
        domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        diagnostics['checks']['domain_definition_method'] = domain_method
        if domain_method not in ['point', 'lumped']:
            diagnostics['reason'] = 'domain_method_not_point_or_lumped'
            return True, diagnostics

        # 6. Lumped with river network routing
        routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')
        diagnostics['checks']['routing_delineation'] = routing_delineation
        if spatial_mode == 'lumped' and routing_delineation == 'river_network':
            diagnostics['reason'] = 'lumped_with_river_network_delineation'
            return True, diagnostics

        # 7. Filesystem check for existing mizuRoute setup
        if settings_dir:
            settings_dir = Path(settings_dir)
            control_exists = self._check_mizuroute_control_exists(settings_dir, model)
            diagnostics['checks']['mizuroute_control_exists'] = control_exists
            if control_exists:
                diagnostics['reason'] = 'mizuroute_control_file_exists'
                return True, diagnostics

        diagnostics['reason'] = 'no_routing_conditions_met'
        return False, diagnostics

    def _check_mizuroute_control_exists(
        self,
        settings_dir: Path,
        model: str
    ) -> bool:
        """
        Check if mizuRoute control file exists.

        Handles directory structure variations for different models.

        Args:
            settings_dir: Settings directory path
            model: Model name

        Returns:
            True if mizuRoute control file exists
        """
        # Standard location
        mizu_control = settings_dir / 'mizuRoute' / 'mizuroute.control'
        if mizu_control.exists():
            logger.debug(f"Found mizuRoute control file at {mizu_control}")
            return True

        # Handle model-specific subdirectory cases (e.g., FUSE settings in subdirectory)
        if settings_dir.name == model.upper():
            parent_mizu = settings_dir.parent / 'mizuRoute' / 'mizuroute.control'
            if parent_mizu.exists():
                logger.debug(f"Found mizuRoute control file at {parent_mizu}")
                return True

        return False


# Module-level instance for convenience
_routing_decider = RoutingDecider()


def needs_routing(
    config: Dict[str, Any],
    model: str,
    settings_dir: Optional[Path] = None
) -> bool:
    """
    Convenience function for routing decision.

    See RoutingDecider.needs_routing for full documentation.
    """
    return _routing_decider.needs_routing(config, model, settings_dir)


def needs_routing_verbose(
    config: Dict[str, Any],
    model: str,
    settings_dir: Optional[Path] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function for routing decision with diagnostics.

    See RoutingDecider.needs_routing_verbose for full documentation.
    """
    return _routing_decider.needs_routing_verbose(config, model, settings_dir)
