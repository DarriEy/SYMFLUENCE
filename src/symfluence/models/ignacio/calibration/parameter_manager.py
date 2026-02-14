"""
IGNACIO Parameter Manager.

Handles FBP parameter bounds, normalization, and configuration file updates
for IGNACIO fire model calibration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.core.parameter_bounds_registry import get_ignacio_bounds
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('IGNACIO')
class IGNACIOParameterManager(BaseParameterManager):
    """
    Parameter manager for IGNACIO FBP calibration.

    Manages six core FBP parameters:
    - ffmc: Fine Fuel Moisture Code (0-101)
    - dmc: Duff Moisture Code (0-200)
    - dc: Drought Code (0-800)
    - fmc: Foliar Moisture Content (50-150%)
    - curing: Grass curing percentage (0-100%)
    - initial_radius: Initial fire radius (1-100 m)
    """

    def __init__(self, config: Dict, logger: logging.Logger, settings_dir: Path):
        super().__init__(config, logger, settings_dir)

        self.domain_name = config.get('DOMAIN_NAME')
        self.project_dir = (
            Path(config.get('SYMFLUENCE_DATA_DIR'))
            / f"domain_{self.domain_name}"
        )

        # Parse parameters to calibrate from config
        params_str = config.get(
            'IGNACIO_PARAMS_TO_CALIBRATE',
            'ffmc,dmc,dc,fmc,curing,initial_radius'
        )
        self.ignacio_params = [
            p.strip() for p in params_str.split(',') if p.strip()
        ]

    def _get_parameter_names(self) -> List[str]:
        """Return list of IGNACIO parameters to calibrate."""
        return self.ignacio_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Load parameter bounds from config or registry defaults.

        Config key IGNACIO_PARAM_BOUNDS takes priority over registry.
        """
        config_bounds = self.config_dict.get('IGNACIO_PARAM_BOUNDS', {})
        registry_bounds = get_ignacio_bounds()

        bounds = {}
        for param in self.ignacio_params:
            if param in config_bounds:
                b = config_bounds[param]
                bounds[param] = {
                    'min': b[0] if isinstance(b, (list, tuple)) else b.get('min', 0),
                    'max': b[1] if isinstance(b, (list, tuple)) else b.get('max', 1),
                    'transform': b.get('transform', 'linear') if isinstance(b, dict) else 'linear',
                }
            elif param in registry_bounds:
                bounds[param] = registry_bounds[param]
            else:
                self.logger.warning(
                    f"No bounds found for IGNACIO param '{param}', using [0, 1]"
                )
                bounds[param] = {'min': 0.0, 'max': 1.0, 'transform': 'linear'}

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """
        Update IGNACIO configuration with calibrated FBP parameters.

        Args:
            params: Dictionary of parameter name -> value

        Returns:
            True if update succeeded.
        """
        try:
            config_path = self._find_config()
            if config_path is None:
                self.logger.debug("No ignacio_config.yaml found (in-memory mode)")
                return True

            import yaml
            with open(config_path, encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            if 'fbp' not in config:
                config['fbp'] = {}

            param_to_yaml = {
                'ffmc': 'ffmc_default',
                'dmc': 'dmc_default',
                'dc': 'dc_default',
                'fmc': 'fmc',
                'curing': 'curing',
            }

            for param_name, value in params.items():
                if param_name == 'initial_radius':
                    if 'simulation' not in config:
                        config['simulation'] = {}
                    config['simulation']['initial_radius'] = float(value)
                else:
                    yaml_key = param_to_yaml.get(param_name, param_name)
                    config['fbp'][yaml_key] = float(value)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False)

            return True

        except Exception as e:
            self.logger.error(f"Error updating IGNACIO config: {e}")
            return False

    def _find_config(self) -> Optional[Path]:
        """Find IGNACIO config file."""
        candidates = [
            self.settings_dir / 'ignacio_config.yaml',
            self.project_dir / 'IGNACIO_input' / 'ignacio_config.yaml',
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """
        Get initial FBP parameter values.

        Reads from config or uses midpoint of bounds.
        """
        bounds = self.param_bounds
        initial = {}
        for param in self.ignacio_params:
            if param in bounds:
                b = bounds[param]
                initial[param] = (b['min'] + b['max']) / 2.0
        return initial if initial else None
