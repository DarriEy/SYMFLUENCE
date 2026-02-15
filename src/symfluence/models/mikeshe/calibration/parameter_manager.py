"""
MIKE-SHE Parameter Manager

Handles MIKE-SHE parameter bounds, normalization, and .she XML file updates
using xml.etree.ElementTree.
"""

import logging
import shutil
import xml.etree.ElementTree as ET  # nosec B405 - parsing trusted MIKE-SHE setup files
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('MIKESHE')
class MIKESHEParameterManager(BaseParameterManager):
    """Handles MIKE-SHE parameter bounds, normalization, and XML file updates.

    MIKE-SHE stores its configuration in XML-based .she files. This manager
    parses and modifies parameter values within the XML tree using standard
    library xml.etree.ElementTree.
    """

    # Maps parameter names to XPath locations in the .she XML
    PARAM_XML_PATHS = {
        'manning_m': './/OverlandFlow/ManningM',
        'detention_storage': './/OverlandFlow/DetentionStorage',
        'Ks_uz': './/UnsaturatedFlow/HydraulicConductivity',
        'theta_sat': './/UnsaturatedFlow/SaturatedMoistureContent',
        'theta_fc': './/UnsaturatedFlow/FieldCapacity',
        'theta_wp': './/UnsaturatedFlow/WiltingPoint',
        'Ks_sz_h': './/SaturatedFlow/HorizontalConductivity',
        'specific_yield': './/SaturatedFlow/SpecificYield',
        'ddf': './/SnowMelt/DegreeDayFactor',
        'snow_threshold': './/SnowMelt/ThresholdTemperature',
        'max_canopy_storage': './/Vegetation/MaxCanopyStorage',
    }

    def __init__(
        self,
        config: Dict,
        logger: logging.Logger,
        mikeshe_settings_dir: Path
    ):
        """
        Initialize MIKE-SHE parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            mikeshe_settings_dir: Path to MIKE-SHE settings directory
        """
        super().__init__(config, logger, mikeshe_settings_dir)

        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse parameters to calibrate from config
        mikeshe_params_str = None
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'mikeshe'):
                mikeshe_params_str = config.model.mikeshe.params_to_calibrate
        except (AttributeError, TypeError):
            pass

        if mikeshe_params_str is None:
            mikeshe_params_str = config.get('MIKESHE_PARAMS_TO_CALIBRATE')

        if mikeshe_params_str is None:
            mikeshe_params_str = (
                'manning_m,detention_storage,Ks_uz,theta_sat,theta_fc,'
                'theta_wp,Ks_sz_h,specific_yield,ddf,snow_threshold,'
                'max_canopy_storage'
            )
            logger.warning(
                f"MIKESHE_PARAMS_TO_CALIBRATE missing; "
                f"using fallback: {mikeshe_params_str}"
            )

        self.mikeshe_params = [
            p.strip() for p in str(mikeshe_params_str).split(',') if p.strip()
        ]

        # Path to .she setup file
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.settings_dir = self.project_dir / 'MIKESHE_input' / 'settings'

        # Get setup file name
        setup_file = 'model.she'
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'mikeshe'):
                setup_file = config.model.mikeshe.setup_file or setup_file
        except (AttributeError, TypeError):
            pass
        self.setup_file = self.settings_dir / setup_file

    def _get_parameter_names(self) -> List[str]:
        """Return MIKE-SHE parameter names from config."""
        return self.mikeshe_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Return MIKE-SHE parameter bounds.

        MIKE-SHE parameters with physically-based calibration ranges.
        """
        from symfluence.models.mikeshe.parameters import PARAM_BOUNDS

        mikeshe_bounds = dict(PARAM_BOUNDS)

        # Check for config overrides
        config_bounds = self.config.get('MIKESHE_PARAM_BOUNDS', {})
        if config_bounds:
            for param_name, bound_list in config_bounds.items():
                if isinstance(bound_list, (list, tuple)) and len(bound_list) == 2:
                    mikeshe_bounds[param_name] = {
                        'min': float(bound_list[0]),
                        'max': float(bound_list[1])
                    }
                    self.logger.debug(
                        f"Using config bounds for {param_name}: "
                        f"[{bound_list[0]}, {bound_list[1]}]"
                    )

        # Log bounds for calibrated parameters
        for param_name in self.mikeshe_params:
            if param_name in mikeshe_bounds:
                b = mikeshe_bounds[param_name]
                self.logger.info(
                    f"MIKESHE param {param_name}: "
                    f"bounds=[{b['min']:.6g}, {b['max']:.6g}]"
                )

        return mikeshe_bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update MIKE-SHE .she XML file with new parameter values."""
        return self.update_she_xml(params)

    def update_she_xml(self, params: Dict[str, float]) -> bool:
        """
        Update MIKE-SHE .she XML setup file with new values.

        Uses xml.etree.ElementTree to parse the XML, locate each
        parameter's element via XPath, and update its text value.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if successful
        """
        try:
            if not self.setup_file.exists():
                self.logger.error(
                    f"MIKE-SHE setup file not found: {self.setup_file}"
                )
                return False

            # Parse the XML
            tree = ET.parse(self.setup_file)  # nosec B314
            root = tree.getroot()

            # Strip namespace for XPath matching if present
            namespace = ''
            if root.tag.startswith('{'):
                namespace = root.tag.split('}')[0] + '}'

            for param_name, value in params.items():
                xpath = self.PARAM_XML_PATHS.get(param_name)
                if xpath is None:
                    self.logger.warning(
                        f"No XML path mapping for parameter '{param_name}'"
                    )
                    continue

                # Try with and without namespace
                element = root.find(xpath)
                if element is None and namespace:
                    # Replace tag names with namespaced versions
                    ns_xpath = xpath
                    for tag in xpath.replace('.//', '').split('/'):
                        ns_xpath = ns_xpath.replace(
                            tag, f'{namespace}{tag}', 1
                        )
                    element = root.find(ns_xpath)

                if element is not None:
                    element.text = f'{value:.6g}'
                    self.logger.debug(
                        f"Updated {param_name} = {value:.6g} "
                        f"at {xpath}"
                    )
                else:
                    self.logger.warning(
                        f"XML element not found for {param_name} "
                        f"at {xpath}"
                    )

            # Validate soil moisture parameter ordering
            validation_error = self._validate_soil_params(params)
            if validation_error:
                self.logger.warning(
                    f"Parameter validation warning: {validation_error}"
                )

            # Write modified XML back (atomic: write tmp then rename)
            temp_file = self.setup_file.with_suffix('.she.tmp')
            tree.write(temp_file, encoding='utf-8', xml_declaration=True)
            temp_file.replace(self.setup_file)

            return True

        except Exception as e:
            self.logger.error(f"Error updating MIKE-SHE setup file: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _validate_soil_params(self, params: Dict[str, float]) -> Optional[str]:
        """
        Validate soil parameter consistency.

        Returns None if valid, or a warning string if inconsistent.
        """
        theta_sat = params.get('theta_sat')
        theta_fc = params.get('theta_fc')
        theta_wp = params.get('theta_wp')

        if theta_sat is not None and theta_fc is not None:
            if theta_fc >= theta_sat:
                return (
                    f"theta_fc ({theta_fc:.4f}) >= theta_sat ({theta_sat:.4f})"
                )

        if theta_fc is not None and theta_wp is not None:
            if theta_wp >= theta_fc:
                return (
                    f"theta_wp ({theta_wp:.4f}) >= theta_fc ({theta_fc:.4f})"
                )

        return None

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from .she file or defaults."""
        try:
            if not self.setup_file.exists():
                return self._get_default_initial_values()

            tree = ET.parse(self.setup_file)  # nosec B314
            root = tree.getroot()

            namespace = ''
            if root.tag.startswith('{'):
                namespace = root.tag.split('}')[0] + '}'

            params = {}

            for param_name in self.mikeshe_params:
                xpath = self.PARAM_XML_PATHS.get(param_name)
                if xpath is None:
                    bounds = self.param_bounds.get(
                        param_name, {'min': 0.1, 'max': 10.0}
                    )
                    params[param_name] = (bounds['min'] + bounds['max']) / 2
                    continue

                element = root.find(xpath)
                if element is None and namespace:
                    ns_xpath = xpath
                    for tag in xpath.replace('.//', '').split('/'):
                        ns_xpath = ns_xpath.replace(
                            tag, f'{namespace}{tag}', 1
                        )
                    element = root.find(ns_xpath)

                if element is not None and element.text:
                    try:
                        params[param_name] = float(element.text)
                    except ValueError:
                        bounds = self.param_bounds.get(
                            param_name, {'min': 0.1, 'max': 10.0}
                        )
                        params[param_name] = (bounds['min'] + bounds['max']) / 2
                else:
                    bounds = self.param_bounds.get(
                        param_name, {'min': 0.1, 'max': 10.0}
                    )
                    params[param_name] = (bounds['min'] + bounds['max']) / 2

            return params

        except Exception as e:
            self.logger.error(f"Error reading initial parameters: {e}")
            return self._get_default_initial_values()

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values from DEFAULT_PARAMS."""
        from symfluence.models.mikeshe.parameters import DEFAULT_PARAMS

        params = {}
        for param_name in self.mikeshe_params:
            if param_name in DEFAULT_PARAMS:
                params[param_name] = DEFAULT_PARAMS[param_name]
            else:
                bounds = self.param_bounds.get(
                    param_name, {'min': 0.1, 'max': 10.0}
                )
                params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params

    def copy_params_to_worker_dir(self, worker_params_dir: Path) -> bool:
        """
        Copy .she setup file to a worker-specific directory for parallel
        calibration.

        Args:
            worker_params_dir: Target directory for worker's setup files

        Returns:
            True if successful
        """
        try:
            worker_params_dir.mkdir(parents=True, exist_ok=True)

            # Copy .she setup file
            if self.setup_file.exists():
                shutil.copy2(
                    self.setup_file,
                    worker_params_dir / self.setup_file.name
                )

            return True

        except Exception as e:
            self.logger.error(
                f"Error copying params to {worker_params_dir}: {e}"
            )
            return False
