"""
WRF-Hydro Parameter Manager

Handles WRF-Hydro parameter bounds, normalization, and Fortran namelist updates.
Parameters are applied to hydro.namelist and namelist.hrldas using regex-based parsing.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('WRFHYDRO')
class WRFHydroParameterManager(BaseParameterManager):
    """Handles WRF-Hydro parameter bounds, normalization, and namelist updates.

    WRF-Hydro uses Fortran namelists for configuration:
    - hydro.namelist: routing parameters (REFKDT, SLOPE, OVROUGHRTFAC, etc.)
    - namelist.hrldas: Noah-MP LSM parameters (BEXP, DKSAT, SMCMAX)

    This manager uses regex-based parsing to find and replace parameter
    values in both namelist files.
    """

    def __init__(self, config: Dict, logger: logging.Logger, wrfhydro_settings_dir: Path):
        """
        Initialize WRF-Hydro parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            wrfhydro_settings_dir: Path to WRF-Hydro settings directory
        """
        super().__init__(config, logger, wrfhydro_settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        # Parse WRF-Hydro parameters to calibrate from config
        params_str = self._get_config_value(lambda: self.config.model.wrfhydro.params_to_calibrate, default=None, dict_key='WRFHYDRO_PARAMS_TO_CALIBRATE')

        if params_str is None:
            params_str = 'REFKDT,SLOPE,OVROUGHRTFAC,RETDEPRTFAC,LKSATFAC,BEXP,DKSAT,SMCMAX'
            logger.warning(
                f"WRFHYDRO_PARAMS_TO_CALIBRATE missing; using fallback: {params_str}"
            )

        self.wrfhydro_params = [p.strip() for p in str(params_str).split(',') if p.strip()]

        # Paths
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.settings_dir = wrfhydro_settings_dir

        # Namelist file names
        self.hydro_namelist_file = self._get_config_value(lambda: None, default='hydro.namelist', dict_key='WRFHYDRO_HYDRO_NAMELIST')
        self.hrldas_namelist_file = self._get_config_value(lambda: None, default='namelist.hrldas', dict_key='WRFHYDRO_NAMELIST_FILE')

    def _get_parameter_names(self) -> List[str]:
        """Return WRF-Hydro parameter names from config."""
        return self.wrfhydro_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return WRF-Hydro parameter bounds."""
        from ..parameters import PARAM_BOUNDS

        bounds = dict(PARAM_BOUNDS)

        # Check for config overrides (preserves transform metadata from registry)
        config_bounds = self._get_config_value(lambda: None, default={}, dict_key='WRFHYDRO_PARAM_BOUNDS')
        if config_bounds:
            self._apply_config_bounds_override(bounds, config_bounds)

        for param_name in self.wrfhydro_params:
            if param_name in bounds:
                b = bounds[param_name]
                self.logger.info(
                    f"WRF-Hydro param {param_name}: bounds=[{b['min']:.6g}, {b['max']:.6g}]"
                )

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update WRF-Hydro namelist files with new parameter values."""
        from ..parameters import WRFHYDRO_PARAM_TARGETS

        try:
            # Separate parameters by target file
            hydro_params = {}
            hrldas_params = {}

            for param_name, value in params.items():
                target_info = WRFHYDRO_PARAM_TARGETS.get(param_name, {})
                target = target_info.get('target', 'hydro_namelist')

                if target == 'hrldas_namelist':
                    hrldas_params[param_name] = value
                else:
                    hydro_params[param_name] = value

            success = True

            # Update hydro.namelist
            if hydro_params:
                hydro_path = self.settings_dir / self.hydro_namelist_file
                if hydro_path.exists():
                    if not self._update_namelist(hydro_path, hydro_params):
                        success = False
                else:
                    self.logger.warning(f"Hydro namelist not found: {hydro_path}")

            # Update namelist.hrldas
            if hrldas_params:
                hrldas_path = self.settings_dir / self.hrldas_namelist_file
                if hrldas_path.exists():
                    if not self._update_namelist(hrldas_path, hrldas_params):
                        success = False
                else:
                    self.logger.warning(f"HRLDAS namelist not found: {hrldas_path}")

            return success

        except Exception as e:
            self.logger.error(f"Error updating WRF-Hydro namelists: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _update_namelist(self, namelist_path: Path, params: Dict[str, float]) -> bool:
        """
        Update a Fortran namelist file with new parameter values.

        Handles the standard format: ``PARAM_NAME = value``

        For parameters not yet present in the namelist, they are inserted
        before the closing ``/`` of the appropriate section.

        Args:
            namelist_path: Path to namelist file
            params: Parameters to update

        Returns:
            True if successful
        """
        try:
            content = namelist_path.read_text(encoding='utf-8')

            for param_name, value in params.items():
                formatted = self._format_namelist_value(value)

                # Try to replace existing parameter
                pattern = re.compile(
                    r'(\s*' + re.escape(param_name) + r'\s*=\s*)([^\s,!/\n]+)',
                    re.IGNORECASE
                )
                match = pattern.search(content)

                if match:
                    content = pattern.sub(r'\g<1>' + formatted, content)
                    self.logger.debug(f"Updated {param_name} = {formatted}")
                else:
                    # Don't insert unknown params — Fortran namelists crash
                    # on variable names the compiled binary doesn't expect.
                    self.logger.debug(
                        f"Skipping {param_name}: not present in {namelist_path.name}"
                    )

            namelist_path.write_text(content, encoding='utf-8')
            return True

        except Exception as e:
            self.logger.error(f"Error updating namelist {namelist_path}: {e}")
            return False

    def _format_namelist_value(self, value: float) -> str:
        """Format a parameter value for Fortran namelist syntax."""
        abs_val = abs(value)
        if abs_val == 0.0:
            return '0.0'
        elif abs_val < 0.001 or abs_val >= 1e6:
            return f'{value:.6e}'
        elif abs_val == int(abs_val) and abs_val < 1e6:
            return f'{value:.1f}'
        else:
            return f'{value:.6f}'

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from namelists or defaults."""
        try:
            params = {}

            # Read from hydro.namelist
            hydro_path = self.settings_dir / self.hydro_namelist_file
            if hydro_path.exists():
                content = hydro_path.read_text(encoding='utf-8')
                for param_name in self.wrfhydro_params:
                    value = self._extract_value_from_content(content, param_name)
                    if value is not None:
                        params[param_name] = value

            # Read from namelist.hrldas for any not yet found
            hrldas_path = self.settings_dir / self.hrldas_namelist_file
            if hrldas_path.exists():
                content = hrldas_path.read_text(encoding='utf-8')
                for param_name in self.wrfhydro_params:
                    if param_name not in params:
                        value = self._extract_value_from_content(content, param_name)
                        if value is not None:
                            params[param_name] = value

            # Fill missing with defaults
            for param_name in self.wrfhydro_params:
                if param_name not in params:
                    bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                    params[param_name] = (bounds['min'] + bounds['max']) / 2

            return params

        except Exception as e:
            self.logger.error(f"Error reading initial parameters: {e}")
            return self._get_default_initial_values()

    def _extract_value_from_content(self, content: str, param_name: str) -> Optional[float]:
        """Extract a parameter value from namelist content."""
        pattern = re.compile(
            r'\s*' + re.escape(param_name) + r'\s*=\s*([^\s,!/\n]+)',
            re.IGNORECASE
        )
        match = pattern.search(content)
        if match:
            try:
                value_str = match.group(1).replace('d', 'e').replace('D', 'E')
                return float(value_str)
            except ValueError:
                return None
        return None

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values."""
        from ..parameters import DEFAULT_PARAMS

        params = {}
        for param_name in self.wrfhydro_params:
            if param_name in DEFAULT_PARAMS:
                params[param_name] = DEFAULT_PARAMS[param_name]
            else:
                bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params

    def copy_params_to_worker_dir(self, worker_settings_dir: Path) -> bool:
        """
        Copy namelist and domain files to a worker-specific directory.

        Args:
            worker_settings_dir: Target directory for worker's files

        Returns:
            True if successful
        """
        import shutil

        try:
            worker_settings_dir.mkdir(parents=True, exist_ok=True)

            # Copy namelist files and TBL files
            for pattern in ['*.namelist', 'namelist.*', '*.nc', '*.TBL']:
                for f in self.settings_dir.glob(pattern):
                    shutil.copy2(f, worker_settings_dir / f.name)

            return True

        except Exception as e:
            self.logger.error(f"Error copying WRF-Hydro files to {worker_settings_dir}: {e}")
            return False
