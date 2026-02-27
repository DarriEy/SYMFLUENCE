"""
WATFLOOD Parameter Manager.

Manages parameters in WATFLOOD .par files which use per-land-class
parameter blocks.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry


@OptimizerRegistry.register_parameter_manager('WATFLOOD')
class WATFLOODParameterManager(BaseParameterManager):
    """Parameter manager for WATFLOOD .par files."""

    def __init__(
        self,
        config: Any,
        logger: logging.Logger,
        settings_dir: Path
    ):
        super().__init__(config, logger, settings_dir)

        params_str = (
            self._get_config_value(lambda: None, default='', dict_key='WATFLOOD_PARAMS_TO_CALIBRATE') or
            'FLZCOEF,PWR,R2N,AK,AKF,REESSION,RETN,AK2,AK2FS,R3,DS,FPET,FTALL,FM,BASE,SUBLIM_FACTOR'
        )
        self.watflood_params = [p.strip() for p in params_str.split(',') if p.strip()]

        self.par_file = self._get_config_value(lambda: None, default='bow.par', dict_key='WATFLOOD_PAR_FILE')

    def _get_parameter_names(self) -> List[str]:
        """Get ordered list of parameter names to calibrate."""
        return list(self.watflood_params)

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Union[float, str]]]:
        """Load parameter bounds from WATFLOOD parameter definitions."""
        from ..parameters import PARAM_BOUNDS
        return {p: PARAM_BOUNDS.get(p, {'min': 0.001, 'max': 10.0})
                for p in self.watflood_params}

    def update_model_files(self, params: Dict[str, Any]) -> bool:
        """Write denormalized parameters to WATFLOOD model files."""
        return self.apply_parameters(params)

    def get_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        return {p: self.param_bounds.get(p, {'min': 0.001, 'max': 10.0})
                for p in self.watflood_params}

    def get_default_parameters(self) -> Dict[str, float]:
        from ..parameters import DEFAULT_PARAMS
        return {p: DEFAULT_PARAMS.get(p, 1.0) for p in self.watflood_params}

    def apply_parameters(self, params: Dict[str, float], target_dir: Optional[Path] = None) -> bool:
        """Apply parameters to WATFLOOD .par file."""
        target = target_dir or self.settings_dir
        # Check basin/ subdirectory first (standard WATFLOOD layout)
        par_path = target / 'basin' / self.par_file
        if not par_path.exists():
            par_path = target / self.par_file

        if not par_path.exists():
            self.logger.warning(f"WATFLOOD .par file not found: {par_path}")
            return False

        try:
            content = par_path.read_text(encoding='utf-8')

            for param_name, value in params.items():
                if param_name in self.watflood_params:
                    content = self._update_par_value(content, param_name, value)

            par_path.write_text(content, encoding='utf-8')
            self.logger.debug(f"Updated WATFLOOD .par file with {len(params)} parameters")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating .par file: {e}")
            return False

    def _update_par_value(self, content: str, param_name: str, value: float) -> str:
        """Update a parameter value in .par file content.

        WATFLOOD .par files use keyword-value format with per-land-class blocks.
        Parameters appear as: KEYWORD value(s)
        Calibration names are mapped to par file keywords via PAR_KEYWORD_MAP.
        """
        from ..parameters import PAR_KEYWORD_MAP

        # Map calibration name to par file keyword (parser format)
        par_keyword = PAR_KEYWORD_MAP.get(param_name, param_name)

        # Parser-format lines: `:keyword, value,` or `:keyword, value`
        pattern = re.compile(
            rf'^(:{re.escape(par_keyword)},\s*)([-+]?[\d.eE+\-]+)',
            re.MULTILINE | re.IGNORECASE
        )

        # Use scientific notation for values in the par file
        def replacer(m):
            prefix = m.group(1)
            return f"{prefix}{value:.3E}"

        new_content, count = pattern.subn(replacer, content)
        if count > 0:
            self.logger.debug(f"Updated {param_name} ({par_keyword}) = {value:.3E} ({count} occurrences)")
        else:
            self.logger.warning(f"Parameter {param_name} (keyword '{par_keyword}') not found in .par file")
        return new_content

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        from ..parameters import DEFAULT_PARAMS
        return {p: DEFAULT_PARAMS.get(p, 1.0) for p in self.watflood_params}
