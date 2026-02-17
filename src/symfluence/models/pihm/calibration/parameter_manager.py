"""
PIHM Parameter Manager

Handles PIHM parameter bounds, normalization, and input file updates.
Parameters are written into PIHM .soil, .calib, and .lc files.

Snow-17 parameters (SCF, MFMAX, PXTEMP) are also supported for
forcing regeneration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional


from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry


PIHM_DEFAULT_BOUNDS = {
    'K_SAT': {
        'min': 1e-8, 'max': 1e-3,
        'transform': 'log',
        'description': 'Saturated hydraulic conductivity (m/s)',
    },
    'POROSITY': {
        'min': 0.05, 'max': 0.6,
        'transform': 'linear',
        'description': 'Total porosity (-)',
    },
    'VG_ALPHA': {
        'min': 0.01, 'max': 10.0,
        'transform': 'log',
        'description': 'van Genuchten alpha (1/m)',
    },
    'VG_N': {
        'min': 1.1, 'max': 5.0,
        'transform': 'linear',
        'description': 'van Genuchten n shape parameter (-)',
    },
    'MACROPORE_K': {
        'min': 1e-7, 'max': 1e-2,
        'transform': 'log',
        'description': 'Macropore hydraulic conductivity (m/s)',
    },
    'MANNINGS_N': {
        'min': 0.005, 'max': 0.3,
        'transform': 'log',
        'description': "Manning's roughness coefficient (s/m^1/3)",
    },
    'SOIL_DEPTH': {
        'min': 0.5, 'max': 10.0,
        'transform': 'linear',
        'description': 'Active soil depth (m)',
    },
    'SNOW17_SCF': {
        'min': 0.7, 'max': 1.4,
        'transform': 'linear',
        'description': 'Snowfall correction factor',
    },
    'SNOW17_MFMAX': {
        'min': 0.5, 'max': 4.0,
        'transform': 'linear',
        'description': 'Max melt factor Jun 21 (mm/C/6hr)',
    },
    'SNOW17_PXTEMP': {
        'min': -4.0, 'max': 3.0,
        'transform': 'linear',
        'description': 'Rain/snow threshold temperature (C)',
    },
}

SNOW17_PARAM_NAMES = {'SNOW17_SCF', 'SNOW17_MFMAX', 'SNOW17_PXTEMP'}


@OptimizerRegistry.register_parameter_manager('PIHM')
class PIHMParameterManager(BaseParameterManager):
    """Handles PIHM parameter bounds, normalization, and input file updates."""

    def __init__(self, config: Dict, logger: logging.Logger, pihm_settings_dir: Path):
        super().__init__(config, logger, pihm_settings_dir)

        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        pihm_params_str = config.get(
            'PIHM_PARAMS_TO_CALIBRATE',
            'K_SAT,POROSITY,VG_ALPHA,VG_N,MACROPORE_K,MANNINGS_N,SOIL_DEPTH'
        )
        self.pihm_params = [p.strip() for p in str(pihm_params_str).split(',') if p.strip()]

    def _get_parameter_names(self) -> List[str]:
        """Return PIHM parameter names from config."""
        return self.pihm_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return PIHM parameter bounds."""
        bounds: Dict[str, Dict[str, float]] = {
            k: {'min': float(v['min']), 'max': float(v['max'])}
            for k, v in PIHM_DEFAULT_BOUNDS.items()
        }

        config_bounds = self.config.get('PIHM_PARAM_BOUNDS') if isinstance(self.config, dict) else None
        if config_bounds and isinstance(config_bounds, dict):
            self.logger.info("Using config-specified PIHM parameter bounds")
            for param_name, param_bounds in config_bounds.items():
                if isinstance(param_bounds, (list, tuple)) and len(param_bounds) == 2:
                    bounds[param_name] = {
                        'min': float(param_bounds[0]),
                        'max': float(param_bounds[1]),
                    }

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update PIHM input files with new parameter values."""
        try:
            subsurface_params = {k: v for k, v in params.items() if k not in SNOW17_PARAM_NAMES}

            # Update .soil file
            self._update_soil_file(subsurface_params)

            # Update .calib file (multipliers)
            self._update_calib_file(subsurface_params)

            # Update .lc file for Manning's n
            if 'MANNINGS_N' in subsurface_params:
                self._update_lc_file(subsurface_params['MANNINGS_N'])

            self.logger.debug(f"Updated PIHM files with {len(params)} parameters")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update PIHM files: {e}")
            return False

    def _update_soil_file(self, params: Dict[str, float]) -> None:
        """Update .soil file with new parameter values."""
        soil_files = list(self.settings_dir.glob('*.soil'))
        if not soil_files:
            return

        soil_path = soil_files[0]
        lines = soil_path.read_text().strip().split('\n')

        if len(lines) < 2:
            return

        # Parse header (number of soil types)
        header = lines[0]
        new_lines = [header]

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 10:
                new_lines.append(line)
                continue

            # Format: id depth k_sat k_satv macro_k macro_depth porosity vg_alpha vg_n s_res
            soil_id = parts[0]
            depth = float(params.get('SOIL_DEPTH', parts[1]))
            k_sat = float(params.get('K_SAT', parts[2]))
            k_satv = k_sat / 10  # vertical K = horizontal K / 10
            macro_k = float(params.get('MACROPORE_K', parts[4]))
            macro_depth = float(parts[5])
            porosity = float(params.get('POROSITY', parts[6]))
            vg_alpha = float(params.get('VG_ALPHA', parts[7]))
            vg_n = float(params.get('VG_N', parts[8]))
            s_res = float(parts[9])

            new_lines.append(
                f"{soil_id} {depth:.4f} {k_sat:.6e} {k_satv:.6e} "
                f"{macro_k:.6e} {macro_depth:.4f} "
                f"{porosity:.4f} {vg_alpha:.4f} {vg_n:.4f} {s_res}"
            )

        soil_path.write_text('\n'.join(new_lines) + '\n')

    def _update_calib_file(self, params: Dict[str, float]) -> None:
        """Update .calib file with calibration multipliers."""
        calib_files = list(self.settings_dir.glob('*.calib'))
        if not calib_files:
            return

        # The calib file uses multipliers. We compute multiplier from
        # the ratio of new value to default.
        defaults = {
            'K_SAT': 1e-5,
            'POROSITY': 0.4,
            'VG_ALPHA': 1.0,
            'VG_N': 2.0,
            'MACROPORE_K': 1e-4,
        }

        calib_path = calib_files[0]
        lines = calib_path.read_text().strip().split('\n')
        new_lines = []

        param_to_calib = {
            'K_SAT': 'KSATH',
            'MACROPORE_K': 'KMACH',
            'POROSITY': 'POROSITY',
            'VG_ALPHA': 'ALPHA',
            'VG_N': 'BETA',
        }

        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                # Check if any param maps to this calib key
                for param_name, calib_key in param_to_calib.items():
                    if key == calib_key and param_name in params:
                        default = defaults.get(param_name, 1.0)
                        if default != 0:
                            multiplier = params[param_name] / default
                        else:
                            multiplier = 1.0
                        line = f"{key} {multiplier:.6f}"
                        break
            new_lines.append(line)

        calib_path.write_text('\n'.join(new_lines) + '\n')

    def _update_lc_file(self, mannings_n: float) -> None:
        """Update .lc file with new Manning's n."""
        lc_files = list(self.settings_dir.glob('*.lc'))
        if not lc_files:
            return

        lc_path = lc_files[0]
        lines = lc_path.read_text().strip().split('\n')
        new_lines = [lines[0]]  # header

        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                new_lines.append(f"{parts[0]} {parts[1]} {mannings_n:.4f} {parts[3] if len(parts) > 3 else '0.0'}")
            else:
                new_lines.append(line)

        lc_path.write_text('\n'.join(new_lines) + '\n')

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from defaults."""
        defaults = {
            'K_SAT': 1e-5,
            'POROSITY': 0.4,
            'VG_ALPHA': 1.0,
            'VG_N': 2.0,
            'MACROPORE_K': 1e-4,
            'MANNINGS_N': 0.03,
            'SOIL_DEPTH': 2.0,
            'SNOW17_SCF': 1.0,
            'SNOW17_MFMAX': 1.0,
            'SNOW17_PXTEMP': 0.0,
        }
        return {k: v for k, v in defaults.items() if k in self.pihm_params}
