"""
HydroGeoSphere Parameter Manager

Handles HGS parameter bounds, normalization, and input file updates.
Parameters are written into HGS .mprops, .oprops, and .grok files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


from symfluence.optimization.core.base_parameter_manager import BaseParameterManager
from symfluence.optimization.registry import OptimizerRegistry


HGS_DEFAULT_BOUNDS = {
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
    'VG_SRES': {
        'min': 0.01, 'max': 0.4,
        'transform': 'linear',
        'description': 'Residual saturation (-)',
    },
    'SS': {
        'min': 1e-7, 'max': 1e-3,
        'transform': 'log',
        'description': 'Specific storage (1/m)',
    },
    'MANNINGS_N': {
        'min': 0.005, 'max': 0.3,
        'transform': 'log',
        'description': "Manning's roughness coefficient (s/m^1/3)",
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


@OptimizerRegistry.register_parameter_manager('HYDROGEOSPHERE')
class HGSParameterManager(BaseParameterManager):
    """Handles HGS parameter bounds, normalization, and input file updates."""

    def __init__(self, config: Dict, logger: logging.Logger, hgs_settings_dir: Path):
        super().__init__(config, logger, hgs_settings_dir)

        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        hgs_params_str = config.get(
            'HGS_PARAMS_TO_CALIBRATE',
            'K_SAT,POROSITY,VG_ALPHA,VG_N,VG_SRES,SS,MANNINGS_N'
        )
        self.hgs_params = [p.strip() for p in str(hgs_params_str).split(',') if p.strip()]

    def _get_parameter_names(self) -> List[str]:
        return self.hgs_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        bounds: Dict[str, Dict[str, Any]] = {
            k: {
                'min': float(v['min']),
                'max': float(v['max']),
                'transform': v.get('transform', 'linear'),
            }
            for k, v in HGS_DEFAULT_BOUNDS.items()
        }

        # Support both dict and Pydantic config objects
        config_bounds = None
        if isinstance(self.config, dict):
            config_bounds = self.config.get('HGS_PARAM_BOUNDS')
        elif hasattr(self.config, 'get'):
            config_bounds = self.config.get('HGS_PARAM_BOUNDS')

        if config_bounds and isinstance(config_bounds, dict):
            for param_name, param_bounds in config_bounds.items():
                if isinstance(param_bounds, (list, tuple)) and len(param_bounds) == 2:
                    transform = bounds.get(param_name, {}).get('transform', 'linear')
                    bounds[param_name] = {
                        'min': float(param_bounds[0]),
                        'max': float(param_bounds[1]),
                        'transform': transform,
                    }

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update HGS input files with new parameter values."""
        try:
            subsurface_params = {k: v for k, v in params.items() if k not in SNOW17_PARAM_NAMES}

            self._update_mprops(subsurface_params)

            if 'MANNINGS_N' in subsurface_params:
                self._update_oprops(subsurface_params['MANNINGS_N'])

            self.logger.debug(f"Updated HGS files with {len(params)} parameters")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update HGS files: {e}")
            return False

    def _update_mprops(self, params: Dict[str, float]) -> None:
        """Update .mprops file with new parameter values."""
        mprops_files = list(self.settings_dir.glob('*.mprops'))
        if not mprops_files:
            return

        mprops_path = mprops_files[0]
        content = mprops_path.read_text()
        lines = content.split('\n')
        new_lines = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == 'k isotropic' and 'K_SAT' in params:
                new_lines.append(lines[i])
                i += 1
                if i < len(lines):
                    new_lines.append(f"{params['K_SAT']:.6e}")
                    i += 1
                continue

            if line == 'porosity' and 'POROSITY' in params:
                new_lines.append(lines[i])
                i += 1
                if i < len(lines):
                    new_lines.append(f"{params['POROSITY']:.4f}")
                    i += 1
                continue

            if line == 'specific storage' and 'SS' in params:
                new_lines.append(lines[i])
                i += 1
                if i < len(lines):
                    new_lines.append(f"{params['SS']:.6e}")
                    i += 1
                continue

            if line == 'van genuchten':
                new_lines.append(lines[i])
                i += 1
                # Next lines: alpha, n, sres, 0.0
                if i < len(lines):
                    vg_alpha = params.get('VG_ALPHA')
                    new_lines.append(f"  {vg_alpha:.4f}" if vg_alpha else lines[i])
                    i += 1
                if i < len(lines):
                    vg_n = params.get('VG_N')
                    new_lines.append(f"  {vg_n:.4f}" if vg_n else lines[i])
                    i += 1
                if i < len(lines):
                    vg_sres = params.get('VG_SRES')
                    new_lines.append(f"  {vg_sres:.4f}" if vg_sres else lines[i])
                    i += 1
                continue

            new_lines.append(lines[i])
            i += 1

        mprops_path.write_text('\n'.join(new_lines))

    def _update_oprops(self, mannings_n: float) -> None:
        """Update .oprops file with new Manning's n."""
        oprops_files = list(self.settings_dir.glob('*.oprops'))
        if not oprops_files:
            return

        oprops_path = oprops_files[0]
        content = oprops_path.read_text()
        lines = content.split('\n')
        new_lines = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if line == 'manning':
                new_lines.append(lines[i])
                i += 1
                if i < len(lines):
                    new_lines.append(f"{mannings_n:.4f}")
                    i += 1
                continue

            new_lines.append(lines[i])
            i += 1

        oprops_path.write_text('\n'.join(new_lines))

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        defaults = {
            'K_SAT': 1e-5,
            'POROSITY': 0.4,
            'VG_ALPHA': 1.0,
            'VG_N': 2.0,
            'VG_SRES': 0.05,
            'SS': 1e-4,
            'MANNINGS_N': 0.03,
            'SNOW17_SCF': 1.0,
            'SNOW17_MFMAX': 1.0,
            'SNOW17_PXTEMP': 0.0,
        }
        return {k: v for k, v in defaults.items() if k in self.hgs_params}
