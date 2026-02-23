"""
GSFLOW Coupling Manager.

Manages the bidirectional coupling between PRMS and MODFLOW-NWT
via SFR (Streamflow-Routing) and UZF (Unsaturated Zone Flow) packages.

In GSFLOW, coupling is handled internally by the GSFLOW binary. This module
provides utilities for configuring SFR/UZF package parameters and managing
the PRMS↔MODFLOW-NWT exchange configuration.
"""

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


class GSFLOWCouplingManager:
    """Manages PRMS↔MODFLOW-NWT coupling configuration for GSFLOW.

    GSFLOW supports three operation modes:
    - PRMS: Surface/soil processes only (PRMS standalone)
    - MODFLOW: Groundwater only (MODFLOW-NWT standalone)
    - COUPLED: Full bidirectional exchange via SFR/UZF (default)
    """

    def __init__(self, config, logger_instance=None):
        self.config = config
        self.logger = logger_instance or logger

        # Resolve GSFLOW mode from typed config
        mode = None
        try:
            gsflow_cfg = config.model.gsflow
            if gsflow_cfg is not None:
                mode = gsflow_cfg.mode
        except (AttributeError, TypeError):
            pass
        self.mode = (mode or 'COUPLED').upper()

    def get_gsflow_mode(self) -> str:
        """Return the GSFLOW operation mode."""
        return self.mode

    def validate_coupling_files(self, settings_dir: Path) -> bool:
        """Validate that required coupling files exist for the selected mode."""
        if self.mode == 'PRMS':
            required = ['control.dat', 'params.dat']
        elif self.mode == 'MODFLOW':
            required = ['modflow/modflow.nam']
        else:  # COUPLED
            required = ['control.dat', 'params.dat', 'modflow/modflow.nam']

        missing = []
        for f in required:
            if not (settings_dir / f).exists():
                missing.append(f)

        if missing:
            self.logger.error(
                f"Missing GSFLOW files for {self.mode} mode: {missing}"
            )
            return False
        return True

    def update_modflow_parameters(
        self,
        settings_dir: Path,
        params: Dict[str, float]
    ) -> bool:
        """Update MODFLOW-NWT UPW package parameters (K, SY).

        Modifies the UPW (Upstream Weighting) package file to update
        hydraulic conductivity and specific yield values.
        """
        modflow_params = {k: v for k, v in params.items() if k in ('K', 'SY')}
        if not modflow_params:
            return True

        # Find UPW package file
        upw_files = list(settings_dir.glob('*.upw'))
        if not upw_files:
            self.logger.warning("No UPW package file found; skipping MODFLOW parameter update")
            return True

        try:
            upw_file = upw_files[0]
            content = upw_file.read_text(encoding='utf-8')
            lines = content.split('\n')

            updated_lines = []
            in_hk_section = False
            in_sy_section = False

            for line in lines:
                stripped = line.strip().upper()
                if 'HK' in stripped and 'LAYER' in stripped:
                    in_hk_section = True
                    in_sy_section = False
                    updated_lines.append(line)
                    continue
                elif 'SY' in stripped and 'LAYER' in stripped:
                    in_sy_section = True
                    in_hk_section = False
                    updated_lines.append(line)
                    continue
                elif stripped.startswith('CONSTANT') and in_hk_section and 'K' in modflow_params:
                    updated_lines.append(f"         CONSTANT  {modflow_params['K']:.6e}")
                    in_hk_section = False
                    continue
                elif stripped.startswith('CONSTANT') and in_sy_section and 'SY' in modflow_params:
                    updated_lines.append(f"         CONSTANT  {modflow_params['SY']:.6e}")
                    in_sy_section = False
                    continue

                updated_lines.append(line)

            upw_file.write_text('\n'.join(updated_lines), encoding='utf-8')
            self.logger.info(f"Updated MODFLOW parameters: {modflow_params}")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error updating MODFLOW parameters: {e}")
            return False
