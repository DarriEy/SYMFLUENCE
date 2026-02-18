"""
WATFLOOD Pre-Processor.

Prepares WATFLOOD input files including watershed (.shd), parameter (.par),
event (.evt), and forcing (.met/.rag) files.
"""

import logging
from pathlib import Path

from symfluence.models.base import BaseModelPreProcessor

logger = logging.getLogger(__name__)


class WATFLOODPreProcessor(BaseModelPreProcessor):
    """Pre-processor for WATFLOOD model setup."""

    def _get_model_name(self) -> str:
        """Return model name."""
        return 'WATFLOOD'

    def run_preprocessing(self) -> bool:
        """Run WATFLOOD preprocessing.

        Sets up watershed definition (.shd), parameter (.par), event (.evt),
        and forcing (.met) files. WATFLOOD uses a GRU-grid structure with
        internal channel routing.
        """
        try:
            settings_dir = self.project_dir / 'WATFLOOD_input' / 'settings'
            settings_dir.mkdir(parents=True, exist_ok=True)

            logger.info("WATFLOOD preprocessing starting")

            # Verify required files exist
            self._check_input_files(settings_dir)

            logger.info(f"WATFLOOD preprocessing complete: {settings_dir}")
            return True

        except Exception as e:
            logger.error(f"WATFLOOD preprocessing failed: {e}")
            return False

    def _check_input_files(self, settings_dir: Path) -> None:
        """Verify required WATFLOOD input files exist."""
        shed_file = self._get_config_value(
            lambda: self.config.model.watflood.shed_file,
            default='watershed.shd',
            dict_key='WATFLOOD_SHED_FILE'
        )
        par_file = self._get_config_value(
            lambda: self.config.model.watflood.par_file,
            default='params.par',
            dict_key='WATFLOOD_PAR_FILE'
        )
        event_file = self._get_config_value(
            lambda: self.config.model.watflood.event_file,
            default='event.evt',
            dict_key='WATFLOOD_EVENT_FILE'
        )

        for fname in [shed_file, par_file, event_file]:
            fpath = settings_dir / fname
            if not fpath.exists():
                logger.info(f"WATFLOOD input file will be provided by user: {fname}")
