"""
HYPE model runner.

Handles HYPE model execution and run-time management.
"""

from pathlib import Path
from typing import Optional, List

from ..registry import ModelRegistry
from ..base import BaseModelRunner


@ModelRegistry.register_runner('HYPE')
class HYPERunner(BaseModelRunner):  # type: ignore[misc]
    """Runner for the HYPE model."""

    MODEL_NAME = "HYPE"

    def _setup_model_specific_paths(self) -> None:
        """Set up HYPE-specific paths."""
        self.setup_dir = self.project_dir / "settings" / "HYPE"

        self.hype_exe = self.get_model_executable(
            install_path_key='HYPE_INSTALL_PATH',
            default_install_subpath='installs/hype/bin',
            exe_name_key='HYPE_EXE',
            default_exe_name='hype',
            typed_exe_accessor=lambda: self.typed_config.model.hype.exe if (self.typed_config and self.typed_config.model.hype) else None,
            must_exist=True
        )

    def _get_output_dir(self) -> Path:
        """HYPE uses custom output path resolution."""
        experiment_id = self.config.domain.experiment_id
        return self.get_config_path('EXPERIMENT_OUTPUT_HYPE', f"simulations/{experiment_id}/HYPE")

    def _build_run_command(self) -> Optional[List[str]]:
        """Build HYPE execution command."""
        return [
            str(self.hype_exe),
            str(self.setup_dir).rstrip('/') + '/'
        ]

    def _get_expected_outputs(self) -> List[str]:
        """Expected HYPE output files."""
        return [
            'timeCOUT.txt',
            'timeEVAP.txt',
            'timeSNOW.txt',
        ]
