"""
FEWS Post-Adapter.

Orchestrates the post-processing phase of a FEWS General Adapter run:
  1. Load model output
  2. Apply reverse ID mapping (SYMFLUENCE -> FEWS variable names)
  3. Write output in FEWS format (PI-XML or NetCDF-CF)
  4. Export state files
  5. Write diagnostics
"""

import logging
from pathlib import Path
from typing import Optional

import xarray as xr

from .config import FEWSConfig
from .exceptions import FEWSAdapterError
from .id_map import IDMapper
from .netcdf_cf import write_fews_netcdf
from .pi_diagnostics import DiagnosticsCollector
from .pi_xml import write_pi_xml_timeseries
from .run_info import RunInfo, parse_run_info
from .state import export_states

logger = logging.getLogger(__name__)


class FEWSPostAdapter:
    """Orchestrates the FEWS post-adapter workflow.

    Args:
        run_info_path: Path to run_info.xml
        config_path: Path to SYMFLUENCE config YAML (for locating output)
        data_format: Data exchange format (``pi-xml`` or ``netcdf-cf``)
        id_map_path: Optional path to YAML ID mapping file
        fews_config: Optional pre-built FEWSConfig
    """

    def __init__(
        self,
        run_info_path: Path,
        config_path: Optional[Path] = None,
        data_format: str = "netcdf-cf",
        id_map_path: Optional[str] = None,
        fews_config: Optional[FEWSConfig] = None,
    ) -> None:
        self.run_info_path = Path(run_info_path)
        self.config_path = config_path
        self.data_format = data_format
        self.id_map_path = id_map_path
        self._fews_config = fews_config

    def run(self, diag: Optional[DiagnosticsCollector] = None) -> Path:
        """Execute the post-adapter workflow.

        Args:
            diag: Optional diagnostics collector

        Returns:
            Path to the FEWS output directory

        Raises:
            FEWSAdapterError: On failure
        """
        # 1. Parse run_info
        run_info = parse_run_info(self.run_info_path)
        logger.info("Post-adapter: output dir = %s", run_info.output_dir)
        if diag:
            diag.info(f"Post-adapter started, output dir: {run_info.output_dir}")

        # 2. Build FEWSConfig
        fews_cfg = self._fews_config or FEWSConfig(
            work_dir=str(run_info.work_dir),
            data_format=self.data_format,
            id_map_file=self.id_map_path,
        )

        # 3. Build ID mapper
        mapper = IDMapper(fews_cfg)

        # 4. Load model output
        output_ds = self._load_model_output(run_info)
        if diag:
            diag.info(f"Loaded model output: {list(output_ds.data_vars)}")

        # 5. Apply reverse ID mapping
        output_ds = mapper.rename_dataset_sym_to_fews(output_ds)
        logger.info("Reverse-mapped variables: %s", list(output_ds.data_vars))
        if diag:
            diag.info(f"Mapped to FEWS names: {list(output_ds.data_vars)}")

        # 6. Write output
        output_dir = run_info.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if fews_cfg.data_format == "pi-xml":
            output_path = output_dir / "timeseries.xml"
            write_pi_xml_timeseries(
                output_ds,
                output_path,
                missing_value=fews_cfg.missing_value,
            )
        else:
            output_path = output_dir / "output.nc"
            write_fews_netcdf(output_ds, output_path)

        logger.info("Wrote FEWS output to %s", output_path)
        if diag:
            diag.info(f"Wrote output to {output_path}")

        # 7. Export states
        if run_info.state_output_dir and fews_cfg.state_dir:
            # Use StateManager if model runner is available and state-capable
            model_runner = getattr(self, '_model_runner', None)
            if model_runner and hasattr(model_runner, 'supports_state') and model_runner.supports_state:
                from symfluence.models.state import StateManager
                StateManager.export_to_fews(model_runner, run_info.state_output_dir)
                if diag:
                    diag.info("Exported state files via StateManager")
            else:
                export_states(Path(fews_cfg.state_dir), run_info.state_output_dir)
                if diag:
                    diag.info("Exported state files")

        return output_dir

    def _load_model_output(self, run_info: RunInfo) -> xr.Dataset:
        """Load model output from the working directory."""
        # Look for NetCDF output in common locations
        search_dirs = [
            run_info.work_dir,
            run_info.work_dir / "output",
            run_info.input_dir,
        ]

        for search_dir in search_dirs:
            if not search_dir.is_dir():
                continue
            nc_files = sorted(search_dir.glob("*output*.nc")) or sorted(search_dir.glob("*.nc"))
            if nc_files:
                try:
                    ds = xr.open_dataset(nc_files[0])
                    logger.info("Loaded model output from %s", nc_files[0])
                    return ds
                except Exception:
                    continue

        raise FEWSAdapterError(
            f"No model output found in {run_info.work_dir}. "
            f"Searched: {[str(d) for d in search_dirs]}"
        )
