"""
Threshold-based stream delineation method.

Uses drainage area threshold to identify stream initiation points.
Optionally performs drop analysis to objectively determine the optimal threshold.

Extracted from geofabric_utils.py (2026-01-01)
"""

from pathlib import Path
from typing import Any, Dict, Optional


class StreamThresholdMethod:
    """
    Threshold-based stream identification.

    The simplest and most commonly used method - streams are identified
    where contributing area exceeds a specified threshold.
    """

    def __init__(self, taudem_executor: Any, config: Dict, logger: Any, interim_dir: Path, reporting_manager: Optional[Any] = None):
        """
        Initialize threshold method.

        Args:
            taudem_executor: TauDEMExecutor instance for running commands
            config: Configuration dictionary
            logger: Logger instance
            interim_dir: Directory for interim TauDEM files
            reporting_manager: ReportingManager instance
        """
        self.taudem = taudem_executor
        self.config = config
        self.logger = logger
        self.interim_dir = interim_dir
        self.reporting_manager = reporting_manager
        self.taudem_dir = taudem_executor.taudem_dir

    def run(self, dem_path: Path, pour_point_path: Path, mpi_prefix: str) -> None:
        """
        Run threshold-based stream identification.

        Optionally performs drop analysis to objectively determine the optimal threshold.

        Args:
            dem_path: Path to the DEM file
            pour_point_path: Path to the pour point shapefile
            mpi_prefix: MPI command prefix
        """
        # Determine threshold
        if self.config.get('USE_DROP_ANALYSIS', False):
            from .drop_analysis import DropAnalysisMethod
            drop_analyzer = DropAnalysisMethod(
                self.taudem, self.config, self.logger, self.interim_dir, self.reporting_manager
            )
            optimal_threshold = drop_analyzer.run(mpi_prefix)

            if optimal_threshold is not None:
                threshold = optimal_threshold
                self.logger.info(f"Using threshold from drop analysis: {threshold}")
            else:
                threshold = self.config.get('STREAM_THRESHOLD', 10000)
                self.logger.warning(f"Drop analysis failed. Using configured threshold: {threshold}")
        else:
            threshold = self.config.get('STREAM_THRESHOLD', 10000)

        max_distance = self.config.get('MOVE_OUTLETS_MAX_DISTANCE', 200)

        steps = [
            f"{mpi_prefix}{self.taudem_dir}/gridnet -p {self.interim_dir}/elv-fdir.tif -plen {self.interim_dir}/elv-plen.tif -tlen {self.interim_dir}/elv-tlen.tif -gord {self.interim_dir}/elv-gord.tif",
            f"{mpi_prefix}{self.taudem_dir}/threshold -ssa {self.interim_dir}/elv-ad8.tif -src {self.interim_dir}/elv-src.tif -thresh {threshold}",
            f"{mpi_prefix}{self.taudem_dir}/moveoutletstostreams -p {self.interim_dir}/elv-fdir.tif -src {self.interim_dir}/elv-src.tif -o {pour_point_path} -om {self.interim_dir}/gauges.shp -md {max_distance}",
            f"{mpi_prefix}{self.taudem_dir}/streamnet -fel {self.interim_dir}/elv-fel.tif -p {self.interim_dir}/elv-fdir.tif -ad8 {self.interim_dir}/elv-ad8.tif -src {self.interim_dir}/elv-src.tif -ord {self.interim_dir}/elv-ord.tif -tree {self.interim_dir}/basin-tree.dat -coord {self.interim_dir}/basin-coord.dat -net {self.interim_dir}/basin-streams.shp -o {self.interim_dir}/gauges.shp -w {self.interim_dir}/elv-watersheds.tif"
        ]

        for step in steps:
            self.taudem.run_command(step)
            self.logger.info("Completed threshold method step")

        self.logger.info(f"Threshold-based stream identification completed with threshold={threshold}")
