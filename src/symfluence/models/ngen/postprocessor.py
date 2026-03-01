# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NGen Model Postprocessor.

Processes simulation outputs from the NOAA NextGen Framework (ngen).
Migrated to use StandardModelPostprocessor with multi-file support (Phase 1.5).
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from symfluence.models.base import StandardModelPostprocessor
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_postprocessor('NGEN')
class NgenPostprocessor(StandardModelPostprocessor):
    """
    Postprocessor for NextGen Framework outputs.

    Handles extraction and analysis of simulation results from multiple nexus
    output files. Uses StandardModelPostprocessor with multi-file aggregation.

    NGEN outputs streamflow to multiple nex-*_output.csv files, one per nexus.
    This postprocessor aggregates them based on CALIBRATION_NEXUS_ID config
    or sums all nexus outputs.

    Special handling:
    - Multi-file glob pattern (nex-*_output.csv)
    - Headerless CSV detection
    - Multiple nexus aggregation

    Attributes:
        model_name: "NGEN"
        output_file_glob: "nex-*_output.csv"
        aggregation_method: "sum" (all nexus outputs summed)
        streamflow_unit: "cms"
    """

    # Model identification
    model_name = "NGEN"

    # Multi-file configuration
    output_file_glob = "nex-*_output.csv"
    aggregation_method = "sum"

    # Text file parsing
    text_file_separator = ","

    # Streamflow is already in cms from NGEN
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        """Return model name for NGEN."""
        return "NGEN"

    def _get_output_dir(self) -> Path:
        """
        Get NGEN output directory.

        Returns:
            Path to ngen output directory within simulations folder
        """
        experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='run_1')
        return self.project_dir / 'simulations' / experiment_id / self.model_name

    def extract_streamflow(self, experiment_id: str = None) -> Optional[Path]:
        """
        Extract streamflow from ngen nexus outputs.

        Handles NGEN's multi-file output format with optional nexus filtering
        based on CALIBRATION_NEXUS_ID configuration.

        Note: NGEN postprocessor accepts an optional experiment_id parameter,
        which differs from the base class signature. This is necessary to support
        NGEN's multi-experiment workflow.

        Args:
            experiment_id: Experiment identifier (default: from config)

        Returns:
            Path to extracted streamflow CSV file, or None if extraction fails
        """
        self.logger.info("Extracting streamflow from ngen outputs")

        if experiment_id is None:
            experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default='run_1')

        # Get output directory
        output_dir = self.project_dir / 'simulations' / experiment_id / self.model_name

        # Find nexus output files
        nexus_files = list(output_dir.glob(self.output_file_glob))

        if not nexus_files:
            self.logger.error(f"No nexus output files found in {output_dir}")
            return None

        # Filter by CALIBRATION_NEXUS_ID if configured
        target_nexus = self._get_config_value(lambda: self.config.model.ngen.calibration_nexus_id, default=None)
        if target_nexus:
            # Normalize ID
            target_files = [
                f for f in nexus_files
                if f.stem == f"{target_nexus}_output" or f.stem == target_nexus
            ]

            if target_files:
                self.logger.info(f"Post-processing restricted to target nexus: {target_nexus}")
                nexus_files = target_files
            else:
                self.logger.warning(
                    f"Configured CALIBRATION_NEXUS_ID '{target_nexus}' not found in output files. "
                    "Processing all files."
                )

        self.logger.info(f"Found {len(nexus_files)} nexus output file(s)")

        # Read and process each nexus file
        all_streamflow: List[pd.DataFrame] = []
        for nexus_file in nexus_files:
            nexus_id = nexus_file.stem.replace('_output', '')

            try:
                df = self._read_ngen_nexus_file(nexus_file)
                if df is not None:
                    df['nexus_id'] = nexus_id
                    all_streamflow.append(df)

            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.error(f"Error processing {nexus_file}: {e}")
                continue

        if not all_streamflow:
            self.logger.error("No streamflow data could be extracted")
            return None

        # Combine all nexus outputs
        combined_streamflow = pd.concat(all_streamflow, ignore_index=True)

        # Aggregate by time (sum for multiple nexuses)
        if self.aggregation_method == "sum":
            aggregated_flow = combined_streamflow.groupby('datetime')['streamflow_cms'].sum()
        elif self.aggregation_method == "mean":
            aggregated_flow = combined_streamflow.groupby('datetime')['streamflow_cms'].mean()
        else:
            # Default to sum
            aggregated_flow = combined_streamflow.groupby('datetime')['streamflow_cms'].sum()

        # Save using standard method
        return self.save_streamflow_to_results(
            aggregated_flow,
            model_column_name=f"NGEN_{experiment_id}_discharge_cms"
        )

    def _read_ngen_nexus_file(self, nexus_file: Path) -> Optional[pd.DataFrame]:
        """
        Read a single NGEN nexus output file.

        Handles NGEN's potentially headerless CSV format by detecting the
        format from the first row.

        Args:
            nexus_file: Path to the nexus output CSV file

        Returns:
            DataFrame with 'datetime' and 'streamflow_cms' columns, or None if failed
        """
        try:
            # First try reading with header
            df = pd.read_csv(nexus_file)

            # Check for standard NGEN headerless format (index, time, flow)
            is_headerless = False
            if len(df.columns) == 3:
                # Check if first row's second column looks like a date
                # (indicating header is actually data)
                try:
                    pd.to_datetime(df.columns[1])
                    is_headerless = True
                except (ValueError, TypeError):
                    pass

            if is_headerless:
                # Reload with header=None
                df = pd.read_csv(
                    nexus_file,
                    header=None,
                    names=['index', 'time', 'flow']
                )
                flow_col = 'flow'
            else:
                # Find flow column from common names
                flow_col = None
                for col_name in ['flow', 'Flow', 'Q_OUT', 'streamflow', 'discharge']:
                    if col_name in df.columns:
                        flow_col = col_name
                        break

            if flow_col is None:
                self.logger.warning(
                    f"No flow column found in {nexus_file}. Columns: {df.columns.tolist()}"
                )
                return None

            # Find time column
            if 'time' in df.columns:
                time = pd.to_datetime(df['time'])
            elif 'Time' in df.columns:
                time = pd.to_datetime(df['Time'], unit='ns')
            else:
                self.logger.warning(f"No time column found in {nexus_file}")
                return None

            # Create standardized output dataframe
            return pd.DataFrame({
                'datetime': time,
                'streamflow_cms': df[flow_col]
            })

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"Error reading {nexus_file}: {e}")
            return None

    def _calculate_nse(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Nash-Sutcliffe Efficiency."""
        # Remove NaN values
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        obs = observed[mask]
        sim = simulated[mask]

        if len(obs) == 0:
            return np.nan

        numerator = np.sum((obs - sim) ** 2)
        denominator = np.sum((obs - np.mean(obs)) ** 2)

        if denominator == 0:
            return np.nan

        return 1 - (numerator / denominator)

    def _calculate_kge(self, observed: np.ndarray, simulated: np.ndarray) -> float:
        """Calculate Kling-Gupta Efficiency."""
        # Remove NaN values
        mask = ~(np.isnan(observed) | np.isnan(simulated))
        obs = observed[mask]
        sim = simulated[mask]

        if len(obs) == 0:
            return np.nan

        # Calculate components
        r = np.corrcoef(obs, sim)[0, 1]  # Correlation
        alpha = np.std(sim) / np.std(obs)  # Variability ratio
        beta = np.mean(sim) / np.mean(obs)  # Bias ratio

        # Calculate KGE
        kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

        return kge
