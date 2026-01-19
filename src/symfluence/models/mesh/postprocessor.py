"""
MESH model postprocessor.

Handles extraction and processing of MESH model simulation results.
Migrated to use StandardModelPostprocessor for reduced boilerplate (Phase 1.3).
"""

from pathlib import Path
from typing import Optional

from ..registry import ModelRegistry
from ..base import StandardModelPostprocessor


@ModelRegistry.register_postprocessor('MESH')
class MESHPostProcessor(StandardModelPostprocessor):
    """
    Postprocessor for the MESH model.

    Handles extraction and processing of MESH model simulation results.
    Uses StandardModelPostprocessor with configuration-based extraction.

    MESH outputs streamflow to MESH_output_streamflow.csv with columns:
    DAY, YEAR, QOMEAS1, QOSIM1, QOMEAS2, QOSIM2, ...

    Special handling:
    - Julian date format (DAY + YEAR columns)
    - Column pattern matching for QOSIM* columns
    - Output directory is forcing/MESH_input (not standard sim_dir)

    Attributes:
        model_name: "MESH"
        output_file_pattern: "MESH_output_streamflow.csv"
        date_parser_type: "julian" for DAY+YEAR columns
        outlet_column_pattern: r"QOSIM\\d+" to match QOSIM1, QOSIM2, etc.
    """

    # Model identification
    model_name = "MESH"

    # Output file configuration
    output_file_pattern = "MESH_output_streamflow.csv"

    # Text file parsing
    text_file_separator = ","
    text_file_skiprows = 0  # MESH has a proper header

    # Julian date parsing (DAY + YEAR columns)
    date_parser_type = "julian"

    # Column pattern matching for QOSIM columns
    outlet_column_pattern = r"QOSIM\d+"
    outlet_selection_method = "pattern"

    # Streamflow is already in cms from MESH
    streamflow_unit = "cms"

    def _get_model_name(self) -> str:
        """Return the model name."""
        return "MESH"

    def _setup_model_specific_paths(self) -> None:
        """Set up MESH-specific paths."""
        self.mesh_setup_dir = self.project_dir / "settings" / "MESH"
        self.forcing_basin_path = self.project_dir / 'forcing' / 'basin_averaged_data'
        self.forcing_mesh_path = self.project_dir / 'forcing' / 'MESH_input'
        self.catchment_path = self._get_default_path('CATCHMENT_PATH', 'shapefiles/catchment')
        self.catchment_name = self.config_dict.get('CATCHMENT_SHP_NAME')
        if self.catchment_name == 'default':
            self.catchment_name = f"{self.domain_name}_HRUs_{self.config_dict.get('DOMAIN_DISCRETIZATION')}.shp"

    def _get_output_dir(self) -> Path:
        """
        Override: MESH outputs to forcing directory, not simulation directory.

        Returns:
            Path to MESH output directory (forcing/MESH_input)
        """
        return self.project_dir / 'forcing' / 'MESH_input'

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow from MESH outputs.

        Overrides base method to handle MESH's alternate file location
        (MESH_output_streamflow_ts.csv as fallback).

        Returns:
            Optional[Path]: Path to processed streamflow file, or None if extraction fails
        """
        self.logger.info("Extracting streamflow from MESH outputs")

        # MESH outputs to the forcing directory where it runs
        mesh_output_file = self._get_output_dir() / self.output_file_pattern

        if not mesh_output_file.exists():
            # Try alternative timestep output
            mesh_output_file = self._get_output_dir() / 'MESH_output_streamflow_ts.csv'
            if not mesh_output_file.exists():
                self.logger.error(f"MESH streamflow output not found at {mesh_output_file}")
                return None

        try:
            # Use StandardModelPostprocessor text extraction
            streamflow = self._extract_from_text(mesh_output_file)

            if streamflow is None:
                return None

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            # Apply unit conversion if needed
            if self.streamflow_unit == "mm_per_day":
                streamflow = self.convert_mm_per_day_to_cms(streamflow)

            # Use inherited save method
            return self.save_streamflow_to_results(
                streamflow,
                model_column_name='MESH_discharge_cms'
            )

        except Exception as e:
            self.logger.error(f"Error extracting MESH streamflow: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
