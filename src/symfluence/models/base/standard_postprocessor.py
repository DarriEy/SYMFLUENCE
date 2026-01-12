"""
Standard Model Postprocessor

Provides a simplified base class for models with standard streamflow extraction patterns.
Most models can inherit from this and only define configuration attributes, reducing
boilerplate from 50-150 lines to 10-20 lines.

Usage:
    @ModelRegistry.register_postprocessor('MYMODEL')
    class MyModelPostprocessor(StandardModelPostprocessor):
        model_name = "MYMODEL"
        output_file_pattern = "{domain}_{experiment}_output.nc"
        streamflow_variable = "discharge"
        streamflow_unit = "mm_per_day"  # or "cms"
        netcdf_selections = {"hru": 0}  # Optional dimension selections
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import xarray as xr

from .base_postprocessor import BaseModelPostProcessor


class StandardModelPostprocessor(BaseModelPostProcessor):
    """
    Simplified postprocessor for models with standard extraction patterns.

    Subclasses only need to define class attributes for configuration.
    Override methods only when custom logic is needed.

    Class Attributes:
        model_name: Name of the model (e.g., "FUSE", "SUMMA")
        output_file_pattern: Pattern for output file name, supports:
            - {domain}: domain_name
            - {experiment}: experiment_id
            - {start_date}: EXPERIMENT_TIME_START date portion
        streamflow_variable: Name of streamflow variable in output file
        streamflow_unit: Unit of streamflow - "mm_per_day" or "cms"
        netcdf_selections: Dict of dimension selections for xr.isel()
        text_file_separator: Separator for text files (default: ",")
        text_file_skiprows: Rows to skip in text files (default: 0)
        text_file_date_column: Name of date column (default: "DATE")
        text_file_flow_column: Name or config key for flow column
        output_dir_override: Override for output directory (default: None, uses sim_dir)
        use_routing_output: Whether to read from mizuRoute output (default: False)

    Example:
        >>> class FUSEPostprocessor(StandardModelPostprocessor):
        ...     model_name = "FUSE"
        ...     output_file_pattern = "{domain}_{experiment}_runs_best.nc"
        ...     streamflow_variable = "q_routed"
        ...     streamflow_unit = "mm_per_day"
        ...     netcdf_selections = {"param_set": 0, "latitude": 0, "longitude": 0}
    """

    # Required: subclass must define
    model_name: str = None

    # NetCDF configuration
    output_file_pattern: str = "{domain}_{experiment}_output.nc"
    streamflow_variable: str = "discharge"
    streamflow_unit: str = "cms"  # "mm_per_day" or "cms"
    netcdf_selections: Dict[str, Any] = {}

    # Text file configuration (for models that output CSV/TSV)
    text_file_separator: str = ","
    text_file_skiprows: int = 0
    text_file_date_column: str = "DATE"
    text_file_flow_column: str = None  # Column name or None to use config

    # Output directory override
    output_dir_override: str = None  # e.g., "mizuRoute" for routing output

    # Routing integration
    use_routing_output: bool = False
    routing_variable: str = "IRFroutedRunoff"
    routing_file_pattern: str = "{experiment}.h.{start_date}-03600.nc"

    # Resampling (e.g., hourly to daily)
    resample_frequency: str = None  # e.g., "D" for daily

    def _get_model_name(self) -> str:
        """Return the model name from class attribute."""
        if self.model_name is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define 'model_name' class attribute"
            )
        return self.model_name

    def _get_output_dir(self) -> Path:
        """Get the output directory for streamflow extraction."""
        if self.output_dir_override:
            return self.project_dir / 'simulations' / self.experiment_id / self.output_dir_override
        if self.use_routing_output:
            return self.project_dir / 'simulations' / self.experiment_id / 'mizuRoute'
        return self.sim_dir

    def _format_pattern(self, pattern: str) -> str:
        """Format a file pattern with available substitutions."""
        start_time = self.config_dict.get('EXPERIMENT_TIME_START', '')
        start_date = start_time.split()[0] if start_time else ''

        return pattern.format(
            domain=self.domain_name,
            experiment=self.experiment_id,
            start_date=start_date,
            model=self.model_name.lower()
        )

    def _get_output_file(self) -> Path:
        """Get the path to the output file."""
        output_dir = self._get_output_dir()

        if self.use_routing_output:
            pattern = self.routing_file_pattern
        else:
            pattern = self.output_file_pattern

        filename = self._format_pattern(pattern)
        return output_dir / filename

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract streamflow using standard patterns.

        Automatically handles:
        - NetCDF files with configurable selections
        - Text files with configurable parsing
        - Unit conversion (mm/day to cms)
        - Resampling (e.g., hourly to daily)

        Override this method only for complex extraction logic.

        Returns:
            Path to saved results CSV, or None if extraction fails
        """
        try:
            self.logger.info(f"Extracting {self.model_name} streamflow results")

            output_file = self._get_output_file()

            if not output_file.exists():
                self.logger.error(f"{self.model_name} output file not found: {output_file}")
                return None

            # Determine file type and extract accordingly
            suffix = output_file.suffix.lower()

            if suffix in ('.nc', '.nc4', '.netcdf'):
                streamflow = self._extract_from_netcdf(output_file)
            elif suffix in ('.csv', '.txt', '.tsv'):
                streamflow = self._extract_from_text(output_file)
            else:
                self.logger.error(f"Unsupported file format: {suffix}")
                return None

            if streamflow is None:
                return None

            # Apply resampling if configured
            if self.resample_frequency:
                streamflow = streamflow.resample(self.resample_frequency).mean()

            # Apply unit conversion if needed
            if self.streamflow_unit == "mm_per_day":
                streamflow = self.convert_mm_per_day_to_cms(streamflow)

            # Save to standard results format
            return self.save_streamflow_to_results(streamflow)

        except Exception as e:
            self.logger.error(f"Error extracting {self.model_name} streamflow: {str(e)}")
            raise

    def _extract_from_netcdf(self, file_path: Path) -> Optional[pd.Series]:
        """
        Extract streamflow from NetCDF file.

        Uses class attributes for variable name and dimension selections.
        """
        variable = self.routing_variable if self.use_routing_output else self.streamflow_variable
        selections = self._get_netcdf_selections()

        try:
            return self.read_netcdf_streamflow(file_path, variable, **selections)
        except Exception as e:
            self.logger.error(f"Error reading NetCDF: {e}")
            return None

    def _get_netcdf_selections(self) -> Dict[str, Any]:
        """
        Get NetCDF dimension selections.

        Override this method to add dynamic selections based on config.
        """
        selections = dict(self.netcdf_selections)  # Copy class attribute

        # Handle routing output reach selection
        if self.use_routing_output:
            sim_reach_id = self.config_dict.get('SIM_REACH_ID')
            if sim_reach_id:
                # This will be handled specially in extract_streamflow for routing
                pass

        return selections

    def _extract_from_text(self, file_path: Path) -> Optional[pd.Series]:
        """
        Extract streamflow from text/CSV file.

        Uses class attributes for separator, skiprows, and column names.
        """
        try:
            df = pd.read_csv(
                file_path,
                sep=self.text_file_separator,
                skiprows=self.text_file_skiprows
            )

            # Handle date column
            if self.text_file_date_column in df.columns:
                df[self.text_file_date_column] = pd.to_datetime(df[self.text_file_date_column])
                df.set_index(self.text_file_date_column, inplace=True)

            # Get flow column
            flow_column = self._get_flow_column(df)
            if flow_column is None or flow_column not in df.columns:
                self.logger.error(f"Flow column '{flow_column}' not found in {file_path}")
                return None

            return df[flow_column]

        except Exception as e:
            self.logger.error(f"Error reading text file: {e}")
            return None

    def _get_flow_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Get the flow column name.

        Override this method for custom column selection logic.
        """
        if self.text_file_flow_column:
            # Check if it's a config key or direct column name
            if self.text_file_flow_column.startswith('config:'):
                config_key = self.text_file_flow_column[7:]  # Remove 'config:' prefix
                return str(self.config_dict.get(config_key))
            return self.text_file_flow_column
        return None


class RoutedModelPostprocessor(StandardModelPostprocessor):
    """
    Specialized postprocessor for models that use mizuRoute routing.

    Handles the common pattern of reading from mizuRoute output files
    and selecting the outlet reach.

    Example:
        >>> class SUMMAPostprocessor(RoutedModelPostprocessor):
        ...     model_name = "SUMMA"
    """

    use_routing_output: bool = True
    routing_variable: str = "IRFroutedRunoff"
    routing_file_pattern: str = "{experiment}.h.{start_date}-03600.nc"
    resample_frequency: str = "D"  # Hourly to daily for routing output
    streamflow_unit: str = "cms"   # Routing output is already in cms

    def _setup_model_specific_paths(self) -> None:
        """Set up routing-specific paths."""
        self.mizuroute_dir = self.project_dir / 'simulations' / self.experiment_id / 'mizuRoute'

    def _extract_from_netcdf(self, file_path: Path) -> Optional[pd.Series]:
        """
        Extract routed streamflow from mizuRoute output.

        Handles reach selection based on SIM_REACH_ID config.
        """
        try:
            ds = xr.open_dataset(file_path, engine='netcdf4')

            # Get reach selection
            sim_reach_id = self.config_dict.get('SIM_REACH_ID')

            if sim_reach_id is not None:
                sim_reach_id = int(sim_reach_id)
                # Select by reach ID
                if 'reachID' in ds:
                    segment_index = ds['reachID'].values == sim_reach_id
                    ds_selected = ds.sel(seg=segment_index)
                else:
                    # Fall back to last segment (outlet)
                    ds_selected = ds.isel(seg=-1)
            else:
                # Default to last segment
                ds_selected = ds.isel(seg=-1)

            # Extract routing variable
            streamflow = ds_selected[self.routing_variable].to_pandas()

            # Round index to hour for proper resampling
            if hasattr(streamflow.index, 'round'):
                streamflow.index = streamflow.index.round(freq='h')

            ds.close()
            return streamflow

        except Exception as e:
            self.logger.error(f"Error reading routing output: {e}")
            return None
