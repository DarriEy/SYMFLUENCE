"""
GR model postprocessor.

Handles extraction and processing of GR (GR4J/CemaNeige) simulation results.
Supports both lumped and distributed modes.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import xarray as xr
import geopandas as gpd

from symfluence.utils.common.constants import UnitConversion
from ..registry import ModelRegistry
from ..base import BaseModelPostProcessor

# Optional R/rpy2 support - only needed for GR models
try:
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except (ImportError, ValueError) as e:
    HAS_RPY2 = False
    robjects = None
    pandas2ri = None
    localconverter = None


@ModelRegistry.register_postprocessor('GR')
class GRPostprocessor(BaseModelPostProcessor):
    """
    Postprocessor for GR (GR4J/CemaNeige) model outputs.
    Handles extraction and processing of simulation results.
    Supports both lumped and distributed modes.
    """

    def _get_model_name(self) -> str:
        """Return the model name."""
        return "GR"

    def _setup_model_specific_paths(self) -> None:
        """Set up GR-specific paths and check dependencies."""
        # Check for R/rpy2 dependency
        if not HAS_RPY2:
            raise ImportError(
                "GR models require R and rpy2. "
                "Please install R and rpy2, or use a different model. "
                "See https://rpy2.github.io/doc/latest/html/overview.html#installation"
            )

        # GR-specific configuration
        self.spatial_mode = self.config.get('GR_SPATIAL_MODE', 'lumped')
        self._output_path = self.sim_dir  # Alias for consistency with existing code

    def extract_streamflow(self) -> Optional[Path]:
        """
        Extract simulated streamflow from GR output and append to results CSV.
        Handles both lumped and distributed modes.
        """
        try:
            self.logger.info(f"Extracting GR streamflow results ({self.spatial_mode} mode)")

            if self.spatial_mode == 'lumped':
                return self._extract_lumped_streamflow()
            else:  # distributed
                return self._extract_distributed_streamflow()

        except Exception as e:
            self.logger.error(f"Error extracting GR streamflow: {str(e)}")
            raise

    def _extract_lumped_streamflow(self) -> Optional[Path]:
        """Extract streamflow from lumped GR4J run"""

        # Check for R data file
        r_results_path = self._output_path / 'GR_results.Rdata'
        if not r_results_path.exists():
            self.logger.error(f"GR results file not found at: {r_results_path}")
            return None

        # Load R data
        robjects.r(f'load("{str(r_results_path)}")')

        # Extract simulated streamflow
        r_script = """
        data.frame(
            date = format(OutputsModel$DatesR, "%Y-%m-%d"),
            flow = OutputsModel$Qsim
        )
        """

        sim_df = robjects.r(r_script)

        # Convert to pandas
        with localconverter(robjects.default_converter + pandas2ri.converter):
            sim_df = robjects.conversion.rpy2py(sim_df)

        sim_df['date'] = pd.to_datetime(sim_df['date'])
        sim_df.set_index('date', inplace=True)

        # Get catchment area
        basin_dir = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')
        basin_name = self.config.get('RIVER_BASINS_NAME')
        if basin_name == 'default' or basin_name is None:
            basin_name = f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
        basin_path = basin_dir / basin_name
        basin_gdf = gpd.read_file(basin_path)

        area_km2 = basin_gdf['GRU_area'].sum() / 1e6
        self.logger.info(f"Total catchment area: {area_km2:.2f} km2")

        # Convert units from mm/day to m3/s (cms)
        q_sim_cms = sim_df['flow'] * area_km2 / UnitConversion.MM_DAY_TO_CMS

        # Read existing results or create new
        output_file = self.results_dir / f"{self.config.get('EXPERIMENT_ID')}_results.csv"
        if output_file.exists():
            results_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
        else:
            results_df = pd.DataFrame(index=q_sim_cms.index)

        # Add GR results
        results_df['GR_discharge_cms'] = q_sim_cms

        # Save updated results
        results_df.to_csv(output_file)

        self.logger.info(f"GR results appended to: {output_file}")
        return output_file

    def _extract_distributed_streamflow(self) -> Optional[Path]:
        """Extract streamflow from distributed GR4J run (after routing)"""

        # Check if routing was performed
        needs_routing = self.config.get('GR_ROUTING_INTEGRATION') == 'mizuRoute'

        if needs_routing:
            # Get routed streamflow from mizuRoute output
            mizuroute_output_dir = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'mizuRoute'

            # Find mizuRoute output file
            output_files = list(mizuroute_output_dir.glob(f"{self.config.get('EXPERIMENT_ID')}*.nc"))

            if not output_files:
                self.logger.error(f"No mizuRoute output files found in {mizuroute_output_dir}")
                return None

            # Use the first output file
            mizuroute_file = output_files[0]
            self.logger.info(f"Reading routed streamflow from: {mizuroute_file}")

            ds = xr.open_dataset(mizuroute_file)

            # Extract streamflow at outlet (typically the last reach)
            # mizuRoute typically names the variable 'IRFroutedRunoff' or similar
            streamflow_vars = ['IRFroutedRunoff', 'dlayRunoff', 'KWTroutedRunoff']
            streamflow_var = None

            for var in streamflow_vars:
                if var in ds.variables:
                    streamflow_var = var
                    break

            if streamflow_var is None:
                self.logger.error(f"Could not find streamflow variable in mizuRoute output. Available: {list(ds.variables)}")
                return None

            # Get streamflow at outlet (last segment)
            q_routed = ds[streamflow_var].isel(seg=-1)

            # Convert to DataFrame
            q_df = q_routed.to_dataframe(name='flow')
            q_df = q_df.reset_index()

            # Convert time if needed
            if 'time' in q_df.columns:
                q_df['time'] = pd.to_datetime(q_df['time'])
                q_df.set_index('time', inplace=True)

        else:
            # No routing - sum all HRU outputs
            gr_output = self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'GR' / \
                        f"{self.domain_name}_{self.config.get('EXPERIMENT_ID')}_runs_def.nc"

            if not gr_output.exists():
                self.logger.error(f"GR output not found: {gr_output}")
                return None

            ds = xr.open_dataset(gr_output)

            # Sum across all GRUs
            routing_var = self.config.get('SETTINGS_MIZU_ROUTING_VAR', 'q_routed')
            q_total = ds[routing_var].sum(dim='gru')

            # Convert to DataFrame
            q_df = q_total.to_dataframe(name='flow')

        # Convert from mm/day to m3/s
        basin_dir = self._get_default_path('RIVER_BASINS_PATH', 'shapefiles/river_basins')
        basin_name = self.config.get('RIVER_BASINS_NAME')
        if basin_name == 'default' or basin_name is None:
            basin_name = f"{self.domain_name}_riverBasins_{self.config.get('DOMAIN_DEFINITION_METHOD')}.shp"
        basin_path = basin_dir / basin_name
        basin_gdf = gpd.read_file(basin_path)

        area_km2 = basin_gdf['GRU_area'].sum() / 1e6
        self.logger.info(f"Total catchment area: {area_km2:.2f} km2")

        # Convert units
        q_cms = q_df['flow'] * area_km2 / UnitConversion.MM_DAY_TO_CMS

        # Save to results
        output_file = self.results_dir / f"{self.config.get('EXPERIMENT_ID')}_results.csv"
        if output_file.exists():
            results_df = pd.read_csv(output_file, index_col=0, parse_dates=True)
        else:
            results_df = pd.DataFrame(index=q_cms.index)

        results_df['GR_discharge_cms'] = q_cms
        results_df.to_csv(output_file)

        self.logger.info(f"Distributed GR results appended to: {output_file}")
        return output_file

    @property
    def output_path(self):
        """Get output path for backwards compatibility"""
        return self.project_dir / 'simulations' / self.config.get('EXPERIMENT_ID') / 'GR'
