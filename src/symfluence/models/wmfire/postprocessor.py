"""
WMFire PostProcessor for SYMFLUENCE

Generates fire perimeter shapefiles from WMFire simulation outputs.
Since WMFire outputs fire sizes (not spatial grids), this postprocessor
creates approximate elliptical perimeters based on:
- Fire size from FireSizes*.txt
- Ignition location from fire.def
- Wind/spread parameters from fire.def
- Grid metadata from patch_grid

The elliptical approximation follows FBP-style fire shape assumptions
where fires spread as ellipses elongated in the wind direction.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from symfluence.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class FireEvent:
    """Container for a single fire event from WMFire output."""
    size_cells: int
    year: int
    month: int
    wind_speed: float
    moisture: float
    spread_events: int

    @property
    def size_ha(self) -> float:
        """Fire size in hectares (assuming 30m grid)."""
        return self.size_cells * 0.09  # 30m * 30m = 900m² = 0.09ha


@dataclass
class FireDefParams:
    """Parameters parsed from fire.def file."""
    ignition_row: int
    ignition_col: int
    n_rows: int
    n_cols: int
    fire_verbose: int = 1
    fire_write: int = 1
    spread_calc_type: int = 9
    mean_log_wind: float = 0.5
    sd_log_wind: float = 0.6
    windmax: float = 1.0


@ModelRegistry.register_postprocessor('WMFire')
class WMFirePostProcessor:
    """
    PostProcessor for WMFire fire spread simulation results.

    Generates approximate fire perimeter shapefiles by creating
    elliptical representations of fires based on size and spread
    parameters.
    """

    def __init__(
        self,
        config,
        logger_instance: Optional[logging.Logger] = None,
        resolution: float = 30.0
    ):
        """
        Initialize WMFire postprocessor.

        Args:
            config: SymfluenceConfig object
            logger_instance: Optional logger
            resolution: Grid resolution in meters (default 30m)
        """
        self.config = config
        self.logger = logger_instance or logger
        self.resolution = resolution

        # Setup paths
        self._setup_paths()

    def _setup_paths(self) -> None:
        """Setup directory paths."""
        # Get project directory
        if hasattr(self.config, 'system') and hasattr(self.config.system, 'data_dir'):
            data_dir = Path(self.config.system.data_dir)
        else:
            data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR', '.'))

        domain_name = self.config.domain.name if hasattr(self.config, 'domain') else self.config.get('DOMAIN_NAME', 'domain')
        experiment_id = self.config.domain.experiment_id if hasattr(self.config, 'domain') else self.config.get('EXPERIMENT_ID', 'default')

        self.project_dir = data_dir / f"domain_{domain_name}"
        self.rhessys_input_dir = self.project_dir / "RHESSys_input"
        self.simulation_dir = self.project_dir / "simulations" / experiment_id / "RHESSys"
        self.output_dir = self.project_dir / "simulations" / experiment_id / "fire_perimeters"

    def run_postprocessing(self) -> bool:
        """
        Run WMFire postprocessing to generate fire perimeters.

        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Running WMFire fire perimeter postprocessing...")

        try:
            # Parse fire.def parameters
            fire_def_path = self.rhessys_input_dir / "defs" / "fire.def"
            fire_params = self._parse_fire_def(fire_def_path)

            if fire_params is None:
                self.logger.error(f"Could not parse fire.def: {fire_def_path}")
                return False

            # Parse fire sizes
            fire_events = self._parse_fire_sizes()

            if not fire_events:
                self.logger.warning("No fire events found in FireSizes output")
                return False

            self.logger.info(f"Found {len(fire_events)} fire events")

            # Get grid metadata for georeferencing
            grid_metadata = self._get_grid_metadata()

            # Generate perimeters for significant fires
            perimeters = self._generate_perimeters(
                fire_events,
                fire_params,
                grid_metadata
            )

            if perimeters:
                # Write perimeter shapefiles
                self._write_perimeters(perimeters, grid_metadata)

                # Write summary
                self._write_summary(fire_events, perimeters)

                self.logger.info(f"Generated {len(perimeters)} fire perimeter(s)")
                return True
            else:
                self.logger.warning("No significant fires to generate perimeters for")
                return True

        except Exception as e:
            self.logger.error(f"WMFire postprocessing failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _parse_fire_def(self, fire_def_path: Path) -> Optional[FireDefParams]:
        """
        Parse fire.def file for fire parameters.

        Args:
            fire_def_path: Path to fire.def file

        Returns:
            FireDefParams or None if parsing fails
        """
        if not fire_def_path.exists():
            return None

        params = {}

        with open(fire_def_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    value = parts[0]
                    key = parts[1]

                    # Parse numeric values
                    try:
                        if '.' in value:
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except ValueError:
                        params[key] = value

        try:
            return FireDefParams(
                ignition_row=params.get('ignition_row', 0),
                ignition_col=params.get('ignition_col', 0),
                n_rows=params.get('n_rows', 100),
                n_cols=params.get('n_cols', 100),
                fire_verbose=params.get('fire_verbose', 1),
                fire_write=params.get('fire_write', 1),
                spread_calc_type=params.get('spread_calc_type', 9),
                mean_log_wind=params.get('mean_log_wind', 0.5),
                sd_log_wind=params.get('sd_log_wind', 0.6),
                windmax=params.get('windmax', 1.0),
            )
        except Exception as e:
            self.logger.error(f"Error creating FireDefParams: {e}")
            return None

    def _parse_fire_sizes(self) -> List[FireEvent]:
        """
        Parse FireSizes*.txt files for fire events.

        Returns:
            List of FireEvent objects
        """
        events = []

        # Find FireSizes files
        fire_size_files = list(self.simulation_dir.glob("FireSizes*.txt"))

        for fsf in fire_size_files:
            try:
                with open(fsf, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) >= 6:
                            try:
                                event = FireEvent(
                                    size_cells=int(parts[0]),
                                    year=int(parts[1]),
                                    month=int(parts[2]),
                                    wind_speed=float(parts[3]),
                                    moisture=float(parts[4]),
                                    spread_events=int(parts[5]),
                                )
                                events.append(event)
                            except (ValueError, IndexError):
                                continue

            except Exception as e:
                self.logger.warning(f"Error parsing {fsf}: {e}")

        return events

    def _get_grid_metadata(self) -> Dict[str, Any]:
        """
        Get grid metadata from patch_grid files.

        Returns:
            Dictionary with grid metadata (transform, CRS, etc.)
        """
        metadata = {
            'resolution': self.resolution,
            'crs': 'EPSG:32611',  # Default UTM 11N, will be updated if available
            'transform': None,
            'n_rows': None,
            'n_cols': None,
        }

        # Try to read from GeoTIFF
        patch_grid_tif = self.rhessys_input_dir / "fire" / "patch_grid.tif"

        if patch_grid_tif.exists():
            try:
                import rasterio

                with rasterio.open(patch_grid_tif) as src:
                    metadata['crs'] = str(src.crs)
                    metadata['transform'] = tuple(src.transform)[:6]
                    metadata['n_rows'] = src.height
                    metadata['n_cols'] = src.width
                    metadata['bounds'] = src.bounds

                self.logger.info(f"Grid metadata from GeoTIFF: {src.height}x{src.width}, {src.crs}")

            except ImportError:
                self.logger.warning("rasterio not available, using default metadata")
            except Exception as e:
                self.logger.warning(f"Error reading patch_grid.tif: {e}")

        # Fall back to text file dimensions
        if metadata['n_rows'] is None:
            patch_grid_txt = self.rhessys_input_dir / "fire" / "patch_grid.txt"
            if patch_grid_txt.exists():
                try:
                    with open(patch_grid_txt, 'r') as f:
                        lines = f.readlines()
                    metadata['n_rows'] = len(lines)
                    if lines:
                        metadata['n_cols'] = len(lines[0].split())
                except Exception:
                    pass

        return metadata

    def _generate_perimeters(
        self,
        fire_events: List[FireEvent],
        fire_params: FireDefParams,
        grid_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate elliptical fire perimeters for significant fires.

        Args:
            fire_events: List of fire events
            fire_params: Fire parameters from fire.def
            grid_metadata: Grid georeferencing metadata

        Returns:
            List of perimeter dictionaries with geometry and attributes
        """
        perimeters: list[dict[str, Any]] = []

        # Filter significant fires (> 10 cells = ~1 hectare)
        significant_fires = [e for e in fire_events if e.size_cells > 10]

        if not significant_fires:
            return perimeters

        # Get ignition point in grid coordinates
        ign_row = fire_params.ignition_row
        ign_col = fire_params.ignition_col

        # Convert to map coordinates if transform available
        if grid_metadata['transform']:
            a, b, c, d, e, f = grid_metadata['transform']
            # x = a * col + c
            # y = e * row + f
            ign_x = a * ign_col + c + a / 2  # Center of cell
            ign_y = e * ign_row + f + e / 2
        else:
            # Use grid coordinates directly
            ign_x = ign_col * self.resolution
            ign_y = ign_row * self.resolution

        self.logger.info(f"Ignition point: row={ign_row}, col={ign_col}, x={ign_x:.1f}, y={ign_y:.1f}")

        # Generate perimeter for each significant fire
        for i, event in enumerate(significant_fires):
            # Calculate approximate fire dimensions
            # Assuming elliptical shape with length-to-breadth ratio based on wind
            area_m2 = event.size_cells * self.resolution * self.resolution

            # Use wind speed to estimate length-to-breadth ratio
            # Higher wind = more elongated fire
            lb_ratio = 1.0 + event.wind_speed * 0.3  # Simple approximation
            lb_ratio = min(max(lb_ratio, 1.0), 4.0)  # Clamp to reasonable range

            # Calculate ellipse semi-axes from area
            # Area = π * a * b, where a/b = lb_ratio
            # Area = π * a * (a/lb_ratio) = π * a² / lb_ratio
            # a = sqrt(Area * lb_ratio / π)
            semi_major = np.sqrt(area_m2 * lb_ratio / np.pi)
            semi_minor = semi_major / lb_ratio

            # Wind direction (use mean from event, convert to radians)
            # Assuming wind_speed column represents direction in some way
            # For now, use a default northward spread (0 degrees)
            wind_dir_rad = np.random.uniform(0, 2 * np.pi)  # Random for now

            # Generate ellipse vertices
            n_vertices = 64
            theta = np.linspace(0, 2 * np.pi, n_vertices)

            # Ellipse in local coordinates (centered at ignition)
            x_local = semi_major * np.cos(theta)
            y_local = semi_minor * np.sin(theta)

            # Rotate by wind direction
            cos_dir = np.cos(wind_dir_rad)
            sin_dir = np.sin(wind_dir_rad)
            x_rot = x_local * cos_dir - y_local * sin_dir
            y_rot = x_local * sin_dir + y_local * cos_dir

            # Translate to ignition point
            # Fire spreads primarily downwind from ignition
            # Offset the ellipse so ignition is at the back
            offset_x = -semi_major * 0.3 * cos_dir  # Back of fire at ignition
            offset_y = -semi_major * 0.3 * sin_dir

            x_coords = x_rot + ign_x + offset_x
            y_coords = y_rot + ign_y + offset_y

            # Create perimeter record
            perimeter = {
                'id': i + 1,
                'year': event.year,
                'month': event.month,
                'size_cells': event.size_cells,
                'size_ha': event.size_ha,
                'wind_speed': event.wind_speed,
                'moisture': event.moisture,
                'coords': list(zip(x_coords, y_coords)),
                'ignition_x': ign_x,
                'ignition_y': ign_y,
                'semi_major': semi_major,
                'semi_minor': semi_minor,
            }

            perimeters.append(perimeter)

            self.logger.info(
                f"Fire {i+1}: {event.year}-{event.month:02d}, "
                f"{event.size_ha:.1f} ha, "
                f"axes={semi_major:.0f}m x {semi_minor:.0f}m"
            )

        return perimeters

    def _write_perimeters(
        self,
        perimeters: List[Dict[str, Any]],
        grid_metadata: Dict[str, Any]
    ) -> None:
        """
        Write fire perimeters to shapefiles.

        Args:
            perimeters: List of perimeter dictionaries
            grid_metadata: Grid georeferencing metadata
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
        except ImportError:
            self.logger.warning("geopandas not available, skipping shapefile output")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create GeoDataFrame
        features = []

        for perim in perimeters:
            geom = Polygon(perim['coords'])

            features.append({
                'geometry': geom,
                'fire_id': perim['id'],
                'year': perim['year'],
                'month': perim['month'],
                'size_cells': perim['size_cells'],
                'size_ha': perim['size_ha'],
                'wind_speed': perim['wind_speed'],
                'moisture': perim['moisture'],
                'ign_x': perim['ignition_x'],
                'ign_y': perim['ignition_y'],
            })

        gdf = gpd.GeoDataFrame(features, crs=grid_metadata['crs'])

        # Write combined shapefile
        combined_path = self.output_dir / "wmfire_perimeters.shp"
        gdf.to_file(combined_path)
        self.logger.info(f"Combined perimeters written: {combined_path}")

        # Write individual shapefiles for each fire
        for perim in perimeters:
            individual_gdf = gdf[gdf['fire_id'] == perim['id']]
            filename = f"wmfire_fire_{perim['year']}_{perim['month']:02d}_{perim['id']}.shp"
            individual_path = self.output_dir / filename
            individual_gdf.to_file(individual_path)

        self.logger.info(f"Individual perimeters written to: {self.output_dir}")

    def _write_summary(
        self,
        fire_events: List[FireEvent],
        perimeters: List[Dict[str, Any]]
    ) -> None:
        """
        Write summary JSON file.

        Args:
            fire_events: All fire events
            perimeters: Generated perimeters
        """
        summary: dict[str, Any] = {
            'total_fires': len(fire_events),
            'significant_fires': len(perimeters),
            'total_area_burned_ha': sum(e.size_ha for e in fire_events),
            'fires_by_year': {},
            'perimeters': []
        }

        # Aggregate by year
        for event in fire_events:
            year = str(event.year)
            if year not in summary['fires_by_year']:
                summary['fires_by_year'][year] = {
                    'count': 0,
                    'total_area_ha': 0,
                    'max_fire_ha': 0
                }
            summary['fires_by_year'][year]['count'] += 1
            summary['fires_by_year'][year]['total_area_ha'] += event.size_ha
            summary['fires_by_year'][year]['max_fire_ha'] = max(
                summary['fires_by_year'][year]['max_fire_ha'],
                event.size_ha
            )

        # Add perimeter info
        for perim in perimeters:
            summary['perimeters'].append({
                'id': perim['id'],
                'year': perim['year'],
                'month': perim['month'],
                'size_ha': perim['size_ha'],
                'ignition': [perim['ignition_x'], perim['ignition_y']],
                'semi_major_m': perim['semi_major'],
                'semi_minor_m': perim['semi_minor'],
            })

        # Write summary
        summary_path = self.output_dir / "wmfire_summary.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Summary written: {summary_path}")
