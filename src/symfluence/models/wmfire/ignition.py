"""
WMFire Ignition Point and Fire Perimeter Module

Handles ignition point management and fire perimeter comparison
for WMFire fire spread modeling validation.
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

if TYPE_CHECKING:
    import geopandas as gpd


logger = logging.getLogger(__name__)


@dataclass
class IgnitionPoint:
    """Container for ignition point data."""
    latitude: float
    longitude: float
    name: str = "ignition"
    date: Optional[datetime] = None
    source: str = "config"  # 'config', 'shapefile', 'random'

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'name': self.name,
            'date': self.date.isoformat() if self.date else None,
            'source': self.source,
        }


class IgnitionManager:
    """
    Manages ignition points for WMFire fire spread modeling.

    Handles:
    - Loading ignition points from shapefile
    - Creating ignition points from coordinates
    - Converting ignition points to grid row/col
    - Writing ignition point shapefiles
    """

    def __init__(self, config, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the IgnitionManager.

        Args:
            config: SymfluenceConfig object with WMFire settings
            logger_instance: Optional logger for status messages
        """
        self.config = config
        self.logger = logger_instance or logger
        self._wmfire_config = self._get_wmfire_config()

    def _get_wmfire_config(self):
        """Extract WMFire configuration from config object."""
        try:
            if (hasattr(self.config, 'model') and
                hasattr(self.config.model, 'rhessys') and
                self.config.model.rhessys is not None):
                return self.config.model.rhessys.wmfire
        except AttributeError:
            pass
        return None

    def get_ignition_point(self) -> Optional[IgnitionPoint]:
        """
        Get ignition point from configuration.

        Priority:
        1. Ignition shapefile (if specified)
        2. Ignition coordinates (if specified)
        3. None (random ignition in fire.def)

        Returns:
            IgnitionPoint or None
        """
        if self._wmfire_config is None:
            return None

        # Check for shapefile first
        if self._wmfire_config.ignition_shapefile:
            shp_path = Path(self._wmfire_config.ignition_shapefile)
            if shp_path.exists():
                return self.load_ignition_from_shapefile(shp_path)
            else:
                self.logger.warning(f"Ignition shapefile not found: {shp_path}")

        # Check for coordinates
        if self._wmfire_config.ignition_point:
            return self.parse_ignition_coords(
                self._wmfire_config.ignition_point,
                name=self._wmfire_config.ignition_name or "ignition",
                date_str=self._wmfire_config.ignition_date
            )

        return None

    def parse_ignition_coords(
        self,
        coords_str: str,
        name: str = "ignition",
        date_str: Optional[str] = None
    ) -> IgnitionPoint:
        """
        Parse ignition coordinates from string.

        Args:
            coords_str: Coordinates as "lat/lon"
            name: Name for the ignition point
            date_str: Optional date string "YYYY-MM-DD"

        Returns:
            IgnitionPoint object
        """
        parts = coords_str.split('/')
        lat, lon = float(parts[0]), float(parts[1])

        date = None
        if date_str:
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                self.logger.warning(f"Could not parse ignition date: {date_str}")

        return IgnitionPoint(
            latitude=lat,
            longitude=lon,
            name=name,
            date=date,
            source='config'
        )

    def load_ignition_from_shapefile(self, shapefile_path: Path) -> Optional[IgnitionPoint]:
        """
        Load ignition point from shapefile.

        Args:
            shapefile_path: Path to ignition point shapefile

        Returns:
            IgnitionPoint or None if loading fails
        """
        try:
            import geopandas as gpd

            gdf = gpd.read_file(shapefile_path)

            if len(gdf) == 0:
                self.logger.warning(f"Empty ignition shapefile: {shapefile_path}")
                return None

            # Get first point
            row = gdf.iloc[0]
            geom = row.geometry

            # Handle Point or PointZ geometry
            if geom.geom_type in ['Point', 'Point Z']:
                # Ensure we're in WGS84 for lat/lon
                if gdf.crs and gdf.crs != 'EPSG:4326':
                    gdf = gdf.to_crs('EPSG:4326')
                    geom = gdf.iloc[0].geometry

                lon, lat = geom.x, geom.y

                # Get name from attributes
                name = "ignition"
                for col in ['Name', 'name', 'NAME', 'id', 'ID']:
                    if col in gdf.columns and row[col]:
                        name = str(row[col])
                        break

                # Get date from attributes if available
                date = None
                for col in ['date', 'Date', 'DATE', 'timestamp', 'begin']:
                    if col in gdf.columns and row[col]:
                        try:
                            date = pd.to_datetime(row[col]).to_pydatetime()
                        except (ValueError, TypeError):
                            pass
                        break

                self.logger.info(f"Loaded ignition point '{name}' from {shapefile_path}: "
                               f"({lat:.4f}, {lon:.4f})")

                return IgnitionPoint(
                    latitude=lat,
                    longitude=lon,
                    name=name,
                    date=date,
                    source='shapefile'
                )
            else:
                self.logger.warning(f"Ignition shapefile contains non-point geometry: {geom.geom_type}")
                return None

        except ImportError:
            self.logger.error("geopandas required for shapefile loading")
            return None
        except Exception as e:
            self.logger.error(f"Error loading ignition shapefile: {e}")
            return None

    def write_ignition_shapefile(
        self,
        ignition: IgnitionPoint,
        output_dir: Path,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Write ignition point to shapefile.

        Args:
            ignition: IgnitionPoint to write
            output_dir: Output directory
            filename: Optional filename (default: {name}.shp)

        Returns:
            Path to written shapefile or None if failed
        """
        try:
            import geopandas as gpd
            from shapely.geometry import Point

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if filename is None:
                filename = f"{ignition.name}.shp"

            output_path = output_dir / filename

            # Create GeoDataFrame
            geom = Point(ignition.longitude, ignition.latitude)
            gdf = gpd.GeoDataFrame({
                'name': [ignition.name],
                'latitude': [ignition.latitude],
                'longitude': [ignition.longitude],
                'date': [ignition.date.isoformat() if ignition.date else None],
                'source': [ignition.source],
                'geometry': [geom]
            }, crs='EPSG:4326')

            gdf.to_file(output_path)
            self.logger.info(f"Ignition shapefile written: {output_path}")

            return output_path

        except ImportError:
            self.logger.error("geopandas required for shapefile writing")
            return None
        except Exception as e:
            self.logger.error(f"Error writing ignition shapefile: {e}")
            return None

    def convert_to_grid_indices(
        self,
        ignition: IgnitionPoint,
        grid_transform: Tuple[float, ...],
        grid_crs: str,
        nrows: int,
        ncols: int
    ) -> Tuple[int, int]:
        """
        Convert ignition point to grid row/column indices.

        Args:
            ignition: IgnitionPoint with lat/lon
            grid_transform: Affine transform tuple (a, b, c, d, e, f)
            grid_crs: CRS of the grid
            nrows: Number of grid rows
            ncols: Number of grid columns

        Returns:
            Tuple of (row, col) indices for fire.def
        """
        try:
            import pyproj
            from shapely.geometry import Point
            from shapely.ops import transform

            # Create point in WGS84
            pt_wgs84 = Point(ignition.longitude, ignition.latitude)

            # Transform to grid CRS
            transformer = pyproj.Transformer.from_crs(
                'EPSG:4326',
                grid_crs,
                always_xy=True
            )
            pt_grid = transform(transformer.transform, pt_wgs84)

            # Get grid coordinates
            x, y = pt_grid.x, pt_grid.y

            # Convert to row/col using inverse transform
            a, b, c, d, e, f = grid_transform
            # x = a * col + c  =>  col = (x - c) / a
            # y = e * row + f  =>  row = (y - f) / e
            col = int((x - c) / a)
            row = int((y - f) / e)

            # Clamp to valid range
            row = max(0, min(row, nrows - 1))
            col = max(0, min(col, ncols - 1))

            self.logger.info(f"Ignition point ({ignition.latitude:.4f}, {ignition.longitude:.4f}) "
                           f"-> grid ({row}, {col})")

            return row, col

        except ImportError:
            self.logger.warning("pyproj required for coordinate transformation")
            return -1, -1
        except Exception as e:
            self.logger.error(f"Error converting ignition to grid indices: {e}")
            return -1, -1


class FirePerimeterValidator:
    """
    Validates simulated fire spread against observed fire perimeters.

    Computes metrics including:
    - Area overlap (Intersection over Union)
    - Perimeter accuracy
    - Commission/Omission errors
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the FirePerimeterValidator.

        Args:
            logger_instance: Optional logger for status messages
        """
        self.logger = logger_instance or logger

    def load_perimeters(self, perimeter_path: Union[str, Path]) -> Optional['gpd.GeoDataFrame']:
        """
        Load observed fire perimeter(s) from shapefile or directory.

        Args:
            perimeter_path: Path to shapefile or directory containing shapefiles

        Returns:
            GeoDataFrame with perimeter polygons or None
        """
        try:
            import geopandas as gpd

            perimeter_path = Path(perimeter_path)

            if perimeter_path.is_file():
                gdf = gpd.read_file(perimeter_path)
                self.logger.info(f"Loaded {len(gdf)} perimeter(s) from {perimeter_path}")
                return gdf

            elif perimeter_path.is_dir():
                # Load all shapefiles in directory
                gdfs = []
                for shp in perimeter_path.glob("*.shp"):
                    try:
                        gdf = gpd.read_file(shp)
                        gdf['source_file'] = shp.name
                        gdfs.append(gdf)
                    except Exception as e:
                        self.logger.warning(f"Could not load {shp}: {e}")

                if gdfs:
                    import pandas as pd
                    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
                    self.logger.info(f"Loaded {len(combined)} perimeter(s) from {perimeter_path}")
                    return combined
                else:
                    self.logger.warning(f"No valid perimeter shapefiles in {perimeter_path}")
                    return None

            else:
                self.logger.warning(f"Perimeter path not found: {perimeter_path}")
                return None

        except ImportError:
            self.logger.error("geopandas required for perimeter loading")
            return None
        except Exception as e:
            self.logger.error(f"Error loading perimeters: {e}")
            return None

    def compare_perimeters(
        self,
        simulated: 'gpd.GeoDataFrame',
        observed: 'gpd.GeoDataFrame',
        buffer_m: float = 0
    ) -> Dict[str, float]:
        """
        Compare simulated fire perimeter with observed.

        Args:
            simulated: GeoDataFrame with simulated fire perimeter
            observed: GeoDataFrame with observed fire perimeter
            buffer_m: Optional buffer distance in meters

        Returns:
            Dictionary with comparison metrics
        """
        try:
            # Ensure same CRS
            if simulated.crs != observed.crs:
                observed = observed.to_crs(simulated.crs)

            # Get union of geometries
            sim_union = simulated.geometry.unary_union
            obs_union = observed.geometry.unary_union

            # Apply buffer if requested
            if buffer_m > 0:
                sim_union = sim_union.buffer(buffer_m)
                obs_union = obs_union.buffer(buffer_m)

            # Calculate metrics
            intersection = sim_union.intersection(obs_union)
            union = sim_union.union(obs_union)

            intersection_area = intersection.area
            union_area = union.area
            sim_area = sim_union.area
            obs_area = obs_union.area

            # Intersection over Union (IoU / Jaccard index)
            iou = intersection_area / union_area if union_area > 0 else 0

            # Sorensen-Dice coefficient
            dice = (2 * intersection_area) / (sim_area + obs_area) if (sim_area + obs_area) > 0 else 0

            # Commission error (false positive): simulated but not observed
            commission = sim_union.difference(obs_union).area
            commission_rate = commission / sim_area if sim_area > 0 else 0

            # Omission error (false negative): observed but not simulated
            omission = obs_union.difference(sim_union).area
            omission_rate = omission / obs_area if obs_area > 0 else 0

            # Area ratio
            area_ratio = sim_area / obs_area if obs_area > 0 else float('inf')

            metrics = {
                'iou': iou,
                'dice': dice,
                'simulated_area_ha': sim_area / 10000,
                'observed_area_ha': obs_area / 10000,
                'intersection_area_ha': intersection_area / 10000,
                'union_area_ha': union_area / 10000,
                'area_ratio': area_ratio,
                'commission_rate': commission_rate,
                'omission_rate': omission_rate,
            }

            self.logger.info(f"Perimeter comparison: IoU={iou:.3f}, Dice={dice:.3f}, "
                           f"Sim={sim_area/10000:.1f}ha, Obs={obs_area/10000:.1f}ha")

            return metrics

        except Exception as e:
            self.logger.error(f"Error comparing perimeters: {e}")
            return {}

    def create_comparison_map(
        self,
        simulated: 'gpd.GeoDataFrame',
        observed: 'gpd.GeoDataFrame',
        output_path: Path,
        title: str = "Fire Perimeter Comparison"
    ) -> Optional[Path]:
        """
        Create a map comparing simulated and observed perimeters.

        Args:
            simulated: GeoDataFrame with simulated fire perimeter
            observed: GeoDataFrame with observed fire perimeter
            output_path: Path for output figure
            title: Map title

        Returns:
            Path to saved figure or None
        """
        try:
            import matplotlib.pyplot as plt

            # Ensure same CRS
            if simulated.crs != observed.crs:
                observed = observed.to_crs(simulated.crs)

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            # Plot observed perimeter
            observed.plot(
                ax=ax,
                facecolor='none',
                edgecolor='red',
                linewidth=2,
                label='Observed'
            )

            # Plot simulated perimeter
            simulated.plot(
                ax=ax,
                facecolor='none',
                edgecolor='blue',
                linewidth=2,
                linestyle='--',
                label='Simulated'
            )

            # Calculate and plot intersection
            sim_union = simulated.geometry.unary_union
            obs_union = observed.geometry.unary_union
            intersection = sim_union.intersection(obs_union)

            if not intersection.is_empty:
                import geopandas as gpd
                gpd.GeoSeries([intersection]).plot(
                    ax=ax,
                    facecolor='purple',
                    alpha=0.3,
                    label='Overlap'
                )

            ax.set_title(title)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels)
            ax.set_xlabel('Easting (m)')
            ax.set_ylabel('Northing (m)')

            # Save figure
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Comparison map saved: {output_path}")
            return output_path

        except ImportError as e:
            self.logger.warning(f"matplotlib required for map creation: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error creating comparison map: {e}")
            return None


# Pandas import for type hints
try:
    import pandas as pd
except ImportError:
    pd = None
