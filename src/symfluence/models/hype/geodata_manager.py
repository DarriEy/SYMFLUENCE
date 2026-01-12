"""
GeoData management utilities for HYPE model.

This module provides the HYPEGeoDataManager class for generating HYPE's geographic
input files. It handles the creation of:

- **GeoData.txt**: Sub-basin properties (topology, area, location, SLC fractions)
- **GeoClass.txt**: Soil-Landcover Class definitions and soil layer depths
- **ForcKey.txt**: Mapping between sub-basin IDs and forcing file station IDs

The manager also performs topological sorting to ensure sub-basins are ordered
from upstream to downstream, which is required for HYPE's internal routing.

Example usage:
    >>> from symfluence.models.hype import HYPEGeoDataManager
    >>> manager = HYPEGeoDataManager(config, logger, output_path, geofabric_mapping)
    >>> land_uses = manager.create_geofiles(
    ...     gistool_output=Path('/path/to/gis_stats'),
    ...     subbasins_shapefile=Path('/path/to/basins.shp'),
    ...     rivers_shapefile=Path('/path/to/rivers.shp'),
    ...     frac_threshold=0.05
    ... )
"""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pint

if TYPE_CHECKING:
    pass


class HYPEGeoDataManager:
    """
    Manager for HYPE geographic and classification data.

    This class handles the creation of HYPE's three geographic input files,
    which define the spatial structure of the model domain:

    **GeoData.txt** columns:
        - subid: Sub-basin identifier (must be > 0)
        - maindown: ID of downstream sub-basin (0 for outlets)
        - area: Sub-basin area (m²)
        - rivlen: River length through sub-basin (m)
        - slope_mean: Mean river slope (m/m)
        - latitude/longitude: Centroid coordinates
        - elev_mean: Mean elevation (m)
        - SLC_1, SLC_2, ...: Fractions of each Soil-Landcover Class

    **GeoClass.txt** columns:
        - SLC: Soil-Landcover Class ID
        - LULC: Land use/cover type (IGBP class 1-17)
        - SOIL TYPE: Soil type ID
        - Vegetation type, soil layer depths, etc.

    **ForcKey.txt** columns:
        - subid: Sub-basin ID
        - stationid: Forcing file station ID (typically same as subid)

    Attributes:
        config: Configuration dictionary.
        logger: Logger instance for status messages.
        output_path: Path to HYPE settings directory.
        geofabric_mapping: Field name mappings for geospatial inputs.
        ureg: Pint unit registry for unit conversions.

    Note:
        HYPE requires sub-basin IDs > 0. If input data uses 0-based IDs,
        this manager automatically shifts all IDs by +1.
    """

    def __init__(
        self,
        config: dict[str, Any],
        logger: logging.Logger | Any | None,
        output_path: Path | str,
        geofabric_mapping: dict[str, Any]
    ) -> None:
        """
        Initialize the HYPE GeoData manager.

        Args:
            config: Configuration dictionary containing domain settings.
            logger: Logger instance for status messages. If None, creates
                a module-level logger.
            output_path: Path to the HYPE settings directory where geographic
                files will be written.
            geofabric_mapping: Dictionary mapping input field names to HYPE
                concepts. Expected keys:
                - 'basinID': {'in_varname': str} - Sub-basin ID field
                - 'nextDownID': {'in_varname': str} - Downstream ID field
                - 'area': {'in_varname': str, 'in_units': str, 'out_units': str}
                - 'rivlen': {'in_varname': str, 'in_units': str, 'out_units': str}
        """
        self.config = config
        self.logger = logger if logger else logging.getLogger(__name__)
        self.output_path = Path(output_path)
        self.geofabric_mapping = geofabric_mapping
        self.ureg = pint.UnitRegistry()

    def create_geofiles(
        self,
        gistool_output: Path,
        subbasins_shapefile: Path,
        rivers_shapefile: Path,
        frac_threshold: float,
        intersect_base_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create GeoData.txt, GeoClass.txt, and ForcKey.txt files.

        Args:
            gistool_output: Path to GIS statistics CSVs
            subbasins_shapefile: Path to catchment shapefile
            rivers_shapefile: Path to river network shapefile
            frac_threshold: Minimum landcover fraction to consider
            intersect_base_path: Optional path to intersection shapefiles

        Returns:
            Array of unique land use IDs for parameter file generation
        """
        self.logger.debug("Generating HYPE geographic files...")
        gistool_output = Path(gistool_output)
        subbasins_shapefile = Path(subbasins_shapefile)
        rivers_shapefile = Path(rivers_shapefile)

        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)

        # 1. Build base topology from river network
        basin_id_col = self.geofabric_mapping['basinID']['in_varname']
        next_down_col = self.geofabric_mapping['nextDownID']['in_varname']

        if rivers_shapefile.exists():
            riv = gpd.read_file(rivers_shapefile)
        else:
            riv = gpd.read_file(subbasins_shapefile)
            if next_down_col not in riv.columns:
                riv[next_down_col] = 0

        base_df = pd.DataFrame({
            'subid': riv[basin_id_col],
            'maindown': riv[next_down_col]
        })

        # 2. River properties
        rivlen_info = self.geofabric_mapping['rivlen']
        if rivlen_info['in_varname'] in riv.columns:
            lengths = riv[rivlen_info['in_varname']].values * self.ureg(rivlen_info['in_units'])
            base_df['rivlen'] = lengths.to(rivlen_info['out_units']).magnitude
        else:
            base_df['rivlen'] = 0

        if 'Slope' in riv.columns:
            base_df['slope_mean'] = riv['Slope']
        else:
            base_df['slope_mean'] = 0.001

        # 3. Catchment properties
        cat = gpd.read_file(subbasins_shapefile)
        area_info = self.geofabric_mapping['area']

        # Calculate centroids in projected CRS
        centroids = self._get_projected_centroids(cat)

        cat_props = pd.DataFrame({
            basin_id_col: cat[basin_id_col],
            'area': cat[area_info['in_varname']].values * self.ureg(area_info['in_units']).to(area_info['out_units']).magnitude,
            'latitude': centroids.y,
            'longitude': centroids.x
        }).set_index(basin_id_col)

        # 4. Load GIS stats
        soil_data, landcover_data, elevation_data = self._load_gis_stats(
            gistool_output, intersect_base_path, basin_id_col
        )

        # 5. SLC processing
        slc_df, base_df = self._process_slc(base_df, landcover_data, soil_data, frac_threshold)

        # 6. Final merging
        base_df = base_df.join(cat_props, on='subid')

        # Robust elevation mapping
        elev_col = 'mean' if 'mean' in elevation_data.columns else 'elev_mean'

        def get_elevation(subid):
            if subid in elevation_data.index:
                return elevation_data.loc[subid, elev_col]
            elif len(elevation_data) == 1:
                return elevation_data[elev_col].iloc[0]
            return 0.0

        base_df['elev_mean'] = base_df['subid'].apply(get_elevation)

        # Normalize SLC fractions
        slc_cols = [col for col in base_df.columns if col.startswith('SLC_')]
        if slc_cols:
            base_df[slc_cols] = base_df[slc_cols].div(base_df[slc_cols].sum(axis=1), axis=0).fillna(0)

        # 7. Handle ID shifting (HYPE requires IDs > 0)
        base_df = self._shift_ids_if_needed(base_df)

        # 8. Sort and save
        sorted_df = self.sort_geodata(base_df)
        sorted_df.to_csv(self.output_path / 'GeoData.txt', sep='\t', index=False)

        # 9. Write ForcKey.txt (required for readobsid=y)
        self._write_forckey(sorted_df)

        # 10. Write GeoClass.txt
        self._write_geoclass(slc_df)

        self.logger.debug("GeoData.txt, GeoClass.txt, and ForcKey.txt created successfully")

        # Return land use information for parameter file generation
        return slc_df['landcover'].unique()

    def _get_projected_centroids(self, gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
        """
        Calculate centroids in a projected CRS and return them in the original CRS.
        This avoids UserWarning about centroids in geographic CRS.
        """
        original_crs = gdf.crs
        if original_crs and original_crs.is_geographic:
            # Project to EPSG:3857 (Web Mercator) for centroid calculation
            gdf_proj = gdf.to_crs(epsg=3857)
            centroids_proj = gdf_proj.geometry.centroid
            return centroids_proj.to_crs(original_crs)
        else:
            return gdf.geometry.centroid

    def _load_gis_stats(
        self,
        gistool_output: Path,
        intersect_base_path: Optional[Path],
        basin_id_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Robustly load GIS statistics from CSV or shapefile fallbacks."""
        def find_data(pattern: str, fallback_shp_path: Optional[str] = None) -> Optional[pd.DataFrame]:
            files = list(gistool_output.glob(pattern))
            if files:
                df = pd.read_csv(files[0])
                # Be robust with index name
                idx_col = basin_id_col if basin_id_col in df.columns else ('ID' if 'ID' in df.columns else df.columns[0])
                return df.set_index(idx_col)

            if intersect_base_path and fallback_shp_path:
                shp_files = list(Path(intersect_base_path).parent.glob(fallback_shp_path))
                if shp_files:
                    gdf = gpd.read_file(shp_files[0])
                    idx_col = basin_id_col if basin_id_col in gdf.columns else ('ID' if 'ID' in gdf.columns else gdf.columns[0])
                    return gdf.set_index(idx_col)
            return None

        soil = find_data('*stats_soil_classes.csv', 'with_soilgrids/*soilclass.shp')
        land = find_data('*stats_*landcover*.csv', 'with_landclass/*landclass.shp')
        elev = find_data('*stats_elv.csv', 'with_dem/*dem.shp')

        if soil is None or land is None or elev is None:
            raise FileNotFoundError(
                f"Required geospatial statistics not found. "
                f"Checked {gistool_output} and {intersect_base_path}"
            )

        return soil, land, elev

    def _process_slc(
        self,
        base_df: pd.DataFrame,
        landcover_data: pd.DataFrame,
        soil_data: pd.DataFrame,
        threshold: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate SLC combinations and fractions."""
        combinations_set: Set[Tuple[int, int]] = set()
        lc_cols = [col for col in landcover_data.columns if col.startswith('IGBP_') or col.startswith('frac_')]

        for basin_id in landcover_data.index:
            # Robust landcover retrieval
            if basin_id in landcover_data.index:
                basin_lc = landcover_data.loc[[basin_id]]
            elif len(landcover_data) == 1:
                basin_lc = landcover_data.iloc[[0]]
            else:
                continue

            active_lc = [col for col in lc_cols if basin_lc[col].values[0] > threshold]

            try:
                lc_values = [int(col.split('_')[1]) for col in active_lc]
            except (ValueError, IndexError):
                lc_values = list(range(1, len(active_lc) + 1))

            # Robust soil retrieval
            if basin_id in soil_data.index:
                basin_soil_data = soil_data.loc[[basin_id]]
            elif len(soil_data) == 1:
                basin_soil_data = soil_data.iloc[[0]]
            else:
                basin_soil_data = None

            if basin_soil_data is not None and 'majority' in basin_soil_data.columns:
                soil_value = [basin_soil_data['majority'].values[0]]
            elif basin_soil_data is not None:
                usgs_cols = [col for col in basin_soil_data.columns if col.startswith('USGS_')]
                if usgs_cols:
                    soil_value = [int(basin_soil_data[usgs_cols].idxmax(axis=1).values[0].split('_')[1])]
                else:
                    soil_value = [1]
            else:
                soil_value = [1]

            combinations_set.update(product(lc_values, soil_value))

        slc_df = pd.DataFrame(list(combinations_set), columns=['landcover', 'soil'])
        # HYPE requires soil types >= 1, so remap 0 to 1
        slc_df['soil'] = slc_df['soil'].replace(0, 1)
        slc_df['SLC'] = range(1, len(slc_df) + 1)

        # Calculate SLC fractions for each basin
        for basin_id in base_df['subid']:
            # Robust landcover row retrieval
            if basin_id in landcover_data.index:
                basin_lc = landcover_data.loc[[basin_id]]
            elif len(landcover_data) == 1:
                basin_lc = landcover_data.iloc[[0]]
            else:
                basin_lc = None

            # Robust soil row retrieval
            if basin_id in soil_data.index:
                basin_soil_data = soil_data.loc[[basin_id]]
            elif len(soil_data) == 1:
                basin_soil_data = soil_data.iloc[[0]]
            else:
                basin_soil_data = None

            if basin_soil_data is not None and 'majority' in basin_soil_data.columns:
                basin_soil = basin_soil_data['majority'].values[0]
            elif basin_soil_data is not None:
                usgs_cols = [col for col in basin_soil_data.columns if col.startswith('USGS_')]
                basin_soil = int(basin_soil_data[usgs_cols].idxmax(axis=1).values[0].split('_')[1]) if usgs_cols else 1
            else:
                basin_soil = 1

            for slc_idx, (lc, soil) in enumerate(zip(slc_df['landcover'], slc_df['soil']), 1):
                lc_val = 0
                if basin_lc is not None:
                    for prefix in ['IGBP_', 'frac_']:
                        col = f'{prefix}{lc}'
                        if col in basin_lc.columns:
                            lc_val = basin_lc[col].values[0]
                            break

                if lc_val > threshold and basin_soil == soil:
                    base_df.loc[base_df['subid'] == basin_id, f'SLC_{slc_idx}'] = lc_val
                else:
                    base_df.loc[base_df['subid'] == basin_id, f'SLC_{slc_idx}'] = 0

        return slc_df, base_df

    def _shift_ids_if_needed(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Shift IDs if they start from 0 (HYPE requires > 0).

        Note: HYPEForcingProcessor also shifts forcing IDs, so this must be consistent.
        """
        if base_df['subid'].min() == 0:
            self.logger.debug("Shifting subids +1 for HYPE compatibility (0-based to 1-based)")

            # Get original IDs for checking connectivity
            original_ids = set(base_df['subid'])

            # Shift subids
            base_df['subid'] = base_df['subid'] + 1

            # Update maindown: map valid connections to shifted ID, set others (outlets) to 0
            def update_downstream(val):
                if val in original_ids:
                    return val + 1
                return 0  # Outlet

            base_df['maindown'] = base_df['maindown'].apply(update_downstream)

        return base_df

    def _write_forckey(self, sorted_df: pd.DataFrame) -> None:
        """
        Write ForcKey.txt (required for readobsid=y).

        Maps subid to the station id in forcing files (which we set to subid).
        """
        forckey_df = pd.DataFrame({
            'subid': sorted_df['subid'],
            'stationid': sorted_df['subid']
        })
        forckey_df.to_csv(self.output_path / 'ForcKey.txt', sep='\t', index=False)
        self.logger.debug("ForcKey.txt created")

    def sort_geodata(self, geodata: pd.DataFrame) -> pd.DataFrame:
        """
        Sort sub-basins from upstream to downstream using topological sorting.

        HYPE requires basins to be ordered such that all upstream basins
        appear before their downstream basins. This uses networkx's
        topological sort which guarantees this ordering.
        """
        try:
            import networkx as nx
        except ImportError:
            self.logger.warning("networkx not installed, skipping topological sort")
            return geodata

        # Create directed graph from subid -> maindown relationships
        G = nx.DiGraph()
        all_subids = set(geodata['subid'].tolist())

        for _, row in geodata.iterrows():
            subid = row['subid']
            maindown = row['maindown']
            # Add all nodes to ensure isolated nodes are included
            G.add_node(subid)
            if maindown > 0 and maindown in all_subids:
                # Edge from upstream to downstream
                G.add_edge(subid, maindown)

        # Find and break cycles if they exist
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                self.logger.warning(f"Found {len(cycles)} circular reference(s) in the network")
                for cycle in cycles:
                    # Find the node in the cycle with the most downstream connections
                    max_downstream = max(cycle, key=lambda n: len(list(nx.descendants(G, n))))
                    cycle_idx = cycle.index(max_downstream)
                    from_node = cycle[cycle_idx - 1]
                    G.remove_edge(from_node, max_downstream)
                    self.logger.warning(f"Breaking cycle at edge: {from_node} -> {max_downstream}")
        except Exception as e:
            self.logger.warning(f"Could not check for cycles: {e}")

        try:
            # Use networkx topological sort - this guarantees upstream before downstream
            # Since edges go from upstream to downstream, topological_sort gives correct order
            final_order = list(nx.topological_sort(G))

            # Handle nodes that weren't in the graph (shouldn't happen, but safety check)
            missing_subids = geodata[~geodata['subid'].isin(final_order)]['subid'].tolist()
            if missing_subids:
                self.logger.warning(f"Found {len(missing_subids)} basins not in network, adding at start")
                final_order = missing_subids + final_order

            # Create a mapping from subid to desired position
            position_map = {subid: pos for pos, subid in enumerate(final_order)}

            # Sort geodata based on the position map
            geodata = geodata.copy()
            geodata['sort_idx'] = geodata['subid'].map(position_map)
            geodata = geodata.sort_values('sort_idx', ignore_index=True)
            geodata = geodata.drop(columns=['sort_idx'])

            # Verify the sorting
            errors = 0
            for i, row in geodata.iterrows():
                if row['maindown'] > 0:
                    downstream_rows = geodata[geodata['subid'] == row['maindown']]
                    if not downstream_rows.empty:
                        downstream_idx = downstream_rows.index[0]
                        if downstream_idx < i:
                            errors += 1
                            if errors <= 3:  # Only log first few
                                self.logger.warning(
                                    f"Basin {row['subid']} (idx={i}) appears after its "
                                    f"downstream basin {row['maindown']} (idx={downstream_idx})"
                                )

            if errors > 0:
                self.logger.error(f"Topological sort failed: {errors} ordering violations found")
            else:
                self.logger.debug("Topological sort successful: all basins correctly ordered")

            return geodata

        except nx.NetworkXUnfeasible:
            self.logger.error("Graph has cycles that could not be resolved")
            return geodata
        except Exception as e:
            self.logger.error(f"Error during topological sorting: {str(e)}")
            return geodata

    def _write_geoclass(self, slc_df: pd.DataFrame) -> None:
        """Write GeoClass.txt file with full metadata and specific formatting."""
        combination = slc_df.copy()
        combination = combination.rename(columns={'landcover': 'LULC', 'soil': 'SOIL TYPE'})
        combination = combination[['SLC', 'LULC', 'SOIL TYPE']]

        combination['Main crop cropid'] = 0
        combination['Second crop cropid'] = 0
        combination['Crop rotation group'] = 0
        combination['Vegetation type'] = 1
        combination['Special class code'] = 0
        combination['Tile depth'] = 0
        combination['Stream depth'] = 2.296
        combination['Number of soil layers'] = 3
        combination['Soil layer depth 1'] = 0.091
        combination['Soil layer depth 2'] = 0.493
        combination['Soil layer depth 3'] = 2.296

        with open(self.output_path / 'GeoClass.txt', 'w') as f:
            f.write(
                "!          SLC\tLULC\tSOIL TYPE\tMain crop cropid\tSecond crop cropid\t"
                "Crop rotation group\tVegetation type\tSpecial class code\tTile depth\t"
                "Stream depth\tNumber of soil layers\tSoil layer depth 1\tSoil layer depth 2\t"
                "Soil layer depth 3 \n"
            )
            combination.to_csv(f, sep='\t', index=False, header=False)

        self.logger.debug("GeoClass.txt created")
