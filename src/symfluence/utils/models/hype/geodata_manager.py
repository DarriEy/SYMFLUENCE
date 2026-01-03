"""
GeoData management utilities for HYPE model.

Handles generation of GeoData.txt and GeoClass.txt files including topological
sorting of sub-basins and calculation of SLC (Soil Landcover) combinations.
"""

# Standard library imports
import glob
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from itertools import product

# Third-party imports
import geopandas as gpd
import pandas as pd
import numpy as np
import pint


class HYPEGeoDataManager:
    """
    Manager for HYPE geographic and classification data.

    Handles:
    - Topological sorting of sub-basins (upstream to downstream)
    - Merging GIS statistics (soil, landcover, elevation) into GeoData format
    - Defining SLC (Soil Landcover Class) combinations
    - Calculating SLC fractions per sub-basin
    - Writing GeoData.txt and GeoClass.txt
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Any,
        output_path: Path,
        geofabric_mapping: Dict[str, Any]
    ):
        """
        Initialize the HYPE GeoData manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            output_path: Path to output HYPE settings directory
            geofabric_mapping: Mapping of input field names to HYPE concepts
        """
        self.config = config
        self.logger = logger
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
    ) -> None:
        """
        Create GeoData.txt and GeoClass.txt files.

        Args:
            gistool_output: Path to GIS statistics CSVs
            subbasins_shapefile: Path to catchment shapefile
            rivers_shapefile: Path to river network shapefile
            frac_threshold: Minimum landcover fraction to consider
            intersect_base_path: Optional path to intersection shapefiles
        """
        self.logger.info("Generating HYPE geographic files...")
        
        # 1. Topology base
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
            
        base_df['slope_mean'] = riv.get('Slope', 0.001)
        
        # 3. Catchment properties
        cat = gpd.read_file(subbasins_shapefile)
        area_info = self.geofabric_mapping['area']
        cat_props = pd.DataFrame({
            basin_id_col: cat[basin_id_col],
            'area': cat[area_info['in_varname']].values * self.ureg(area_info['in_units']).to(area_info['out_units']).magnitude,
            'latitude': cat.centroid.y,
            'longitude': cat.centroid.x
        }).set_index(basin_id_col)
        
        # 4. Load GIS stats
        soil_data, landcover_data, elevation_data = self._load_gis_stats(gistool_output, intersect_base_path, basin_id_col)
        
        # 5. SLC processing
        slc_df, base_df = self._process_slc(base_df, landcover_data, soil_data, frac_threshold)
        
        # 6. Final merging and sorting
        base_df = base_df.join(cat_props, on='subid')
        elev_col = 'mean' if 'mean' in elevation_data.columns else 'elev_mean'
        base_df['elev_mean'] = base_df['subid'].map(elevation_data[elev_col])
        
        # Normalize SLC fractions
        slc_cols = [col for col in base_df.columns if col.startswith('SLC_')]
        if slc_cols:
            base_df[slc_cols] = base_df[slc_cols].div(base_df[slc_cols].sum(axis=1), axis=0).fillna(0)
            
        sorted_df = self.sort_geodata(base_df)
        sorted_df.to_csv(self.output_path / 'GeoData.txt', sep='\t', index=False)
        
        self._write_geoclass(slc_df)
        self.logger.info("GeoData.txt and GeoClass.txt created successfully")

    def _load_gis_stats(self, gistool_output: Path, intersect_base_path: Optional[Path], basin_id_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Robustly load GIS statistics from CSV or shapefile fallbacks."""
        def find_data(pattern, fallback_shp_path=None):
            files = list(Path(gistool_output).glob(pattern))
            if files:
                return pd.read_csv(files[0]).set_index(basin_id_col)
            
            if intersect_base_path:
                shp_files = list(Path(intersect_base_path).parent.glob(fallback_shp_path))
                if shp_files:
                    return gpd.read_file(shp_files[0]).set_index(basin_id_col)
            return None

        soil = find_data('*stats_soil_classes.csv', 'with_soilgrids/*soilclass.shp')
        land = find_data('*stats_*landcover*.csv', 'with_landclass/*landclass.shp')
        elev = find_data('*stats_elv.csv', 'with_dem/*dem.shp')

        if soil is None or land is None or elev is None:
            raise FileNotFoundError(f"Required geospatial statistics not found. Checked {gistool_output}")
            
        return soil, land, elev

    def _process_slc(self, base_df: pd.DataFrame, landcover_data: pd.DataFrame, soil_data: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate SLC combinations and fractions."""
        combinations_set = set()
        lc_cols = [col for col in landcover_data.columns if col.startswith('IGBP_') or col.startswith('frac_')]
        
        for basin_id in landcover_data.index:
            basin_lc = landcover_data.loc[[basin_id]]
            active_lc = [col for col in lc_cols if basin_lc[col].values[0] > threshold]
            
            try:
                lc_values = [int(col.split('_')[1]) for col in active_lc]
            except (ValueError, IndexError):
                lc_values = range(1, len(active_lc) + 1)
            
            if 'majority' in soil_data.columns:
                soil_value = [soil_data.loc[basin_id, 'majority']]
            else:
                usgs_cols = [col for col in soil_data.columns if col.startswith('USGS_')]
                soil_value = [int(soil_data.loc[basin_id, usgs_cols].idxmax().split('_')[1])] if usgs_cols else [1]
            
            combinations_set.update(product(lc_values, soil_value))
            
        slc_df = pd.DataFrame(combinations_set, columns=['landcover', 'soil'])
        slc_df['SLC'] = range(1, len(slc_df) + 1)
        
        # Calculate fractions
        for basin_id in base_df['subid']:
            if 'majority' in soil_data.columns:
                basin_soil = soil_data.loc[basin_id, 'majority']
            else:
                usgs_cols = [col for col in soil_data.columns if col.startswith('USGS_')]
                basin_soil = int(soil_data.loc[basin_id, usgs_cols].idxmax().split('_')[1]) if usgs_cols else 1

            for slc_idx, (lc, soil) in enumerate(zip(slc_df['landcover'], slc_df['soil']), 1):
                lc_val = 0
                for prefix in ['IGBP_', 'frac_']:
                    col = f'{prefix}{lc}'
                    if col in landcover_data.columns:
                        lc_val = landcover_data.loc[basin_id, col]
                        break
                
                base_df.loc[base_df['subid'] == basin_id, f'SLC_{slc_idx}'] = lc_val if lc_val > threshold and basin_soil == soil else 0
                
        return slc_df, base_df

    def sort_geodata(self, geodata: pd.DataFrame) -> pd.DataFrame:
        """Sort sub-basins from upstream to downstream using topological sorting."""
        try:
            import networkx as nx
        except ImportError:
            self.logger.warning("networkx not installed, skipping topological sort")
            return geodata
        
        G = nx.DiGraph()
        for _, row in geodata.iterrows():
            if row['maindown'] > 0:
                G.add_edge(row['subid'], row['maindown'])
        
        cycles = list(nx.simple_cycles(G))
        if cycles:
            self.logger.warning(f"Found {len(cycles)} circular reference(s) in the network")
            for cycle in cycles:
                max_downstream = max(cycle, key=lambda n: len(list(nx.descendants(G, n))))
                cycle_idx = cycle.index(max_downstream)
                from_node = cycle[cycle_idx-1]
                G.remove_edge(from_node, max_downstream)
        
        try:
            headwaters = [n for n in G.nodes() if G.in_degree(n) == 0]
            ordered_subids = []
            visited = set()
            
            def traverse(node, depth=0):
                if node in visited: return
                visited.add(node)
                for successor in G.successors(node):
                    traverse(successor, depth + 1)
                ordered_subids.append((node, depth))
            
            for hw in headwaters: traverse(hw)
            ordered_subids.sort(key=lambda x: x[1])
            final_order = [x[0] for x in ordered_subids]
            
            missing = geodata[~geodata['subid'].isin(final_order)]['subid'].tolist()
            position_map = {subid: pos for pos, subid in enumerate(missing + final_order)}
            
            geodata['sort_idx'] = geodata['subid'].map(position_map)
            return geodata.sort_values('sort_idx').drop(columns=['sort_idx']).reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Error during sorting: {str(e)}")
            return geodata

    def _write_geoclass(self, slc_df: pd.DataFrame) -> None:
        """Write GeoClass.txt with standard formatting."""
        combo = slc_df.copy().rename(columns={'landcover': 'LULC', 'soil': 'SOIL TYPE'})
        cols_to_add = {
            'Main crop cropid': 0, 'Second crop cropid': 0, 'Crop rotation group': 0,
            'Vegetation type': 1, 'Special class code': 0, 'Tile depth': 0,
            'Stream depth': 2.296, 'Number of soil layers': 3,
            'Soil layer depth 1': 0.091, 'Soil layer depth 2': 0.493, 'Soil layer depth 3': 2.296
        }
        for col, val in cols_to_add.items(): combo[col] = val
        
        with open(self.output_path / 'GeoClass.txt', 'w') as f:
            f.write("!          SLC\tLULC\tSOIL TYPE\tMain crop cropid\tSecond crop cropid\tCrop rotation group\tVegetation type\tSpecial class code\tTile depth\tStream depth\tNumber of soil layers\tSoil layer depth 1\tSoil layer depth 2\tSoil layer depth 3 \n")
            combo[['SLC', 'LULC', 'SOIL TYPE'] + list(cols_to_add.keys())].to_csv(f, sep='\t', index=False, header=False)
