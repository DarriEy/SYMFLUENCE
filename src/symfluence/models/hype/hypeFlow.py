# Load needed packages
import xarray as xr
import pint
import glob
import netCDF4 as nc4
import os
import cdo
import pandas as pd
import easymore
import numpy       as      np
import geopandas   as      gpd
import sys
from   itertools   import  product
from tqdm import tqdm
import shutil
from pathlib import Path

def _create_easymore_instance():
    """Create an EASYMORE instance handling different module structures."""
    if hasattr(easymore, "Easymore"):
        return easymore.Easymore()
    if hasattr(easymore, "easymore"):
        return easymore.easymore()
    raise AttributeError("easymore module does not expose an Easymore class")

# sort geodata from upstream to downstream
def sort_geodata(geodata):
    """Sort sub-basins from upstream to downstream using topological sorting.
    Handles cycles by breaking them at the highest downstream point."""
    
    try:
        import networkx as nx
    except ImportError:
        print("Warning: networkx not installed, skipping topological sort")
        return geodata
    
    # Create directed graph from subid -> maindown relationships
    G = nx.DiGraph()
    for _, row in geodata.iterrows():
        if row['maindown'] > 0:  # Only add valid downstream connections
            G.add_edge(row['subid'], row['maindown'])
    
    # Find and break cycles if they exist
    cycles = list(nx.simple_cycles(G))
    if cycles:
        print(f"Warning: Found {len(cycles)} circular reference(s) in the network")
        for cycle in cycles:
            # Find the node in the cycle with the most downstream connections
            # and break the cycle there
            max_downstream = max(cycle, 
                key=lambda n: len(list(nx.descendants(G, n))))
            # Find the edge pointing to this node within the cycle
            cycle_idx = cycle.index(max_downstream)
            from_node = cycle[cycle_idx-1]
            # Remove this edge to break the cycle
            G.remove_edge(from_node, max_downstream)
            print(f"Breaking cycle at edge: {from_node} -> {max_downstream}")
    
    try:
        # Find all nodes with no incoming edges (headwaters)
        headwaters = [n for n in G.nodes() if G.in_degree(n) == 0]
        
        # For each headwater, find all downstream nodes and their distances
        ordered_subids = []
        visited = set()
        
        def traverse_downstream(node, depth=0):
            """Recursively traverse downstream, tracking depth"""
            if node in visited:
                return
            visited.add(node)
            # Get all downstream nodes
            downstream = list(G.successors(node))
            if downstream:
                # Recursively process downstream nodes
                for next_node in downstream:
                    traverse_downstream(next_node, depth + 1)
            # Add node to ordered list with its depth
            ordered_subids.append((node, depth))
        
        # Process each headwater
        for hw in headwaters:
            traverse_downstream(hw)
        
        # Sort by depth (upstream to downstream)
        ordered_subids.sort(key=lambda x: x[1])
        ordered_subids = [x[0] for x in ordered_subids]  # Extract just the subids
        
        # Handle nodes that weren't reached (isolated nodes)
        missing_subids = geodata[~geodata['subid'].isin(ordered_subids)]['subid'].tolist()
        
        # Add missing subids at the start
        final_order = missing_subids + ordered_subids
        
        # Create a mapping from subid to desired position
        position_map = {subid: pos for pos, subid in enumerate(final_order)}
        
        # Sort geodata based on the position map
        geodata['sort_idx'] = geodata['subid'].map(position_map)
        geodata = geodata.sort_values('sort_idx', ignore_index=True)
        geodata = geodata.drop(columns=['sort_idx'])
        
        # Verify the sorting
        for i, row in geodata.iterrows():
            if row['maindown'] > 0:
                downstream_idx = geodata.index[geodata['subid'] == row['maindown']].tolist()
                if downstream_idx and downstream_idx[0] < i:
                    print(f"Warning: Basin {row['subid']} appears before its downstream basin {row['maindown']}")
        
        return geodata
        
    except Exception as e:
        print(f"Error during sorting: {str(e)}")
        return geodata  # Return unsorted data if we can't resolve the ordering

# write HYPE forcing from easymore nc files
def write_hype_forcing(easymore_output, timeshift, forcing_units, geofabric_mapping, path_to_save, cache_path):
    path_to_save = str(path_to_save).rstrip('/') + '/'
    cache_path = str(cache_path).rstrip('/')
    if not os.path.isdir(path_to_save):
        os.makedirs(path_to_save)
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)
        
    # function to get daily values from hourly timeseries
    def convert_hourly_to_daily (input_file_name,
                                 variable_in,
                                 variable_out,
                                 variable_out_long_name = None,
                                 var_unit_conversion = None,
                                 var_time = 'time',
                                 var_id = 'id',
                                 time_diff = 0,
                                 stat = 'max', 
                                 output_file_name_nc = None,
                                 output_file_name_txt = None,
                                 Fill_value = -9999.0): # 'max', 'min', 'mean'

        # read the input houtly nc file
        ds = xr.open_dataset(input_file_name)
        # set id as integer and preserve values before resampling
        hru_id_values = None
        if var_id in ds.coords:
            ds.coords[var_id] = ds.coords[var_id].astype(int)
            hru_id_values = ds.coords[var_id].values.copy()
        elif var_id in ds.data_vars:
            ds[var_id] = ds[var_id].astype(int)
            hru_id_values = ds[var_id].values.copy()

        # drop all the other variables except the mentioned varibale, time and id
        variables_to_keep = [variable_in, var_time]
        if not var_id is None:
            variables_to_keep.append(var_id)

        # Drop all variables except the specified ones
        ds = ds.drop([v for v in ds.variables if v not in variables_to_keep])

        # roll the time based on hour of difference to have more accurate
        if time_diff !=0:
            ds[var_time] = ds[var_time].roll(time=time_diff)
            # Remove the first or last roll_steps time steps
            if time_diff < 0:
                ds = ds.isel(time=slice( None, time_diff))
            elif time_diff > 0:
                ds = ds.isel(time=slice( time_diff, None))

        # to create the xarray dataframe with daily time
        if stat == 'max':
            ds_daily = ds.resample(time='D').max()
        elif stat == 'min':
            ds_daily = ds.resample(time='D').min()
        elif stat == 'mean':
            ds_daily = ds.resample(time='D').mean()
        elif stat == 'sum':
            ds_daily = ds.resample(time='D').sum()
        else:
            sys.exit('input stat should be max, min, mean or sum')

        # rename the variable in
        if variable_in in ds_daily:
            ds_daily = ds_daily.rename({variable_in: variable_out})

        # add long name
        if not variable_out_long_name is None:
            ds_daily[variable_out].attrs['long_name'] = variable_out_long_name

        # transpose the variable
        if 'time' in ds_daily[variable_out].dims and var_id in ds_daily[variable_out].dims:
            ds_daily[variable_out] = ds_daily[variable_out].transpose(var_id, 'time')

        # this section is written to avoid issues with netcdf and HYPE!
        df = ds_daily[variable_out].to_dataframe()
        df = df.unstack(level=var_id if var_id in df.index.names else -1)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)
        # Use the hruId values saved before resampling (not the 0-based index from unstacking)
        if hru_id_values is not None:
            df.columns = hru_id_values.astype(int)
        df.columns.name = None
        df.index.name = var_time
        if not output_file_name_txt is None:
            df.to_csv(output_file_name_txt,\
                      sep='\t', na_rep='', index_label='time', float_format='%.3f')
        
        # return
        return ds_daily
    ############
    print('Merging easymore outputs to one NetCDF file \n')
    easymore_nc_files = sorted(glob.glob(os.path.join(easymore_output, '*.nc')))
    if not easymore_nc_files:
        print(f"Warning: No forcing files found in {easymore_output}")
        return

    # split the files in batches as cdo cannot mergetime long list of file names
    batch_size = 20
    if(len(easymore_nc_files) < batch_size):
        batch_size = len(easymore_nc_files)
    
    files_split = np.array_split(easymore_nc_files, batch_size)
    cdo_obj = cdo.Cdo()  # CDO object
    intermediate_files = []

    merged_forcing_path = os.path.join(cache_path, 'merged_forcing.nc')
    for i in tqdm(range(batch_size), desc="Processing easymore outputs"):
        batch_files = files_split[i].tolist()
        batch_output = os.path.join(cache_path, f"forcing_batch_{i}.nc")
        cdo_obj.mergetime(input=batch_files, output=batch_output)
        intermediate_files.append(batch_output)

    # Combine intermediate results into one netcdf file
    cdo_obj.mergetime(input=intermediate_files, output=merged_forcing_path)

    # Clean up intermediate files
    for f in intermediate_files:
        if os.path.exists(f):
            os.remove(f)

    # open the forcing file
    forcing = xr.open_dataset(merged_forcing_path)
    forcing = forcing.convert_calendar('standard')
    forcing['time'] = forcing['time'] + pd.Timedelta(hours=timeshift)
    forcing.to_netcdf(merged_forcing_path + '.tmp')
    forcing.close()
    os.replace(merged_forcing_path + '.tmp', merged_forcing_path)
    
    ############
    print('Get average daily values for HYPE \n')
    
    # helper to find variable in dataset
    def get_in_var(key):
        return forcing_units[key]['in_varname']

    convert_hourly_to_daily(merged_forcing_path,
                                get_in_var('temperature'),
                                'TMAXobs',
                                var_unit_conversion = None,
                                var_time = 'time',
                                var_id = 'hruId',
                                time_diff = 0,
                                stat = 'max',
                                output_file_name_txt = os.path.join(path_to_save, 'TMAXobs.txt'))

    convert_hourly_to_daily(merged_forcing_path,
                                get_in_var('temperature'),
                                'TMINobs',
                                var_unit_conversion = None,
                                var_time = 'time',
                                var_id = 'hruId',
                                time_diff = 0,
                                stat = 'min',
                                output_file_name_txt = os.path.join(path_to_save, 'TMINobs.txt'))

    convert_hourly_to_daily(merged_forcing_path,
                                get_in_var('temperature'),
                                'Tobs',
                                var_unit_conversion = None,
                                var_time = 'time',
                                var_id = 'hruId',
                                time_diff = 0,
                                stat = 'mean',
                                output_file_name_txt = os.path.join(path_to_save, 'Tobs.txt'))

    convert_hourly_to_daily(merged_forcing_path,
                                get_in_var('precipitation'),
                                'Pobs',
                                var_unit_conversion = None,
                                var_time = 'time',
                                var_id = 'hruId',
                                time_diff = 0,
                                stat = 'sum',
                                output_file_name_txt = os.path.join(path_to_save, 'Pobs.txt'))
    
    if os.path.exists(merged_forcing_path):
        os.remove(merged_forcing_path)

def _get_projected_centroids(gdf):
    """
    Calculate centroids in a projected CRS and return them in the original CRS.
    This avoids UserWarning about centroids in geographic CRS.
    """
    original_crs = gdf.crs
    if original_crs and original_crs.is_geographic:
        # Project to EPSG:3857 (Web Mercator) for centroid calculation
        # This is a generic projected CRS suitable for avoiding the warning
        gdf_proj = gdf.to_crs(epsg=3857)
        centroids_proj = gdf_proj.geometry.centroid
        return centroids_proj.to_crs(original_crs)
    else:
        return gdf.geometry.centroid

# write GeoData and GeoClass files
def write_hype_geo_files(gistool_output, subbasins_shapefile, rivers_shapefile, frac_threshold, geofabric_mapping, path_to_save, intersect_base_path=None):
    gistool_output = str(gistool_output).rstrip('/') + '/'
    path_to_save = str(path_to_save).rstrip('/') + '/'
    
    # Extract mapping variables
    basinID = geofabric_mapping['basinID']['in_varname']
    NextDownID = geofabric_mapping['nextDownID']['in_varname']
    
    # 1. Start with river network as our base - it defines the topology
    if os.path.exists(rivers_shapefile):
        riv = gpd.read_file(rivers_shapefile)
    else:
        riv = gpd.read_file(subbasins_shapefile)
        if NextDownID not in riv.columns:
            riv[NextDownID] = 0
    
    # 2. Create base dataframe with correct topology
    base_df = pd.DataFrame({
        'subid': riv[basinID],
        'maindown': riv[NextDownID]
    })
    
    # 3. Add river properties
    rivlen_name = geofabric_mapping['rivlen']['in_varname']
    ureg = pint.UnitRegistry()
    if rivlen_name in riv.columns:
        lengthm = riv[rivlen_name].values * ureg(geofabric_mapping['rivlen']['in_units'])
        base_df['rivlen'] = lengthm.to(geofabric_mapping['rivlen']['out_units']).magnitude
    else:
        base_df['rivlen'] = 0
        
    if 'Slope' in riv.columns:
        base_df['slope_mean'] = riv['Slope']
    else:
        base_df['slope_mean'] = 0.001
    
    # 4. Add catchment properties
    cat = gpd.read_file(subbasins_shapefile)
    
    # Calculate centroids using projected CRS to avoid warning
    centroids = _get_projected_centroids(cat)
    
    cat_props = pd.DataFrame({
        basinID: cat[basinID],
        'area': cat[geofabric_mapping['area']['in_varname']].values * ureg(geofabric_mapping['area']['in_units']).to(geofabric_mapping['area']['out_units']).magnitude,
        'latitude': centroids.y,
        'longitude': centroids.x
    }).set_index(basinID)
    
    # 5. Add soil, landcover and elevation data - be robust with filenames
    def find_data(pattern, fallback_shp_path=None):
        files = glob.glob(gistool_output + pattern)
        if files:
            df = pd.read_csv(files[0])
            # Be robust with index name
            idx_col = basinID if basinID in df.columns else ('ID' if 'ID' in df.columns else df.columns[0])
            return df.set_index(idx_col)
        
        if intersect_base_path:
            shp_files = glob.glob(str(Path(intersect_base_path).parent) + fallback_shp_path)
            if shp_files:
                gdf = gpd.read_file(shp_files[0])
                idx_col = basinID if basinID in gdf.columns else ('ID' if 'ID' in gdf.columns else gdf.columns[0])
                return gdf.set_index(idx_col)
        return None

    soil_data = find_data('*stats_soil_classes.csv', '/with_soilgrids/*soilclass.shp')
    landcover_data = find_data('*stats_*landcover*.csv', '/with_landclass/*landclass.shp')
    elevation_data = find_data('*stats_elv.csv', '/with_dem/*dem.shp')

    if soil_data is None or landcover_data is None or elevation_data is None:
        raise FileNotFoundError(f"Required geospatial statistics not found. Checked {gistool_output} and {intersect_base_path}")

    # 6. Calculate SLC combinations
    combinations_set_all = set()
    lc_cols = [col for col in landcover_data.columns if col.startswith('IGBP_') or col.startswith('frac_')]
    
    for basin_id in landcover_data.index:
        # Robust retrieval of basin row
        if basin_id in landcover_data.index:
            basin_lc = landcover_data.loc[[basin_id]]
        elif len(landcover_data) == 1:
            basin_lc = landcover_data.iloc[[0]]
        else:
            continue

        active_lc = [col for col in lc_cols if basin_lc[col].values[0] > frac_threshold]
        try:
            lc_values = [int(col.split('_')[1]) for col in active_lc]
        except (ValueError, IndexError):
            lc_values = range(1, len(active_lc) + 1)
        
        # Robust retrieval of soil data
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
        
        combinations_set_all.update(product(lc_values, soil_value))
    
    slc_df = pd.DataFrame(combinations_set_all, columns=['landcover', 'soil'])
    # HYPE requires soil types >= 1, so remap 0 to 1
    slc_df['soil'] = slc_df['soil'].replace(0, 1)
    slc_df['SLC'] = range(1, len(slc_df) + 1)
    
    # 7. Calculate SLC fractions
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
            
            if lc_val > frac_threshold and basin_soil == soil:
                base_df.loc[base_df['subid'] == basin_id, f'SLC_{slc_idx}'] = lc_val
            else:
                base_df.loc[base_df['subid'] == basin_id, f'SLC_{slc_idx}'] = 0
    
    # 8. Add remaining properties
    base_df = base_df.join(cat_props, on='subid')
    
    elev_col = 'mean' if 'mean' in elevation_data.columns else 'elev_mean'
    
    # Robust elevation mapping
    def get_elevation(subid):
        if subid in elevation_data.index:
            return elevation_data.loc[subid, elev_col]
        elif len(elevation_data) == 1:
            return elevation_data[elev_col].iloc[0]
        return 0.0
        
    base_df['elev_mean'] = base_df['subid'].apply(get_elevation)
    
    # 9. Normalize SLC fractions
    slc_cols = [col for col in base_df.columns if col.startswith('SLC_')]
    if slc_cols:
        base_df[slc_cols] = base_df[slc_cols].div(base_df[slc_cols].sum(axis=1), axis=0).fillna(0)
    
    # 10. Sort and save
    sorted_df = sort_geodata(base_df)
    sorted_df.to_csv(path_to_save+'GeoData.txt', sep='\t', index=False)
    
    # Write GeoClass file
    write_geoclass(slc_df, path_to_save)

    # Return land use information for parameter file generation
    return slc_df['landcover'].unique()

def _generate_landuse_params(land_uses):
    """
    Generate land-use-dependent parameter values for HYPE.

    Args:
        land_uses: Array of land use type IDs present in the domain

    Returns:
        Dictionary mapping parameter names to value strings
    """
    # Base parameter values for common IGBP land use types
    # These are typical values that work across different land uses
    # Index by land use ID (1-17 for IGBP classification)
    base_values = {
        'ttmp': {  # Snowmelt threshold temperature (deg)
            1: -0.9253,   # Evergreen Needleleaf
            2: -1.5960,   # Evergreen Broadleaf
            3: -0.9620,   # Deciduous Needleleaf
            4: -0.9620,   # Deciduous Broadleaf
            5: -0.9620,   # Mixed Forest
            6: -2.7121,   # Closed Shrubland
            7: -2.7121,   # Open Shrubland
            8: -0.9620,   # Woody Savanna
            9: -0.9620,   # Savanna
            10: -0.9253,  # Grassland
            11: 2.6945,   # Permanent Wetland
            12: -0.9253,  # Cropland
            13: -1.5960,  # Urban
            14: -2.7121,  # Cropland/Natural Mosaic
            15: 2.6945,   # Snow/Ice
            16: -2.7121,  # Barren
            17: 0.0,      # Water
        },
        'cmlt': {  # Snowmelt degree day coefficient
            1: 9.6497, 2: 9.2928, 3: 9.8897, 4: 9.8897, 5: 9.8897,
            6: 5.5393, 7: 5.5393, 8: 9.8897, 9: 9.8897, 10: 9.6497,
            11: 2.5333, 12: 9.6497, 13: 9.2928, 14: 5.5393, 15: 2.5333,
            16: 5.5393, 17: 0.0,
        },
        'cevp': {  # Evapotranspiration coefficient
            1: 0.4689, 2: 0.7925, 3: 0.6317, 4: 0.6317, 5: 0.6317,
            6: 0.1699, 7: 0.1699, 8: 0.6317, 9: 0.6317, 10: 0.4689,
            11: 0.4506, 12: 0.4689, 13: 0.7925, 14: 0.1699, 15: 0.4506,
            16: 0.1699, 17: 0.0,
        },
        'ttrig': {  # Soil temperature threshold for transpiration
            1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
            10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0,
        },
        'treda': {  # Root water uptake coefficient
            1: 0.84, 2: 0.84, 3: 0.84, 4: 0.84, 5: 0.84, 6: 0.84, 7: 0.84,
            8: 0.84, 9: 0.84, 10: 0.84, 11: 0.95, 12: 0.84, 13: 0.84,
            14: 0.84, 15: 0.95, 16: 0.84, 17: 0.0,
        },
        'tredb': {  # Root water uptake coefficient B
            1: 0.4, 2: 0.4, 3: 0.4, 4: 0.4, 5: 0.4, 6: 0.4, 7: 0.4,
            8: 0.4, 9: 0.4, 10: 0.4, 11: 0.4, 12: 0.4, 13: 0.4,
            14: 0.4, 15: 0.4, 16: 0.4, 17: 0.0,
        },
        'fepotsnow': {  # Fraction of PET for snow sublimation
            1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.8, 6: 0.8, 7: 0.8,
            8: 0.8, 9: 0.8, 10: 0.8, 11: 0.8, 12: 0.8, 13: 0.8,
            14: 0.8, 15: 0.8, 16: 0.8, 17: 0.0,
        },
        'srrcs': {  # Surface runoff coefficient
            1: 0.0673, 2: 0.1012, 3: 0.1984, 4: 0.1984, 5: 0.1984,
            6: 0.0202, 7: 0.0202, 8: 0.1984, 9: 0.1984, 10: 0.0673,
            11: 0.0202, 12: 0.0673, 13: 0.1012, 14: 0.0202, 15: 0.0202,
            16: 0.0202, 17: 0.0,
        },
        'surfmem': {  # Upper soil temperature memory
            1: 17.8, 2: 17.8, 3: 17.8, 4: 17.8, 5: 17.8, 6: 17.8, 7: 17.8,
            8: 17.8, 9: 17.8, 10: 17.8, 11: 5.15, 12: 17.8, 13: 17.8,
            14: 17.8, 15: 5.15, 16: 17.8, 17: 5.15,
        },
        'depthrel': {  # Depth relation for soil temperature
            1: 1.1152, 2: 1.1152, 3: 1.1152, 4: 1.1152, 5: 1.1152, 6: 1.1152, 7: 1.1152,
            8: 1.1152, 9: 1.1152, 10: 1.1152, 11: 2.47, 12: 1.1152, 13: 1.1152,
            14: 1.1152, 15: 2.47, 16: 1.1152, 17: 2.47,
        },
        'frost': {  # Frost depth parameter
            1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2,
            10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2,
        },
    }

    # Get maximum land use ID to determine array size
    max_lu = int(max(land_uses))

    # Generate parameter strings
    param_strings = {}
    for param_name, value_dict in base_values.items():
        values = []
        for lu_id in range(1, max_lu + 1):
            if lu_id in value_dict:
                values.append(value_dict[lu_id])
            else:
                # Use a sensible default for missing land uses
                values.append(value_dict.get(1, 0.0))

        # Format as tab-separated string
        param_strings[param_name] = '\t'.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in values)

    return param_strings, max_lu

# write par.txt file
def write_hype_par_file(path_to_save, params=None, template_file=None, land_uses=None):
    output_file = os.path.join(path_to_save, 'par.txt')
    if os.path.isfile(output_file):
        os.remove(output_file)
    
    # If land uses not provided, try to read from GeoClass.txt
    if land_uses is None:
        geoclass_file = os.path.join(path_to_save, 'GeoClass.txt')
        if os.path.exists(geoclass_file):
            try:
                geoclass_df = pd.read_csv(geoclass_file, sep='\t', skiprows=1, header=None)
                land_uses = geoclass_df.iloc[:, 1].unique()  # Second column is LULC
            except Exception as e:
                print(f"Warning: Could not read GeoClass.txt for land uses: {e}")
                land_uses = None

    # Generate dynamic land use parameters
    if land_uses is not None and len(land_uses) > 0:
        lu_params, max_lu = _generate_landuse_params(land_uses)
        lu_header = '\t'.join([f'LU{i}' for i in range(1, max_lu + 1)])
    else:
        # Fallback to default 5 land uses if we can't determine them
        lu_params = None
        max_lu = 5
        lu_header = 'LU1\tLU2\tLU3\tLU4\tLU5'

    if template_file and os.path.exists(template_file):
        with open(template_file, 'r') as f:
            par_content = f.read()
    else:
        # Build parameter file content with dynamic land use parameters
        # Use default values if no land use params provided
        ttmp_val = lu_params['ttmp'] if lu_params else ' -0.9253  -1.5960  -0.9620  -2.7121   2.6945'
        cmlt_val = lu_params['cmlt'] if lu_params else '  9.6497   9.2928   9.8897   5.5393   2.5333'
        cevp_val = lu_params['cevp'] if lu_params else ' 0.4689  0.7925  0.6317  0.1699  0.4506'
        ttrig_val = lu_params['ttrig'] if lu_params else '0\t0\t0\t0\t0'
        treda_val = lu_params['treda'] if lu_params else '0.84\t0.84\t0.84\t0.84\t0.95'
        tredb_val = lu_params['tredb'] if lu_params else '0.4\t0.4\t0.4\t0.4\t0.4'
        fepotsnow_val = lu_params['fepotsnow'] if lu_params else '0.8\t0.8\t0.8\t0.8\t0.8'
        srrcs_val = lu_params['srrcs'] if lu_params else ' 0.0673  0.1012  0.1984  0.0202  0.0202'
        surfmem_val = lu_params['surfmem'] if lu_params else '17.8\t17.8\t17.8\t17.8\t5.15'
        depthrel_val = lu_params['depthrel'] if lu_params else '1.1152\t1.1152\t1.1152\t1.1152\t2.47'
        frost_val = lu_params['frost'] if lu_params else '2\t2\t2\t2\t2'

        par_content = f"""!!	=======================================================================================================
!! Parameter file for:
!! HYPE -- Generated by the Model Agnostic Framework (hypeflow)
!!	=======================================================================================================
!!
!!	------------------------
!!
!!	=======================================================================================================
!!	SNOW - MELT, ACCUMULATION, AND DISTRIBUTION
!!	-----
!!	General snow accumulation and melt related parameters (baseline values from SHYPE)
ttpi	1.7083	!! width of the temperature interval with mixed precipitation
sdnsnew	0.13	!! density of fresh snow (kg/dm3)
snowdensdt	0.0016	!! snow densification parameter
fsceff	1	!! efficiency of fractional snow cover to reduce melt and evap
cmrefr	0.2	!! snow refreeze capacity (fraction of degreeday melt factor)
!!	-----
!!	Landuse dependent snow melt parameters
!!LUSE:	{lu_header}
ttmp	{ttmp_val}	!! Snowmelt threshold temperature (deg)
cmlt	{cmlt_val}	!! Snowmelt degree day coef (mm/deg/timestep)
!!	-----
!!	=======================================================================================================
!!	EVAPOTRANSPIRATION PARAMETERS
!!	-----
!!	General evapotranspiration parameters
lp	    0.6613	!! Threshold for water content reduction of transpiration
epotdist	   4.7088	!! Coefficient in exponential function for PET depth dependency
!!	-----
!!
!!LUSE:	{lu_header}
cevp	{cevp_val}
ttrig	{ttrig_val}	!! Soil temperature threshold to allow transpiration
treda	{treda_val}	!! Coefficient in soil temperature response function
tredb	{tredb_val}	!! Coefficient in soil temperature response fuction
fepotsnow	{fepotsnow_val}	!! Fraction of PET used for snow sublimation
!!
!! Frozen soil infiltration parameters
!! SOIL:	S1	S2
bfroznsoil  3.7518  3.2838  !! frozen soil infiltration parameter
logsatmp	1.15	!! saturated matric potential
bcosby	    11.2208	    19.6669	!! Cosby B parameter
!!	=======================================================================================================
!!	SOIL/LAND HYDRAULIC RESPONSE PARAMETERS
!!	-----
!!	Soil-class parameters
!!	S1	S2
rrcs1   0.4345  0.5985	!! recession coefficients uppermost layer
rrcs2   0.1201  0.1853	!! recession coefficients bottom layer
rrcs3	    0.0939	!! Recession coefficient slope dependance
sfrost  1   1	!! frost depth parameter (cm/degree Celsius)
wcwp    0.1171  0.0280	!! Soil water content at wilting point
wcfc    0.3771  0.2009	!! Field capacity
wcep    0.4047  0.4165	!! Effective porosity
!!	-----
!!	Landuse-class parameters
!!LUSE:	{lu_header}
srrcs	{srrcs_val}	!! Runoff coefficient for surface runoff
!!	-----
!!	Regional groundwater outflow
rcgrw	0	!! recession coefficient for regional groundwater outflow
!!	=======================================================================================================
!!	SOIL TEMPERATURE AND SOIL FROST DEPT
!!	-----
!!	General
deepmem	1000	!! temperature memory of deep soil (days)
!!-----
!!LUSE:	{lu_header}
surfmem	{surfmem_val}	!! upper soil layer soil temperature memory (days)
depthrel	{depthrel_val}	!! depth relation for soil temperature memory
frost	{frost_val}	!! frost depth parameter
!!	-----
!!	=======================================================================================================
!!	LAKE DISCHARGE
!!	-----
!!	ILAKE and OLAKE REGIONAL PARAMETERS
!!	ILAKE parameters
!! ilRegion	PPR 1
ilratk  149.9593
ilratp  4.9537
illdepth    0.33
ilicatch    1.0
!!
!!	=======================================================================================================
!!	RIVER ROUTING
!!	-----
damp	   0.2719	!! fraction of delay in the watercourse which also causes damping
rivvel	     9.7605	!! celerity of flood in watercourse
qmean 	200	!! initial value for calculation of mean flow (mm/yr)"""

    # Substitution logic for calibration parameters
    if params:
        for key, value in params.items():
            # Handle list/array values (often for landuse/soil specific params)
            if isinstance(value, (list, np.ndarray)):
                val_str = "  ".join(map(str, value))
            else:
                val_str = str(value)
            
            # Simple substitution for now: look for key followed by whitespace or tab
            # This is a bit naive for HYPE par files but works for basic cases
            import re
            par_content = re.sub(fr'^({key}\s+)[^\!]*', fr'\g<1>{val_str}  ', par_content, flags=re.MULTILINE)

    with open(output_file, 'w') as file:
            file.write(par_content)

# write info and filedir files
def write_hype_info_filedir_files(path_to_save, spinup_days, hype_results_dir,
                                   experiment_start=None, experiment_end=None):
    """
    Write HYPE info.txt and filedir.txt files.

    Args:
        path_to_save: Directory to save files
        spinup_days: Number of spinup days
        hype_results_dir: Directory for HYPE results
        experiment_start: Start date for experiment (defaults to first date in Pobs.txt)
        experiment_end: End date for experiment (defaults to last date in Pobs.txt)
    """
    output_file = os.path.join(path_to_save, 'filedir.txt')
    if os.path.isfile(output_file):
        os.remove(output_file)
    with open(output_file, 'w') as file:
            file.write('./')

    output_file = os.path.join(path_to_save, 'info.txt')
    if os.path.isfile(output_file):
        os.remove(output_file)

    # Use experiment dates if provided, otherwise read from Pobs.txt
    if experiment_start is None or experiment_end is None:
        Pobs_path = os.path.join(path_to_save, 'Pobs.txt')
        Pobs = pd.read_csv(Pobs_path, sep='\t', parse_dates=['time'])
        Pobs['time'] = pd.to_datetime(Pobs['time']).dt.date
        if experiment_start is None:
            start_date = Pobs['time'].iloc[0]
        else:
            start_date = pd.to_datetime(experiment_start).date()
        if experiment_end is None:
            end_date = Pobs['time'].iloc[-1]
        else:
            # For HYPE's daily timestep, if end time is not midnight, include the full day
            end_dt = pd.to_datetime(experiment_end)
            end_date = end_dt.date()
            if end_dt.hour > 0 or end_dt.minute > 0 or end_dt.second > 0:
                # If there's a time component beyond midnight, include the full day
                end_date = end_date + pd.Timedelta(days=0)
    else:
        start_date = pd.to_datetime(experiment_start).date()
        # For HYPE's daily timestep, if end time is not midnight, include the full day
        end_dt = pd.to_datetime(experiment_end)
        end_date = end_dt.date()
        if end_dt.hour > 0 or end_dt.minute > 0 or end_dt.second > 0:
            # If there's a time component beyond midnight, include the full day
            end_date = end_date + pd.Timedelta(days=0)

    spinup_date = start_date + pd.Timedelta(days=spinup_days)

    # HYPE requires at least one day of simulation: if start==end, increment end by 1 day
    if start_date == end_date:
        end_date = end_date + pd.Timedelta(days=1)

    print(f"DEBUG HYPE dates: start={start_date}, end={end_date}, spinup_days={spinup_days}, spinup_date={spinup_date}")

    s1 = """!! ----------------------------------------------------------------------------
!!
!! HYPE - Model Agnostic Framework
!!							
!! -----------------------------------------------------------------------------							
!! Check Indata during first runs (deactivate after first runs) 
indatacheckonoff 	2						
indatachecklevel	2		
!! -----------------------------------------------------------------------------							
!!
!! -----------------------------------------------------------------------------							
!!						
!! Simulation settings:							
!!							
!! -----------------	 """
    
    df2_row = ['bdate', 'cdate', 'edate', 'resultdir', 'instate', 'warning']
    df2_val = [start_date, spinup_date, end_date, hype_results_dir, 'n', 'y']
    df2 = pd.DataFrame(df2_val, index=df2_row, columns=None)

    s3 = """readdaily 	y						
submodel 	n						
calibration	n						
readobsid   n							
soilstretch	n						
!! Soilstretch enable the use of soilcorr parameters (strech soildepths in layer 2 and 3)
steplength	1d							
!! -----------------------------------------------------------------------------							
!!							
!! Enable/disable optional input files
!!							
!! -----------------							
readsfobs	n	!! For observed snowfall fractions in SFobs.txt							
readswobs	n	!! For observed shortwave radiation in SWobs.txt
readuobs	n	!! For observed wind speeds in Uobs.txt
readrhobs	n	!! For observed relative humidity in RHobs.txt					
readtminobs	y	!! For observed min air temperature in TMINobs.txt				
readtmaxobs	y	!! For observed max air temperature in TMAXobs.txt
soiliniwet	n	!! initiates soil water to porosity instead of field capacity which is default (N). Set Y to use porosity.
usestop84	n	!! flag to use the old return code 84 for a successful run					
!! -----------------------------------------------------------------------------							
!!							
!! Define model options (optional)
!!							
!! -----------------							
!!snowfallmodel:								
!!                  0 threshold temperature model							
!!                  1 inputdata (SFobs.txt)							
!!snowmeltmodel:							
!!                  0,1 temperature index             (with/without snowcover scaling)							
!!                  2   temperature + radiation index (with/without snowcover scaling)							
!!							
!!  snowevapmodel   0 off							
!!                  1 on							
!!                   							
!!  petmodel:  (potential evapotranspiration) (is shown in geodata for WWH)							
!!                  0 original HYPE temperature model (with Xobs epot replacement)							
!!                  1 original HYPE temperature model (without Xobs epot replacement)							
!!                  2 Modified Jensen-Haise 							
!!                  3 Modified Hargreaves-Samani							
!!                  4 Priestly-Taylor							
!!                  5 FAo Penman-Monteith							
!!							
!! lakeriverice:							
!!                  0 off							
!!                  1 on, old (simple) air-water heat exchange              (requires T2 water temperature model)							
!!                  2 on, new heatbalance model for air-water heat exchange (requires T2 water temperature model)							
!!							
!! substance T2     switching on the new water temperature trace model							
!!							
!! deepground       0   off    Deep groundwater (Aquifer) model options							
!!                  1,2 on
!! Glacierini	0 off 1 on	(1 used for statefile preparation)	
!! Floodplain		0, 1, 2, 3 (3 used for WWH)					
!! -----------------							
modeloption snowfallmodel	0						
modeloption snowdensity	0
modeloption snowfalldist	2
modeloption snowheat	0
modeloption snowmeltmodel	0	
modeloption	snowevapmodel	1				
modeloption snowevaporation	1					
modeloption lakeriverice	0									
modeloption deepground	0 	
modeloption glacierini	1
modeloption floodmodel 0
modeloption frozensoil 2
modeloption infiltration 3
modeloption surfacerunoff 0
modeloption petmodel	1
modeloption wetlandmodel 2		
modeloption connectivity 0					
!! ------------------------------------------------------------------------------------							
!!							
!! Define outputs
!!							
!! -----------------							
!! meanperiod 1=daymean, 2=weekmean, 3=monthmean, 4=yearmean, 5=total period mean							
!! output variables: see http://www.smhi.net/hype/wiki/doku.php?id=start:hype_file_reference:info.txt:variables 
!! -----------------							
!! TIME outputs 
!! -----------------	
timeoutput variable cout	evap	snow
timeoutput meanperiod	1
timeoutput decimals	3					
!! ------------------------------------------------------------------------------------							
!!							
!! Select criteria for model evaluation and automatic calibration
!!							
!! -----------------							
!! crit 1 criterion	MKG
!! crit 1 cvariable	cout
!! crit 1 rvariable	rout
!! crit 1 weight	1"""

    with open(output_file, 'w') as file:
        file.write(s1 + '\n')
        df2.to_csv(file, sep='\t', index=True, header=False)
        file.write(s3 + '\n')

def write_geoclass(slc_df, path_to_save):
    """Write GeoClass.txt file for HYPE model with full metadata and specific formatting."""
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

    output_file = os.path.join(path_to_save, 'GeoClass.txt')
    with open(output_file, 'w') as file:
        file.write("!          SLC	LULC	SOIL TYPE	Main crop cropid	Second crop cropid	Crop rotation group	Vegetation type	Special class code	Tile depth	Stream depth	Number of soil layers	Soil layer depth 1	Soil layer depth 2	Soil layer depth 3 \n")
        combination.to_csv(file, sep='\t', index=False, header=False)