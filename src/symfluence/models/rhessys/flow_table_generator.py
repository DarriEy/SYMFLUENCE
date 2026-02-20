"""
RHESSys Flow Table Generator

Handles generation of RHESSys flow routing tables, which define how water
moves between patches in the landscape hierarchy.

Flow table format (per construct_routing_topology.c):
    <num_patches>
    <patch_ID zone_ID hill_ID x y z area area drainage_type gamma num_neighbors>
    [<neighbor_patch_ID neighbor_zone_ID neighbor_hill_ID gamma> for each neighbor]

    drainage_type: 0=LAND, 1=STREAM, 2=ROAD

Extracted from RHESSysPreProcessor for modularity.
"""
import logging
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)


class RHESSysFlowTableGenerator:
    """
    Generates RHESSys flow routing tables.

    Creates routing tables that define lateral water movement between patches.
    Supports both lumped (single-patch) and distributed (multi-patch) domains.

    Args:
        preprocessor: Parent RHESSysPreProcessor instance providing access
            to configuration, paths, and helper methods.
    """

    def __init__(self, preprocessor):
        self.pp = preprocessor

    def generate_flow_table(self):
        """
        Generate the RHESSys flow table for routing.

        For distributed models, creates routing based on elevation gradient.
        For lumped models, single patch drains to stream outlet.

        Flow table format (per construct_routing_topology.c):
        <num_patches>
        <patch_ID zone_ID hill_ID x y z area area drainage_type gamma num_neighbors>
        [<neighbor_patch_ID neighbor_zone_ID neighbor_hill_ID gamma> for each neighbor]

        drainage_type: 0=LAND, 1=STREAM, 2=ROAD
        """
        logger.info("Generating flow table...")

        flow_file = self.pp.routing_dir / f"{self.pp.domain_name}.routing"

        # Check if we have distributed patches from worldfile generation
        if hasattr(self.pp, '_distributed_patches') and len(self.pp._distributed_patches) > 1:
            self.generate_distributed_flow_table(flow_file)
            return

        # Fall back to single-patch flow table
        self._generate_lumped_flow_table(flow_file)

    def _generate_lumped_flow_table(self, flow_file: Path):
        """
        Generate a lumped single-patch flow table.

        Args:
            flow_file: Output path for the flow table file.
        """
        try:
            catchment_path = self.pp.get_catchment_path()

            # Search alternate experiment dirs if not found
            if not catchment_path.exists():
                catchment_dir = self.pp.project_dir / 'shapefiles' / 'catchment'
                if catchment_dir.exists():
                    for shp_file in catchment_dir.rglob('*.shp'):
                        if self.pp.domain_name in shp_file.name:
                            catchment_path = shp_file
                            logger.info(f"Found catchment shapefile in alternate location: {shp_file}")
                            break

            if not catchment_path.exists():
                raise FileNotFoundError(
                    f"Catchment shapefile not found for flow table generation. "
                    f"Searched: {self.pp.project_dir / 'shapefiles' / 'catchment'}"
                )

            gdf = gpd.read_file(catchment_path)
            utm_crs = self.pp._get_utm_crs_from_bounds(gdf)
            lon, lat = self.pp._get_centroid_lon_lat(gdf, utm_crs)
            elev_col = getattr(self.pp, 'catchment_elev_col', 'elev_mean')
            elev = float(gdf.get(elev_col, [1500])[0]) if elev_col in gdf.columns else 1500.0

            gdf_proj = (gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf).to_crs(utm_crs)
            area_m2 = gdf_proj.geometry.area.sum()
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError(
                f"Failed to read catchment for flow table: {e}"
            ) from e

        patch_id = 1
        zone_id = 1
        hill_id = 1
        num_hillslopes = 1
        num_patches = 1

        # For lumped single-patch model, patch is treated as outlet (drainage_type=1=STREAM)
        # gamma=1.0 means 100% of water leaves the system as streamflow
        # num_neighbors=0 since there's no downstream patch
        content = f"""{num_hillslopes}
{hill_id}
{num_patches}
{patch_id} {zone_id} {hill_id} {lon:.8f} {lat:.8f} {elev:.8f} {area_m2:.8f} {area_m2:.8f} 1 1.0 0
"""

        flow_file.write_text(content, encoding='utf-8')
        logger.info(f"Flow table written: {flow_file}")

    def generate_distributed_flow_table(self, flow_file: Path):
        """
        Generate flow table for distributed domain with multiple patches.

        Routes water based on elevation gradient - each patch drains to the
        lowest elevation neighbor, with the lowest overall patch being the outlet.

        Args:
            flow_file: Output path for flow table
        """
        patches = self.pp._distributed_patches
        num_patches = len(patches)
        logger.info(f"Generating distributed flow table for {num_patches} patches")

        # Sort patches by elevation (lowest = outlet)
        sorted_patches = sorted(patches, key=lambda p: p['elev'])

        # Find outlet (lowest elevation patch)
        outlet_patch = sorted_patches[0]
        outlet_id = outlet_patch['patch_id']

        # Build adjacency based on elevation - each patch drains to outlet
        # For simple approach: all patches drain directly to outlet
        # More sophisticated: chain drainage based on elevation
        lines = []
        lines.append("1")  # num_hillslopes
        lines.append("1")  # hillslope_ID
        lines.append(str(num_patches))

        for patch in patches:
            pid = patch['patch_id']
            zid = patch['zone_id']
            hid = patch['hill_id']
            lon = patch['lon']
            lat = patch['lat']
            elev = patch['elev']
            area = patch['area']

            if pid == outlet_id:
                # Outlet patch: drainage_type=1 (STREAM), no neighbors
                lines.append(
                    f"{pid} {zid} {hid} {lon:.8f} {lat:.8f} {elev:.8f} "
                    f"{area:.8f} {area:.8f} 1 0.0 0"
                )
            else:
                # Find downstream neighbor (next lower elevation patch)
                # Simple approach: drain directly to outlet
                downstream_id = outlet_id
                downstream_zone = outlet_patch['zone_id']
                downstream_hill = outlet_patch['hill_id']

                # More sophisticated: find nearest lower elevation patch
                for p in sorted_patches:
                    if p['elev'] < elev and p['patch_id'] != pid:
                        downstream_id = p['patch_id']
                        downstream_zone = p['zone_id']
                        downstream_hill = p['hill_id']
                        break

                # drainage_type=0 (LAND), gamma=1.0 (100% to neighbor), 1 neighbor
                lines.append(
                    f"{pid} {zid} {hid} {lon:.8f} {lat:.8f} {elev:.8f} "
                    f"{area:.8f} {area:.8f} 0 1.0 1"
                )
                # Neighbor line: patch_id zone_id hill_id gamma
                lines.append(f"{downstream_id} {downstream_zone} {downstream_hill} 1.0")

        content = '\n'.join(lines)
        flow_file.write_text(content, encoding='utf-8')
        logger.info(f"Distributed flow table written: {flow_file} ({num_patches} patches, outlet={outlet_id})")
