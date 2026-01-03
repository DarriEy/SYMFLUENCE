from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
import logging
from typing import Optional, Union

def clean_geometry(
    geometry: Union[Polygon, MultiPolygon, GeometryCollection], 
    logger: Optional[logging.Logger] = None
) -> Optional[Union[Polygon, MultiPolygon]]:
    """
    Clean and validate geometries, ensuring only Polygon or MultiPolygon.
    
    Args:
        geometry: Shapely geometry object
        logger: Optional logger for debug messages
        
    Returns:
        Cleaned Polygon or MultiPolygon, or None if invalid/empty
    """
    if geometry is None or geometry.is_empty:
        return None

    try:
        # Handle GeometryCollection - extract only Polygons
        if isinstance(geometry, GeometryCollection):
            polygons = []
            for geom in geometry.geoms:
                if (
                    isinstance(geom, Polygon)
                    and geom.is_valid
                    and not geom.is_empty
                ):
                    polygons.append(geom)
                elif isinstance(geom, MultiPolygon):
                    for poly in geom.geoms:
                        if (
                            isinstance(poly, Polygon)
                            and poly.is_valid
                            and not poly.is_empty
                        ):
                            polygons.append(poly)

            if not polygons:
                return None
            elif len(polygons) == 1:
                geometry = polygons[0]
            else:
                geometry = MultiPolygon(polygons)

        # Ensure we have a valid Polygon or MultiPolygon
        if not isinstance(geometry, (Polygon, MultiPolygon)):
            return None

        # Fix invalid geometries
        if not geometry.is_valid:
            geometry = geometry.buffer(0)

            # Check again after buffer
            if not isinstance(geometry, (Polygon, MultiPolygon)):
                return None

        return geometry if geometry.is_valid and not geometry.is_empty else None

    except Exception as e:
        if logger:
            logger.debug(f"Error cleaning geometry: {str(e)}")
        return None
