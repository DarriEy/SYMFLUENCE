# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""MERIT-Basins Vector Data Acquisition Handler

Resolves catchment and river network vector data from the MERIT-Basins
dataset (Lin et al., 2019). Data is organized by Pfafstetter Level-1
hydrological regions.

The historical bulk-download host (hydrology.princeton.edu) is gone (DNS
no longer resolves); distribution moved to a public Google Drive folder and
Globus (see reachhydro.org). Neither channel supports scripted direct
downloads, so this handler resolves locally staged files and, when they are
missing, fails with an actionable message listing the current distribution
channels and the staging directory.

Workflow:
    1. Map pour point / domain bbox to Pfafstetter Level-1 code(s) using the
       official region table
    2. Resolve locally staged catchment + river shapefiles for the region(s)
    3. The subsetter's spatial join handles precise selection afterwards

Source:
    reachhydro.org — Pfafstetter-organized shapefiles
    (Google Drive folder + Globus; see _MERIT_CHANNELS below)

Column Convention (matches subsetter MERIT type):
    COMID, NextDownID, up1, up2, up3

References:
    - Lin et al. (2019). Global reconstruction of naturalized river flows at
      2.94 million reaches. Water Resources Research, 55, 6499-6516
    - Yamazaki et al. (2019). MERIT Hydro. Water Resources Research, 55, 5053-5073
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from symfluence.core.registries import R

from ..base import BaseAcquisitionHandler

# Current MERIT-Basins distribution channels (the legacy bulk host
# hydrology.princeton.edu is DNS-dead).
_MERIT_DRIVE_URL = "https://drive.google.com/drive/folders/1uCQFmdxFbjwoT9OYJxw-pXaP8q_GYH1a"
_MERIT_INFO_URL = "https://www.reachhydro.org/home/params/merit-basins"

# Upstream shapefile basenames inside the per-region zips
_CAT_NAME = "pfaf_{code}_MERIT_Hydro_v07_Basins_v01"
_RIV_NAME = "pfaf_{code}_MERIT_Hydro_v07_Basins_v01_rivernet"

# Pfafstetter Level-1 continental regions with approximate bounding boxes,
# following the OFFICIAL MERIT-Basins ReadMe mapping:
#   1 Africa, 2 Europe, 3 North Asia/Siberia, 4 South/Central Asia,
#   5 Oceania + South Asian islands, 6 South America, 7 North America,
#   8 Arctic (North American Arctic), 9 Greenland.
# (A previous version of this table shipped a different, incorrect mapping
# — e.g. 9 was labelled Australia while the shipped pfaf_9 extent is
# Greenland — which selected the wrong region archives.)
# Format: {code: (lat_min, lon_min, lat_max, lon_max)}
_PFAFSTETTER_L1_BBOXES = {
    1: (-36.0, -20.0, 38.0, 56.0),    # Africa
    2: (12.0, -25.0, 72.0, 70.0),     # Europe (incl. Middle East fringe)
    3: (40.0, 55.0, 84.0, 180.0),     # North Asia / Siberia
    4: (0.0, 55.0, 56.0, 150.0),      # South Asia (Middle East to E Asia)
    5: (-56.0, 90.0, 21.0, 180.0),    # Oceania + South Asian islands
    6: (-56.0, -93.0, 15.0, -32.0),   # South America
    7: (5.0, -170.0, 62.0, -52.0),    # North America
    8: (50.0, -180.0, 84.0, -50.0),   # Arctic Region (N. American Arctic)
    9: (58.0, -75.0, 84.0, -10.0),    # Greenland
}


@R.acquisition_handlers.add('MERIT_BASINS')
class MERITBasinsAcquirer(BaseAcquisitionHandler):
    """MERIT-Basins vector data resolution from locally staged archives.

    Maps the domain to Pfafstetter-organized catchment and river network
    shapefiles using the official region table — the subsetter handles
    precise spatial selection afterwards. Because the current distribution
    channels (Google Drive folder, Globus) cannot be scripted, missing
    region files raise an actionable error instead of attempting a download
    from the dead legacy host.

    Accepted staged files (per region ``{code}``), in the geofabric
    directory ``<domain>/attributes/geofabric/merit_basins/``:
        pfaf_{code}_MERIT_Hydro_v07_Basins_v01.shp          — catchments
        pfaf_{code}_MERIT_Hydro_v07_Basins_v01_rivernet.shp — rivers
        merit_cat_pfaf_{code}.shp / merit_riv_pfaf_{code}.shp (legacy names)
    """

    def download(self, output_dir: Path) -> Path:
        """Resolve MERIT-Basins data for the domain from local staging.

        Args:
            output_dir: Base output directory

        Returns:
            Path to the directory containing the geofabric files

        Raises:
            ValueError: If the pour point matches no Pfafstetter region
            RuntimeError: If region files are not staged locally (with the
                current distribution channels listed)
        """
        geofabric_dir = self._attribute_dir("geofabric") / "merit_basins"
        geofabric_dir.mkdir(parents=True, exist_ok=True)

        lat, lon = self._get_pour_point_coords()
        self.logger.info(
            f"Resolving MERIT-Basins data for pour point ({lat:.4f}, {lon:.4f})"
        )

        # Find matching Pfafstetter region(s)
        pfaf_codes = self._find_pfafstetter_regions(lat, lon)
        if not pfaf_codes:
            raise ValueError(
                f"Pour point ({lat}, {lon}) does not fall within any "
                "Pfafstetter Level-1 region"
            )
        self.logger.info(f"Matched Pfafstetter L1 region(s): {pfaf_codes}")

        missing = [
            code for code in pfaf_codes
            if not self._region_files_present(geofabric_dir, code)
        ]
        if missing:
            raise RuntimeError(self._staging_error_message(geofabric_dir, missing))

        self.logger.info(f"MERIT-Basins data resolved in: {geofabric_dir}")
        return geofabric_dir

    @staticmethod
    def _region_files_present(geofabric_dir: Path, code: int) -> bool:
        """Check whether both shapefiles for a region are staged locally.

        Accepts the upstream archive names and the legacy ``merit_cat_``/
        ``merit_riv_`` names, anywhere below the geofabric directory.
        """
        cat_patterns = (
            f"**/{_CAT_NAME.format(code=code)}.shp",
            f"**/merit_cat_pfaf_{code}.shp",
        )
        riv_patterns = (
            f"**/{_RIV_NAME.format(code=code)}.shp",
            f"**/merit_riv_pfaf_{code}.shp",
        )
        has_cat = any(next(geofabric_dir.glob(p), None) for p in cat_patterns)
        has_riv = any(next(geofabric_dir.glob(p), None) for p in riv_patterns)
        return has_cat and has_riv

    @staticmethod
    def _staging_error_message(geofabric_dir: Path, missing: List[int]) -> str:
        """Build the actionable failure message for missing region files."""
        wanted = "\n".join(
            f"    {_CAT_NAME.format(code=c)}.zip  and  "
            f"{_RIV_NAME.format(code=c)}.zip"
            for c in missing
        )
        return (
            f"MERIT-Basins region files for Pfafstetter region(s) {missing} "
            "are not staged locally, and the legacy bulk-download host "
            "(hydrology.princeton.edu) no longer exists. MERIT-Basins is now "
            "distributed through channels that cannot be scripted, so the "
            "files must be staged manually:\n"
            "\n"
            "  1. Download the region archives from one of:\n"
            f"     - Google Drive folder: {_MERIT_DRIVE_URL}\n"
            "       (level_01 catchments + river networks, MERIT_Hydro_v07_Basins_v01)\n"
            "     - Globus (bulk transfer; endpoint linked from reachhydro.org)\n"
            f"     - Documentation: {_MERIT_INFO_URL}\n"
            "\n"
            "  2. Required archives:\n"
            f"{wanted}\n"
            "\n"
            "  3. Extract the shapefile sets (.shp/.dbf/.shx/.prj) into:\n"
            f"     {geofabric_dir}\n"
            "\n"
            "Re-run the acquisition step afterwards; staged files are picked "
            "up automatically."
        )

    def _get_pour_point_coords(self) -> tuple:
        """Extract pour point lat/lon from config.

        Returns:
            Tuple of (lat, lon)
        """
        pour_point_str = self._get_config_value(lambda: self.config.domain.pour_point_coords, default=None)
        if pour_point_str:
            parts = str(pour_point_str).replace('/', ',').split(',')
            return float(parts[0].strip()), float(parts[1].strip())

        lat = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2
        lon = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2
        return lat, lon

    def _find_pfafstetter_regions(self, lat: float, lon: float) -> List[int]:
        """Find Pfafstetter L1 region(s) containing the pour point.

        Uses coarse bounding boxes — may return multiple regions for
        points near boundaries. Also checks the full bounding box if
        available.

        Args:
            lat: Latitude of pour point
            lon: Longitude of pour point

        Returns:
            List of matching Pfafstetter L1 codes
        """
        matching = []
        for code, (lat_min, lon_min, lat_max, lon_max) in _PFAFSTETTER_L1_BBOXES.items():
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                matching.append(code)

        # Also check full domain bbox for large domains spanning regions
        if self.bbox:
            for code, (lat_min, lon_min, lat_max, lon_max) in _PFAFSTETTER_L1_BBOXES.items():
                if code not in matching:
                    # Check if the domain bbox intersects this region
                    if (self.bbox['lat_min'] <= lat_max and
                            self.bbox['lat_max'] >= lat_min and
                            self.bbox['lon_min'] <= lon_max and
                            self.bbox['lon_max'] >= lon_min):
                        matching.append(code)

        return sorted(set(matching))
