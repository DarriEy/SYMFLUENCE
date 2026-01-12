"""
MESH Data Preprocessor

Handles shapefile and landcover data preparation.
"""

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, Any, List

import geopandas as gpd
import pandas as pd


class MESHDataPreprocessor:
    """
    Prepares shapefiles and landcover data for MESH preprocessing.

    Handles:
    - Copying and sanitizing shapefiles
    - Fixing outlet segments in river network
    - Detecting GRU classes from landcover stats
    - Sanitizing landcover stats for meshflow
    - Copying settings files
    """

    def __init__(
        self,
        forcing_dir: Path,
        setup_dir: Path,
        config: Dict[str, Any],
        logger: logging.Logger = None
    ):
        """
        Initialize data preprocessor.

        Args:
            forcing_dir: Directory for MESH files
            setup_dir: Directory containing settings files
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.forcing_dir = forcing_dir
        self.setup_dir = setup_dir
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def copy_shapefile(self, src: str, dst: Path) -> None:
        """Copy all files associated with a shapefile."""
        src_path = Path(src)
        for f in src_path.parent.glob(f"{src_path.stem}.*"):
            shutil.copy2(f, dst.parent / f"{dst.stem}{f.suffix}")

    def sanitize_shapefile(self, shp_path: str) -> None:
        """Remove or rename problematic fields from shapefile."""
        if not shp_path:
            return

        path = Path(shp_path)
        if not path.exists():
            return

        try:
            gdf = gpd.read_file(path)

            if 'ID' in gdf.columns:
                self.logger.info(f"Sanitizing {path.name}: renaming 'ID' to 'ORIG_ID'")
                gdf = gdf.rename(columns={'ID': 'ORIG_ID'})
                temp_path = path.with_suffix('.tmp.shp')
                gdf.to_file(temp_path)
                shutil.move(temp_path, path)

                for ext in ['.shx', '.dbf', '.prj', '.cpg']:
                    temp_ext = temp_path.with_suffix(ext)
                    if temp_ext.exists():
                        shutil.move(temp_ext, path.with_suffix(ext))

        except Exception as e:
            self.logger.warning(f"Failed to sanitize shapefile {path}: {e}")

    def fix_outlet_segment(self, shp_path: str, outlet_value: int = 0) -> None:
        """Fix outlet segment in river network shapefile."""
        if not shp_path:
            return

        path = Path(shp_path)
        if not path.exists():
            return

        try:
            gdf = gpd.read_file(path)

            if 'LINKNO' not in gdf.columns or 'DSLINKNO' not in gdf.columns:
                return

            valid_linknos = set(gdf['LINKNO'].values)
            outlet_mask = ~gdf['DSLINKNO'].isin(valid_linknos) & (gdf['DSLINKNO'] != outlet_value)

            if outlet_mask.any():
                gdf.loc[outlet_mask, 'DSLINKNO'] = outlet_value
                gdf.to_file(path)
                self.logger.info(f"Fixed {outlet_mask.sum()} outlet segment(s) in {path.name}")

        except Exception as e:
            self.logger.warning(f"Failed to fix outlet segment: {e}")

    def ensure_gru_id(self, shp_path: str) -> None:
        """
        Ensure shapefile has GRU_ID column.

        If missing:
        - For single feature (lumped): set to 1
        - For multiple features: set to range 1..N
        """
        if not shp_path:
            return

        path = Path(shp_path)
        if not path.exists():
            return

        try:
            gdf = gpd.read_file(path)

            if 'GRU_ID' not in gdf.columns:
                self.logger.info(f"Adding GRU_ID to {path.name}")
                if len(gdf) == 1:
                    gdf['GRU_ID'] = 1
                else:
                    gdf['GRU_ID'] = range(1, len(gdf) + 1)
                
                gdf.to_file(path)

        except Exception as e:
            self.logger.warning(f"Failed to ensure GRU_ID: {e}")

    def detect_gru_classes(self, landcover_path: Path) -> List[int]:
        """
        Detect which GRU classes exist in the landcover stats file.

        Args:
            landcover_path: Path to landcover stats CSV

        Returns:
            List of GRU class numbers that exist in the data
        """
        if not landcover_path or not Path(landcover_path).exists():
            self.logger.warning(f"Landcover file not found: {landcover_path}")
            return []

        try:
            df = pd.read_csv(landcover_path)

            frac_cols = [col for col in df.columns if col.startswith('frac_')]
            igbp_cols = [col for col in df.columns if col.startswith('IGBP_')]

            gru_classes = set()

            for col in frac_cols:
                match = re.match(r'frac_(\d+)', col)
                if match:
                    gru_classes.add(int(match.group(1)))

            for col in igbp_cols:
                match = re.match(r'IGBP_(\d+)', col)
                if match:
                    gru_classes.add(int(match.group(1)))

            result = sorted(list(gru_classes))
            self.logger.debug(f"Detected GRU classes from {landcover_path.name}: {result}")
            return result

        except Exception as e:
            self.logger.warning(f"Failed to detect GRU classes: {e}")
            return []

    def sanitize_landcover_stats(self, csv_path: str) -> str:
        """Sanitize landcover stats CSV for meshflow compatibility."""
        if not csv_path:
            return csv_path

        path = Path(csv_path)
        if not path.exists():
            return csv_path

        try:
            df = pd.read_csv(path)

            # Remove unnamed columns
            cols_to_drop = [col for col in df.columns if 'Unnamed' in col]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            # Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_rows:
                self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

            # Convert IGBP_* to frac_*
            igbp_cols = [col for col in df.columns if col.startswith('IGBP_')]
            if igbp_cols:
                count_data = df[igbp_cols].fillna(0)
                row_totals = count_data.sum(axis=1)

                for col in igbp_cols:
                    class_num = col.replace('IGBP_', '')
                    frac_col = f'frac_{class_num}'
                    df[frac_col] = count_data[col] / row_totals.replace(0, 1)
                    df = df.drop(columns=[col])

            temp_path = self.forcing_dir / f"temp_{path.name}"
            df.to_csv(temp_path, index=False)
            return str(temp_path)

        except Exception as e:
            self.logger.warning(f"Failed to sanitize landcover stats: {e}")
            return csv_path

    def copy_settings_to_forcing(self) -> None:
        """Copy MESH settings files from setup_dir to forcing_dir."""
        import os

        self.logger.info(f"Copying MESH settings from {self.setup_dir} to {self.forcing_dir}")

        try:
            if self.setup_dir.resolve() == self.forcing_dir.resolve():
                self.logger.info("Settings and forcing directories are the same, skipping")
                return
        except Exception:
            pass

        skip_files = [
            "MESH_input_run_options.ini",
            "MESH_forcing.nc",
            "MESH_drainage_database.nc",
            "MESH_forcing_safe.nc"
        ]

        for settings_file in self.setup_dir.glob("*"):
            if settings_file.is_file():
                if settings_file.name in skip_files:
                    continue

                dest_file = self.forcing_dir / settings_file.name

                try:
                    if settings_file.resolve() == dest_file.resolve():
                        continue
                    if dest_file.exists() and os.path.samefile(settings_file, dest_file):
                        continue
                except (OSError, FileNotFoundError):
                    pass

                shutil.copy2(settings_file, dest_file)
