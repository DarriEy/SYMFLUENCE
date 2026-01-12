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

        If missing or invalid:
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

            needs_update = False
            if 'GRU_ID' not in gdf.columns:
                self.logger.info(f"Adding GRU_ID to {path.name}")
                if len(gdf) == 1:
                    gdf['GRU_ID'] = 1
                else:
                    gdf['GRU_ID'] = range(1, len(gdf) + 1)
                needs_update = True
            
            # Ensure it is integer type and has no NaNs
            if 'GRU_ID' in gdf.columns:
                if len(gdf) == 1 and gdf['GRU_ID'].iloc[0] != 1:
                    self.logger.info(f"Forcing lumped GRU_ID to 1 in {path.name}")
                    gdf['GRU_ID'] = 1
                    needs_update = True

                if not pd.api.types.is_integer_dtype(gdf['GRU_ID']):
                    self.logger.info(f"Converting GRU_ID to integer in {path.name}")
                    gdf['GRU_ID'] = pd.to_numeric(gdf['GRU_ID'], errors='coerce').fillna(1).astype(int)
                    needs_update = True

            if needs_update:
                gdf.to_file(path)

        except Exception as e:
            self.logger.warning(f"Failed to ensure GRU_ID: {e}")

    def ensure_hru_id(self, shp_path: str, hru_col: str, main_id_col: str = 'GRU_ID') -> None:
        """
        Ensure shapefile has the HRU dimension column (e.g., 'subbasin').
        Meshflow uses this for indexing the DDB.
        """
        if not shp_path or not hru_col:
            return

        path = Path(shp_path)
        if not path.exists():
            return

        try:
            gdf = gpd.read_file(path)
            needs_update = False

            if hru_col not in gdf.columns:
                self.logger.info(f"Adding {hru_col} to {path.name}")
                if len(gdf) == 1:
                    gdf[hru_col] = 1
                elif main_id_col in gdf.columns:
                    gdf[hru_col] = gdf[main_id_col]
                else:
                    gdf[hru_col] = range(1, len(gdf) + 1)
                needs_update = True
            
            # Ensure it's not empty/NaN and matches lumped ID 1
            if len(gdf) == 1 and gdf[hru_col].iloc[0] != 1:
                self.logger.info(f"Forcing lumped {hru_col} to 1 in {path.name}")
                gdf[hru_col] = 1
                needs_update = True

            if gdf[hru_col].isnull().any():
                self.logger.info(f"Fixing NaNs in {hru_col} for {path.name}")
                if main_id_col in gdf.columns:
                    gdf[hru_col] = gdf[hru_col].fillna(gdf[main_id_col])
                gdf[hru_col] = gdf[hru_col].fillna(1)
                needs_update = True

            if needs_update:
                gdf.to_file(path)

        except Exception as e:
            self.logger.warning(f"Failed to ensure HRU ID ({hru_col}): {e}")

    def fix_missing_columns(self, shp_path: str, column_mapping: Dict[str, str]) -> None:
        """
        Ensure all columns required by meshflow exist in the shapefile.
        
        Args:
            shp_path: Path to the shapefile
            column_mapping: Mapping of standard names to file-specific names
        """
        if not shp_path:
            return

        path = Path(shp_path)
        if not path.exists():
            return

        try:
            gdf = gpd.read_file(path)
            needs_update = False

            # Required standard names we need to ensure exist in some form
            required_cols = {
                'river_slope': 0.001,
                'river_length': 1000.0,
                'river_class': 1,
            }

            for std_name, default_val in required_cols.items():
                target_col = column_mapping.get(std_name)
                if target_col and target_col not in gdf.columns:
                    self.logger.info(f"Adding missing column '{target_col}' to {path.name} with default {default_val}")
                    gdf[target_col] = default_val
                    needs_update = True

            if needs_update:
                gdf.to_file(path)

        except Exception as e:
            self.logger.warning(f"Failed to fix missing columns in {path.name}: {e}")

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
            # Read without specifying index to see all columns
            df = pd.read_csv(path)

            # If the first column is unnamed (common in exported CSVs), rename it to GRU_ID
            if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
                self.logger.info(f"Renaming first column '{df.columns[0]}' to 'GRU_ID'")
                df = df.rename(columns={df.columns[0]: 'GRU_ID'})

            # Ensure GRU_ID is integer
            if 'GRU_ID' in df.columns:
                df['GRU_ID'] = pd.to_numeric(df['GRU_ID'], errors='coerce').fillna(1).astype(int)
            else:
                # If still no GRU_ID, try to use the first column anyway if it's numeric-like
                self.logger.warning(f"No 'GRU_ID' column found in {path.name}. Using first column '{df.columns[0]}' as ID.")
                df = df.rename(columns={df.columns[0]: 'GRU_ID'})
                df['GRU_ID'] = pd.to_numeric(df['GRU_ID'], errors='coerce').fillna(1).astype(int)

            # For lumped cases (1 row), force ID to 1 to match shapefile
            if len(df) == 1:
                self.logger.info("Forcing single landcover row ID to 1")
                df['GRU_ID'] = 1

            # Remove unnamed columns (except the one we just renamed if it was the first)
            cols_to_drop = [col for col in df.columns if 'Unnamed' in col and col != 'GRU_ID']
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

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

            # IMPORTANT: meshflow's digit-only stripping for columns means we MUST 
            # remove any column that is not 'GRU_ID' or starts with 'frac_'
            # otherwise a column named 'count' becomes '', which causes ValueError in int()
            cols_to_keep = ['GRU_ID'] + [col for col in df.columns if col.startswith('frac_')]
            
            # Final check that we have columns to keep
            actual_cols_to_keep = [c for c in cols_to_keep if c in df.columns]
            df = df[actual_cols_to_keep]

            # Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates()
            if len(df) < initial_rows:
                self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

            temp_path = self.forcing_dir / f"temp_{path.name}"
            df.to_csv(temp_path, index=False)
            return str(temp_path)

        except Exception as e:
            self.logger.warning(f"Failed to sanitize landcover stats: {e}")
            return csv_path

    def convert_landcover_to_maf(self, csv_path: str) -> None:
        """
        Ensure landcover CSV is MAF (Mesh Attribute File) compliant.
        Crucial for MESH 1.5 to actually 'see' the GRU fractions.
        """
        if not csv_path or not Path(csv_path).exists():
            return

        try:
            df = pd.read_csv(csv_path)
            
            # MAF format expects columns like: GRU_ID, frac_1, frac_2, ...
            # where frac_* values sum to 1.0 for each row.
            
            modified = False
            
            # If we only have IGBP_x columns, convert to frac_x
            igbp_cols = [c for c in df.columns if c.startswith('IGBP_')]
            if igbp_cols:
                for col in igbp_cols:
                    class_num = col.split('_')[1]
                    new_col = f'frac_{class_num}'
                    if new_col not in df.columns:
                        total = df[igbp_cols].sum(axis=1).replace(0, 1)
                        df[new_col] = df[col] / total
                        modified = True
            
            # If we have no frac_ columns but some numeric columns, try to guess
            frac_cols = [c for c in df.columns if c.startswith('frac_')]
            if not frac_cols:
                # Fallback: if there's only one non-ID column, treat it as the GRU fraction
                other_cols = [c for c in df.columns if c != 'GRU_ID' and not c.startswith('Unnamed')]
                if len(other_cols) == 1:
                    self.logger.info(f"Guessing {other_cols[0]} is the primary GRU fraction")
                    df['frac_10'] = 1.0 # Default to grassland if unknown
                    modified = True
            
            if modified:
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Converted {csv_path} to MAF format")

        except Exception as e:
            self.logger.warning(f"Failed to convert to MAF: {e}")

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
