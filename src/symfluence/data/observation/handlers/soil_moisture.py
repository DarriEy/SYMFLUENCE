import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Optional
from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

@ObservationRegistry.register('SMAP')
class SMAPHandler(BaseObservationHandler):
    """
    Handles SMAP Soil Moisture data.
    """

    def acquire(self) -> Path:
        """Locate SMAP data."""
        config = self.config_dict
        data_access = str(config.get('DATA_ACCESS', 'local')).lower()
        smap_path = config.get('SMAP_PATH', 'default')
        if isinstance(smap_path, str) and smap_path.lower() == 'default':
            smap_dir = self.project_dir / "observations" / "soil_moisture" / "smap"
        else:
            smap_dir = Path(smap_path)
        if not smap_dir.exists():
            smap_dir.mkdir(parents=True, exist_ok=True)
        force_download = str(config.get('FORCE_DOWNLOAD', False)).lower() == 'true' if isinstance(config.get('FORCE_DOWNLOAD', False), str) else bool(config.get('FORCE_DOWNLOAD', False))
        use_opendap = bool(config.get('SMAP_USE_OPENDAP', False))
        if list(smap_dir.glob("*.nc")) and not force_download:
            return smap_dir
        if not use_opendap and not force_download:
            for pattern in ("*.h5", "*.hdf5"):
                if list(smap_dir.glob(pattern)):
                    return smap_dir
        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for SMAP soil moisture")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('SMAP', config, self.logger)
            return acquirer.download(smap_dir)
        return smap_dir

    def process(self, input_path: Path) -> Path:
        """Process SMAP NetCDF data."""
        self.logger.info(f"Processing SMAP Soil Moisture for domain: {self.domain_name}")
        
        nc_files = list(input_path.glob("*.nc"))
        if not nc_files:
            for pattern in ("*.h5", "*.hdf5"):
                nc_files.extend(input_path.glob(pattern))
        if not nc_files:
            self.logger.warning("No SMAP NetCDF files found")
            return input_path
            
        # Strategy: spatial average over bounding box if multiple pixels
        # For simplicity in this implementation, we take the mean of the first file
        results = []
        for f in nc_files:
            try:
                try:
                    ds = xr.open_dataset(f, engine='netcdf4')
                except Exception:
                    ds = xr.open_dataset(f, engine='h5netcdf')
            except Exception as exc:
                self.logger.warning(f"Skipping unreadable SMAP file {f.name}: {exc}")
                continue
            with ds:
                # SMAP variables often named 'soil_moisture', 'sm_surface', or 'sm_rootzone'
                var_names = [
                    v for v in ds.data_vars
                    if 'soil_moisture' in v.lower() or 'sm_surface' in v.lower() or 'sm_rootzone' in v.lower()
                ]
                if not var_names:
                    continue

                file_frames = []
                for var_name in var_names:
                    output_name = var_name
                    if 'sm_surface' in var_name.lower():
                        output_name = 'surface_sm'
                    elif 'rootzone' in var_name.lower():
                        output_name = 'rootzone_sm'

                    # Spatial average
                    mean_sm = ds[var_name].mean(dim=[d for d in ds[var_name].dims if d != 'time'])
                    df_ts = mean_sm.to_dataframe().reset_index()
                    df_ts = df_ts.rename(columns={var_name: output_name})
                    if 'time' in df_ts.columns:
                        df_ts = df_ts.set_index('time')[[output_name]]
                    file_frames.append(df_ts)

                if file_frames:
                    results.append(pd.concat(file_frames, axis=1))
        
        if not results:
            self.logger.warning("No SMAP data could be extracted")
            return input_path
            
        df = pd.concat(results).sort_index()
        if 'time' in df.columns:
            df = df.set_index('time')
        df = df.groupby(level=0).mean().sort_index()
        if self.start_date is not None and self.end_date is not None:
            df = df.loc[(df.index >= self.start_date) & (df.index <= self.end_date)]
        
        output_dir = self.project_dir / "observations" / "soil_moisture" / "smap" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_smap_processed.csv"
        df.to_csv(output_file)
        legacy_dir = self.project_dir / "observations" / "soil_moisture" / "preprocessed"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_file = legacy_dir / f"{self.domain_name}_smap_processed.csv"
        df.to_csv(legacy_file)
        
        self.logger.info(f"SMAP processing complete: {output_file}")
        return output_file


@ObservationRegistry.register('ISMN')
class ISMNHandler(BaseObservationHandler):
    """
    Handles ISMN soil moisture data.
    """

    def acquire(self) -> Path:
        """Locate or download ISMN data."""
        config = self.config_dict
        data_access = str(config.get('DATA_ACCESS', 'local')).lower()
        ismn_path = config.get('ISMN_PATH', 'default')
        if isinstance(ismn_path, str) and ismn_path.lower() == 'default':
            ismn_dir = self.project_dir / "observations" / "soil_moisture" / "ismn"
        else:
            ismn_dir = Path(ismn_path)
        ismn_dir.mkdir(parents=True, exist_ok=True)

        force_download = str(config.get('FORCE_DOWNLOAD', False)).lower() == 'true' if isinstance(config.get('FORCE_DOWNLOAD', False), str) else bool(config.get('FORCE_DOWNLOAD', False))
        if list(ismn_dir.glob("*.csv")) and not force_download:
            return ismn_dir

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for ISMN soil moisture")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('ISMN', config, self.logger)
            return acquirer.download(ismn_dir)
        return ismn_dir

    def process(self, input_path: Path) -> Path:
        """Process ISMN station data to a basin-average time series."""
        self.logger.info(f"Processing ISMN Soil Moisture for domain: {self.domain_name}")

        files = []
        for pattern in ("*.csv", "*.txt", "*.dat"):
            files.extend(input_path.glob(pattern))
        if not files:
            self.logger.warning("No ISMN files found")
            return input_path

        target_depth = self._get_target_depth()
        series_list = []
        for f in files:
            df = self._read_station_file(f)
            if df is None or df.empty:
                continue

            date_col = self._find_date_column(df.columns)
            if not date_col:
                continue

            sm_col = self._find_soil_moisture_column(df.columns)
            if not sm_col:
                continue

            df['DateTime'] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=['DateTime'])
            df = df.set_index('DateTime')

            depth_col = self._find_depth_column(df.columns)
            if depth_col:
                df['depth_m'] = pd.to_numeric(df[depth_col], errors='coerce')
                df['depth_m'] = df['depth_m'].where(df['depth_m'].notna(), pd.NA)
                df['depth_m'] = df['depth_m'].apply(self._normalize_depth)
                df = df.dropna(subset=['depth_m'])
                if not df.empty:
                    depth_values = df['depth_m'].unique()
                    closest_depth = min(depth_values, key=lambda x: abs(x - target_depth))
                    df = df[df['depth_m'] == closest_depth]

            series = pd.to_numeric(df[sm_col], errors='coerce').dropna()
            if series.empty:
                continue
            series_list.append(series)

        if not series_list:
            self.logger.warning("No ISMN soil moisture data could be extracted")
            return input_path

        combined = pd.concat(series_list, axis=1)
        combined = combined.groupby(level=0).mean()
        combined = combined.sort_index()

        if self.start_date is not None and self.end_date is not None:
            combined = combined.loc[(combined.index >= self.start_date) & (combined.index <= self.end_date)]

        aggregation = self.config_dict.get('ISMN_TEMPORAL_AGGREGATION', 'daily_mean')
        if aggregation == 'daily_mean':
            combined = combined.resample('D').mean().dropna()

        col_name = f"sm_{target_depth:.2f}"
        output_df = combined.to_frame(name=col_name)

        output_dir = self.project_dir / "observations" / "soil_moisture" / "ismn" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_ismn_processed.csv"
        output_df.to_csv(output_file)

        self.logger.info(f"ISMN processing complete: {output_file}")
        return output_file

    def _read_station_file(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(path)
        except Exception:
            try:
                return pd.read_csv(path, delim_whitespace=True)
            except Exception as exc:
                self.logger.warning(f"Skipping unreadable ISMN file {path.name}: {exc}")
                return None

    def _find_date_column(self, columns):
        candidates = [
            'timestamp', 'datetime', 'DateTime', 'date', 'Date', 'time', 'Time'
        ]
        for candidate in candidates:
            if candidate in columns:
                return candidate
        for col in columns:
            lower = col.lower()
            if any(term in lower for term in ['timestamp', 'datetime', 'date', 'time']):
                return col
        return None

    def _find_soil_moisture_column(self, columns):
        for col in columns:
            lower = col.lower()
            if any(term in lower for term in ['soil_moisture', 'soilmoisture', 'volumetric', 'vsm', 'theta']):
                return col
        for col in columns:
            lower = col.lower()
            if lower.startswith('sm') and 'flag' not in lower and 'qc' not in lower:
                return col
        return None

    def _find_depth_column(self, columns):
        for col in columns:
            if 'depth' in col.lower():
                return col
        return None

    def _normalize_depth(self, depth):
        try:
            depth_val = float(depth)
        except Exception:
            return pd.NA
        if depth_val > 10:
            return depth_val / 100.0
        return depth_val

    def _get_target_depth(self) -> float:
        target_depth = self.config_dict.get('ISMN_TARGET_DEPTH_M', self.config_dict.get('SM_TARGET_DEPTH', 0.05))
        try:
            return float(target_depth)
        except Exception:
            return 0.05

@ObservationRegistry.register('ESA_CCI_SM')
class ESACCISMHandler(BaseObservationHandler):
    """
    Handles ESA CCI Soil Moisture data.
    """

    def acquire(self) -> Path:
        """Locate ESA CCI SM data."""
        esa_dir = Path(self.config.get('ESA_CCI_SM_PATH', self.project_dir / "observations" / "soil_moisture" / "esa_cci"))
        if not esa_dir.exists():
            esa_dir.mkdir(parents=True, exist_ok=True)
        return esa_dir

    def process(self, input_path: Path) -> Path:
        """Process ESA CCI SM NetCDF data."""
        self.logger.info(f"Processing ESA CCI Soil Moisture for domain: {self.domain_name}")
        
        nc_files = list(input_path.glob("*.nc"))
        if not nc_files:
            self.logger.warning("No ESA CCI SM NetCDF files found")
            return input_path
            
        results = []
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                # ESA CCI SM variable is usually 'sm'
                if 'sm' not in ds.data_vars:
                    continue
                
                # Spatial average
                mean_sm = ds['sm'].mean(dim=[d for d in ds['sm'].dims if d != 'time'])
                df_ts = mean_sm.to_dataframe().reset_index()
                results.append(df_ts)
        
        if not results:
            self.logger.warning("No ESA CCI SM data could be extracted")
            return input_path
            
        df = pd.concat(results).sort_values('time').set_index('time')
        
        output_dir = self.project_dir / "observations" / "soil_moisture" / "preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_esa_cci_sm_processed.csv"
        df.to_csv(output_file)
        
        self.logger.info(f"ESA CCI SM processing complete: {output_file}")
        return output_file
