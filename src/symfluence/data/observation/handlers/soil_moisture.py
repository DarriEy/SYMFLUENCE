"""
SMAP soil moisture observation handler.

Provides acquisition and preprocessing of NASA SMAP satellite soil moisture
data for multivariate hydrological model calibration and validation.
"""

import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Any, Optional
from ..base import BaseObservationHandler
from ..registry import ObservationRegistry

@ObservationRegistry.register('smap')
class SMAPHandler(BaseObservationHandler):
    """
    Handles SMAP Soil Moisture data.
    """

    obs_type = "soil_moisture"
    source_name = "NASA_SMAP"

    def acquire(self) -> Path:
        """Locate SMAP data."""
        data_access = self._get_config_value(
            lambda: self.config.domain.data_access, default='local'
        ).lower()

        smap_path = self._get_config_value(
            lambda: self.config.evaluation.smap.path, default='default'
        )
        if isinstance(smap_path, str) and smap_path.lower() == 'default':
            smap_dir = self.project_dir / "observations" / "soil_moisture" / "smap"
        else:
            smap_dir = Path(smap_path)

        if not smap_dir.exists():
            smap_dir.mkdir(parents=True, exist_ok=True)

        force_download = self._get_config_value(
            lambda: self.config.data.force_download, default=False
        )
        use_opendap = self._get_config_value(
            lambda: self.config.evaluation.smap.use_opendap, default=False
        )

        if list(smap_dir.glob("*.nc")) and not force_download:
            return smap_dir
        if not use_opendap and not force_download:
            for pattern in ("*.h5", "*.hdf5"):
                if list(smap_dir.glob(pattern)):
                    return smap_dir

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for SMAP soil moisture")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('SMAP', self.config, self.logger)
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
                except (OSError, ValueError):
                    # netcdf4 engine failed, try h5netcdf fallback
                    ds = xr.open_dataset(f, engine='h5netcdf')
            except (OSError, ValueError) as exc:
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


@ObservationRegistry.register('ismn')
class ISMNHandler(BaseObservationHandler):
    """
    Handles ISMN soil moisture data.
    """

    obs_type = "soil_moisture"
    source_name = "ISMN"

    def acquire(self) -> Path:
        """Locate or download ISMN data."""
        data_access = self._get_config_value(
            lambda: self.config.domain.data_access, default='local'
        ).lower()

        ismn_path = self._get_config_value(
            lambda: self.config.evaluation.ismn.path, default='default'
        )
        if isinstance(ismn_path, str) and ismn_path.lower() == 'default':
            ismn_dir = self.project_dir / "observations" / "soil_moisture" / "ismn"
        else:
            ismn_dir = Path(ismn_path)
        ismn_dir.mkdir(parents=True, exist_ok=True)

        force_download = self._get_config_value(
            lambda: self.config.data.force_download, default=False
        )
        if list(ismn_dir.glob("*.csv")) and not force_download:
            return ismn_dir

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for ISMN soil moisture")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('ISMN', self.config, self.logger)
            return acquirer.download(ismn_dir)
        return ismn_dir

    def process(self, input_path: Path) -> Path:
        """Process ISMN station data to a basin-average time series."""
        self.logger.info(f"Processing ISMN Soil Moisture for domain: {self.domain_name}")

        files: list[Any] = []
        for pattern in ("*.csv", "*.txt", "*.dat"):
            files.extend(input_path.glob(pattern))
        if not files:
            self.logger.warning("No ISMN files found")
            return input_path

        target_depth = self._get_target_depth()
        series_list = []
        depth_series = []
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
            depth_value = None
            if depth_col:
                df['depth_m'] = pd.to_numeric(df[depth_col], errors='coerce')
                df['depth_m'] = df['depth_m'].where(df['depth_m'].notna(), pd.NA)  # type: ignore[call-overload]
                df['depth_m'] = df['depth_m'].apply(self._normalize_depth)
                df = df.dropna(subset=['depth_m'])
                if not df.empty:
                    depth_values = df['depth_m'].unique()
                    closest_depth = min(depth_values, key=lambda x: abs(x - target_depth))
                    df = df[df['depth_m'] == closest_depth]
                    depth_value = float(closest_depth)

            series = pd.to_numeric(df[sm_col], errors='coerce').dropna()  # type: ignore[call-overload, index]
            if series.empty:
                continue
            series_list.append(series)
            if depth_value is not None:
                depth_series.append((series, depth_value))

        if depth_series:
            min_diff = min(abs(d - target_depth) for _, d in depth_series)
            series_list = [s for s, d in depth_series if abs(d - target_depth) == min_diff]

        if not series_list:
            self.logger.warning("No ISMN soil moisture data could be extracted")
            return input_path

        combined = pd.concat(series_list, axis=1)
        combined = combined.groupby(level=0).mean().sort_index()
        if isinstance(combined, pd.DataFrame):
            combined = combined.mean(axis=1)

        if self.start_date is not None and self.end_date is not None:
            combined = combined.loc[(combined.index >= self.start_date) & (combined.index <= self.end_date)]

        aggregation = self._get_config_value(
            lambda: self.config.evaluation.ismn.temporal_aggregation, default='daily_mean'
        )
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
                return pd.read_csv(path, delim_whitespace=True)  # type: ignore[call-overload]
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
        target_depth = self._get_config_value(
            lambda: self.config.evaluation.ismn.target_depth_m, default=0.05
        )
        try:
            return float(target_depth)
        except Exception:
            return 0.05

@ObservationRegistry.register('esa_cci_sm')
class ESACCISMHandler(BaseObservationHandler):
    """
    Handles ESA CCI Soil Moisture data.
    """

    obs_type = "soil_moisture"
    source_name = "ESA_CCI"

    def acquire(self) -> Path:
        """Locate ESA CCI SM data."""
        # ESA CCI SM path - fallback to default location
        esa_path = self.config_dict.get('ESA_CCI_SM_PATH')
        if esa_path:
            esa_dir = Path(esa_path)
        else:
            esa_dir = self.project_dir / "observations" / "soil_moisture" / "esa_cci"
        esa_dir.mkdir(parents=True, exist_ok=True)

        data_access = self._get_config_value(
            lambda: self.config.domain.data_access, default='local'
        ).lower()
        force_download = self._get_config_value(
            lambda: self.config.data.force_download, default=False
        )

        if list(esa_dir.glob("*.nc")) and not force_download:
            return esa_dir

        if data_access == 'cloud':
            self.logger.info("Triggering cloud acquisition for ESA CCI soil moisture")
            from ...acquisition.registry import AcquisitionRegistry
            acquirer = AcquisitionRegistry.get_handler('ESA_CCI_SM', self.config, self.logger)
            return acquirer.download(esa_dir)

        return esa_dir

    def process(self, input_path: Path) -> Path:
        """Process ESA CCI SM NetCDF data."""
        self.logger.info(f"Processing ESA CCI Soil Moisture for domain: {self.domain_name}")

        nc_files = list(input_path.glob("*.nc"))
        if not nc_files:
            self.logger.warning("No ESA CCI SM NetCDF files found")
            return input_path

        # Use bbox from base class (already parsed in __init__)
        north = west = south = east = None
        target_lat = target_lon = None
        if self.bbox:
            try:
                north = self.bbox.get('lat_max')
                south = self.bbox.get('lat_min')
                west = self.bbox.get('lon_min')
                east = self.bbox.get('lon_max')
                if all(v is not None for v in [north, south, west, east]):
                    target_lat = (north + south) / 2.0
                    if west <= east:
                        target_lon = (west + east) / 2.0
                    else:
                        # Dateline-crossing bbox; pick midpoint on wrapped domain
                        target_lon = ((west + 360.0 + east) / 2.0) % 360.0
                        if target_lon > 180.0:
                            target_lon -= 360.0
            except Exception:
                self.logger.warning("Failed to parse bbox for ESA CCI SM subsetting")

        results = []
        for f in nc_files:
            with xr.open_dataset(f) as ds:
                # ESA CCI SM variable is usually 'sm'
                if 'sm' not in ds.data_vars:
                    continue

                sm = ds['sm']
                if target_lat is not None and target_lon is not None and {'lat', 'lon'}.issubset(sm.dims):
                    # Use nearest pixel to reduce spatial smoothing
                    sm = sm.sel(lat=target_lat, lon=target_lon, method='nearest')
                elif None not in (north, west, south, east) and {'lat', 'lon'}.issubset(sm.dims):
                    lat_desc = sm['lat'][0] > sm['lat'][-1]
                    lat_slice = slice(north, south) if lat_desc else slice(south, north)
                    sm = sm.sel(lat=lat_slice)
                    if west <= east:
                        sm = sm.sel(lon=slice(west, east))
                    else:
                        sm = sm.where((sm['lon'] >= west) | (sm['lon'] <= east), drop=True)

                # Filter invalid values
                sm = sm.where((sm >= 0) & (sm <= 1))

                # Spatial average if still gridded, else keep point
                mean_sm = sm.mean(dim=[d for d in sm.dims if d != 'time'])
                df_ts = mean_sm.to_dataframe().reset_index()
                results.append(df_ts)

        if not results:
            self.logger.warning("No ESA CCI SM data could be extracted")
            return input_path

        df = pd.concat(results).sort_values('time')
        df = df.rename(columns={'sm': 'soil_moisture'})

        output_dir = self.project_dir / "observations" / "soil_moisture" / "esa_sm" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{self.domain_name}_esa_processed.csv"
        df.to_csv(output_file, index=False)

        self.logger.info(f"ESA CCI SM processing complete: {output_file}")
        return output_file
