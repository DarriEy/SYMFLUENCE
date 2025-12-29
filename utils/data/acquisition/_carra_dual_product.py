"""
New CARRA dual-product download implementation.
This will replace the existing _download_carra method in cloud_downloader.py
"""

# This is the new method body to replace lines 2312-2593 in cloud_downloader.py
NEW_CARRA_IMPLEMENTATION = '''
        if not HAS_CDSAPI:
            raise ImportError(
                "cdsapi package is required for CARRA downloads. "
                "Install with: pip install cdsapi"
            )

        self.logger.info("Downloading CARRA data via CDS API (dual-product strategy)")
        self.logger.info(f"  Bounding box: {self.bbox}")
        self.logger.info(f"  Time period: {self.start_date} to {self.end_date}")
        self.logger.info("  Strategy: Analysis (meteorology) + Forecast (fluxes) + Calculated (longwave)")

        # Initialize CDS client
        try:
            c = cdsapi.Client()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize CDS API client: {e}\\n"
                "Please ensure ~/.cdsapirc is configured with your CDS API key.\\n"
                "See: https://cds.climate.copernicus.eu/how-to-api"
            )

        # CARRA domain selection
        domain = self.config.get("CARRA_DOMAIN", "west_domain")
        if domain not in ["west_domain", "east_domain"]:
            self.logger.warning(f"Invalid CARRA_DOMAIN '{domain}', using 'west_domain'")
            domain = "west_domain"

        # Build time range for CDS API
        years = list(range(self.start_date.year, self.end_date.year + 1))
        months = [f"{m:02d}" for m in range(1, 13)]
        days = [f"{d:02d}" for d in range(1, 32)]
        hours = [f"{h:02d}:00" for h in range(0, 24)]

        # For precise time range, filter months/days
        if len(years) == 1:
            months = [f"{m:02d}" for m in range(self.start_date.month, self.end_date.month + 1)]
            if self.start_date.month == self.end_date.month:
                days = [f"{d:02d}" for d in range(self.start_date.day, self.end_date.day + 1)]

        # Spatial extent for local subsetting (CDS API area parameter doesn't work with CARRA)
        area_bbox = {
            "north": self.bbox["lat_max"],
            "west": self.bbox["lon_min"],
            "south": self.bbox["lat_min"],
            "east": self.bbox["lon_max"],
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        domain_name = self.config.get("DOMAIN_NAME", "domain")

        # ============================================================================
        # STEP 1: Download ANALYSIS product (meteorological variables)
        # ============================================================================
        self.logger.info("Step 1/3: Downloading CARRA ANALYSIS product (meteorology)")

        analysis_vars = [
            "2m_temperature",
            "2m_relative_humidity",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
        ]

        analysis_request = {
            "domain": domain,
            "level_type": "surface_or_atmosphere",
            "product_type": "analysis",
            "variable": analysis_vars,
            "year": [str(y) for y in years],
            "month": months,
            "day": days,
            "time": hours,
            "data_format": "netcdf",
        }

        analysis_file = output_dir / f"{domain_name}_CARRA_analysis_temp.nc"

        self.logger.info(f"  Requesting: {analysis_vars}")
        try:
            c.retrieve("reanalysis-carra-single-levels", analysis_request, str(analysis_file))
            self.logger.info(f"  ✓ Analysis product downloaded: {analysis_file.stat().st_size / 1024 / 1024:.1f} MB")
        except Exception as e:
            raise RuntimeError(f"Failed to download CARRA analysis product: {e}")

        # ============================================================================
        # STEP 2: Download FORECAST product (precipitation and radiation)
        # ============================================================================
        self.logger.info("Step 2/3: Downloading CARRA FORECAST product (fluxes)")

        forecast_vars = [
            "total_precipitation",
            "surface_solar_radiation_downwards",
        ]

        forecast_request = {
            "domain": domain,
            "level_type": "surface_or_atmosphere",
            "product_type": "forecast",
            "leadtime_hour": ["1"],  # 1-hour forecast
            "variable": forecast_vars,
            "year": [str(y) for y in years],
            "month": months,
            "day": days,
            "time": hours,
            "data_format": "netcdf",
        }

        forecast_file = output_dir / f"{domain_name}_CARRA_forecast_temp.nc"

        self.logger.info(f"  Requesting: {forecast_vars} (1-hour leadtime)")
        try:
            c.retrieve("reanalysis-carra-single-levels", forecast_request, str(forecast_file))
            self.logger.info(f"  ✓ Forecast product downloaded: {forecast_file.stat().st_size / 1024 / 1024:.1f} MB")
        except Exception as e:
            raise RuntimeError(f"Failed to download CARRA forecast product: {e}")

        # ============================================================================
        # STEP 3: Merge, process, and calculate derived variables
        # ============================================================================
        self.logger.info("Step 3/3: Merging products and processing variables")

        # Load both datasets
        with xr.open_dataset(analysis_file) as ds_analysis, xr.open_dataset(forecast_file) as ds_forecast:

            # Standardize time dimension name
            for ds, name in [(ds_analysis, 'analysis'), (ds_forecast, 'forecast')]:
                time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
                if time_dim != 'time':
                    self.logger.info(f"  Renaming '{time_dim}' -> 'time' in {name} product")
                    ds = ds.rename({time_dim: 'time'})
                    if name == 'analysis':
                        ds_analysis = ds
                    else:
                        ds_forecast = ds

            # Temporal subsetting
            ds_analysis = ds_analysis.sel(time=slice(self.start_date, self.end_date))
            ds_forecast = ds_forecast.sel(time=slice(self.start_date, self.end_date))

            self.logger.info(f"  Time range: {len(ds_analysis.time)} timesteps")

            # Spatial subsetting (both products have same grid)
            if 'latitude' in ds_analysis.coords and 'longitude' in ds_analysis.coords:
                lat = ds_analysis['latitude'].values
                lon = ds_analysis['longitude'].values

                mask = (
                    (lat >= area_bbox["south"]) & (lat <= area_bbox["north"]) &
                    (lon >= area_bbox["west"]) & (lon <= area_bbox["east"])
                )

                y_idx, x_idx = np.where(mask)
                if len(y_idx) > 0:
                    y_min, y_max = y_idx.min(), y_idx.max()
                    x_min, x_max = x_idx.min(), x_idx.max()

                    if 'y' in ds_analysis.dims and 'x' in ds_analysis.dims:
                        ds_analysis = ds_analysis.isel(y=slice(y_min, y_max+1), x=slice(x_min, x_max+1))
                        ds_forecast = ds_forecast.isel(y=slice(y_min, y_max+1), x=slice(x_min, x_max+1))
                        self.logger.info(f"  Spatially subset to {(y_max-y_min+1) * (x_max-x_min+1)} grid cells")
                else:
                    self.logger.warning("  No grid points in bounding box, keeping full domain")

            # ========================================================================
            # Variable processing and merging
            # ========================================================================
            self.logger.info("  Processing and renaming variables...")

            # Start with analysis variables
            ds_merged = ds_analysis.copy()

            # Add forecast variables to merged dataset
            for var in ['tp', 'ssrd']:
                if var in ds_forecast.variables:
                    ds_merged[var] = ds_forecast[var]

            # Rename to SUMMA standard names
            rename_map = {
                't2m': 'airtemp',
                'sp': 'airpres',
                'u10': 'windspd_u',
                'v10': 'windspd_v',
                'tp': 'pptrate',
                'ssrd': 'SWRadAtm',
            }

            existing_renames = {old: new for old, new in rename_map.items() if old in ds_merged.variables}
            ds_merged = ds_merged.rename(existing_renames)
            self.logger.info(f"    Renamed: {list(existing_renames.keys())}")

            # ========================================================================
            # Calculate derived variables
            # ========================================================================

            # 1. Wind speed from u/v components
            if 'windspd_u' in ds_merged and 'windspd_v' in ds_merged:
                self.logger.info("  Calculating wind speed from u/v components")
                ds_merged['windspd'] = np.sqrt(ds_merged['windspd_u']**2 + ds_merged['windspd_v']**2)
                ds_merged['windspd'].attrs = {
                    'units': 'm s-1',
                    'long_name': 'wind speed',
                    'standard_name': 'wind_speed',
                }

            # 2. Specific humidity from relative humidity + temperature + pressure
            if 'r2' in ds_merged and 'airtemp' in ds_merged and 'airpres' in ds_merged:
                self.logger.info("  Converting relative humidity to specific humidity")

                T = ds_merged['airtemp']  # K
                RH = ds_merged['r2']      # %
                P = ds_merged['airpres']  # Pa

                # Saturation vapor pressure (Magnus formula)
                es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
                e = (RH / 100.0) * es
                q = (0.622 * e) / (P - 0.378 * e)

                ds_merged['spechum'] = q
                ds_merged['spechum'].attrs = {
                    'units': 'kg kg-1',
                    'long_name': 'specific humidity',
                    'standard_name': 'specific_humidity',
                }

                # Keep vapor pressure for longwave calculation
                ds_merged['vapor_pressure_hPa'] = e / 100.0  # Convert Pa to hPa

                ds_merged = ds_merged.drop_vars('r2')

            # 3. Longwave radiation using Brutsaert (1975) formula
            if 'airtemp' in ds_merged and 'vapor_pressure_hPa' in ds_merged:
                self.logger.info("  Calculating longwave radiation (Brutsaert 1975 formula)")

                T = ds_merged['airtemp']  # K
                e_a = ds_merged['vapor_pressure_hPa']  # hPa

                # Brutsaert (1975) clear-sky emissivity
                epsilon_a = 1.24 * (e_a / T) ** (1.0/7.0)

                # Stefan-Boltzmann constant
                sigma = 5.67e-8  # W m^-2 K^-4

                # Longwave radiation (clear-sky)
                LW = epsilon_a * sigma * T**4

                ds_merged['LWRadAtm'] = LW
                ds_merged['LWRadAtm'].attrs = {
                    'units': 'W m-2',
                    'long_name': 'downward longwave radiation at surface (calculated)',
                    'standard_name': 'surface_downwelling_longwave_flux_in_air',
                    'method': 'Brutsaert (1975) clear-sky formula',
                    'note': 'Calculated from air temperature and vapor pressure',
                }

                # Clean up temporary variable
                ds_merged = ds_merged.drop_vars('vapor_pressure_hPa')

            # 4. Convert precipitation from kg/m² to m/s
            if 'pptrate' in ds_merged:
                self.logger.info("  Converting precipitation units")
                # CARRA tp is accumulated over 1-hour forecast, in kg/m²
                # 1 kg/m² = 1 mm = 0.001 m
                # For hourly data: m/hour to m/s
                ds_merged['pptrate'] = (ds_merged['pptrate'] * 0.001) / 3600.0
                ds_merged['pptrate'].attrs = {
                    'units': 'm s-1',
                    'long_name': 'precipitation rate',
                    'standard_name': 'precipitation_flux',
                }

            # 5. Convert shortwave radiation from J/m² to W/m²
            if 'SWRadAtm' in ds_merged:
                self.logger.info("  Converting shortwave radiation units")
                # CARRA ssrd is accumulated over 1-hour in J/m²
                # Convert to W/m²: J/m² / 3600 s = W/m²
                ds_merged['SWRadAtm'] = ds_merged['SWRadAtm'] / 3600.0
                ds_merged['SWRadAtm'].attrs = {
                    'units': 'W m-2',
                    'long_name': 'downward shortwave radiation at surface',
                    'standard_name': 'surface_downwelling_shortwave_flux_in_air',
                }

            # Update air temperature and pressure attributes
            if 'airtemp' in ds_merged:
                ds_merged['airtemp'].attrs.update({
                    'units': 'K',
                    'long_name': 'air temperature',
                    'standard_name': 'air_temperature',
                })

            if 'airpres' in ds_merged:
                ds_merged['airpres'].attrs.update({
                    'units': 'Pa',
                    'long_name': 'surface air pressure',
                    'standard_name': 'surface_air_pressure',
                })

            # ========================================================================
            # Final metadata and save
            # ========================================================================
            ds_merged.attrs["source"] = "CARRA (Copernicus Arctic Regional Reanalysis)"
            ds_merged.attrs["source_url"] = "https://cds.climate.copernicus.eu/datasets/reanalysis-carra-single-levels"
            ds_merged.attrs["downloaded_by"] = "SYMFLUENCE cloud_downloader (CDS API)"
            ds_merged.attrs["download_date"] = pd.Timestamp.now().isoformat()
            ds_merged.attrs["bbox"] = str(self.bbox)
            ds_merged.attrs["carra_domain"] = domain
            ds_merged.attrs["processing"] = "Dual-product merge (analysis+forecast), calculated longwave, derived variables"
            ds_merged.attrs["longwave_method"] = "Brutsaert (1975) clear-sky emissivity"

            # Save final merged file
            final_file = output_dir / f"{domain_name}_CARRA_{self.start_date.year}-{self.end_date.year}.nc"
            ds_merged.to_netcdf(final_file)

            # Log final variable list
            summa_vars = [v for v in ds_merged.data_vars if v in ['airpres', 'LWRadAtm', 'SWRadAtm', 'pptrate', 'airtemp', 'spechum', 'windspd']]
            self.logger.info(f"  ✓ SUMMA variables in output: {summa_vars}")
            self.logger.info(f"✓ CARRA data saved to: {final_file}")
            self.logger.info(f"  File size: {final_file.stat().st_size / 1024 / 1024:.1f} MB")

        # Clean up temporary files
        if analysis_file.exists():
            analysis_file.unlink()
        if forecast_file.exists():
            forecast_file.unlink()

        return final_file
'''
