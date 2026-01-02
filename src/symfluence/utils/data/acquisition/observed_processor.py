import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import pandas as pd # type: ignore

from symfluence.utils.common.constants import UnitConversion
from symfluence.utils.exceptions import DataAcquisitionError

class ObservedDataProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.data_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.forcing_time_step_size = int(self.config.get('FORCING_TIME_STEP_SIZE'))
        self.data_provider = self.config.get('STREAMFLOW_DATA_PROVIDER', 'USGS').upper()

        self.streamflow_raw_path = self._get_file_path('STREAMFLOW_RAW_PATH', 'observations/streamflow/raw_data', '')
        self.streamflow_processed_path = self._get_file_path('STREAMFLOW_PROCESSED_PATH', 'observations/streamflow/preprocessed', '')
        self.streamflow_raw_name = self.config.get('STREAMFLOW_RAW_NAME')

    def _get_file_path(self, file_type, file_def_path, file_name):
        if self.config.get(f'{file_type}') == 'default':
            return self.project_dir / file_def_path / file_name
        else:
            return Path(self.config.get(f'{file_type}'))

    def get_resample_freq(self):
        if self.forcing_time_step_size == UnitConversion.SECONDS_PER_HOUR:
            return 'h'
        if self.forcing_time_step_size == 10800: # 3 hours in seconds
            return 'h'
        elif self.forcing_time_step_size == UnitConversion.SECONDS_PER_DAY:
            return 'D'
        else:
            return f'{self.forcing_time_step_size}s'

    def process_streamflow_data(self):
        try:
            if self.config.get('PROCESS_CARAVANS', False):
                self._process_caravans_data()
            elif self.data_provider == 'USGS':
                if self.config.get('DOWNLOAD_USGS_DATA') == True:
                    self._download_and_process_usgs_data()
                else:
                    self._process_usgs_data()
            elif self.data_provider == 'WSC':
                if self.config.get('DOWNLOAD_WSC_DATA') == True:
                    self._extract_and_process_hydat_data()
                else:
                    self._process_wsc_data()
            elif self.data_provider == 'VI':
                self._process_vi_data()
            else:
                self.logger.error(f"Unsupported streamflow data provider: {self.data_provider}")
                raise DataAcquisitionError(f"Unsupported streamflow data provider: {self.data_provider}")
        except Exception as e:
            self.logger.error(f'Issue in streamflow data preprocessing: {e}')

    def _extract_and_process_hydat_data(self):
        """
        Process Water Survey of Canada (WSC) streamflow data by fetching it directly from the HYDAT SQLite database.
        
        This function fetches discharge data for the specified WSC station,
        processes it, and resamples it to the configured time step.
        """
        import sqlite3
        import pandas as pd
        from datetime import datetime, timedelta
        from pathlib import Path
        
        self.logger.info("Processing WSC streamflow data from HYDAT database")
        
        # Get configuration parameters
        station_id = self.config.get('STATION_ID')
        hydat_path = self.config.get('HYDAT_PATH')
        if hydat_path == 'default':
            # Default path assumes geospatial-data is one level up from the domain's data directory
            hydat_path = str(self.project_dir.parent.parent / 'geospatial-data' / 'hydat' / 'Hydat.sqlite3')
        
        # Check if HYDAT_PATH exists
        if not hydat_path or not Path(hydat_path).exists():
            self.logger.error(f"HYDAT database not found at: {hydat_path}")
            raise FileNotFoundError(f"HYDAT database not found at: {hydat_path}")
        
        # Parse and format the start date properly
        start_date_raw = self.config.get('EXPERIMENT_TIME_START')
        try:
            # Try to parse the date string with various formats
            for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    parsed_date = datetime.strptime(start_date_raw, fmt)
                    start_date = parsed_date.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                # If none of the formats match, use a default format
                self.logger.warning(f"Could not parse start date: {start_date_raw}. Using first 10 characters as YYYY-MM-DD.")
                start_date = start_date_raw[:10]
        except Exception as e:
            self.logger.warning(f"Error parsing start date: {e}. Using default date format.")
            start_date = start_date_raw[:10]
        
        # Parse the date components for SQL queries
        try:
            start_year = int(start_date.split('-')[0])
            end_year = datetime.now().year
            self.logger.info(f"Querying data from year {start_year} to {end_year}")
        except Exception as e:
            self.logger.warning(f"Error parsing date components: {e}. Using default range.")
            start_year = 1900
            end_year = datetime.now().year
        
        # Log the station and date range
        self.logger.info(f"Retrieving discharge data for WSC station {station_id} from HYDAT database")
        self.logger.info(f"Database path: {hydat_path}")
        self.logger.info(f"Time period: {start_year} to {end_year}")
        
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(hydat_path)
            
            # First, check if the station exists in the database
            station_query = "SELECT * FROM STATIONS WHERE STATION_NUMBER = ?"
            station_df = pd.read_sql_query(station_query, conn, params=(station_id,))
            
            if station_df.empty:
                self.logger.error(f"Station {station_id} not found in HYDAT database")
                raise DataAcquisitionError(f"Station {station_id} not found in HYDAT database")
            
            self.logger.info(f"Found station {station_id} in HYDAT database")
            if 'STATION_NAME' in station_df.columns:
                self.logger.info(f"Station name: {station_df['STATION_NAME'].iloc[0]}")
            
            # Query for daily discharge data
            # HYDAT stores discharge data in DLY_FLOWS table
            # The column names are like FLOW1, FLOW2, ... FLOW31 for each day of the month
            query = """
            SELECT * FROM DLY_FLOWS 
            WHERE STATION_NUMBER = ? 
            AND YEAR >= ? AND YEAR <= ?
            ORDER BY YEAR, MONTH
            """
            
            self.logger.info(f"Executing SQL query for daily flows...")
            dly_flow_df = pd.read_sql_query(query, conn, params=(station_id, start_year, end_year))
            
            if dly_flow_df.empty:
                self.logger.error(f"No flow data found for station {station_id} in the specified date range")
                raise DataAcquisitionError(f"No flow data found for station {station_id} in the specified date range")
            
            self.logger.info(f"Retrieved {len(dly_flow_df)} monthly records from HYDAT")
            
            # Now we need to reshape the data from the HYDAT format to a time series
            # HYDAT stores each month as a row, with columns FLOW1, FLOW2, ... FLOW31
            
            # Create an empty list to store the time series data
            time_series_data = []
            
            # Process each row (each row is a month of data)
            for _, row in dly_flow_df.iterrows():
                year = row['YEAR']
                month = row['MONTH']
                
                # Days in the month (accounting for leap years)
                days_in_month = 31  # Default max
                if month == 2:  # February
                    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):  # Leap year
                        days_in_month = 29
                    else:
                        days_in_month = 28
                elif month in [4, 6, 9, 11]:  # April, June, September, November
                    days_in_month = 30
                
                # Extract flow values for each day and create a date
                for day in range(1, days_in_month + 1):
                    flow_col = f'FLOW{day}'
                    if flow_col in row and not pd.isna(row[flow_col]):
                        date = f"{year}-{month:02d}-{day:02d}"
                        flow = row[flow_col]
                        
                        # Check for data flags - HYDAT has flags for data quality
                        symbol_col = f'SYMBOL{day}'
                        symbol = row.get(symbol_col, '')
                        
                        # Skip values with certain flags if needed
                        # E.g., 'E' for Estimate, 'A' for Partial Day, etc.
                        # Uncomment if you want to filter based on symbols
                        # if symbol in ['B', 'D', 'E']:
                        #     continue
                        
                        time_series_data.append({'date': date, 'flow': flow, 'symbol': symbol})
            
            # Convert to DataFrame
            df = pd.DataFrame(time_series_data)
            
            if df.empty:
                self.logger.error("No valid flow data found after processing")
                raise DataAcquisitionError("No valid flow data found after processing")
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            # Sort index to ensure chronological order
            df.sort_index(inplace=True)
            
            # Filter to the exact date range we want
            start_datetime = pd.to_datetime(start_date)
            end_datetime = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
            df = df[(df.index >= start_datetime) & (df.index <= end_datetime)]
            
            # Check if we have data after filtering
            if df.empty:
                self.logger.error("No data available after filtering to the specified date range")
                raise DataAcquisitionError("No data available for the specified date range")
            
            self.logger.info(f"Processed {len(df)} daily flow records")
            
            # Create the discharge_cms column (HYDAT data is in m³/s)
            df['discharge_cms'] = df['flow']
            
            # Basic statistics for logging
            self.logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            self.logger.info(f"Min flow: {df['discharge_cms'].min()} m³/s")
            self.logger.info(f"Max flow: {df['discharge_cms'].max()} m³/s")
            self.logger.info(f"Mean flow: {df['discharge_cms'].mean()} m³/s")
            
            # Call the resampling and saving function
            self._resample_and_save(df['discharge_cms'])
            
            self.logger.info(f"Successfully processed WSC data for station {station_id}")
            
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error processing WSC data from HYDAT: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        
    def _process_vi_data(self):
        self.logger.info("Processing VI (Iceland) streamflow data")

        vi_files = list(self.streamflow_raw_path.glob('*.csv'))
        if not vi_files:
            self.logger.error(f"No CSV files found in {self.streamflow_raw_path} for VI data.")
            return
        vi_file = vi_files[0] # Assuming the first CSV is the one we need
        
        try:
            vi_data = pd.read_csv(vi_file, 
                                  sep=';', 
                                  header=None, 
                                  names=['YYYY', 'MM', 'DD', 'qobs', 'qc_flag'],
                                  parse_dates={'datetime': ['YYYY', 'MM', 'DD']},
                                  na_values=['', 'NA', 'NaN'], # Explicitly list common NA values
                                  skiprows = 1)

            vi_data['discharge_cms'] = pd.to_numeric(vi_data['qobs'], errors='coerce')
            vi_data.set_index('datetime', inplace=True)

            # Filter out data with qc_flag values indicating unreliable measurements
            # The exact meaning of qc_flag values can vary, so this might need adjustment
            # For now, let's assume lower values are more reliable. This is a placeholder.
            # Example: Keep data where qc_flag is None or <= 100
            # reliable_data = vi_data[vi_data['qc_flag'].isna() | (vi_data['qc_flag'] <= 100)]
            # For now, we'll just use all data after conversion
            
            # Filter out rows where discharge_cms could not be converted
            vi_data = vi_data.dropna(subset=['discharge_cms'])

            self._resample_and_save(vi_data['discharge_cms'])
            self.logger.info(f"Successfully processed VI data from {vi_file}")

        except FileNotFoundError:
            self.logger.error(f"VI data file not found at {vi_file}")
        except Exception as e:
            self.logger.error(f"Error processing VI data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def process_usgs_groundwater_data(self):
        """
        Process USGS groundwater level data by fetching it directly from USGS API.
        
        This method:
        1. Checks if USGS groundwater data acquisition is enabled in configuration
        2. Downloads groundwater level data for the specified USGS station from the API
        3. Processes the JSON response to extract relevant data
        4. Saves processed data to the project directory's observations/groundwater folder
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Processing USGS groundwater level data")
        
        # Check if USGS groundwater processing is enabled
        if self.config.get('DOWNLOAD_USGS_GW') != 'true':
            self.logger.info("USGS groundwater data processing is disabled in configuration")
            return False
        
        try:
            # Get configuration parameters
            station_id = self.config.get('USGS_STATION')
            
            if not station_id:
                self.logger.error("Missing USGS_STATION in configuration")
                return False
            
            # If station ID includes a prefix, extract just the numeric part for the API
            if '-' in str(station_id):
                station_numeric = str(station_id).split('-')[-1]
            else:
                station_numeric = str(station_id)
            
            # Create directory for processed data if it doesn't exist
            output_dir = self.project_dir / 'observations' / 'groundwater'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output file path
            output_file = output_dir / f"{self.domain_name}_groundwater_processed.csv"
            
            # Construct the URL for JSON formatted groundwater level data
            # Using NWIS IV (Instantaneous Values) endpoint for potentially more recent data
            # Or use NWIS GWLevels endpoint if available and preferred
            # For groundwater levels, 'gwlevels' is more appropriate.
            url = f"https://waterservices.usgs.gov/nwis/gwlevels/?format=json&sites={station_numeric}&siteStatus=all&parameterCd=72019" # 72019 is for groundwater level
            
            self.logger.info(f"Retrieving groundwater level data for USGS station {station_id}")
            self.logger.info(f"API URL: {url}")
            
            # Fetch data from USGS API
            import requests
            response = requests.get(url, timeout=60) # Increased timeout
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # Parse JSON response
            data = response.json()
            
            # Extract the relevant data from the response
            if 'value' not in data or 'timeSeries' not in data['value']:
                self.logger.error("No groundwater data found in the API response (missing 'value' or 'timeSeries')")
                return False
            
            # Check if we have any time series data
            time_series = data['value']['timeSeries']
            if not time_series:
                self.logger.error(f"No groundwater level data found for station {station_id} in the API response")
                return False
            
            self.logger.info(f"Found {len(time_series)} time series in response")
            
            # Create lists to store data
            dates = []
            values = []
            units = []
            qualifiers = []
            parameter_names = []
            
            # Process each time series (there might be multiple parameter codes)
            for ts in time_series:
                # Extract parameter information
                try:
                    parameter_name = ts['variable']['variableName']
                    unit_code = ts['variable']['unit']['unitCode']
                    self.logger.info(f"Processing time series: {parameter_name}, unit: {unit_code}")
                    
                    # We are specifically looking for groundwater level
                    # Parameter code 72019 is for 'Ground water level' (use this for filtering)
                    # VariableName might contain 'level', 'depth', etc.
                    if '72019' not in ts.get('variable', {}).get('parameterCode', ''):
                        self.logger.info(f"Skipping time series not related to groundwater level (parameter code mismatch): {parameter_name}")
                        continue
                    
                    # Extract the values
                    if 'values' in ts and ts['values']:
                        value_data = ts['values'][0]
                        if 'value' in value_data:
                            values_list = value_data['value']
                            
                            for value_obj in values_list:
                                dates.append(value_obj['dateTime'])
                                values.append(float(value_obj['value']))
                                units.append(unit_code)
                                
                                # Extract qualifiers (e.g., A for approved, P for provisional)
                                qualifier_list = value_obj.get('qualifiers', [])
                                qualifiers.append(','.join(qualifier_list))
                                parameter_names.append(parameter_name)
                        else:
                            self.logger.warning(f"No 'value' field found in time series data for {parameter_name}")
                    else:
                        self.logger.warning(f"No 'values' field found for time series: {parameter_name}")

                except (KeyError, IndexError, ValueError, TypeError) as e:
                    self.logger.warning(f"Error processing time series details: {e}")
                    continue
            
            if not dates:
                self.logger.error("No valid groundwater level data found after processing time series.")
                return False
            
            # Create a DataFrame
            import pandas as pd
            df = pd.DataFrame({
                'datetime': pd.to_datetime(dates),
                'groundwater_level': values,
                'unit': units,
                'qualifier': qualifiers,
                'parameter': parameter_names
            })
            
            # Sort by date
            df.sort_values('datetime', inplace=True)
            
            # Handle units - convert to consistent units if needed
            # Most common units: ft (feet) or m (meters) below land surface
            if df['unit'].nunique() > 1:
                self.logger.warning(f"Multiple units found in groundwater data: {df['unit'].unique()}")
                # TODO: Implement unit conversion if necessary. For now, assume consistent units or log warning.
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Select only the groundwater level column for resampling
            # If multiple parameters were found, we might need to decide which one to keep or average
            # For now, assume the first one found is sufficient or that they are consistent
            df_processed = df[['groundwater_level', 'unit', 'qualifier', 'parameter']]

            # Basic statistics for logging
            self.logger.info(f"Date range: {df_processed.index.min()} to {df_processed.index.max()}")
            self.logger.info(f"Number of records: {len(df_processed)}")
            self.logger.info(f"Min level: {df_processed['groundwater_level'].min()} {df_processed['unit'].iloc[0]}")
            self.logger.info(f"Max level: {df_processed['groundwater_level'].max()} {df_processed['unit'].iloc[0]}")
            self.logger.info(f"Mean level: {df_processed['groundwater_level'].mean()} {df_processed['unit'].iloc[0]}")
            
            # Resample to regular intervals if needed
            resample_freq = self.get_resample_freq()
            # Groundwater levels are typically daily or sub-daily. Resampling to hourly or daily is common.
            # If resample_freq is 'D' or 'h', proceed.
            if resample_freq in ['D', 'h']:
                self.logger.info(f"Resampling groundwater level data to {resample_freq} frequency")
                # Use mean for resampling, interpolate afterwards
                resampled_df = df_processed.resample(resample_freq).mean()
                # Fill missing values with linear interpolation, limit to avoid excessive extrapolation
                resampled_df = resampled_df.interpolate(method='linear', limit_direction='both', limit=30) # Limit interpolation to 30 periods
                df_processed = resampled_df
            else:
                self.logger.warning(f"Resampling frequency '{resample_freq}' might be too coarse for groundwater data. Skipping resampling.")

            # Save to CSV
            df_processed.to_csv(output_file)
            self.logger.info(f"Processed groundwater level data saved to {output_file}")
            
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data from USGS API: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"Error processing USGS groundwater data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def process_fluxnet_data(self):
        """
        Process FLUXNET data by copying relevant station files to the project directory.
        
        This method:
        1. Checks if FLUXNET data acquisition is enabled in configuration
        2. Locates files containing the specified station ID
        3. Copies them to the project directory's observations/fluxnet folder
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Processing FLUXNET data")
        
        # Check if FLUXNET processing is enabled
        if self.config.get('DOWNLOAD_FLUXNET') != 'true':
            self.logger.info("FLUXNET data processing is disabled in configuration")
            return False
        
        try:
            # Get FLUXNET configuration parameters
            fluxnet_path_str = self.config.get('FLUXNET_PATH')
            station_id = self.config.get('FLUXNET_STATION')
            
            if not fluxnet_path_str or not station_id:
                self.logger.error("Missing FLUXNET_PATH or FLUXNET_STATION in configuration")
                return False
            
            fluxnet_path = Path(fluxnet_path_str)
            
            # Create directory for FLUXNET data if it doesn't exist
            output_dir = self.project_dir / 'observations' / 'fluxnet'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Looking for FLUXNET files with station ID: {station_id} in {fluxnet_path}")
            
            # Find files containing the station ID
            import shutil
            import glob
            
            # Check if the path exists
            if not fluxnet_path.exists():
                self.logger.error(f"FLUXNET path does not exist: {fluxnet_path}")
                return False
                
            # Find all files in the directory (including subdirectories) that match the station ID
            matching_files = []
            # Use rglob for recursive search
            for file_path in fluxnet_path.rglob('*'):
                if file_path.is_file() and station_id in file_path.name:
                    matching_files.append(file_path)
                    
            if not matching_files:
                self.logger.warning(f"No FLUXNET files found for station ID: {station_id} in {fluxnet_path}")
                return False
                
            self.logger.info(f"Found {len(matching_files)} FLUXNET files for station {station_id}")
            
            # Copy files to the project directory
            for file_path in matching_files:
                dest_file = output_dir / file_path.name
                try:
                    shutil.copy2(file_path, dest_file)
                    self.logger.info(f"Copied {file_path.name} to {dest_file}")
                except Exception as copy_e:
                    self.logger.error(f"Failed to copy {file_path.name}: {copy_e}")
            
            self.logger.info(f"Successfully processed FLUXNET data for station {station_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing FLUXNET data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def process_snotel_data(self):
        """
        Process SNOTEL snow water equivalent data.
        
        This method:
        1. Checks if SNOTEL data download is enabled in configuration
        2. Finds the appropriate SNOTEL CSV file based on station ID
        3. Extracts date and SWE columns
        4. Saves processed data to project directory
        
        Returns:
            bool: True if successful, False otherwise
        """
        self.logger.info("Processing SNOTEL data")
        
        # Check if SNOTEL processing is enabled
        if self.config.get('DOWNLOAD_SNOTEL') != 'true':
            self.logger.info("SNOTEL data processing is disabled in configuration")
            return False
        
        try:
            # Get SNOTEL configuration parameters
            snotel_path_str = self.config.get('SNOTEL_PATH')
            snotel_station_id = self.config.get('SNOTEL_STATION')
            domain_name = self.config.get('DOMAIN_NAME')
            
            if not snotel_path_str or not snotel_station_id:
                self.logger.error("Missing SNOTEL_PATH or SNOTEL_STATION in configuration")
                return False
            
            snotel_path = Path(snotel_path_str)
            
            # Create directory for processed data if it doesn't exist
            project_dir = Path(self.config.get('SYMFLUENCE_DATA_DIR')) / f"domain_{domain_name}"
            output_dir = project_dir / 'observations' / 'snow' / 'swe'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define output file path
            output_file = output_dir / f"{domain_name}_swe_processed.csv"
            
            # Find the appropriate SNOTEL file based on station ID
            snotel_file = None
            
            # Search for files containing the station ID
            # Use rglob for recursive search
            for file in snotel_path.rglob(f'*{snotel_station_id}*.csv'):
                snotel_file = file
                break
            
            if not snotel_file:
                self.logger.error(f"No SNOTEL file found for station ID: {snotel_station_id} in {snotel_path}")
                return False
            
            self.logger.info(f"Found SNOTEL file: {snotel_file}")
            
            # Read the SNOTEL data file
            import pandas as pd
            
            # Read the data, skipping header rows until we find the actual data
            # Usually headers end when we find a line starting with "Date"
            header_line_num = -1
            with open(snotel_file, 'r') as f:
                for i, line in enumerate(f):
                    if line.startswith('Date'):
                        header_line_num = i
                        break
            
            if header_line_num == -1:
                self.logger.error(f"Could not find header line starting with 'Date' in {snotel_file}")
                return False

            # Read the data starting from the identified line
            df = pd.read_csv(snotel_file, skiprows=header_line_num)
            
            # Extract just the Date and SWE columns
            # The column name might vary, so we'll try to identify it
            swe_column = None
            for col in df.columns:
                if 'Snow Water Equivalent' in col:
                    swe_column = col
                    break
            
            if not swe_column:
                self.logger.error("Could not find 'Snow Water Equivalent' column in SNOTEL data")
                return False
            
            # Create a new DataFrame with just Date and SWE
            processed_df = pd.DataFrame()
            processed_df['Date'] = df['Date']
            processed_df['SWE'] = df[swe_column]
            
            # Try to parse dates with different formats
            try:
                # Attempt to infer format first
                processed_df['Date'] = pd.to_datetime(processed_df['Date'], infer_datetime_format=True, errors='coerce')
            except Exception as date_error:
                self.logger.warning(f"Flexible date parsing failed: {str(date_error)}")
                # Fallback to specific formats if inference fails
                try:
                    processed_df['Date'] = pd.to_datetime(processed_df['Date'], format='%m/%d/%Y', errors='coerce') # MM/DD/YYYY
                except:
                    try:
                        processed_df['Date'] = pd.to_datetime(processed_df['Date'], format='%Y-%m-%d', errors='coerce') # YYYY-MM-DD
                    except:
                        self.logger.error("Could not parse Date column with known formats.")
                        return False

            # Ensure the Date column is formatted consistently (YYYY-MM-DD)
            processed_df['Date'] = processed_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Convert SWE to numeric, coercing errors to NaN
            processed_df['SWE'] = pd.to_numeric(processed_df['SWE'], errors='coerce')
            
            # Drop rows with invalid dates or SWE values
            processed_df = processed_df.dropna(subset=['Date', 'SWE'])
            
            # Save the processed data
            processed_df.to_csv(output_file, index=False)
            
            self.logger.info(f"Processed SNOTEL data saved to {output_file}")
            return True
        
        except FileNotFoundError:
            self.logger.error(f"SNOTEL file not found at {snotel_file}")
            return False
        except Exception as e:
            self.logger.error(f"Error processing SNOTEL data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _process_caravans_data(self):
        """
        Process CARAVANS streamflow data.
        
        This function reads CARAVANS CSV data, processes it, and converts from mm/d to m³/s
        using the basin area from the shapefile.
        """
        # Check if CARAVANS processing is enabled
        if not self.config.get('PROCESS_CARAVANS', False):
            self.logger.info("CARAVANS data processing is disabled in configuration")
            return
        
        self.logger.info("Processing CARAVANS streamflow data")
        
        try:
            # Determine input and output paths
            input_file_name = self.streamflow_raw_name
            if not input_file_name:
                self.logger.error("STREAMFLOW_RAW_NAME not specified in config for CARAVANS data.")
                return
            
            input_file = self.streamflow_raw_path / input_file_name
            output_file = self.streamflow_processed_path / f'{self.domain_name}_streamflow_processed.csv'
            
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Reading CARAVANS data from: {input_file}")
            
            # Read the CSV file
            try:
                # Try reading with standard format
                caravans_data = pd.read_csv(input_file, sep=',', header=0)
            except Exception as e:
                self.logger.warning(f"Standard parsing failed: {e}. Trying alternative format...")
                try:
                    # Try with flexible parsing (handles multiple delimiters)
                    caravans_data = pd.read_csv(input_file, sep='[,\s]+', engine='python', header=0)
                except Exception as e2:
                    self.logger.error(f"Alternative parsing also failed: {e2}")
                    raise DataAcquisitionError(f"Could not parse CARAVANS data file: {input_file}")
            
            # Identify date and discharge columns
            date_col_name = None
            discharge_col_name = None
            
            for col in caravans_data.columns:
                col_lower = col.lower()
                if 'date' in col_lower and not date_col_name:
                    date_col_name = col
                if ('discharge' in col_lower or 'flow' in col_lower or 'm3s' in col_lower or 'mm/d' in col_lower) and not discharge_col_name:
                    discharge_col_name = col
            
            if not date_col_name:
                self.logger.error("No date column identified in CARAVANS data. Please check column names.")
                raise DataAcquisitionError("No date column found in CARAVANS data")
            if not discharge_col_name:
                self.logger.error("No discharge column identified in CARAVANS data. Please check column names.")
                raise DataAcquisitionError("No discharge column found in CARAVANS data")
            
            self.logger.info(f"Using date column: '{date_col_name}', discharge column: '{discharge_col_name}'")
            
            # Rename columns and select only necessary ones
            caravans_data = caravans_data.rename(columns={date_col_name: 'date', discharge_col_name: 'discharge_value'})
            caravans_data = caravans_data[['date', 'discharge_value']]
            
            # Convert discharge to numeric, handling errors
            caravans_data['discharge_value'] = pd.to_numeric(caravans_data['discharge_value'], errors='coerce')
            
            # Convert date to datetime
            try:
                # Try parsing with common formats, prioritizing European format if applicable
                caravans_data['datetime'] = pd.to_datetime(caravans_data['date'], dayfirst=True, infer_datetime_format=True, errors='coerce')
            except Exception as e:
                self.logger.warning(f"Date parsing with dayfirst=True failed: {e}. Trying without.")
                caravans_data['datetime'] = pd.to_datetime(caravans_data['date'], infer_datetime_format=True, errors='coerce')
            
            # Drop rows with invalid dates
            na_date_count = caravans_data['datetime'].isna().sum()
            if na_date_count > 0:
                self.logger.warning(f"Dropping {na_date_count} rows with invalid date values")
                caravans_data = caravans_data.dropna(subset=['datetime'])
                
            # Set datetime as index
            caravans_data.set_index('datetime', inplace=True)
            
            # Sort index
            caravans_data.sort_index(inplace=True)
            
            # Now drop rows with NaN discharge values
            na_count = caravans_data['discharge_value'].isna().sum()
            if na_count > 0:
                self.logger.warning(f"Dropping {na_count} rows with missing or non-numeric discharge values")
                caravans_data = caravans_data.dropna(subset=['discharge_value'])
            
            # Determine if discharge is in mm/d or m³/s based on column name or config
            discharge_unit = 'mm/d' # Default assumption
            if 'm3s' in discharge_col_name.lower() or 'cms' in discharge_col_name.lower():
                discharge_unit = 'm³/s'
            elif 'cfs' in discharge_col_name.lower():
                discharge_unit = 'cfs'
            
            self.logger.info(f"Detected discharge unit: {discharge_unit}")

            # Convert discharge to m³/s if necessary
            if discharge_unit == 'mm/d':
                # Get the basin area from the shapefile
                try:
                    # Determine the shapefile path
                    subbasins_name = self.config.get('RIVER_BASINS_NAME')
                    if subbasins_name == 'default':
                        subbasins_name = f"{self.config.get('DOMAIN_NAME')}_riverBasins.shp"
                    
                    shapefile_path_str = self.config.get('RIVER_BASIN_SHP_PATH')
                    if shapefile_path_str:
                        shapefile_path = Path(shapefile_path_str)
                    else:
                        # Try default locations
                        shapefile_path = self.project_dir / "shapefiles/river_basins" / subbasins_name
                        if not shapefile_path.exists():
                            alt_shapefile_path = self.project_dir / "shapefiles/catchment" / f"{self.config.get('DOMAIN_NAME')}_catchment.shp"
                            if alt_shapefile_path.exists():
                                shapefile_path = alt_shapefile_path
                                self.logger.info(f"Using alternative shapefile: {shapefile_path}")
                            else:
                                raise FileNotFoundError(f"Cannot find shapefile at {shapefile_path} or {alt_shapefile_path}")
                    
                    # Read the shapefile
                    import geopandas as gpd
                    gdf = gpd.read_file(shapefile_path)
                    
                    # Get area column from the shapefile
                    area_column = self.config.get('RIVER_BASIN_SHP_AREA', 'GRU_area')
                    
                    # If area column not found, try alternative names
                    if area_column not in gdf.columns:
                        area_alternatives = ['GRU_area', 'area', 'Area', 'AREA', 'basin_area', 'HRU_area', 'catchment_area']
                        for alt in area_alternatives:
                            if alt in gdf.columns:
                                area_column = alt
                                self.logger.info(f"Using alternative area column: {area_column}")
                                break
                        
                        # If still not found, calculate area from geometry
                        if area_column not in gdf.columns:
                            self.logger.warning("No area column found, calculating from geometry...")
                            # Ensure CRS is suitable for area calculation (e.g., projected CRS)
                            # If CRS is geographic (lat/lon), reproject to an equal-area projection
                            if gdf.crs and gdf.crs.is_geographic:
                                self.logger.info(f"Reprojecting to an equal-area CRS for area calculation: {gdf.crs}")
                                # Use a common equal-area projection, e.g., Albers Equal Area
                                gdf_projected = gdf.to_crs('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs')
                            else:
                                gdf_projected = gdf # Assume it's already projected
                            
                            gdf['calculated_area'] = gdf_projected.geometry.area
                            area_column = 'calculated_area'
                            # Area is now in square meters, convert to square km
                            gdf[area_column] = gdf[area_column] / 1e6
                    
                    # Sum the areas to get total basin area in km²
                    basin_area_km2 = gdf[area_column].sum() # Assuming area is already in km² or m²
                    
                    # Check units and convert if necessary (assuming area column might be in m²)
                    # If the sum is very large (e.g., > 1,000,000 km²), it might be in m²
                    if basin_area_km2 > 1000000:
                        self.logger.warning(f"Basin area sum ({basin_area_km2:.2f}) seems large, assuming units are m² and converting to km².")
                        basin_area_km2 = basin_area_km2 / 1e6
                    elif basin_area_km2 < 0.01:
                        self.logger.warning(f"Basin area sum ({basin_area_km2:.2f}) seems small, assuming units are m² and converting to km².")
                        basin_area_km2 = basin_area_km2 * 1e6 # If it's very small, maybe it's km² and needs conversion to m² for calculation, then back to km²
                        # This logic needs careful review based on expected units.
                        # For now, let's assume the area column is in km² or m² and sum it.
                        # If it's m², we'll convert later.

                    # Convert discharge from mm/d to m³/s
                    # Formula: m³/s = (mm/d × basin_area_km² × 1000) / SECONDS_PER_DAY
                    # 1000: convert km² to m²
                    # SECONDS_PER_DAY: seconds in a day (86400)
                    
                    # Ensure basin_area_km2 is in km² for the formula
                    # If the area column was in m², we need to convert it first
                    if area_column == 'calculated_area' or 'm2' in area_column.lower(): # Heuristic check for m²
                        self.logger.info(f"Area column '{area_column}' seems to be in m², converting to km².")
                        basin_area_km2 = basin_area_km2 / 1e6
                    
                    conversion_factor = (basin_area_km2 * 1000) / UnitConversion.SECONDS_PER_DAY
                    caravans_data['discharge_cms'] = caravans_data['discharge_value'] * conversion_factor
                    
                    self.logger.info(f"Basin area: {basin_area_km2:.2f} km²")
                    self.logger.info(f"Converted discharge from mm/d to m³/s using conversion factor: {conversion_factor:.6f}")

                except FileNotFoundError as fnf_e:
                    self.logger.error(f"Shapefile not found: {fnf_e}. Cannot convert mm/d to m³/s.")
                    raise DataAcquisitionError("Shapefile not found for basin area calculation.") from fnf_e
                except Exception as basin_error:
                    self.logger.error(f"Error determining basin area or converting units: {basin_error}")
                    self.logger.warning("Falling back to assuming discharge is already in m³/s.")
                    caravans_data['discharge_cms'] = caravans_data['discharge_value'] # Assume it's already m³/s
            elif discharge_unit == 'm³/s' or discharge_unit == 'cms':
                self.logger.info("Discharge unit is already m³/s, no conversion needed.")
                caravans_data['discharge_cms'] = caravans_data['discharge_value']
            elif discharge_unit == 'cfs':
                self.logger.info("Discharge unit is cfs, converting to m³/s.")
                caravans_data['discharge_cms'] = caravans_data['discharge_value'] * UnitConversion.CFS_TO_CMS
            else:
                self.logger.warning(f"Unknown discharge unit '{discharge_unit}'. Assuming it's already in m³/s.")
                caravans_data['discharge_cms'] = caravans_data['discharge_value']

            # Verify we have a DatetimeIndex
            if not isinstance(caravans_data.index, pd.DatetimeIndex):
                self.logger.error("Failed to create DatetimeIndex. Index type is: " + str(type(caravans_data.index)))
                # Try a last-resort conversion
                try:
                    caravans_data.index = pd.to_datetime(caravans_data.index)
                except Exception as idx_e:
                    self.logger.error(f"Final attempt to convert index to datetime failed: {idx_e}")
                    raise DataAcquisitionError("Failed to create a valid DatetimeIndex.")
            
            self.logger.info(f"Data date range: {caravans_data.index.min()} to {caravans_data.index.max()}")
            self.logger.info(f"Number of records after processing: {len(caravans_data)}")
            self.logger.info(f"Min discharge: {caravans_data['discharge_cms'].min():.4f} m³/s")
            self.logger.info(f"Max discharge: {caravans_data['discharge_cms'].max():.4f} m³/s")
            self.logger.info(f"Mean discharge: {caravans_data['discharge_cms'].mean():.4f} m³/s")
            
            # Resample and save the data
            self._resample_and_save(caravans_data['discharge_cms'])
            
            self.logger.info(f"Successfully processed CARAVANS data")
            
        except FileNotFoundError:
            self.logger.error(f"CARAVANS input file not found at {input_file}")
        except DataAcquisitionError as dae:
            self.logger.error(f"Data Acquisition Error during CARAVANS processing: {dae}")
        except Exception as e:
            self.logger.error(f"Error processing CARAVANS data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
        
    def _download_and_process_usgs_data(self):
        """
        Process USGS streamflow data by fetching it directly from USGS API.
        
        This function fetches discharge data for the specified USGS station,
        converts it from cubic feet per second (cfs) to cubic meters per second (cms),
        and resamples it to the configured time step.
        """
        import requests
        import io
        from datetime import datetime, timedelta
        import time
        import pandas as pd

        self.logger.info("Processing USGS streamflow data directly from API")
        
        # Get configuration parameters
        station_id = self.config.get('STATION_ID')
        
        # Format station ID - ensure it's a string and pad with leading zeros if needed
        try:
            # If it's a numeric ID, format it with leading zeros (typically 8 digits for USGS)
            if str(station_id).isdigit():
                # Try to ensure proper USGS station ID format (typically 8 digits)
                if len(str(station_id)) < 8:
                    station_id = str(station_id).zfill(8)
                    self.logger.info(f"Formatted station ID to 8 digits: {station_id}")
        except (AttributeError, ValueError):
            self.logger.warning(f"Could not format station ID: {station_id}. Using as is.")
        
        # Parse and format the start date properly
        start_date_raw = self.config.get('EXPERIMENT_TIME_START')
        try:
            # Try to parse the date string with various formats
            for fmt in ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                try:
                    parsed_date = datetime.strptime(start_date_raw, fmt)
                    start_date = parsed_date.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                # If none of the formats match, use a default format
                self.logger.warning(f"Could not parse start date: {start_date_raw}. Using first 10 characters as YYYY-MM-DD.")
                start_date = start_date_raw[:10]
        except Exception as e:
            self.logger.warning(f"Error parsing start date: {e}. Using default date format.")
            start_date = start_date_raw[:10]
        
        # Format end date as YYYY-MM-DD
        end_date = datetime.now().strftime("%Y-%m-%d")
        parameter_cd = "00060"  # Discharge parameter code (cubic feet per second)
        
        # Conversion factor from cubic feet per second (cfs) to cubic meters per second (cms)
        CFS_TO_CMS = UnitConversion.CFS_TO_CMS
        
        self.logger.info(f"Retrieving discharge data for station {station_id}")
        self.logger.info(f"Time period: {start_date} to {end_date}")
        self.logger.info(f"Converting from cfs to cms using factor: {CFS_TO_CMS}")
        
        # Log the formatted dates
        self.logger.info(f"Using formatted start date: {start_date}")
        self.logger.info(f"Using formatted end date: {end_date}")
        
        # Use the correct URL with the 'nwis' prefix
        base_url = "https://nwis.waterservices.usgs.gov/nwis/iv/" 
        
        # Construct the URL for tab-delimited data - ensure no spaces in the URL
        url = f"{base_url}?site={station_id}&format=rdb&parameterCd={parameter_cd}&startDT={start_date}&endDT={end_date}"
        
        self.logger.info(f"Fetching data from: {url}")
        
        try:
            response = requests.get(url, timeout=60)  # Add timeout
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            # The RDB format has comment lines starting with #
            lines = response.text.split('\n')
            
            # Find the header line (it's after the comments and has field names)
            header_line_num = -1
            for i, line in enumerate(lines):
                if not line.startswith('#') and '\t' in line:
                    header_line_num = i
                    break
            
            if header_line_num == -1:
                self.logger.error("Could not find header line in the response. Response might be empty or malformed.")
                # Check if the response is empty or contains an error message
                if not response.text.strip() or 'No data available' in response.text:
                    self.logger.error(f"USGS API returned no data for station {station_id} in the specified range.")
                else:
                    self.logger.error(f"Response content: {response.text[:500]}...") # Log snippet of response
                raise DataAcquisitionError(f"Could not find header line in USGS response for station {station_id}.")
            
            # Skip the header line and the line after (which contains format info)
            data_start_line = header_line_num + 2
            
            # Create a data string with just the header and data rows
            data_str = '\n'.join([lines[header_line_num]] + lines[data_start_line:])
            
            # Parse the tab-delimited data
            df = pd.read_csv(io.StringIO(data_str), sep='\t', comment='#', skipinitialspace=True)
            
            # Find the discharge column (usually contains the parameter code)
            discharge_cols = [col for col in df.columns if parameter_cd in col]
            datetime_col = None
            
            # Find the datetime column (usually named 'datetime' or similar)
            datetime_candidates = ['datetime', 'date_time', 'dateTime', 'Timestamp']
            for col in df.columns:
                if col.lower() in [c.lower() for c in datetime_candidates]:
                    datetime_col = col
                    break
            
            if not discharge_cols:
                self.logger.error(f"Could not find column with parameter code {parameter_cd} in USGS data.")
                # Try to guess based on typical column names if parameter code fails
                value_cols = [col for col in df.columns if 'value' in col.lower()]
                if value_cols:
                    discharge_cols = [value_cols[0]]
                    self.logger.info(f"Using column '{discharge_cols[0]}' as discharge values based on name.")
                else:
                    raise DataAcquisitionError(f"Could not identify discharge column in USGS data for station {station_id}.")
            
            if not datetime_col:
                self.logger.error("Could not find datetime column in USGS data.")
                # Try to guess based on column data type and content
                for col in df.columns:
                    # Check if column contains date-like strings and is object type
                    if df[col].dtype == 'object' and df[col].astype(str).str.contains('-').any() and df[col].astype(str).str.contains(':').any():
                        datetime_col = col
                        self.logger.info(f"Using column '{datetime_col}' as datetime based on content.")
                        break
                if not datetime_col:
                    raise DataAcquisitionError(f"Could not identify datetime column in USGS data for station {station_id}.")
            
            discharge_col = discharge_cols[0]
            self.logger.info(f"Using discharge column: '{discharge_col}'")
            self.logger.info(f"Using datetime column: '{datetime_col}'")
            
            # Keep only the necessary columns
            df_clean = df[[datetime_col, discharge_col]].copy()
            
            # Convert datetime column to datetime type
            df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col], errors='coerce')
            
            # Convert discharge values to numeric, forcing errors to NaN
            df_clean[discharge_col] = pd.to_numeric(df_clean[discharge_col], errors='coerce')
            
            # Drop rows with NaN datetime or discharge values
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=[datetime_col, discharge_col])
            na_count = initial_rows - len(df_clean)
            if na_count > 0:
                self.logger.warning(f"Dropped {na_count} rows with invalid datetime or discharge values.")
            
            # Create a new column with the discharge in cubic meters per second (cms)
            df_clean['discharge_cms'] = df_clean[discharge_col] * CFS_TO_CMS
            
            # Set datetime as index
            df_clean.set_index(datetime_col, inplace=True)
            
            # Call the resampling and saving function
            self._resample_and_save(df_clean['discharge_cms'])
            
            self.logger.info(f"Successfully processed USGS data for station {station_id}")
            self.logger.info(f"Retrieved {len(df_clean)} valid records from {df_clean.index.min()} to {df_clean.index.max()}")
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching data from USGS API: {str(e)}")
            
            # Attempt fallback URLs or strategies if the primary URL fails
            fallback_strategies = [
                # Try without 'nwis' prefix
                lambda: url.replace("nwis.waterservices.usgs.gov", "waterservices.usgs.gov"),
                # Try a shorter date range (e.g., last year)
                lambda: url.replace(f"startDT={start_date}", f"startDT={(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')}"),
                # Try a different parameter code if 00060 fails (less likely for discharge)
            ]

            for strategy in fallback_strategies:
                try:
                    fallback_url = strategy()
                    self.logger.info(f"Trying fallback URL: {fallback_url}")
                    fallback_response = requests.get(fallback_url, timeout=60)
                    fallback_response.raise_for_status()
                    
                    # Re-process the response using the same logic as above
                    lines = fallback_response.text.split('\n')
                    header_line_num = -1
                    for i, line in enumerate(lines):
                        if not line.startswith('#') and '\t' in line:
                            header_line_num = i
                            break
                    
                    if header_line_num == -1:
                        self.logger.warning("Fallback URL also returned no valid header.")
                        continue # Try next strategy
                    
                    data_start_line = header_line_num + 2
                    data_str = '\n'.join([lines[header_line_num]] + lines[data_start_line:])
                    df = pd.read_csv(io.StringIO(data_str), sep='\t', comment='#', skipinitialspace=True)

                    # Re-identify columns (might differ slightly in fallback responses)
                    discharge_cols = [col for col in df.columns if parameter_cd in col or 'value' in col.lower()]
                    datetime_col = None
                    for col in df.columns:
                        if col.lower() in ['datetime', 'date_time', 'datetime', 'timestamp']:
                            datetime_col = col
                            break
                    
                    if not discharge_cols or not datetime_col:
                        self.logger.warning("Could not reliably identify columns in fallback response.")
                        continue # Try next strategy

                    discharge_col = discharge_cols[0]
                    df_clean = df[[datetime_col, discharge_col]].copy()
                    df_clean[datetime_col] = pd.to_datetime(df_clean[datetime_col], errors='coerce')
                    df_clean[discharge_col] = pd.to_numeric(df_clean[discharge_col], errors='coerce')
                    df_clean = df_clean.dropna(subset=[datetime_col, discharge_col])
                    df_clean['discharge_cms'] = df_clean[discharge_col] * CFS_TO_CMS
                    df_clean.set_index(datetime_col, inplace=True)
                    
                    self._resample_and_save(df_clean['discharge_cms'])
                    self.logger.info(f"Successfully processed USGS data using fallback strategy for station {station_id}")
                    return # Success, exit function

                except Exception as fallback_e:
                    self.logger.warning(f"Fallback strategy failed: {str(fallback_e)}")
                    continue # Try next strategy
            
            # If all fallbacks fail
            self.logger.error("All attempts to retrieve USGS data failed.")
            raise DataAcquisitionError(f"Could not retrieve USGS data for station {station_id} after multiple attempts.") from e
                
        except Exception as e:
            self.logger.error(f"Unexpected error processing USGS data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _process_usgs_data(self):
        self.logger.info("Processing USGS streamflow data from local file")
        try:
            file_path = self.streamflow_raw_path / self.streamflow_raw_name
            if not file_path.exists():
                self.logger.error(f"USGS raw data file not found at {file_path}")
                raise FileNotFoundError(f"USGS raw data file not found: {file_path}")

            # Read the CSV file, handling comments and potential header issues
            # USGS RDB format often has comments starting with '#'
            # The actual data starts after a line like '# Data columns:'
            
            # Find the line number where data starts
            header_line_num = -1
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if line.startswith('# Data columns:') or line.startswith('USGS'): # Common header indicators
                        header_line_num = i
                        break
            
            if header_line_num == -1:
                self.logger.error(f"Could not find data header in {file_path}. File might be malformed.")
                return

            # Read the data using pandas, specifying comment character and separator
            # Use skipinitialspace=True to handle spaces after delimiters
            usgs_data = pd.read_csv(file_path, 
                                    comment='#', 
                                    sep='\t', 
                                    header=header_line_num, # Use the found header line
                                    skipinitialspace=True, 
                                    low_memory=False)

            # Identify datetime and discharge columns
            datetime_col = None
            discharge_col = None
            
            # Common datetime column names
            datetime_candidates = ['datetime', 'date_time', 'dateTime', 'Timestamp']
            for col in usgs_data.columns:
                if col.lower() in [c.lower() for c in datetime_candidates]:
                    datetime_col = col
                    break
            
            # Common discharge column names (often contain parameter code like 00060)
            discharge_candidates = ['00060', 'discharge', 'flow', 'value']
            for col in usgs_data.columns:
                for candidate in discharge_candidates:
                    if candidate in col.lower():
                        discharge_col = col
                        break
                if discharge_col: break

            if not datetime_col:
                self.logger.error("Could not find datetime column in USGS data file.")
                return
            if not discharge_col:
                self.logger.error("Could not find discharge column in USGS data file.")
                return

            self.logger.info(f"Using datetime column: '{datetime_col}', discharge column: '{discharge_col}'")

            # Convert datetime column
            usgs_data[datetime_col] = pd.to_datetime(usgs_data[datetime_col], errors='coerce')
            
            # Convert discharge column to numeric
            usgs_data[discharge_col] = pd.to_numeric(usgs_data[discharge_col], errors='coerce')
            
            # Drop rows with invalid datetime or discharge values
            usgs_data = usgs_data.dropna(subset=[datetime_col, discharge_col])
            
            # Rename datetime column for consistency
            usgs_data.rename(columns={datetime_col: 'datetime'}, inplace=True)
            usgs_data.set_index('datetime', inplace=True)
            
            # Convert discharge from cfs to cms
            usgs_data['discharge_cms'] = usgs_data[discharge_col] * UnitConversion.CFS_TO_CMS
            
            self._resample_and_save(usgs_data['discharge_cms'])
            self.logger.info(f"Successfully processed local USGS data from {file_path}")

        except FileNotFoundError:
            self.logger.error(f"USGS raw data file not found at {file_path}")
        except Exception as e:
            self.logger.error(f"Error processing local USGS data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _process_wsc_data(self):
        self.logger.info("Processing WSC streamflow data from local file")
        try:
            file_path = self.streamflow_raw_path / self.streamflow_raw_name
            if not file_path.exists():
                self.logger.error(f"WSC raw data file not found at {file_path}")
                raise FileNotFoundError(f"WSC raw data file not found: {file_path}")

            # Read the CSV file, handling comments and potential header issues
            # WSC RDB format often has comments starting with '#'
            wsc_data = pd.read_csv(file_path, 
                                   comment='#', 
                                   low_memory=False)

            # Identify datetime and discharge columns
            datetime_col = None
            discharge_col = None

            # Common datetime column names
            datetime_candidates = ['ISO 8601 UTC', 'datetime', 'date_time', 'Timestamp']
            for col in wsc_data.columns:
                if col in datetime_candidates:
                    datetime_col = col
                    break
            
            # Common discharge column names
            discharge_candidates = ['Value', 'discharge', 'flow', 'discharge_cms']
            for col in wsc_data.columns:
                for candidate in discharge_candidates:
                    if candidate.lower() in col.lower():
                        discharge_col = col
                        break
                if discharge_col: break

            if not datetime_col:
                self.logger.error("Could not find datetime column in WSC data file.")
                return
            if not discharge_col:
                self.logger.error("Could not find discharge column in WSC data file.")
                return

            self.logger.info(f"Using datetime column: '{datetime_col}', discharge column: '{discharge_col}'")

            # Convert datetime column, handling potential timezone issues
            wsc_data[datetime_col] = pd.to_datetime(wsc_data[datetime_col], errors='coerce')
            # If timezone info is present (e.g., 'UTC'), remove it for consistency if needed, or convert to local time
            if wsc_data[datetime_col].dt.tz is not None:
                self.logger.info(f"Detected timezone '{wsc_data[datetime_col].dt.tz}' in WSC datetime. Converting to local time and removing tz info.")
                wsc_data[datetime_col] = wsc_data[datetime_col].dt.tz_convert('America/Edmonton').dt.tz_localize(None)
            
            # Convert discharge column to numeric
            wsc_data[discharge_col] = pd.to_numeric(wsc_data[discharge_col], errors='coerce')
            
            # Drop rows with invalid datetime or discharge values
            wsc_data = wsc_data.dropna(subset=[datetime_col, discharge_col])
            
            # Rename datetime column for consistency
            wsc_data.rename(columns={datetime_col: 'datetime'}, inplace=True)
            wsc_data.set_index('datetime', inplace=True)
            
            # Rename discharge column to 'discharge_cms' for consistency
            wsc_data.rename(columns={discharge_col: 'discharge_cms'}, inplace=True)

            self._resample_and_save(wsc_data['discharge_cms'])
            self.logger.info(f"Successfully processed local WSC data from {file_path}")

        except FileNotFoundError:
            self.logger.error(f"WSC raw data file not found at {file_path}")
        except Exception as e:
            self.logger.error(f"Error processing local WSC data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _resample_and_save(self, data):
        resample_freq = self.get_resample_freq()
        
        # Ensure data is sorted by index before resampling
        data = data.sort_index()

        # Resample the data
        resampled_data = data.resample(resample_freq).mean()
        
        # Interpolate missing values
        # Use time-based interpolation for potentially irregular time series
        # Limit interpolation to avoid excessive extrapolation
        resampled_data = resampled_data.interpolate(method='time', limit_direction='both', limit=30) # Limit interpolation to 30 periods
        
        # Optionally, drop remaining NaNs if interpolation didn't fill everything
        # resampled_data = resampled_data.dropna()

        output_file = self.streamflow_processed_path / f'{self.domain_name}_streamflow_processed.csv'
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for writing, ensuring datetime format is consistent
        data_to_write = []
        for dt, value in resampled_data.items():
            # Format datetime to YYYY-MM-DD HH:MM:SS
            formatted_datetime = dt.strftime('%Y-%m-%d %H:%M:%S')
            data_to_write.append([formatted_datetime, value])

        # Write to CSV
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write header
                csv_writer.writerow(['datetime', 'discharge_cms'])
                # Write data rows
                csv_writer.writerows(data_to_write)

            self.logger.info(f"Processed streamflow data saved to: {output_file}")
            self.logger.info(f"Total rows in processed data: {len(resampled_data)}")
            self.logger.info(f"Number of non-null values: {resampled_data.count()}")
            self.logger.info(f"Number of null values after interpolation: {resampled_data.isnull().sum()}")

        except IOError as e:
            self.logger.error(f"Failed to write processed data to {output_file}: {e}")
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during file writing: {e}")
