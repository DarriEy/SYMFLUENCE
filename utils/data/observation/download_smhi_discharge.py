#!/usr/bin/env python3
"""
Download discharge data from SMHI (Swedish Meteorological and Hydrological Institute)
for Swedish catchments.

SMHI Open Data API: https://opendata.smhi.se/apidocs/hydroapi/
"""

import requests
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime
import argparse


def get_smhi_stations(parameter_key=2):
    """
    Get list of available SMHI hydrological stations.

    Parameters:
    -----------
    parameter_key : int
        2 = Discharge (Vattenföring)
        1 = Water level (Vattennivå)

    Returns:
    --------
    list of dict: Station information
    """
    url = f"https://opendata-download-hydroobs.smhi.se/api/version/latest/parameter/{parameter_key}.json"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('station', [])
    except Exception as e:
        print(f"Error fetching stations: {e}")
        return []


def find_station_by_name(station_name, parameter_key=2):
    """Find station by name (case-insensitive partial match)."""
    stations = get_smhi_stations(parameter_key)

    matches = []
    for station in stations:
        if station_name.lower() in station.get('name', '').lower():
            matches.append(station)

    return matches


def download_discharge_data(station_id, parameter_key=2, period='corrected-archive'):
    """
    Download discharge data for a specific station.

    Parameters:
    -----------
    station_id : int
        SMHI station ID
    parameter_key : int
        2 = Discharge (Vattenföring)
    period : str
        'latest-months' - Latest 3 months
        'corrected-archive' - Quality-controlled historical data
        'latest-day' - Latest day

    Returns:
    --------
    pandas.DataFrame: Discharge data
    """
    url = f"https://opendata-download-hydroobs.smhi.se/api/version/latest/parameter/{parameter_key}/station/{station_id}/period/{period}/data.json"

    try:
        print(f"Downloading from: {url}")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract time series data
        values = data.get('value', [])

        if not values:
            print("No data available for this station/period combination")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(values)
        df['date'] = pd.to_datetime(df['date'] / 1000, unit='s')  # SMHI uses milliseconds
        df = df.rename(columns={'value': 'discharge_m3s', 'quality': 'quality_code'})

        # Get station metadata
        station_info = data.get('station', {})

        return df, station_info

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error {e.response.status_code}: {e}")
        return None, None
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None, None


def save_discharge_data(df, station_info, output_dir, station_name=""):
    """Save discharge data and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Format dates for filename
    start_date = df['date'].min().strftime('%Y-%m-%d')
    end_date = df['date'].max().strftime('%Y-%m-%d')

    # Save CSV
    csv_file = output_dir / f"{station_name}_discharge_daily_{start_date}_{end_date}.csv"

    # Convert to standard format (YYYY, MM, DD, qobs, qc_flag)
    df_out = pd.DataFrame({
        'YYYY': df['date'].dt.year,
        'MM': df['date'].dt.month,
        'DD': df['date'].dt.day,
        'qobs': df['discharge_m3s'],
        'qc_flag': df['quality_code']
    })

    df_out.to_csv(csv_file, index=False, sep=';')
    print(f"✓ Saved discharge data: {csv_file}")

    # Save metadata
    metadata = {
        'station': {
            'id': station_info.get('id'),
            'name': station_info.get('name'),
            'owner': station_info.get('owner'),
            'country': 'Sweden'
        },
        'coordinates': {
            'latitude': station_info.get('latitude'),
            'longitude': station_info.get('longitude'),
            'epsg': 4326  # WGS84
        },
        'data': {
            'source': 'SMHI Open Data',
            'parameter': 'Discharge (Vattenföring)',
            'units': 'm3/s',
            'temporal_resolution': 'daily',
            'period': {
                'start': start_date,
                'end': end_date,
                'total_days': len(df)
            }
        },
        'format': {
            'columns': {
                'YYYY': 'year',
                'MM': 'month',
                'DD': 'day',
                'qobs': 'observed discharge (m3/s)',
                'qc_flag': 'quality control flag'
            },
            'delimiter': ';'
        },
        'quality_codes': {
            'G': 'Green - Approved data',
            'Y': 'Yellow - Preliminary data',
            'R': 'Red - Uncertain data'
        },
        'download_date': datetime.now().isoformat(),
        'api_url': f"https://opendata-download-hydroobs.smhi.se/",
        'references': [
            'SMHI Open Data: https://www.smhi.se/data/utforskaren-oppna-data',
            'API Documentation: https://opendata.smhi.se/apidocs/hydroapi/'
        ]
    }

    metadata_file = output_dir.parent / 'metadata' / f"{station_name}_discharge_metadata.yaml"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Saved metadata: {metadata_file}")

    return csv_file, metadata_file


def main():
    parser = argparse.ArgumentParser(description='Download SMHI discharge data')
    parser.add_argument('--station-name', type=str, help='Station name (e.g., Flottsund, Fyris)')
    parser.add_argument('--station-id', type=int, help='SMHI station ID')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for discharge data')
    parser.add_argument('--period', type=str, default='corrected-archive',
                       choices=['corrected-archive', 'latest-months', 'latest-day'],
                       help='Data period to download')
    parser.add_argument('--list-stations', action='store_true', help='List all available stations')
    parser.add_argument('--search', type=str, help='Search for stations by name')

    args = parser.parse_args()

    # List all stations
    if args.list_stations:
        print("\n" + "="*80)
        print("Available SMHI Discharge Stations")
        print("="*80)
        stations = get_smhi_stations(parameter_key=2)
        for station in stations:
            print(f"ID: {station['id']:6d} | {station['name']:40s} | Lat: {station.get('latitude', 'N/A'):8.4f} Lon: {station.get('longitude', 'N/A'):8.4f}")
        return

    # Search stations
    if args.search:
        print(f"\nSearching for stations matching '{args.search}'...")
        matches = find_station_by_name(args.search)
        if matches:
            print(f"\nFound {len(matches)} matching station(s):")
            for station in matches:
                print(f"  ID: {station['id']:6d} | {station['name']:40s} | Lat: {station.get('latitude', 'N/A'):8.4f} Lon: {station.get('longitude', 'N/A'):8.4f}")
        else:
            print(f"No stations found matching '{args.search}'")
        return

    # Download data
    if args.station_id:
        station_id = args.station_id
    elif args.station_name:
        # Try to find station by name
        matches = find_station_by_name(args.station_name)
        if not matches:
            print(f"Error: No station found matching '{args.station_name}'")
            print("Use --search to find available stations")
            return
        if len(matches) > 1:
            print(f"Warning: Multiple stations found. Using first match:")
        station_id = matches[0]['id']
        print(f"Using station: {matches[0]['name']} (ID: {station_id})")
    else:
        print("Error: Must provide either --station-id or --station-name")
        return

    print(f"\nDownloading discharge data for station {station_id}...")
    df, station_info = download_discharge_data(station_id, period=args.period)

    if df is not None and not df.empty:
        print(f"✓ Downloaded {len(df)} records")
        print(f"  Period: {df['date'].min()} to {df['date'].max()}")
        print(f"  Mean discharge: {df['discharge_m3s'].mean():.2f} m³/s")

        # Save data
        station_name = station_info.get('name', f'station_{station_id}').lower().replace(' ', '_').replace('ä', 'a').replace('ö', 'o').replace('å', 'a')
        csv_file, metadata_file = save_discharge_data(df, station_info, args.output_dir, station_name)

        print(f"\n✓ Download complete!")
        print(f"  Data file: {csv_file}")
        print(f"  Metadata: {metadata_file}")
    else:
        print("Failed to download data")


if __name__ == "__main__":
    main()
