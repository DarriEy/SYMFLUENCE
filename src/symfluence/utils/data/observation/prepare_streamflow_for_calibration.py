#!/usr/bin/env python3
"""
Convert discharge observations to SYMFLUENCE calibration format.

Converts raw discharge CSV files to the standardized format expected by
SYMFLUENCE calibration system.
"""

import pandas as pd
from pathlib import Path
import argparse


def prepare_streamflow_data(input_file: Path, output_file: Path, domain_name: str):
    """
    Convert discharge data to SYMFLUENCE calibration format.

    Expected input format: YYYY;MM;DD;qobs;qc_flag
    Output format: datetime, discharge_m3s, quality_flag
    """
    print(f"Reading input file: {input_file}")

    # Read CSV
    df = pd.read_csv(input_file, sep=';')

    # Create datetime column
    if 'YYYY' in df.columns:
        df['datetime'] = pd.to_datetime(df[['YYYY', 'MM', 'DD']].rename(
            columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}))
    else:
        raise ValueError(f"Expected columns YYYY, MM, DD in {input_file}")

    # Rename discharge column
    df_out = pd.DataFrame({
        'datetime': df['datetime'],
        'discharge_m3s': df['qobs'],
        'quality_flag': df['qc_flag'] if 'qc_flag' in df.columns else 'G'
    })

    # Filter by quality if needed (40 = good quality for LamaH, G = good for SMHI)
    if 'qc_flag' in df.columns:
        good_mask = (df['qc_flag'] == 40) | (df['qc_flag'] == 'G')
        df_out = df_out[good_mask]
        print(f"Filtered to {len(df_out)} good quality observations")

    # Sort by date
    df_out = df_out.sort_values('datetime')

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save
    df_out.to_csv(output_file, index=False)

    print(f"✓ Saved to: {output_file}")
    print(f"  Period: {df_out['datetime'].min()} to {df_out['datetime'].max()}")
    print(f"  Records: {len(df_out)}")
    print(f"  Mean discharge: {df_out['discharge_m3s'].mean():.2f} m³/s")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Prepare streamflow data for SYMFLUENCE calibration')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--domain', type=str, required=True, help='Domain name')

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    prepare_streamflow_data(input_file, output_file, args.domain)


if __name__ == "__main__":
    main()
