from pathlib import Path
from typing import Dict, Any
import pandas as pd # type: ignore
import numpy as np # type: ignore

from symfluence.core.exceptions import DataAcquisitionError

class DataCleanupProcessor:
    def __init__(self, config: Dict[str, Any], logger: Any):
        self.config = config
        self.logger = logger
        self.root_path = Path(self.config.get('SYMFLUENCE_DATA_DIR'))
        self.domain_name = self.config.get('DOMAIN_NAME')
        self.project_dir = self.root_path / f"domain_{self.domain_name}"

    def cleanup_and_checks(self):
        """Perform cleanup and checks on the MAF output."""
        self.logger.info("Performing cleanup and checks on MAF output")
        
        # Define paths
        path_soil_type = self.project_dir / f'attributes/soil_class/domain_{self.domain_name}_stats_soil_classes.csv'
        path_landcover_type = self.project_dir / f'attributes/land_class/domain_{self.domain_name}_stats_NA_NALCMS_landcover_2020_30m.csv'
        path_elevation_mean = self.project_dir / f'attributes/elevation/domain_{self.domain_name}_stats_elv.csv'

        # Read files
        soil_type = pd.read_csv(path_soil_type)
        landcover_type = pd.read_csv(path_landcover_type)
        elevation_mean = pd.read_csv(path_elevation_mean)

        # Sort by COMID
        soil_type = soil_type.sort_values(by='COMID').reset_index(drop=True)
        landcover_type = landcover_type.sort_values(by='COMID').reset_index(drop=True)
        elevation_mean = elevation_mean.sort_values(by='COMID').reset_index(drop=True)

        # Check if COMIDs are the same across all files
        if not (len(soil_type) == len(landcover_type) == len(elevation_mean) and
                (soil_type['COMID'] == landcover_type['COMID']).all() and
                (landcover_type['COMID'] == elevation_mean['COMID']).all()):
            raise DataAcquisitionError("COMIDs are not consistent across soil, landcover, and elevation files")

        # Process soil type
        majority_value = soil_type['majority'].replace(0, np.nan).mode().iloc[0]
        soil_type['majority'] = soil_type['majority'].replace(0, majority_value).fillna(majority_value)
        if self.config.get('UNIFY_SOIL', False):
            soil_type['majority'] = majority_value

        # Process landcover
        min_land_fraction = self.config.get('MINIMUM_LAND_FRACTION', 0.01)
        for col in landcover_type.columns:
            if col.startswith('frac_'):
                landcover_type[col] = landcover_type[col].apply(lambda x: 0 if x < min_land_fraction else x)
        
        for index, row in landcover_type.iterrows():
            frac_columns = [col for col in landcover_type.columns if col.startswith('frac_')]
            row_sum = row[frac_columns].sum()
            if row_sum > 0:
                for col in frac_columns:
                    landcover_type.at[index, col] /= row_sum

        num_land_cover = self.config.get('NUM_LAND_COVER', 20)
        missing_columns = [f"frac_{i}" for i in range(1, num_land_cover+1) if f"frac_{i}" not in landcover_type.columns]
        for col in missing_columns:
            landcover_type[col] = 0

        frac_columns = [col for col in landcover_type.columns if col.startswith('frac_')]
        frac_columns.sort(key=lambda x: int(x.split('_')[1]))
        sorted_columns = [col for col in landcover_type.columns if col not in frac_columns] + frac_columns
        landcover_type = landcover_type.reindex(columns=sorted_columns)

        for col in frac_columns:
            if landcover_type.loc[0, col] < 0.00001:
                landcover_type.loc[0, col] = 0.00001

        # Process elevation
        elevation_mean['mean'].fillna(0, inplace=True)

        # Save modified files
        soil_type.to_csv(self.project_dir / 'attributes/soil_class/modified_domain_stats_soil_classes.csv', index=False)
        landcover_type.to_csv(self.project_dir / 'attributes/land_class/modified_domain_stats_NA_NALCMS_landcover_2020_30m.csv', index=False)
        elevation_mean.to_csv(self.project_dir / 'attributes/elevation/modified_domain_stats_elv.csv', index=False)

        self.logger.info("Cleanup and checks completed")
