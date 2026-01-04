
import logging
import sys
from pathlib import Path
import shutil

# Add src to path
sys.path.append('src')

from symfluence.utils.data.acquisition.handlers.cds_datasets import CERRAAcquirer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CERRA_Test')

# Mock config
config = {
    'DOMAIN_NAME': 'test_cerra_fix',
    'BOUNDING_BOX_COORDS': '59.88/17.59/59.86/17.61', # Fyris coords
    'EXPERIMENT_TIME_START': '2015-12-31 00:00',
    'EXPERIMENT_TIME_END': '2016-01-01 23:00',
    'SYMFLUENCE_DATA_DIR': 'test_data_dir'
}

def test_cerra_acquisition():
    print("Testing CERRA acquisition with year-by-year chunking...")
    
    # Initialize handler
    handler = CERRAAcquirer(config, logger)
    
    # Output directory
    output_dir = Path('test_data_dir') / 'domain_test_cerra_fix' / 'forcing' / 'raw_data'
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run download
        result_path = handler.download(output_dir)
        
        print(f"\nSUCCESS: Download completed. Result saved to: {result_path}")
        
        # Verify content
        import xarray as xr
        ds = xr.open_dataset(result_path)
        print("\nDataset Info:")
        print(ds)
        print(f"\nTime range: {ds.time.min().values} to {ds.time.max().values}")
        ds.close()
        
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cerra_acquisition()
