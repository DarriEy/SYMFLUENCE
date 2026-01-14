import pytest
import numpy as np
import netCDF4 as nc
import logging
from symfluence.optimization.core.transformers import SoilDepthTransformer

def test_soil_depth_transformation(tmp_path):
    """
    Test the SoilDepthTransformer with a real (mock) NetCDF file.
    """
    # 1. Create a dummy coldState.nc
    coldstate_path = tmp_path / "coldState.nc"
    with nc.Dataset(coldstate_path, 'w') as ds:
        ds.createDimension('hru', 1)
        ds.createDimension('midToto', 6)
        ds.createDimension('ifcToto', 7)

        depths = ds.createVariable('mLayerDepth', 'f8', ('midToto', 'hru'))
        heights = ds.createVariable('iLayerHeight', 'f8', ('ifcToto', 'hru'))

        original_depth_values = np.array([0.1, 0.2, 0.3, 0.4, 1.0, 2.0])
        depths[:, 0] = original_depth_values

        h_vals = np.zeros(7)
        for i in range(6):
            h_vals[i+1] = h_vals[i] + original_depth_values[i]
        heights[:, 0] = h_vals

    # 2. Setup transformer
    config = {'SETTINGS_SUMMA_COLDSTATE': 'coldState.nc'}
    import logging
    logger = logging.getLogger('test')
    transformer = SoilDepthTransformer(config, logger)

    # 3. Apply transformation
    params = {'total_soil_depth_multiplier': 1.5, 'shape_factor': 1.0}
    success = transformer.apply(params, tmp_path)

    assert success is True

    # 4. Verify results
    with nc.Dataset(coldstate_path, 'r') as ds:
        new_depths = ds.variables['mLayerDepth'][:, 0]
        expected_depths = original_depth_values * 1.5
        assert np.allclose(new_depths, expected_depths)

        new_heights = ds.variables['iLayerHeight'][:, 0]
        assert new_heights[-1] == pytest.approx(np.sum(expected_depths))

def test_transformation_manager(tmp_path):

    """Test the TransformationManager orchestrator."""

    from symfluence.optimization.core.transformers import TransformationManager



    coldstate_path = tmp_path / "coldState.nc"

    with nc.Dataset(coldstate_path, 'w') as ds:

        ds.createDimension('hru', 1)

        ds.createDimension('midToto', 2)

        ds.createDimension('ifcToto', 3)

        ds.createVariable('mLayerDepth', 'f8', ('midToto', 'hru'))

        ds.createVariable('iLayerHeight', 'f8', ('ifcToto', 'hru'))

        ds.variables['mLayerDepth'][:, 0] = [1.0, 2.0]

        ds.variables['iLayerHeight'][:, 0] = [0, 1, 3]



    mgr = TransformationManager({}, logging.getLogger('test'))

    success = mgr.transform({'total_soil_depth_multiplier': 2.0}, tmp_path)



    assert success is True

    with nc.Dataset(coldstate_path, 'r') as ds:

        assert ds.variables['mLayerDepth'][0, 0] == 2.0

        assert ds.variables['mLayerDepth'][1, 0] == 4.0
