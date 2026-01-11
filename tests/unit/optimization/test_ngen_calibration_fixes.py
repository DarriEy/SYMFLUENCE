
import pytest
from pathlib import Path
import logging
from symfluence.optimization.parameter_managers.ngen_parameter_manager import NgenParameterManager

def test_soil_depth_mapping(tmp_path):
    """Test that soil_depth is correctly mapped to soil_params.depth in CFE."""
    logger = logging.getLogger('test')
    
    # Setup mock ngen structure
    ngen_dir = tmp_path / "NGEN"
    cfe_dir = ngen_dir / "CFE"
    cfe_dir.mkdir(parents=True)
    
    cfe_file = cfe_dir / "cat-1_bmi_config_cfe_pass.txt"
    cfe_file.write_text("soil_params.depth=2.0[m]\nsoil_params.smcmax=0.4[m/m]\n")
    
    config = {
        'DOMAIN_NAME': 'test',
        'EXPERIMENT_ID': 'run_1',
        'NGEN_MODULES_TO_CALIBRATE': 'CFE',
        'NGEN_CFE_PARAMS_TO_CALIBRATE': 'soil_depth,maxsmc'
    }
    
    manager = NgenParameterManager(config, logger, ngen_dir)
    
    # Update parameters
    params = {'CFE.soil_depth': 5.0, 'CFE.maxsmc': 0.45}
    success = manager.update_model_files(params)
    
    assert success
    content = cfe_file.read_text()
    assert "soil_params.depth=5[m]" in content or "soil_params.depth=5.0[m]" in content
    assert "soil_params.smcmax=0.45[m/m]" in content

def test_noah_tbl_updates(tmp_path):
    """Test that NOAH parameters are updated in TBL files."""
    logger = logging.getLogger('test')
    
    # Setup mock ngen structure
    ngen_dir = tmp_path / "NGEN"
    noah_dir = ngen_dir / "NOAH"
    params_dir = noah_dir / "parameters"
    params_dir.mkdir(parents=True)
    
    # Create mock TBL files
    genparm = params_dir / "GENPARM.TBL"
    genparm.write_text("General Parameters\nREFKDT_DATA\n3.0\n")
    
    soilparm = params_dir / "SOILPARM.TBL"
    soilparm.write_text("Soil Parameters\nSTAS\n19,1 'BB MAXSMC SATDK'\n3, 4.74, 0.434, 5.23E-6, 'SANDY LOAM'\n")
    
    # Create mock cat-1.input to set isltyp
    cat_input = noah_dir / "cat-1.input"
    cat_input.write_text("&structure\n isltyp = 3\n/\n")
    
    config = {
        'DOMAIN_NAME': 'test',
        'EXPERIMENT_ID': 'run_1',
        'NGEN_MODULES_TO_CALIBRATE': 'NOAH',
        'NGEN_NOAH_PARAMS_TO_CALIBRATE': 'refkdt,smcmax,dksat'
    }
    
    manager = NgenParameterManager(config, logger, ngen_dir)
    
    # Update parameters
    params = {
        'NOAH.refkdt': 1.5,
        'NOAH.smcmax': 0.5,
        'NOAH.dksat': 1.0E-5
    }
    success = manager.update_model_files(params)
    
    assert success
    
    # Check GENPARM
    gen_content = genparm.read_text()
    assert "1.5" in gen_content
    
    # Check SOILPARM
    soil_content = soilparm.read_text()
    # Row 3 should be updated. Column indices in our mock are BB=1, MAXSMC=2, SATDK=3 (0-indexed)
    # But manager uses BB=2, MAXSMC=5, SATDK=8 by default for real TBL.
    # In our mock SOILPARM we have 5 parts: "3," "4.74," "0.434," "5.23E-6," "'SANDY LOAM'"
    # If we want to test our specific mappings we should match the real TBL structure.

def test_noah_tbl_real_structure(tmp_path):
    """Test NOAH TBL updates with real-world structure."""
    logger = logging.getLogger('test')
    ngen_dir = tmp_path / "NGEN"
    noah_dir = ngen_dir / "NOAH"
    params_dir = noah_dir / "parameters"
    params_dir.mkdir(parents=True)
    
    soilparm = params_dir / "SOILPARM.TBL"
    # Real structure: BB=2, DRYSMC=3, F11=4, MAXSMC=5, REFSMC=6, SATPSI=7, SATDK=8
    soilparm.write_text("Soil Parameters\nSTAS\n19,1 'BB DRYSMC F11 MAXSMC REFSMC SATPSI SATDK'\n3, 4.74, 0.047, -0.569, 0.434, 0.312, 0.141, 5.23E-6, 'SANDY LOAM'\n")
    
    cat_input = noah_dir / "cat-1.input"
    cat_input.write_text("&structure\n isltyp = 3\n/\n")
    
    manager = NgenParameterManager({'DOMAIN_NAME':'t','NGEN_MODULES_TO_CALIBRATE':'NOAH'}, logger, ngen_dir)
    
    # Update smcmax (col 5) and dksat (col 8)
    manager.update_model_files({'NOAH.smcmax': 0.55, 'NOAH.dksat': 2.0E-5})
    
    content = soilparm.read_text()
    parts = [p.rstrip(',') for p in content.splitlines()[-1].split()]
    assert parts[0] == "3"
    assert float(parts[5]) == 0.55
    assert "2.0000E-05" in parts[8] or "2e-05" in parts[8].lower()
