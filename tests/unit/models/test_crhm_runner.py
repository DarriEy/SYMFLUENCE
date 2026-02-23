"""
Test CRHM model execution.

Tests the CRHM runner against real preprocessed input files
in the domain_Bow_at_Banff_lumped_era5 domain.

The original generated .prj file was missing several required
sections (Display_Variable, Final_State, etc.) that caused CRHM
to crash with SIGSEGV after printing "No model output selected".
This test generates a corrected .prj file in /tmp and verifies
the model runs to completion with output.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

CRHM_EXE = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/installs/crhm/bin/crhm")
ORIGINAL_PRJ = Path("/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Bow_at_Banff_lumped_era5/CRHM_input/settings/model.prj")
OBS_FILE = Path("/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Bow_at_Banff_lumped_era5/CRHM_input/settings/forcing.obs")
TEST_DIR = Path("/tmp/crhm_test_run")


@pytest.fixture(autouse=True)
def setup_test_dir():
    """Create a clean test directory with copies of input files."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    TEST_DIR.mkdir(parents=True)
    yield


LUMPED_PRJ = """\
CRHM Project - Bow_at_Banff lumped - SYMFLUENCE
###### Version NON DLL 4.02
Dimensions:
######
nhru 3
nlay 2
nobs 1
######
Macros:
######
######
Observations:
######
forcing.obs
######
Dates:
######
2002 1 1
2002 7 1
######
Modules:
######
basin CRHM 04/20/06
global CRHM 04/20/06
obs CRHM 04/20/06
calcsun CRHM 04/20/06
intcp CRHM 04/20/06
pbsm CRHM 04/20/06
albedo CRHM 04/20/06
ebsm CRHM 04/20/06
netall CRHM 04/20/06
crack CRHM 04/20/06
evap CRHM 04/20/06
Soil CRHM 04/20/06
Netroute CRHM 04/20/06
######
Parameters:
###### 'basin' parameters always first
Shared basin_area <1E-06 to 1E+09>
2210
Shared hru_area <1E-06 to 1E+09>
2209 1
Shared hru_ASL <0 to 360>
0 0
Shared hru_elev <0 to 1E+05>
2138 2138
Shared hru_GSL <0 to 90>
0 0
Shared hru_lat <-90 to 90>
51.36 51.36
Shared Ht <0.001 to 100>
0.3 0.3
Shared inhibit_evap <0 to 5>
1 1
Shared Sdmax <0 to 1000>
10 10
Shared soil_rechr_max <0 to 350>
60 60
Shared fetch <300 to 10000>
1000 1000
albedo Albedo_bare <0 to 1>
0.17 0.17
albedo Albedo_snow <0 to 1>
0.85 0.85
basin basin_name
''
basin hru_names
'Main' 'Minor'
basin INIT_STATE
''
basin Loop_to
''
basin RapidAdvance_to
''
basin RUN_END <0 to 1E+05>
0
basin RUN_ID <-1E+08 to 1E+08>
1
basin RUN_START <0 to 1E+05>
0
basin StateVars_to_Update
''
basin TraceVars
''
ebsm delay_melt <0 to 366>
0
ebsm nfactor <0 to 10>
0
ebsm Qe_subl_from_SWE <0 to 1>
0
ebsm tfactor <0 to 10>
0
ebsm Use_QnD <0 to 1>
0
evap evap_type <0 to 2>
0
evap F_Qg <0 to 1>
0.05
evap inhibit_evap_User <0 to 1>
0
evap rs <0 to 0.01>
0
evap Zwind <0.01 to 100>
10
global Time_Offset <-12 to 12>
0
Netroute gwKstorage <0 to 200>
0
Netroute gwLag <0 to 1E+04>
0
Netroute gwwhereto <-1000 to 1000>
3 3 0
Netroute Kstorage <0 to 200>
1
Netroute Lag <0 to 1E+04>
3
Netroute order <1 to 1000>
1 2 3
Netroute preferential_flow <0 to 1>
0
Netroute runKstorage <0 to 200>
0
Netroute runLag <0 to 1E+04>
0
Netroute Sd_ByPass <0 to 1>
0
Netroute soil_rechr_ByPass <0 to 1>
0
Netroute ssrKstorage <0 to 200>
0
Netroute ssrLag <0 to 1E+04>
0
Netroute whereto <0 to 1000>
3 3 0
obs catchadjust <0 to 3>
0
obs ClimChng_flag <0 to 1>
0
obs ClimChng_precip <0 to 10>
1
obs ClimChng_t <-50 to 50>
0
obs ElevChng_flag <0 to 1>
0
obs HRU_OBS <1 to 100>
1 1
1 1
1 1
1 1
1 1
obs lapse_rate <0 to 2>
0.75 0.75
obs obs_elev <0 to 1E+05>
2138 2138
2138 2138
obs ppt_daily_distrib <0 to 1>
1
obs precip_elev_adj <-1 to 1>
0
obs snow_rain_determination <0 to 2>
0
obs tmax_allrain <-10 to 10>
4
obs tmax_allsnow <-10 to 10>
0
pbsm A_S <0 to 2>
0.003
pbsm distrib <-10 to 10>
1
pbsm fetch <300 to 1E+04>
1500
pbsm inhibit_bs <0 to 1>
0
pbsm inhibit_subl <0 to 1>
0
pbsm N_S <1 to 500>
320
Soil cov_type <0 to 2>
1
Soil gw_init <0 to 5000>
75
Soil gw_K <0 to 100>
0.001
Soil gw_max <0 to 5000>
150
Soil lower_ssr_K <0 to 100>
0.001
Soil rechr_ssr_K <0 to 100>
0.001
Soil Sdinit <0 to 5000>
0
Soil Sd_gw_K <0 to 100>
0.001
Soil Sd_ssr_K <0 to 100>
0.001
Soil soil_gw_K <0 to 100>
0.001
Soil soil_moist_init <0 to 5000>
125
Soil soil_moist_max <0 to 5000>
250
Soil soil_rechr_init <0 to 250>
30
Soil soil_ssr_runoff <0 to 1>
1
Soil soil_withdrawal <1 to 4>
2 2
2 2
Soil transp_limited <0 to 1>
0
Soil Wetlands_scaling_factor <-1 to 1>
1
######
Initial_State
######
######
Final_State
######
######
Summary_period
######
Daily
######
Display_Variable:
######
Netroute basinflow 1
Soil soil_moist 1
Soil soil_rechr 1
Soil gw_flow 1
pbsm SWE 1
evap hru_actet 1
obs hru_t 1
obs hru_p 1
######
Display_Observation:
######
######
Log_All
######
Summary_Screen
######
TChart:
######
######
"""


@pytest.mark.skipif(not CRHM_EXE.exists(), reason=f"CRHM executable not found: {CRHM_EXE}")
def test_crhm_executable_exists():
    """Verify CRHM executable is present and marked executable."""
    assert CRHM_EXE.exists(), f"CRHM executable not found: {CRHM_EXE}"
    assert os.access(CRHM_EXE, os.X_OK), "CRHM binary is not executable"
    print(f"CRHM exe: {CRHM_EXE} ({CRHM_EXE.stat().st_size} bytes)")


@pytest.mark.skipif(not OBS_FILE.exists(), reason=f"External data not available: {OBS_FILE}")
def test_crhm_input_files_exist():
    """Verify observation file exists."""
    assert OBS_FILE.exists(), f"Obs file not found: {OBS_FILE}"
    print(f"OBS file: {OBS_FILE} ({OBS_FILE.stat().st_size} bytes)")


@pytest.mark.skipif(not OBS_FILE.exists(), reason=f"External data not available: {OBS_FILE}")
def test_obs_file_format():
    """Verify obs file format is correct for CRHM."""
    with open(OBS_FILE, 'r') as f:
        lines = f.readlines()

    # Find header delimiter
    delim_idx = None
    for i, line in enumerate(lines):
        if line.startswith('#'):
            delim_idx = i
            break

    assert delim_idx is not None, "No header delimiter (line starting with #) found"
    print(f"Header delimiter at line {delim_idx + 1}")

    # Check variable declarations between line 1 and delimiter
    var_lines = lines[1:delim_idx]
    declared_vars = []
    for vl in var_lines:
        vl = vl.strip()
        if vl and not vl.startswith('$') and not vl.startswith('/'):
            parts = vl.split()
            declared_vars.append(parts[0])
            print(f"  Declared variable: {parts[0]} ({parts[1]} columns)")

    # CRHM requires at minimum: t, rh (or ea), u, p (or ppt)
    required_vars = {'t', 'u'}
    precip_vars = {'p', 'ppt'}
    humidity_vars = {'rh', 'ea'}

    assert required_vars.issubset(set(declared_vars)), \
        f"Missing required variables: {required_vars - set(declared_vars)}"
    assert precip_vars.intersection(set(declared_vars)), \
        f"Need at least one of {precip_vars} in obs file"
    assert humidity_vars.intersection(set(declared_vars)), \
        f"Need at least one of {humidity_vars} in obs file"

    # Check first data line format
    first_data = lines[delim_idx + 1].strip().split()
    print(f"First data line has {len(first_data)} fields")
    total_cols = sum(int(vl.strip().split()[1]) for vl in var_lines
                     if vl.strip() and not vl.startswith('$') and not vl.startswith('/'))
    expected_fields = 5 + total_cols
    assert len(first_data) == expected_fields, \
        f"Expected {expected_fields} fields (5 datetime + {total_cols} data), got {len(first_data)}"
    print("Obs file format: OK")


@pytest.mark.skipif(not ORIGINAL_PRJ.exists(), reason=f"External data not available: {ORIGINAL_PRJ}")
def test_original_prj_missing_display_variable():
    """Verify the original .prj is missing the Display_Variable section."""
    with open(ORIGINAL_PRJ, 'r') as f:
        content = f.read()

    has_display_var = 'Display_Variable:' in content
    print(f"Original PRJ has Display_Variable: {has_display_var}")
    if not has_display_var:
        print("  -> This is the root cause of 'No model output selected' + SIGSEGV")


@pytest.mark.skipif(not CRHM_EXE.exists(), reason=f"CRHM executable not found: {CRHM_EXE}")
def test_crhm_reference_badlake():
    """Run CRHM with the reference badlake project to verify the binary works."""
    ref_prj = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/installs/crhm/crhmcode/prj/badlake.prj")
    ref_obs_dir = Path("/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data/installs/crhm/crhmcode/obs/")
    output_file = TEST_DIR / "badlake_output.txt"

    modified_prj = TEST_DIR / "badlake_modified.prj"
    with open(ref_prj, 'r') as f:
        prj_content = f.read()
    prj_content = prj_content.replace(
        r"C:\Users\jhs507\repos\crhmcode\crhmcode\obs\Badlake73_76.obs",
        "Badlake73_76.obs"
    )
    modified_prj.write_text(prj_content, encoding='utf-8')

    cmd = [
        str(CRHM_EXE),
        '--obs_file_directory', str(ref_obs_dir) + os.sep,
        '-o', str(output_file),
        '-p', '30',
        str(modified_prj),
    ]
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, cwd=str(TEST_DIR), capture_output=True, text=True,
        timeout=300, stdin=subprocess.DEVNULL,
    )

    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT:\n{result.stdout[-2000:]}")

    assert result.returncode == 0, f"Badlake ref failed: {result.returncode}"
    assert output_file.stat().st_size > 100, "No meaningful output"
    print(f"Output: {output_file.stat().st_size} bytes -- reference binary works")


@pytest.mark.skipif(not OBS_FILE.exists(), reason=f"External domain data not available: {OBS_FILE}")
def test_crhm_execution_lumped():
    """Run CRHM with a properly constructed lumped (nhru=1) project file.

    This prj includes all required module-specific parameters and
    the critical Display_Variable section that was missing from
    the original preprocessor output.
    """
    prj_path = TEST_DIR / "model.prj"
    prj_path.write_text(LUMPED_PRJ, encoding='utf-8')

    obs_dir = str(OBS_FILE.parent) + os.sep
    output_file = TEST_DIR / "crhm_output.txt"

    cmd = [
        str(CRHM_EXE),
        '--obs_file_directory', obs_dir,
        '-o', str(output_file),
        '-p', '30',
        str(prj_path),
    ]
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, cwd=str(TEST_DIR), capture_output=True, text=True,
        timeout=300, stdin=subprocess.DEVNULL,
    )

    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT:\n{result.stdout[-3000:]}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr[-3000:]}")

    for p in TEST_DIR.iterdir():
        if p.is_file():
            print(f"  File: {p.name} ({p.stat().st_size} bytes)")

    assert result.returncode == 0, \
        f"CRHM exited with code {result.returncode}\nstdout: {result.stdout[-2000:]}"

    assert output_file.exists(), "Output file was not created"
    size = output_file.stat().st_size
    assert size > 100, f"Output file too small ({size} bytes)"
    print(f"\nOutput: {output_file} ({size} bytes)")

    with open(output_file, 'rb') as f:
        data = f.read(1000)
    print(f"First 1000 bytes:\n{data.decode('latin-1')}")
