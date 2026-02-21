"""
GSFLOW build instructions for SYMFLUENCE.

GSFLOW is a USGS coupled model (PRMS + MODFLOW-NWT). The build compiles
the combined GSFLOW binary from Fortran source using the top-level Makefile
that builds MODFLOW -> PRMS -> GSFLOW in sequence.

The upstream GSFLOW Makefiles have several issues that require patching:
  1. Circular cross-directory file-path prerequisites in .mod rules
  2. Missing prms_it0_vars.mod rule in modflow/Makefile
  3. Missing gwflow_inactive_cell.o in gsflow/Makefile
  4. Incomplete .mod dependency lists for some .o targets

Repository: https://github.com/rniswon/gsflow_v2
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import get_common_build_environment


@BuildInstructionsRegistry.register('gsflow')
def get_gsflow_build_instructions():
    """Get GSFLOW build instructions."""
    common_env = get_common_build_environment()

    return {
        'description': 'USGS GSFLOW coupled PRMS + MODFLOW-NWT model',
        'config_path_key': 'GSFLOW_INSTALL_PATH',
        'config_exe_key': 'GSFLOW_EXE',
        'default_path_suffix': 'installs/gsflow/bin',
        'default_exe': 'gsflow',
        'repository': 'https://github.com/rniswon/gsflow_v2.git',
        'branch': 'master',
        'install_dir': 'gsflow',
        'build_commands': [
            common_env,
            r'''
set -e

echo "=== GSFLOW Build Starting ==="

if ! command -v gfortran >/dev/null 2>&1; then
    echo "ERROR: gfortran not found."
    echo "  macOS: brew install gcc"
    echo "  Ubuntu: sudo apt-get install gfortran"
    echo "  HPC: module load gcc"
    exit 1
fi

echo "gfortran version: $(gfortran --version | head -1)"

GFORT_VER=$(gfortran -dumpversion | cut -d. -f1)
COMPAT_FLAGS=""
if [ "$GFORT_VER" -ge 10 ] 2>/dev/null; then
    COMPAT_FLAGS="-fallow-argument-mismatch"
    echo "Adding -fallow-argument-mismatch for gfortran >= 10"
fi

UNAME_S=$(uname -s)

# Find GSFLOW source directory
SRC_DIR=""
for d in "GSFLOW/src" "gsflow_develop/src" "src"; do
    if [ -d "$d" ] && [ -f "$d/Makefile" ] && [ -f "$d/makelist" ]; then
        SRC_DIR="$d"
        break
    fi
done

if [ -z "$SRC_DIR" ]; then
    echo "ERROR: GSFLOW source directory with Makefile not found"
    find . -name "Makefile" -maxdepth 3 2>/dev/null || true
    exit 1
fi

echo "Found GSFLOW source at: $SRC_DIR"
GSFLOW_INSTALL_ROOT="$(pwd)"
cd "$SRC_DIR"

# === Patch makelist for modern gfortran + macOS ===
echo "Patching makelist..."
cp makelist makelist.bak

python3 -c "
import re
with open('makelist') as f:
    txt = f.read()
uname = '$UNAME_S'
compat = '$COMPAT_FLAGS'
if compat and '-fallow-argument-mismatch' not in txt:
    txt = re.sub(
        r'^(FFLAGS\s*=\s*\\\$\(OPTLEVEL\))',
        r'\1 ' + compat,
        txt, flags=re.MULTILINE
    )
if uname == 'Darwin':
    txt = txt.replace('-Bstatic', '')
with open('makelist', 'w') as f:
    f.write(txt)
print('  makelist patched')
"

# === Patch Makefiles to fix upstream build issues ===
# Write the patcher as a heredoc to avoid bash escaping issues
cat > /tmp/_gsflow_patch.py << 'PYEOF'
import re, sys

def patch_file(path, func):
    with open(path) as f:
        orig = f.read()
    txt = func(orig)
    if txt != orig:
        with open(path, 'w') as f:
            f.write(txt)
        print(f'  {path} patched')

def patch_gsflow_makefile(txt):
    # Fix 1: Remove cross-directory file-path prerequisites from .mod rules
    # e.g. "prms_constants.mod: $(PRMSDIR)/prms_constants.mod" -> "prms_constants.mod:"
    txt = re.sub(r'^(\S+\.mod):\s+\$\(PRMSDIR\)/\S+', r'\1:', txt, flags=re.MULTILINE)
    txt = re.sub(r'^(\S+\.mod):\s+\$\(MODFLOWDIR\)/\S+', r'\1:', txt, flags=re.MULTILINE)

    # Fix 3: Add gwflow_inactive_cell.o to MODOBJS if missing
    if 'gwflow_inactive_cell.o' not in txt:
        txt = txt.replace(
            'gsflow_module.o \\\n\t\tgsflow_prms.o',
            'gsflow_module.o \\\n\t\tgwflow_inactive_cell.o \\\n\t\tgsflow_prms.o'
        )

    # Add compile rule for gwflow_inactive_cell.o if missing
    if 'gwflow_inactive_cell.o:' not in txt:
        rule = (
            'gwflow_inactive_cell.o: gwflow_inactive_cell.f90 prms_constants.mod '
            'prms_module.mod prms_mmfapi.mod prms_read_param_file.mod prms_basin.mod '
            'prms_flowvars.mod prms_cascade.mod prms_soilzone.mod prms_set_time.mod '
            'prms_srunoff.mod prms_water_use.mod prms_utils.mod gsflow_module.o\n'
            '\t$(FC) -c $(FFLAGS) gwflow_inactive_cell.f90\n'
            '\n'
            'prms_gwflow_inactive_cell.mod: gwflow_inactive_cell.o\n'
        )
        txt = txt.replace(
            'gsflow_fortran.o: gsflow_fortran.f90',
            rule + '\ngsflow_fortran.o: gsflow_fortran.f90'
        )

    # Add prms_gwflow_inactive_cell.mod dep to gsflow_prms2mf.o
    if 'prms_gwflow_inactive_cell.mod' not in txt:
        txt = txt.replace(
            'gsflow_prms2mf.o: gsflow_prms2mf.f90',
            'gsflow_prms2mf.o: gsflow_prms2mf.f90 prms_gwflow_inactive_cell.mod'
        )
    return txt

def patch_prms_makefile(txt):
    # Fix 1: Remove $(GSFLOWDIR)/xxx.mod file-path prerequisites
    txt = re.sub(r'^(\S+\.mod):\s+\$\(GSFLOWDIR\)/\S+', r'\1:', txt, flags=re.MULTILINE)

    # Fix 4: Add missing gsfmodflow.mod dependency to nhru_summary.o
    if 'nhru_summary.o:' in txt and 'gsfmodflow.mod' not in txt.split('nhru_summary.o:')[1].split('\n')[0]:
        txt = txt.replace(
            'nhru_summary.o: nhru_summary.f90',
            'nhru_summary.o: nhru_summary.f90 gsfmodflow.mod gwfbasmodule.mod'
        )
        # Also need rules to fetch MODFLOW modules into prms/
        if 'gwfbasmodule.mod:' not in txt:
            txt += '\ngwfbasmodule.mod:\n\t$(CP) $(MODFLOWDIR)/gwfbasmodule.mod .\n'
    return txt

def patch_modflow_makefile(txt):
    # Fix 2: Add missing prms_it0_vars.mod rule
    if 'prms_it0_vars.mod:' not in txt:
        txt = txt.replace(
            'de47_NWT.o: de47_NWT.f',
            'prms_it0_vars.mod:\n'
            '\t$(CD) ../prms;make prms_it0_vars.mod\n'
            '\t$(CP) ../prms/prms_it0_vars.mod .\n'
            '\nde47_NWT.o: de47_NWT.f'
        )
    return txt

patch_file('gsflow/Makefile', patch_gsflow_makefile)
patch_file('prms/Makefile', patch_prms_makefile)
patch_file('modflow/Makefile', patch_modflow_makefile)
PYEOF

echo "Patching Makefiles for upstream build issues..."
python3 /tmp/_gsflow_patch.py
rm -f /tmp/_gsflow_patch.py

# === Build (step by step with .mod file propagation) ===
# The three subdirectories have circular .mod dependencies.
# Build MODFLOW first (recursive make resolves basic cross-deps),
# then copy .mod files between directories before each subsequent step.

mkdir -p lib

echo "Step 1/3: Building MODFLOW-NWT..."
cd modflow && make -j1 FC=gfortran 2>&1 && cd ..

echo "Step 2/3: Building PRMS..."
cp modflow/*.mod prms/ 2>/dev/null || true
cp gsflow/*.mod prms/ 2>/dev/null || true
cd prms && make -j1 FC=gfortran 2>&1 && cd ..

echo "Step 3/3: Building GSFLOW..."
cp modflow/*.mod gsflow/ 2>/dev/null || true
cp prms/*.mod gsflow/ 2>/dev/null || true
cd gsflow && make -j1 FC=gfortran 2>&1 && cd ..

cd "$GSFLOW_INSTALL_ROOT"

# Find the built executable
GSFLOW_EXE=""
for exe_path in "$SRC_DIR/../bin/gsflow" "$SRC_DIR/gsflow/gsflow" \
                "GSFLOW/bin/gsflow" "bin/gsflow"; do
    if [ -f "$exe_path" ]; then
        GSFLOW_EXE="$exe_path"
        break
    fi
done

if [ -z "$GSFLOW_EXE" ]; then
    GSFLOW_EXE=$(find . -name "gsflow" -type f \( -perm +111 -o -perm /111 \) 2>/dev/null \
        | grep -v '\.o$' | grep -v '\.mod$' | head -1)
fi

if [ -z "$GSFLOW_EXE" ]; then
    echo "ERROR: GSFLOW executable not found after build"
    find . -maxdepth 4 -name "gsflow*" -type f 2>/dev/null || true
    exit 1
fi

echo "Build successful! Found: $GSFLOW_EXE"

mkdir -p bin
cp "$GSFLOW_EXE" bin/gsflow
chmod +x bin/gsflow

echo "=== GSFLOW Build Complete ==="
echo "Installed to: bin/gsflow"

if [ -f "bin/gsflow" ]; then
    file bin/gsflow
    echo "Verification: gsflow exists"
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['gfortran'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/gsflow'],
            'check_type': 'exists'
        },
        'order': 23,
        'optional': True,
    }
