"""
WATFLOOD/CHARM build instructions for SYMFLUENCE.

WATFLOOD (CHARM — Canadian Hydrological And Routing Model) is an open-source
distributed flood forecasting model developed at the University of Waterloo.

The GitHub repository (https://github.com/watflood/model) contains the model
subroutines but requires the area_watflood module from the full WATFLOOD
distribution. If building from the full distribution, place the complete
source in the install directory and re-run.

Tier 1: Build from full WATFLOOD source (if area_watflood module present).
Tier 2: Clone from GitHub and attempt build with dependency resolution.
Tier 3: Manual installation guidance.
"""

from symfluence.cli.services import BuildInstructionsRegistry
from symfluence.cli.services import get_common_build_environment


@BuildInstructionsRegistry.register('watflood')
def get_watflood_build_instructions():
    """Get WATFLOOD/CHARM build instructions."""
    common_env = get_common_build_environment()

    return {
        'description': 'WATFLOOD/CHARM distributed flood forecasting model',
        'config_path_key': 'WATFLOOD_INSTALL_PATH',
        'config_exe_key': 'WATFLOOD_EXE',
        'default_path_suffix': 'installs/watflood/bin',
        'default_exe': 'watflood',
        'repository': 'https://github.com/watflood/model.git',
        'branch': 'master',
        'install_dir': 'watflood',
        'build_commands': [
            common_env,
            r'''
set -e

echo "=== WATFLOOD/CHARM Build Starting ==="

INSTALL_DIR="${INSTALL_DIR:-.}"
mkdir -p "${INSTALL_DIR}/bin"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"

# Check for gfortran
if ! command -v gfortran >/dev/null 2>&1; then
    echo "ERROR: gfortran not found."
    echo "  macOS: brew install gcc"
    echo "  Ubuntu: sudo apt-get install gfortran"
    echo "  HPC: module load gcc"
    exit 1
fi

echo "gfortran version: $(gfortran --version | head -1)"

# gfortran >=10 compatibility
GFORT_VER=$(gfortran -dumpversion | cut -d. -f1)
COMPAT_FLAGS=""
if [ "$GFORT_VER" -ge 10 ] 2>/dev/null; then
    COMPAT_FLAGS="-fallow-argument-mismatch"
fi

# macOS: remove -Bstatic
UNAME_S=$(uname -s)

FFLAGS="-O2 -ffree-line-length-none -fmax-identifier-length=63 $COMPAT_FLAGS"

# === Check if user has a pre-compiled binary ===
if [ -f "watflood" ] && file "watflood" | grep -qi "executable\|Mach-O\|ELF"; then
    echo "Found pre-compiled watflood binary"
    cp watflood "${INSTALL_DIR}/bin/watflood"
    chmod +x "${INSTALL_DIR}/bin/watflood"
    echo "=== WATFLOOD Installation Complete (pre-compiled) ==="
    exit 0
fi

# === Check if user has a Makefile (full distribution) ===
if [ -f "Makefile" ] || [ -f "makefile" ]; then
    echo "Found Makefile from full WATFLOOD distribution"
    NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)

    # Patch for modern gfortran
    for mf in Makefile makefile; do
        if [ -f "$mf" ] && [ -n "$COMPAT_FLAGS" ]; then
            if ! grep -q "fallow-argument-mismatch" "$mf"; then
                sed -i.bak "s/FFLAGS\s*=/FFLAGS = $COMPAT_FLAGS /" "$mf" 2>/dev/null || \
                sed -i '' "s/FFLAGS *= */FFLAGS = $COMPAT_FLAGS /" "$mf" 2>/dev/null || true
            fi
        fi
    done

    make -j${NCORES} FC=gfortran 2>&1 || make FC=gfortran 2>&1

    # Find executable
    for exe in watflood charm CHARM; do
        if [ -f "$exe" ] && [ -x "$exe" ]; then
            cp "$exe" "${INSTALL_DIR}/bin/watflood"
            chmod +x "${INSTALL_DIR}/bin/watflood"
            echo "=== WATFLOOD Build Complete (Makefile) ==="
            exit 0
        fi
    done
fi

# === Check if area_watflood module exists (full source) ===
HAS_AREA_MODULE=false
if ls *.mod 2>/dev/null | grep -qi "area_watflood"; then
    HAS_AREA_MODULE=true
fi

# Look for area_watflood source file
for f in area_watflood.f90 area_watflood.f AREA_WATFLOOD.f90 area.f90; do
    if [ -f "$f" ]; then
        HAS_AREA_MODULE=true
        echo "Found area_watflood source: $f"
        echo "Compiling area_watflood module..."
        gfortran -c $FFLAGS -o area_watflood.o "$f" 2>&1
        break
    fi
done

if [ "$HAS_AREA_MODULE" = "true" ]; then
    echo "area_watflood module available — attempting full compilation..."

    OBJ_DIR="obj"
    mkdir -p "$OBJ_DIR"

    # Multi-pass compilation for module dependency resolution
    ALL_SRC=$(find . -maxdepth 1 \( -name "*.f90" -o -name "*.f" -o -name "*.F90" -o -name "*.for" \) \
        ! -name "CHARM.f90" ! -name "charm.f90" | sort)

    COMPILED_OBJS=""
    UNCOMPILED="$ALL_SRC"

    for pass in 1 2 3 4 5; do
        STILL_UNCOMPILED=""
        COUNT=0

        for src in $UNCOMPILED; do
            [ -f "$src" ] || continue
            base=$(basename "$src")
            obj="$OBJ_DIR/${base%.*}.o"

            if [ -f "$obj" ]; then
                continue
            fi

            if gfortran -c $FFLAGS -J"$OBJ_DIR" -I"$OBJ_DIR" -I. -o "$obj" "$src" 2>/dev/null; then
                COMPILED_OBJS="$COMPILED_OBJS $obj"
                COUNT=$((COUNT + 1))
            else
                STILL_UNCOMPILED="$STILL_UNCOMPILED $src"
            fi
        done

        echo "  Pass $pass: compiled $COUNT files"
        UNCOMPILED="$STILL_UNCOMPILED"

        if [ "$COUNT" -eq 0 ]; then
            break
        fi
    done

    # Link
    MAIN_SRC=""
    for f in CHARM.f90 charm.f90; do
        [ -f "$f" ] && MAIN_SRC="$f" && break
    done

    if [ -n "$MAIN_SRC" ] && [ -n "$COMPILED_OBJS" ]; then
        echo "Linking $(echo $COMPILED_OBJS | wc -w) object files with $MAIN_SRC..."
        gfortran $FFLAGS -J"$OBJ_DIR" -I"$OBJ_DIR" -I. -o watflood "$MAIN_SRC" $COMPILED_OBJS 2>&1

        if [ -f "watflood" ]; then
            cp watflood "${INSTALL_DIR}/bin/watflood"
            chmod +x "${INSTALL_DIR}/bin/watflood"
            echo "=== WATFLOOD Build Complete ==="
            exit 0
        fi
    fi
fi

# === Tier 3: Manual installation guidance ===
echo ""
echo "======================================================================"
echo "  WATFLOOD source compilation requires the 'area_watflood' module"
echo "  which is not included in the GitHub repository."
echo ""
echo "  To complete installation:"
echo ""
echo "  Option A: Obtain full source from University of Waterloo"
echo "    Website: https://www.civil.uwaterloo.ca/watflood/"
echo "    Contact: watflood@uwaterloo.ca"
echo "    Place complete source in: ${INSTALL_DIR}/"
echo "    Then re-run: symfluence binary install watflood --force"
echo ""
echo "  Option B: If you have a pre-compiled binary"
echo "    Copy it to: ${INSTALL_DIR}/bin/watflood"
echo "    chmod +x ${INSTALL_DIR}/bin/watflood"
echo ""
echo "  Option C: If you have the full source with area_watflood"
echo "    Place all .f/.f90 files in: ${INSTALL_DIR}/"
echo "    Then re-run: symfluence binary install watflood --force"
echo "======================================================================"

# Create placeholder so the install dir exists for later
mkdir -p "${INSTALL_DIR}/bin"
exit 1
            '''.strip()
        ],
        'dependencies': ['gfortran'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/watflood'],
            'check_type': 'exists'
        },
        'order': 24,
        'optional': True,
    }
