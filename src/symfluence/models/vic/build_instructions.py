# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
VIC build instructions for SYMFLUENCE.

This module defines how to build VIC from source, including:
- Repository and branch information
- Build commands (Makefile-based with MPI)
- Installation verification criteria

VIC 5.x uses Makefiles for building. The image driver is built by default,
which supports NetCDF input/output for distributed simulations.
"""

from symfluence.cli.services import (
    BuildInstructionsRegistry,
    get_common_build_environment,
    get_netcdf_detection,
)


@BuildInstructionsRegistry.register('vic')
def get_vic_build_instructions():
    """
    Get VIC build instructions.

    VIC requires NetCDF-C library and MPI. The build uses Makefiles and produces
    the vic_image.exe executable for grid-based simulations.

    Returns:
        Dictionary with complete build configuration for VIC.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()

    return {
        'description': 'Variable Infiltration Capacity Model',
        'config_path_key': 'VIC_INSTALL_PATH',
        'config_exe_key': 'VIC_EXE',
        'default_path_suffix': 'installs/vic/bin',
        'default_exe': 'vic_image.exe',
        'repository': 'https://github.com/UW-Hydro/VIC.git',
        'branch': 'develop',  # Use develop branch for VIC 5.x
        'install_dir': 'vic',
        'build_commands': [
            common_env,
            netcdf_detect,
            r'''
# VIC Build Script for SYMFLUENCE
# Builds VIC 5.x with image driver (NetCDF support)
# VIC uses Makefiles, not CMake

echo "=== VIC Build Starting ==="
echo "Building VIC image driver with NetCDF support"

# Ensure we have NetCDF
if [ -z "$NETCDF_C" ]; then
    echo "ERROR: NetCDF-C not found. Please install netcdf-c."
    exit 1
fi

echo "Using NetCDF from: $NETCDF_C"

# Platform detection
UNAME_S=$(uname -s)
echo "Platform: $UNAME_S"

# Windows/MSYS2: Create pwd.h polyfill (VIC uses getpwuid for logging only)
case "$UNAME_S" in
    MSYS*|MINGW*|CYGWIN*)
        echo "Windows/MSYS2 detected - creating POSIX polyfills..."
        mkdir -p vic/vic_run/include/win_compat
        cat > vic/vic_run/include/win_compat/pwd.h << 'PWDEOF'
#ifndef _WIN_PWD_H_POLYFILL
#define _WIN_PWD_H_POLYFILL
/* Minimal pwd.h polyfill for Windows/MinGW */
#include <stdlib.h>
struct passwd { char *pw_name; };
static inline struct passwd *getpwuid(int uid) {
    (void)uid;
    static struct passwd pw;
    static char name[256];
    char *env = getenv("USERNAME");
    if (env) { strncpy(name, env, 255); name[255] = 0; }
    else { name[0] = '?'; name[1] = 0; }
    pw.pw_name = name;
    return &pw;
}
#endif
PWDEOF
        cat > vic/vic_run/include/win_compat/execinfo.h << 'EXECEOF'
#ifndef _WIN_EXECINFO_H_POLYFILL
#define _WIN_EXECINFO_H_POLYFILL
/* Stub execinfo.h for Windows/MinGW (no backtrace support) */
static inline int backtrace(void **buffer, int size) {
    (void)buffer; (void)size; return 0;
}
static inline char **backtrace_symbols(void *const *buffer, int size) {
    (void)buffer; (void)size; return (char **)0;
}
#endif
EXECEOF
        # Comprehensive POSIX compat header for VIC on Windows
        cat > vic/vic_run/include/win_compat/vic_win_compat.h << 'WINEOF'
#ifndef _VIC_WIN_COMPAT_H
#define _VIC_WIN_COMPAT_H
#ifdef __MINGW32__
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* uid_t type (not in MinGW) */
#ifndef _UID_T_DEFINED
typedef int uid_t;
#define _UID_T_DEFINED
#endif

/* geteuid stub (not in MinGW) */
static inline uid_t geteuid(void) { return 0; }

/* gethostname - use COMPUTERNAME env var */
#define gethostname(name, len) \
    (strncpy((name), getenv("COMPUTERNAME") ? getenv("COMPUTERNAME") : "unknown", (len)-1), \
     (name)[(len)-1] = 0, 0)

/* sysconf and _SC_NPROCESSORS_ONLN (not in MinGW) */
#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 84
#endif
static inline long _vic_sysconf(int name) {
    (void)name;
    return 1; /* fallback: report 1 processor */
}
#define sysconf _vic_sysconf

/* strptime - minimal implementation for VIC date parsing (not in MinGW) */
static inline char *strptime(const char *s, const char *format, struct tm *tm) {
    int y, m, d, H, M, S;
    (void)format;
    /* VIC uses "%Y-%m-%d %H:%M:%S" and "%Y-%m-%d" formats */
    if (sscanf(s, "%d-%d-%d %d:%d:%d", &y, &m, &d, &H, &M, &S) == 6) {
        tm->tm_year = y - 1900; tm->tm_mon = m - 1; tm->tm_mday = d;
        tm->tm_hour = H; tm->tm_min = M; tm->tm_sec = S;
        return (char *)s + strlen(s);
    }
    if (sscanf(s, "%d-%d-%d", &y, &m, &d) == 3) {
        tm->tm_year = y - 1900; tm->tm_mon = m - 1; tm->tm_mday = d;
        tm->tm_hour = 0; tm->tm_min = 0; tm->tm_sec = 0;
        return (char *)s + strlen(s);
    }
    return NULL;
}
#endif /* __MINGW32__ */
#endif
WINEOF
        export VIC_WIN_COMPAT="-I$(pwd)/vic/vic_run/include/win_compat"
        # Patch vic_def.h to include our compat header (included by every VIC source file)
        sed -i '/#include <pwd.h>/a #include "vic_win_compat.h"' vic/vic_run/include/vic_def.h
        export VIC_WIN_COMPAT="-I$(pwd)/vic/vic_run/include/win_compat"
        ;;
esac

# Navigate to the image driver directory
cd vic/drivers/image

if [ ! -f "Makefile" ]; then
    echo "ERROR: Makefile not found in vic/drivers/image"
    exit 1
fi

echo "Building in: $(pwd)"

# Clean any previous build
make clean 2>/dev/null || true

# Backup original Makefile
cp Makefile Makefile.orig

# VIC has global variables defined in headers causing duplicate symbols
# Add -fcommon to allow this (was default behavior before GCC 10)
sed -i.bak 's/CFLAGS  =  ${INCLUDES}/CFLAGS  =  -fcommon ${INCLUDES}/' Makefile

# Windows/MSYS2: inject win_compat include path for pwd.h polyfill
if [ -n "${VIC_WIN_COMPAT:-}" ]; then
    echo "Injecting Windows compat include path into Makefile..."
    # Prepend win_compat path before other includes so pwd.h polyfill is found
    sed -i "s|CFLAGS  =  -fcommon|CFLAGS  = -fcommon ${VIC_WIN_COMPAT}|" Makefile
fi

# Platform-specific configuration
if [ "$UNAME_S" = "Darwin" ]; then
    echo "macOS detected - configuring for clang/OpenMP..."

    # Check for MPI
    if command -v mpicc >/dev/null 2>&1; then
        export MPICC="mpicc"
        echo "Found MPI compiler: $MPICC"
    else
        echo "WARNING: mpicc not found. Install open-mpi: brew install open-mpi"
        export MPICC="clang"
    fi

    # Check for libomp (OpenMP for clang)
    LIBOMP_PATH=""
    if [ -d "/opt/homebrew/opt/libomp" ]; then
        LIBOMP_PATH="/opt/homebrew/opt/libomp"
    elif [ -d "/usr/local/opt/libomp" ]; then
        LIBOMP_PATH="/usr/local/opt/libomp"
    fi

    if [ -n "$LIBOMP_PATH" ]; then
        echo "Found libomp at: $LIBOMP_PATH"
        # Replace -fopenmp with clang-compatible version
        sed -i.bak 's/-fopenmp/-Xpreprocessor -fopenmp/g' Makefile
        # Add libomp library path
        sed -i.bak "s|LIBRARY = -lm \${NC_LIBS}|LIBRARY = -lm -L${LIBOMP_PATH}/lib -lomp \${NC_LIBS}|g" Makefile
        # Add libomp include path to NC_CFLAGS
        export NC_CFLAGS="$(nc-config --cflags 2>/dev/null || echo "-I${NETCDF_C}/include") -I${LIBOMP_PATH}/include"
    else
        echo "WARNING: libomp not found. Install it: brew install libomp"
        echo "Disabling OpenMP..."
        sed -i.bak 's/-fopenmp//g' Makefile
        export NC_CFLAGS="$(nc-config --cflags 2>/dev/null || echo "-I${NETCDF_C}/include")"
    fi

    export NC_LIBS="$(nc-config --libs 2>/dev/null || echo "-L${NETCDF_C}/lib -lnetcdf")"
elif echo "$UNAME_S" | grep -qE "^(MSYS|MINGW|CYGWIN)"; then
    echo "Windows/MSYS2 detected - configuring for MinGW + MS-MPI..."

    # mpicc wrapper for MS-MPI
    if command -v mpicc >/dev/null 2>&1; then
        export MPICC="mpicc"
        echo "Found MPI compiler: $MPICC"
    else
        echo "ERROR: mpicc not found. MS-MPI must be installed."
        exit 1
    fi

    # nc-config is broken on Windows (MSVC paths in bash script), use direct paths
    export NC_LIBS="-L${NETCDF_C}/lib -lnetcdf"
    export NC_CFLAGS="-I${NETCDF_C}/include"

    # Disable OpenMP on MinGW - MS-MPI already provides parallelism
    # and OpenMP with MPI on MinGW can cause issues
    sed -i 's/-fopenmp//g' Makefile
else
    # Linux - use mpicc if available
    if command -v mpicc >/dev/null 2>&1; then
        export MPICC="mpicc"
        echo "Found MPI compiler: $MPICC"
    elif command -v gcc >/dev/null 2>&1; then
        export MPICC="gcc"
        echo "Using gcc (parallel execution disabled)"
    else
        echo "ERROR: No C compiler found (need mpicc or gcc)"
        exit 1
    fi

    export NC_LIBS="$(nc-config --libs 2>/dev/null || echo "-L${NETCDF_C}/lib -lnetcdf")"
    export NC_CFLAGS="$(nc-config --cflags 2>/dev/null || echo "-I${NETCDF_C}/include")"
fi

echo "NC_LIBS: $NC_LIBS"
echo "NC_CFLAGS: $NC_CFLAGS"
echo "MPICC: $MPICC"

# Build VIC
echo "Running make..."
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
make -j${NCORES}

# Check build result
if [ $? -ne 0 ]; then
    echo "VIC build failed"
    exit 1
fi

# Check if executable was created
if [ ! -f "vic_image.exe" ]; then
    echo "ERROR: vic_image.exe not found after build"
    ls -la
    exit 1
fi

echo "Build successful!"

# Create bin directory and install
mkdir -p ../../../bin
cp vic_image.exe ../../../bin/
chmod +x ../../../bin/vic_image.exe

echo "=== VIC Build Complete ==="
echo "Installed to: bin/vic_image.exe"

# Verify installation
if [ -f "../../../bin/vic_image.exe" ]; then
    echo "Verification: vic_image.exe exists"
else
    echo "ERROR: Installation verification failed"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['nc-config', 'mpicc'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/vic_image.exe'],
            'check_type': 'exists'
        },
        'order': 15,  # After core models
        'optional': True,  # Not installed by default with --install
    }
