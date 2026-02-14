# System Requirements

This document describes the system requirements for running SYMFLUENCE,
including pre-built binary packages (via npm) and building from source.

> **Canonical machine-readable source:** All system dependencies are defined in
> [`src/symfluence/resources/system_deps.yml`](../src/symfluence/resources/system_deps.yml).
> Run `symfluence binary doctor` to check your environment against this registry.

---

## Python

- **Version**: Python 3.11 or later
- **Package manager**: pip, uv, pipx, or conda

---

## Platform Support

| Platform | Architecture | Pre-built Binaries | Build from Source |
|----------|-------------|-------------------|-------------------|
| Linux (Ubuntu 22.04+, RHEL 9+, Debian 12+) | x86_64 | Yes | Yes |
| macOS 12+ (Monterey) | ARM64 (Apple Silicon M1/M2/M3/M4) | Yes | Yes |
| macOS (Intel) | x86_64 | No | Yes (untested) |
| Linux (ARM64) | aarch64 | No | Possible (untested) |
| Windows | x86_64 | No | No (use WSL2) |

Pre-built binaries are distributed via npm (`npm install -g symfluence`) and
require the runtime libraries listed below.

---

## Runtime Libraries (Pre-built Binaries)

Pre-built binaries are dynamically linked against system libraries. You must
install the following before the binaries will run.

### Linux (apt — Ubuntu 22.04+ / Debian 12+)

```bash
sudo apt-get update
sudo apt-get install -y \
    libnetcdf19 libnetcdff7 libhdf5-103 \
    libgdal32 libproj25 libgeos3.11.1 \
    libopenmpi3
```

**glibc requirement**: >= 2.35 (ships with Ubuntu 22.04, Debian 12, RHEL 9).

### Linux (dnf — RHEL 9+ / Fedora 36+)

```bash
sudo dnf install -y \
    netcdf netcdf-fortran hdf5 \
    gdal-libs proj geos \
    openmpi
```

### macOS (Homebrew)

```bash
brew install netcdf netcdf-fortran hdf5 gdal proj geos open-mpi
```

### conda-forge (any platform)

```bash
conda install -c conda-forge \
    netcdf4 netcdf-fortran hdf5 \
    gdal proj geos \
    openmpi
```

---

## System Libraries for Python Packages

The Python framework (`pip install symfluence`) requires development headers
for several native libraries.

### GDAL (Required)

GDAL is the most common source of installation issues. The Python bindings
must match the system library version exactly.

```bash
# Ubuntu/Debian
sudo apt-get install -y gdal-bin libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# macOS (Homebrew)
brew install gdal

# Install matching Python bindings
pip install gdal==$(gdal-config --version)
```

> **Tip**: Run `gdal-config --version` and `python -c "from osgeo import gdal; print(gdal.VersionInfo())"` — the major.minor versions should match.

### NetCDF / HDF5

```bash
# Ubuntu/Debian
sudo apt-get install -y libnetcdf-dev libhdf5-dev

# RHEL/Fedora
sudo dnf install -y netcdf-devel hdf5-devel

# macOS
brew install netcdf hdf5
```

### PROJ / GEOS

```bash
# Ubuntu/Debian
sudo apt-get install -y libproj-dev libgeos-dev

# RHEL/Fedora
sudo dnf install -y proj-devel geos-devel

# macOS
brew install proj geos
```

### MPI (for parallel calibration)

```bash
# Ubuntu/Debian
sudo apt-get install -y libopenmpi-dev openmpi-bin

# RHEL/Fedora
sudo dnf install -y openmpi-devel

# macOS
brew install open-mpi
```

---

## Build Toolchain (Building from Source)

To compile model binaries from source (`symfluence binary install`), you need:

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| GCC / Clang | GCC 12+ | C/C++ compiler |
| gfortran | 12+ | Fortran compiler (SUMMA, mizuRoute, FUSE, MESH) |
| CMake | 3.20+ | Build system |
| Make | any | Build automation |
| MPI (OpenMPI or MPICH) | OpenMPI 4.x+ | Parallel execution |
| BLAS/LAPACK | any | Linear algebra (OpenBLAS recommended) |

```bash
# Ubuntu/Debian
sudo apt-get install -y build-essential gfortran cmake \
    libopenmpi-dev openmpi-bin libopenblas-dev liblapack-dev \
    libnetcdf-dev libnetcdff-dev libhdf5-dev

# RHEL/Fedora
sudo dnf install -y gcc gcc-c++ gcc-gfortran cmake make \
    openmpi-devel openblas-devel lapack-devel \
    netcdf-devel netcdf-fortran-devel hdf5-devel

# macOS (Homebrew)
brew install cmake gcc open-mpi openblas lapack \
    netcdf netcdf-fortran hdf5
```

---

## HPC Module Recipes

### Compute Canada (FIR / Narval / Cedar)

```bash
module load StdEnv/2023
module load gcc/12.3
module load python/3.11.5
module load gdal/3.9.1
module load r/4.5.0
module load cdo/2.2.2
module load mpi4py/4.0.3
module load netcdf-fortran/4.6.1
module load openblas/0.3.24
```

### ARC (University of Calgary)

```bash
. /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
module unuse $MODULEPATH
module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

module load gcc/14.2.0
module load cmake
module load netcdf-fortran/4.6.1
module load netcdf-c/4.9.2
module load openblas/0.3.27
module load hdf5/1.14.3
module load gdal/3.9.2
module load openmpi/4.1.6
module load python/3.11.7
module load r/4.4.1
```

### Anvil (Purdue RCAC)

```bash
module load r/4.4.1
module load gcc/14.2.0
module load openmpi/4.1.6
module load gdal/3.10.0
module load conda/2024.09
module load openblas/0.3.17
module load netcdf-fortran/4.5.3
module load udunits/2.2.28
```

### Generic HPC

Adapt the following to your module system:

```bash
module load gcc          # or intel
module load openmpi      # or mpich
module load python/3.11
module load netcdf-fortran
module load hdf5
module load gdal
module load cmake
```

Then run:

```bash
./scripts/symfluence-bootstrap --install
```

---

## R (Optional)

R is required only for models that use rpy2 integration (e.g., some statistical post-processing).

```bash
# Ubuntu/Debian
sudo apt-get install -y r-base r-base-dev

# RHEL/Fedora
sudo dnf install -y R R-devel

# macOS
brew install r
```

---

## Diagnostics

Run the built-in diagnostics to verify your environment:

```bash
# Full system check (environment + binaries + libraries)
symfluence doctor

# Binary-only checks
symfluence binary doctor

# Validate installed binaries
symfluence binary validate
```

---

## Troubleshooting

### GDAL version mismatch

```bash
# Check system version
gdal-config --version

# Check Python version
python -c "from osgeo import gdal; print(gdal.VersionInfo())"

# Reinstall matching version
pip install --force-reinstall gdal==$(gdal-config --version)
```

### "version `GLIBC_2.35' not found" (Linux)

Your system has an older glibc. Options:
- Upgrade to Ubuntu 22.04+ / RHEL 9+ / Debian 12+
- Build from source: `symfluence binary install`
- Use Docker or Singularity with a compatible base image

### "dyld: Library not loaded" (macOS)

Install missing libraries via Homebrew:

```bash
brew install netcdf netcdf-fortran hdf5 gdal
```

### "libnetcdf.so.19: not found" (Linux)

```bash
sudo apt-get install libnetcdf19 libnetcdff7 libhdf5-103
```
