# SYMFLUENCE System Requirements

This document specifies the minimum system requirements for running SYMFLUENCE pre-built binaries.

---

## Overview

SYMFLUENCE distributes pre-compiled binaries for:
- **Linux** (x86_64): Ubuntu 22.04+, RHEL 9+, Debian 12+
- **macOS** (ARM64): macOS 12 (Monterey)+ for M1/M2/M3 chips

**Installation Method**: `npm install -g symfluence` (Phase 3) or manual tarball download

---

## Linux Requirements (x86_64)

### Supported Distributions

| Distribution | Minimum Version | Recommended | Notes |
|--------------|----------------|-------------|-------|
| **Ubuntu** | 22.04 LTS | 24.04 LTS | Primary test platform |
| **Debian** | 12 (Bookworm) | 12 | Compatible with Ubuntu 22.04 |
| **RHEL/Rocky/Alma** | 9.0 | 9.3+ | Compatible glibc ≥ 2.34 |
| **Fedora** | 36+ | Latest | Rolling release |

**Why these versions?**
- SYMFLUENCE binaries require **glibc ≥ 2.35**
- Ubuntu 22.04 ships with glibc 2.35
- Older distributions (Ubuntu 20.04, RHEL 8) have glibc 2.31-2.34 and are **not supported**

### System Libraries (Required)

These libraries must be installed on the target system:

#### NetCDF and HDF5
```bash
# Ubuntu/Debian
sudo apt-get install libnetcdf19 libnetcdff7 libhdf5-103

# RHEL/Rocky/Alma
sudo dnf install netcdf hdf5

# Check versions
nc-config --version  # Should be ≥ 4.8.0
h5cc -showconfig | grep "HDF5 Version"  # Should be ≥ 1.10.0
```

**Minimum Versions**:
- **NetCDF**: ≥ 4.8.0
- **NetCDF-Fortran**: ≥ 4.5.0
- **HDF5**: ≥ 1.10.0

#### GDAL (for geospatial processing)
```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal32

# RHEL/Rocky/Alma
sudo dnf install gdal gdal-libs

# Check version
gdal-config --version  # Should be ≥ 3.0
```

**Minimum Version**: GDAL ≥ 3.0

#### MPI (Optional - for parallel execution)
```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin libopenmpi3

# RHEL/Rocky/Alma
sudo dnf install openmpi openmpi-devel

# Check version
mpirun --version  # OpenMPI ≥ 4.0 or MPICH ≥ 3.3
```

**MPI Support**:
- **Optional**: SYMFLUENCE binaries work without MPI (serial mode)
- **Recommended for large domains**: OpenMPI 4.1+ or MPICH 4.0+
- HPC users: Use system-provided MPI modules

### System Tools

```bash
# Basic tools (usually pre-installed)
bash --version    # ≥ 4.0
python3 --version # ≥ 3.11 (for SYMFLUENCE Python package)
tar --version     # GNU tar for extraction
```

### Verification

Check if your system meets requirements:

```bash
# Download and run system check script
wget https://raw.githubusercontent.com/DarriEy/SYMFLUENCE/main/scripts/check_system_requirements.sh
chmod +x check_system_requirements.sh
./check_system_requirements.sh
```

Or manually:

```bash
# Check glibc version
ldd --version | head -1
# Output should be: ldd (Ubuntu GLIBC 2.35-0ubuntu3.x) 2.35

# Check NetCDF
nc-config --version
nf-config --version

# Check HDF5
h5cc -showconfig | grep "HDF5 Version"

# Check GDAL
gdal-config --version
```

---

## macOS Requirements (ARM64)

### Supported Versions

| macOS Version | Codename | Supported | Notes |
|---------------|----------|-----------|-------|
| **macOS 15** | Sequoia | ✅ | Latest |
| **macOS 14** | Sonoma | ✅ | Fully tested |
| **macOS 13** | Ventura | ✅ | Compatible |
| **macOS 12** | Monterey | ✅ | Minimum version |
| **macOS 11** | Big Sur | ❌ | Not tested |

**Architecture**: Apple Silicon (M1/M2/M3) only
- Intel Macs (x86_64) are **not supported** in current builds
- Use Rosetta 2 as workaround (not recommended)

### Homebrew Dependencies

Install required libraries via Homebrew:

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install SYMFLUENCE dependencies
brew install netcdf netcdf-fortran hdf5 gdal udunits

# Optional: MPI for parallel execution
brew install open-mpi

# Verify installations
brew list --versions netcdf netcdf-fortran hdf5 gdal
```

**Minimum Versions** (Homebrew usually provides latest):
- **NetCDF**: ≥ 4.9.0
- **NetCDF-Fortran**: ≥ 4.6.0
- **HDF5**: ≥ 1.12.0
- **GDAL**: ≥ 3.5.0

### Python

```bash
# macOS ships with Python 3.9+, but we recommend:
brew install python@3.11

# Verify
python3.11 --version  # Should be ≥ 3.11.0
```

### Verification

```bash
# Check library installations
brew list netcdf netcdf-fortran hdf5 gdal | grep -E "\.(dylib|a)$"

# Check if libraries are found
nc-config --prefix
nf-config --prefix
h5cc -showconfig | head -5
gdal-config --prefix
```

---

## HPC Cluster Requirements

### Module System

Most HPC systems use environment modules. Load required dependencies:

```bash
# Example module loads (varies by cluster)
module load gcc/11.4.0
module load netcdf/4.9.0
module load hdf5/1.12.2
module load gdal/3.6.0
module load openmpi/4.1.5

# Verify
module list
```

### Pre-built Binaries vs System Tools

**Option 1: Use SYMFLUENCE pre-built binaries** (Default)
```bash
npm install -g symfluence
# Uses bundled SUMMA, mizuRoute, FUSE, NGEN, TauDEM
```

**Option 2: Use system-compiled tools** (Recommended for HPC)
```bash
# Configure SYMFLUENCE to use cluster-optimized builds
symfluence config --use-system-tools

# Point to module-provided binaries
export SUMMA_EXE=/path/to/cluster/summa
export MIZUROUTE_EXE=/path/to/cluster/mizuroute
```

**Why use system tools on HPC?**
- Optimized for specific hardware (e.g., Intel vs AMD)
- Linked against high-performance MPI
- May have vendor optimizations (Intel MKL, BLAS)

---

## Compatibility Matrix

### glibc Version Requirements (Linux)

| SYMFLUENCE Version | Min glibc | Rationale |
|--------------------|-----------|-----------|
| **v0.7.0+** | 2.35 | Built on Ubuntu 22.04 |
| **Future (v0.8.0+)** | 2.34 | Planned RHEL 9 support |

**Check your glibc**:
```bash
ldd --version | head -1
```

If you have glibc < 2.35, you have **two options**:
1. **Upgrade your OS** to Ubuntu 22.04+ / RHEL 9+
2. **Compile from source** (see `docs/BUILDING.md`)

### NetCDF/HDF5 Compatibility

SYMFLUENCE binaries dynamically link against NetCDF/HDF5. Version ranges:

| Library | Min Version | Max Tested | Notes |
|---------|-------------|------------|-------|
| **NetCDF-C** | 4.8.0 | 4.9.2 | Backward compatible |
| **NetCDF-Fortran** | 4.5.0 | 4.6.1 | Must match NetCDF-C major version |
| **HDF5** | 1.10.0 | 1.14.3 | Forward compatible within 1.x |

**Troubleshooting**:
```bash
# If you see "version `NETCDF_4.9.0` not found"
nc-config --version  # Check your installed version

# Solution 1: Install matching version
sudo apt-get install libnetcdf19=4.9.0-1

# Solution 2: Use LD_LIBRARY_PATH to point to compatible version
export LD_LIBRARY_PATH=/opt/netcdf-4.9.0/lib:$LD_LIBRARY_PATH
```

---

## Hardware Requirements

### Minimum

- **CPU**: 2 cores (x86_64 or ARM64)
- **RAM**: 4 GB
- **Disk**: 2 GB free space (for installation + small domains)

### Recommended

- **CPU**: 8+ cores (for parallel model execution)
- **RAM**: 16 GB (for large domains or long simulations)
- **Disk**: 50+ GB (for multi-year simulations, multiple domains)

### Large-Scale Modeling

- **CPU**: 32+ cores (HPC cluster)
- **RAM**: 64+ GB
- **Disk**: 500+ GB (for ensemble runs, calibration)
- **Network**: Low-latency interconnect (Infiniband) for MPI

---

## Verification Scripts

### Automated System Check

SYMFLUENCE provides a system requirements checker:

```bash
# Option 1: After npm install
symfluence doctor

# Option 2: Standalone script
wget https://raw.githubusercontent.com/DarriEy/SYMFLUENCE/main/scripts/check_system_requirements.sh
bash check_system_requirements.sh
```

**Output Example**:
```
SYMFLUENCE System Requirements Check
=====================================
✓ glibc: 2.35 (≥ 2.35 required)
✓ NetCDF: 4.9.0 (≥ 4.8.0 required)
✓ HDF5: 1.12.2 (≥ 1.10.0 required)
✓ GDAL: 3.4.3 (≥ 3.0.0 required)
✓ Python: 3.11.7 (≥ 3.11.0 required)
⚠ MPI: Not found (optional for parallel execution)
=====================================
System is compatible with SYMFLUENCE!
```

---

## Troubleshooting

### Common Issues

#### 1. "version `GLIBC_2.35' not found"

**Cause**: Your system has an older glibc.

**Solution**:
```bash
# Check your version
ldd --version | head -1

# If < 2.35, upgrade OS or compile from source
# Ubuntu:
sudo do-release-upgrade  # Upgrade to 22.04 or later

# Or build from source (see docs/BUILDING.md)
```

#### 2. "libnetcdf.so.19: cannot open shared object file"

**Cause**: NetCDF library not installed.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libnetcdf19 libnetcdff7

# macOS
brew install netcdf netcdf-fortran

# Verify
ldconfig -p | grep netcdf  # Linux
otool -L /path/to/summa | grep netcdf  # macOS
```

#### 3. MPI version mismatch

**Cause**: SYMFLUENCE binary built with different MPI than system.

**Solution**:
```bash
# Option 1: Use serial mode (no MPI)
symfluence run --no-mpi

# Option 2: Use system MPI (HPC)
symfluence config --use-system-mpi

# Option 3: Install compatible MPI
# Ubuntu:
sudo apt-get install openmpi-bin libopenmpi3
```

---

## Docker Alternative

If your system doesn't meet requirements, use Docker:

```bash
# Pull SYMFLUENCE container
docker pull ghcr.io/darrieey/symfluence:latest

# Run interactively
docker run -it -v $(pwd):/workspace ghcr.io/darrieey/symfluence:latest

# Inside container:
symfluence --version
```

**Advantages**:
- Guaranteed compatible environment
- No system dependency issues
- Portable across platforms

**Disadvantages**:
- Slower I/O for large datasets
- Cannot use system MPI (limited to container)

---

## Getting Help

If you encounter compatibility issues:

1. **Check this document** for known solutions
2. **Run system checker**: `symfluence doctor`
3. **Open an issue**: https://github.com/DarriEy/SYMFLUENCE/issues
   - Include output of `symfluence doctor`
   - Include OS and library versions
4. **Build from source**: See `docs/BUILDING.md`

---

## Summary Table

| Requirement | Linux | macOS |
|-------------|-------|-------|
| **OS** | Ubuntu 22.04+, RHEL 9+, Debian 12+ | macOS 12+ (Monterey) |
| **Architecture** | x86_64 | ARM64 (Apple Silicon) |
| **glibc** | ≥ 2.35 | N/A |
| **NetCDF** | ≥ 4.8.0 | ≥ 4.9.0 |
| **HDF5** | ≥ 1.10.0 | ≥ 1.12.0 |
| **GDAL** | ≥ 3.0.0 | ≥ 3.5.0 |
| **Python** | ≥ 3.11 | ≥ 3.11 |
| **MPI** | Optional (OpenMPI 4.1+) | Optional (OpenMPI 4.1+) |

---

*Last Updated: 2025-12-30*
*SYMFLUENCE Version: 0.7.0+*
