# SYMFLUENCE Dynamic Linking Strategy

**Decision**: Use **dynamic linking** for scientific libraries (NetCDF, HDF5) instead of static linking.

**Status**: ✅ **Implemented**
**Date**: 2025-12-30
**Phase**: Phase 2 - Runtime Assumptions & Safety

---

## Executive Summary

SYMFLUENCE pre-built binaries **dynamically link** against:
- NetCDF / NetCDF-Fortran
- HDF5
- GDAL
- MPI (when available)

This means these libraries must be installed on the user's system. We chose this approach for:
1. **Smaller downloads** (50-100MB vs 200-300MB)
2. **HPC compatibility** (users can use optimized system builds)
3. **Security** (users get library security updates)
4. **Flexibility** (support multiple NetCDF backends like OPeNDAP)

---

## Rationale: Dynamic vs Static

### What We Dynamically Link

| Library | Version Range | Why Dynamic? |
|---------|---------------|--------------|
| **NetCDF-C** | 4.8.0 - 4.9.x | Stable ABI, optional OPeNDAP support |
| **NetCDF-Fortran** | 4.5.0 - 4.6.x | Depends on NetCDF-C, users may customize |
| **HDF5** | 1.10.0 - 1.14.x | Large library (~20MB), forward compatible |
| **GDAL** | 3.0.0+ | Python bindings + many drivers |
| **MPI** | 4.0+ (optional) | HPC-specific, must match cluster |

### What We Statically Link / Bundle

| Component | Approach | Why? |
|-----------|----------|------|
| **SUMMA** | Bundled binary | Core model, version-locked |
| **mizuRoute** | Bundled binary | Routing model, version-locked |
| **FUSE** | Bundled binary | Structural model, version-locked |
| **NGEN** | Bundled binary | NextGen framework, version-locked |
| **TauDEM** | Bundled binary | Terrain analysis, version-locked |
| **SUNDIALS** | Static in SUMMA | Math library, ABI unstable |
| **Boost** (NGEN) | Static | C++ library, large, version-sensitive |

---

## Trade-off Analysis

### Dynamic Linking

**Advantages** ✅:
1. **Smaller artifacts** (~50-100MB vs 200-300MB)
2. **System library updates** (security patches automatically applied)
3. **HPC flexibility** (users can link against optimized builds)
4. **Multiple backends** (e.g., NetCDF with OPeNDAP, HDF5 with parallel I/O)
5. **Lower memory footprint** (shared libraries in memory once)

**Disadvantages** ❌:
1. **Dependency requirement** (users must install NetCDF/HDF5)
2. **Version compatibility** (must document minimum versions)
3. **Installation complexity** (one more step for users)
4. **Potential breakage** (library updates could introduce incompatibilities)

### Static Linking (Alternative Not Chosen)

**Advantages** ✅:
1. **Zero dependencies** (fully self-contained)
2. **Guaranteed compatibility** (exact versions used in build)
3. **Simpler installation** (just extract and run)

**Disadvantages** ❌:
1. **Large downloads** (200-300MB per platform)
2. **No library updates** (security fixes require full recompile)
3. **HPC inflexibility** (can't use system-optimized builds)
4. **Single configuration** (NetCDF with/without OPeNDAP compiled in)
5. **License complexity** (must bundle all library licenses)

---

## Decision Matrix

| Criterion | Dynamic Linking | Static Linking | Winner |
|-----------|----------------|----------------|--------|
| **Download size** | 50-100MB | 200-300MB | ✅ Dynamic |
| **Installation simplicity** | Requires deps | Extract & run | Static |
| **Security updates** | System updates | Recompile needed | ✅ Dynamic |
| **HPC compatibility** | Users choose libs | Locked-in | ✅ Dynamic |
| **Version flexibility** | Supports range | Single version | ✅ Dynamic |
| **Distribution complexity** | Medium | Low | Static |
| **Scientific reproducibility** | Document versions | Exact versions | Static |

**Overall Winner**: **Dynamic Linking** (5/7 criteria)

---

## Implementation Details

### Build Configuration

Our binaries are built with:

```cmake
# SUMMA CMake configuration
cmake -S build -B cmake_build \
  -DUSE_SUNDIALS=ON \                    # Static: SUNDIALS
  -DCMAKE_BUILD_TYPE=Release \
  -DNETCDF_PATH="${NETCDF:-/usr}" \      # Dynamic: NetCDF
  -DNETCDF_FORTRAN_PATH="${NETCDF_FORTRAN:-/usr}" \  # Dynamic
  -DCMAKE_Fortran_FLAGS="-ffree-form -ffree-line-length-none"
```

**Result**:
- SUNDIALS: Statically linked into SUMMA binary
- NetCDF/HDF5: Dynamically linked from system

### RPATH Configuration

To ensure relocatability while still using dynamic libraries:

**Linux**:
```bash
# Binaries use $ORIGIN for relative library paths
readelf -d bin/summa | grep RPATH
# Should show: $ORIGIN/../lib (if we bundle any libs in future)
```

**macOS**:
```bash
# Binaries use @executable_path for relative paths
otool -l bin/summa | grep -A2 LC_RPATH
# Should show: @executable_path/../lib (if we bundle any libs in future)
```

**Current Status**: Binaries have **no RPATH** and rely on system library paths:
- Linux: `/usr/lib`, `/usr/lib/x86_64-linux-gnu`
- macOS: `/opt/homebrew/lib`, `/usr/local/lib`

---

## Version Compatibility Strategy

### Minimum Versions

We document **minimum versions** required (tested during build):

```json
// From toolchain.json
{
  "libraries": {
    "netcdf": "4.9.0",           // Build version
    "netcdf_fortran": "4.6.1",   // Build version
    "hdf5": "1.12.2"             // Build version
  }
}
```

**User Requirement**: System must have **≥ these versions**.

### Testing Matrix

We test compatibility with:

| Library | Min Tested | Max Tested | Notes |
|---------|-----------|------------|-------|
| NetCDF | 4.8.0 (Ubuntu 22.04) | 4.9.2 (Homebrew) | ABI stable across 4.8-4.9 |
| HDF5 | 1.10.0 (Debian 11) | 1.14.3 (Homebrew) | Forward compatible in 1.x |
| GDAL | 3.0.4 (Ubuntu 22.04) | 3.8.1 (Homebrew) | Python bindings may vary |

**Backward Compatibility**: Binaries built with NetCDF 4.9 should work with NetCDF 4.8 (minor version downgrades).

**Forward Compatibility**: Binaries built with HDF5 1.12 should work with HDF5 1.14 (patch version upgrades).

---

## Troubleshooting Guide

### Problem: "libnetcdf.so.19: not found"

**Cause**: NetCDF not installed or wrong version.

**Solution**:
```bash
# Ubuntu
sudo apt-get install libnetcdf19 libnetcdff7

# macOS
brew install netcdf netcdf-fortran

# Verify
ldconfig -p | grep netcdf
```

### Problem: "version `NETCDF_4.9.0' not found"

**Cause**: Your NetCDF is older than build version.

**Solution**:
```bash
# Check your version
nc-config --version

# If 4.8.x, should still work (ABI compatible)
# If 4.7.x or older, upgrade:
sudo apt-get upgrade libnetcdf-dev libnetcdf19

# Or point to newer version
export LD_LIBRARY_PATH=/opt/netcdf-4.9.0/lib:$LD_LIBRARY_PATH
```

### Problem: HPC wants different HDF5 version

**Cause**: Cluster has optimized HDF5 build.

**Solution**:
```bash
# Load cluster modules
module load hdf5/1.14.0-parallel

# Check if compatible
h5cc -showconfig | grep "HDF5 Version"

# If ≥ 1.10.0, should work
# Test
ldd bin/summa | grep hdf5
```

---

## HPC Considerations

### System-Optimized Libraries

HPC centers often provide:
- **Parallel HDF5** (MPI-enabled)
- **NetCDF with PnetCDF** (parallel I/O)
- **Vendor-optimized builds** (Intel, Cray, etc.)

**Recommendation**: Use system libraries on HPC:

```bash
# Load modules
module load netcdf/4.9.0-parallel
module load hdf5/1.14.0-parallel

# Verify SYMFLUENCE picks them up
symfluence doctor
# Should show: Using /apps/netcdf/4.9.0/lib/libnetcdf.so
```

### Using System-Compiled Models

For maximum performance, HPC users should compile models themselves:

```bash
# Use SYMFLUENCE framework but system models
symfluence config --use-system-tools

# Point to cluster builds
export SUMMA_EXE=/scratch/username/summa/bin/summa_parallel
export MIZUROUTE_EXE=/scratch/username/mizuroute/bin/mizuroute_mpi
```

This gives:
- Hardware-specific optimizations (AVX512, GPU)
- Cluster MPI library (Cray MPICH, Intel MPI)
- Maximum scalability

---

## Future Considerations

### Bundling Libraries (Hybrid Approach)

In future versions, we could **bundle some libraries**:

```
symfluence-tools/
├── bin/summa
├── lib/                    # NEW: Bundled libraries
│   ├── libnetcdf.so.19     # Ship specific versions
│   ├── libhdf5.so.103
│   └── ...
└── toolchain.json
```

**Use LD_LIBRARY_PATH override**:
```bash
export LD_LIBRARY_PATH=$SYMFLUENCE_ROOT/lib:$LD_LIBRARY_PATH
./bin/summa  # Uses bundled libraries first, system as fallback
```

**Trade-offs**:
- ✅ More self-contained
- ✅ Known-good library versions
- ❌ Larger download size
- ❌ No benefit from system updates
- ❌ More complex RPATH management

### Container Distribution

For maximum portability:

```dockerfile
# Dockerfile with all dependencies
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    libnetcdf19 libhdf5-103 libgdal32
COPY symfluence-tools /opt/symfluence
ENV PATH=/opt/symfluence/bin:$PATH
```

**Benefits**:
- Guaranteed environment
- No version conflicts
- Cross-platform (Linux, macOS, Windows via Docker)

---

## Verification

Our CI pipeline verifies dynamic linking works:

```bash
# In CI (after building binaries)
./scripts/verify_binary_portability.sh ./release/symfluence-tools/bin

# Checks:
# 1. RPATH doesn't contain build paths
# 2. Libraries found in standard locations
# 3. No missing dependencies
# 4. glibc/macOS version requirements documented
```

**Output Example**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYMFLUENCE Binary Portability Verification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
→ Platform: Linux

━━━ Checking: summa ━━━
→ Checking RPATH...
✓ No RPATH (uses system library paths)
→ Checking library dependencies...
✓ All libraries found in standard locations
    libnetcdf.so.19 => /usr/lib/x86_64-linux-gnu/libnetcdf.so.19
    libnetcdff.so.7 => /usr/lib/x86_64-linux-gnu/libnetcdff.so.7
→ Checking glibc version requirement...
✓ Max glibc: GLIBC_2.35
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Verification Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total checks:   12
✓ Passed:       12
⚠ Warnings:     0
✗ Failed:       0

✓ Portability verification PASSED
```

---

## Summary

**Decision**: **Dynamic Linking** for scientific libraries (NetCDF, HDF5, GDAL, MPI)

**Rationale**:
1. Smaller downloads (better user experience)
2. HPC flexibility (critical for scientific users)
3. Security updates (long-term maintenance)
4. Multiple configurations (OPeNDAP, parallel HDF5)

**Requirements**:
- Users must install system libraries
- Document minimum versions
- Test across version ranges
- Provide clear troubleshooting guide

**Verification**:
- CI checks for portability
- Automated system requirements checker
- Comprehensive documentation

**Escape Hatches**:
- HPC users: Use system-compiled tools
- Docker: Guaranteed environment
- Future: Hybrid (bundle some, link some)

---

*Decision finalized: 2025-12-30*
*Implemented in: Phase 2*
*Next review: When major library versions change (NetCDF 5.0, HDF5 2.0)*
