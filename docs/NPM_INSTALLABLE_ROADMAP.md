# SYMFLUENCE npm-Installable Transformation - Assessment & Roadmap

## Executive Summary

**Goal**: Transform SYMFLUENCE's CI artifacts into npm-distributable, relocatable binaries that enable zero-compilation installation via `npm install symfluence`.

**Current State**: ✅ Well-positioned with robust cross-platform CI
- Ubuntu 22.04 and macOS ARM (M1/M2) builds working
- 10 external tools (SUMMA, mizuRoute, FUSE, NGEN, TauDEM, etc.) build successfully
- Comprehensive validation and testing infrastructure
- Tool installation managed via BinaryManager with dependency resolution

**Readiness Assessment**: 🟡 **60% Ready** - Strong foundation, needs packaging layer

---

## Current Infrastructure Analysis

### ✅ What We Have (Strong Foundation)

1. **Cross-Platform CI** (`.github/workflows/cross-platform.yml`, `install-validate.yml`)
   - Linux: Ubuntu 22.04 (x86_64)
   - macOS: ARM64 (M1/M2)
   - Automated building and caching of tools
   - Binary validation tests in place

2. **Structured Tool Definitions** (`external_tools_config.py`)
   - 10 tools with complete build specifications
   - Dependency resolution (e.g., SUMMA requires SUNDIALS)
   - Installation order management
   - Verification criteria for each tool

3. **BinaryManager** (`binary_manager.py`)
   - Handles installation, validation, and execution
   - Understands tool dependencies
   - Can force reinstall or skip existing installations

4. **Consistent Installation Structure**
   ```
   $SYMFLUENCE_DATA/installs/
   ├── sundials/install/sundials/  # SUNDIALS libraries
   ├── summa/bin/summa.exe         # SUMMA executable
   ├── mizuRoute/route/bin/        # mizuRoute routing
   ├── fuse/bin/fuse.exe           # FUSE framework
   ├── ngen/cmake_build/ngen       # NGEN framework
   └── TauDEM/bin/                 # Terrain analysis tools
   ```

5. **Test Data Management**
   - GitHub Releases for example data (v0.6.0)
   - Automated download and extraction in tests
   - Checksum validation (implicitly via downloads)

### 🟡 What Needs Work (Packaging Gap)

1. **No Toolchain Metadata** ❌
   - No `toolchain.json` generation
   - Can't track:
     - Tool commit hashes (SUMMA, mizuRoute versions)
     - Compiler versions (gfortran, gcc)
     - NetCDF/HDF5 library versions
     - MPI presence (or "none")
     - Build timestamp

2. **Non-Relocatable Binaries** ⚠️
   - RPATH leakage possible (not verified)
   - Hard-coded library paths in build
   - No smoke tests for "extract and run elsewhere"

3. **No Standardized Artifacts** ❌
   - CI caches per-tool directories
   - No single tarball per platform
   - No versioned naming scheme
   - No GitHub Release automation

4. **Missing npm Wrapper** ❌
   - No `npm/` subdirectory or separate repo
   - No `package.json` defining:
     - Platform/architecture constraints
     - Binary download logic
     - Checksum verification

5. **Incomplete Portability Testing** ⚠️
   - Binary validation tests existence, not relocation
   - No glibc version documentation (Linux)
   - No "run on fresh machine" CI test

---

## Phase-by-Phase Implementation Plan

### Phase 1: CI Artifacts Become "Products" 🎯 **Start Here**

**Goal**: Make CI outputs installable, relocatable, versioned tarballs.

#### 1.1 Generate `toolchain.json` During Build

**Implementation**: Add to CI workflows after build step

```yaml
# .github/workflows/install-validate.yml (Linux)
- name: Generate toolchain metadata
  run: |
    set -e

    # Get tool versions
    SUMMA_COMMIT=$(cd ${{ env.SYMFLUENCE_DATA }}/installs/summa && git rev-parse HEAD 2>/dev/null || echo "unknown")
    MIZU_COMMIT=$(cd ${{ env.SYMFLUENCE_DATA }}/installs/mizuRoute && git rev-parse HEAD 2>/dev/null || echo "unknown")
    FUSE_COMMIT=$(cd ${{ env.SYMFLUENCE_DATA }}/installs/fuse && git rev-parse HEAD 2>/dev/null || echo "unknown")
    NGEN_COMMIT=$(cd ${{ env.SYMFLUENCE_DATA }}/installs/ngen && git rev-parse HEAD 2>/dev/null || echo "unknown")

    # Get compiler versions
    FC_VERSION=$(gfortran --version | head -1 || echo "unknown")
    CC_VERSION=$(gcc --version | head -1 || echo "unknown")

    # Get library versions
    NETCDF_VERSION=$(nc-config --version 2>/dev/null || echo "unknown")
    HDF5_VERSION=$(h5cc -showconfig 2>/dev/null | grep "HDF5 Version" || echo "unknown")

    # MPI detection
    if command -v mpirun >/dev/null 2>&1; then
      MPI_VERSION=$(mpirun --version 2>&1 | head -1 || echo "OpenMPI unknown")
    else
      MPI_VERSION="none"
    fi

    # Build timestamp
    BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Generate JSON
    cat > ${{ env.SYMFLUENCE_DATA }}/installs/toolchain.json <<EOF
    {
      "symfluence_version": "${{ github.ref_name }}",
      "build_date": "$BUILD_DATE",
      "platform": "linux-x86_64",
      "tools": {
        "summa": {
          "commit": "$SUMMA_COMMIT",
          "branch": "develop_sundials",
          "executable": "bin/summa.exe"
        },
        "mizuroute": {
          "commit": "$MIZU_COMMIT",
          "branch": "serial",
          "executable": "route/bin/mizuRoute.exe"
        },
        "fuse": {
          "commit": "$FUSE_COMMIT",
          "executable": "bin/fuse.exe"
        },
        "ngen": {
          "commit": "$NGEN_COMMIT",
          "branch": "ngiab",
          "executable": "cmake_build/ngen"
        }
      },
      "compilers": {
        "fortran": "$FC_VERSION",
        "c": "$CC_VERSION",
        "mpi": "$MPI_VERSION"
      },
      "libraries": {
        "netcdf": "$NETCDF_VERSION",
        "hdf5": "$HDF5_VERSION"
      }
    }
    EOF

    cat ${{ env.SYMFLUENCE_DATA }}/installs/toolchain.json
```

**Similar for macOS** in `cross-platform.yml`

#### 1.2 Standardize Artifact Structure

**Create staging script** (`scripts/stage_release_artifacts.sh`):

```bash
#!/bin/bash
# Stage release artifacts for npm distribution
set -e

PLATFORM="${1:-linux-x86_64}"  # linux-x86_64 or macos-arm64
INSTALLS_DIR="${2:-$SYMFLUENCE_DATA/installs}"
OUTPUT_DIR="${3:-./release}"

echo "Staging SYMFLUENCE tools for $PLATFORM"

mkdir -p "$OUTPUT_DIR/symfluence-tools"
cd "$OUTPUT_DIR/symfluence-tools"

# Create standard structure
mkdir -p bin share LICENSES

# Stage SUMMA
if [ -f "$INSTALLS_DIR/summa/bin/summa.exe" ]; then
  cp "$INSTALLS_DIR/summa/bin/summa.exe" bin/summa
  cp "$INSTALLS_DIR/summa/LICENSE" LICENSES/LICENSE-SUMMA 2>/dev/null || true
fi

# Stage mizuRoute
if [ -f "$INSTALLS_DIR/mizuRoute/route/bin/mizuRoute.exe" ]; then
  cp "$INSTALLS_DIR/mizuRoute/route/bin/mizuRoute.exe" bin/mizuroute
  cp "$INSTALLS_DIR/mizuRoute/LICENSE" LICENSES/LICENSE-mizuRoute 2>/dev/null || true
fi

# Stage FUSE
if [ -f "$INSTALLS_DIR/fuse/bin/fuse.exe" ]; then
  cp "$INSTALLS_DIR/fuse/bin/fuse.exe" bin/fuse
  cp "$INSTALLS_DIR/fuse/LICENSE" LICENSES/LICENSE-FUSE 2>/dev/null || true
fi

# Stage NGEN
if [ -f "$INSTALLS_DIR/ngen/cmake_build/ngen" ]; then
  cp "$INSTALLS_DIR/ngen/cmake_build/ngen" bin/ngen
  cp "$INSTALLS_DIR/ngen/LICENSE" LICENSES/LICENSE-NGEN 2>/dev/null || true
fi

# Stage TauDEM (multiple binaries)
if [ -d "$INSTALLS_DIR/TauDEM/bin" ]; then
  cp "$INSTALLS_DIR/TauDEM/bin/"* bin/ 2>/dev/null || true
  cp "$INSTALLS_DIR/TauDEM/LICENSE" LICENSES/LICENSE-TauDEM 2>/dev/null || true
fi

# Copy toolchain metadata
cp "$INSTALLS_DIR/toolchain.json" .

# List contents
echo "Staged binaries:"
ls -lh bin/
echo ""
echo "Toolchain metadata:"
cat toolchain.json
```

#### 1.3 Create Tarballs in CI

**Add to workflows after staging**:

```yaml
- name: Stage release artifacts
  run: |
    chmod +x ./scripts/stage_release_artifacts.sh
    ./scripts/stage_release_artifacts.sh \
      linux-x86_64 \
      ${{ env.SYMFLUENCE_DATA }}/installs \
      ./release

- name: Create tarball
  run: |
    cd release
    VERSION="${{ github.ref_name }}"
    tar -czf "symfluence-tools-${VERSION}-linux-x86_64.tar.gz" symfluence-tools/

    # Generate checksum
    sha256sum "symfluence-tools-${VERSION}-linux-x86_64.tar.gz" > \
      "symfluence-tools-${VERSION}-linux-x86_64.tar.gz.sha256"

- name: Upload artifacts
  uses: actions/upload-artifact@v4
  with:
    name: symfluence-tools-linux-x86_64
    path: |
      release/symfluence-tools-*.tar.gz
      release/symfluence-tools-*.sha256
```

#### 1.4 Smoke Test: Extract and Run

**Add verification step**:

```yaml
- name: Test artifact relocatability
  run: |
    set -e

    # Extract to temporary location
    mkdir -p /tmp/symfluence-test
    cd /tmp/symfluence-test
    tar -xzf $GITHUB_WORKSPACE/release/symfluence-tools-*.tar.gz

    # Test binaries run without original install directory
    echo "Testing relocated binaries..."
    ./symfluence-tools/bin/summa --version || echo "SUMMA version check failed"
    ./symfluence-tools/bin/ngen --help || echo "NGEN help check failed"

    echo "Relocatability test complete"
```

#### 1.5 Upload to GitHub Releases

**Create new workflow** (`.github/workflows/release-binaries.yml`):

```yaml
name: Release Binaries

on:
  release:
    types: [created]
  workflow_dispatch:
    inputs:
      tag:
        description: 'Release tag (e.g., v0.7.0)'
        required: true

jobs:
  build-and-release:
    strategy:
      matrix:
        include:
          - os: ubuntu-22.04
            platform: linux-x86_64
          - os: macos-latest
            platform: macos-arm64

    runs-on: ${{ matrix.os }}

    steps:
      # ... (build steps from install-validate.yml) ...

      - name: Upload to Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: ${{ github.event.inputs.tag || github.ref_name }}
          files: |
            release/symfluence-tools-*.tar.gz
            release/symfluence-tools-*.sha256
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**Estimated Effort**: 2-3 days
**Deliverables**:
- ✅ `toolchain.json` generation
- ✅ Standardized artifact structure
- ✅ Platform-specific tarballs
- ✅ Automated GitHub Release uploads
- ✅ Relocatability smoke tests

---

### Phase 2: Runtime Assumptions & Safety 🔒 **Portability**

**Goal**: Ensure binaries work on user machines.

#### 2.1 RPATH Verification

**Add to build scripts** (in `external_tools_config.py`):

```python
# For SUMMA build:
cmake -S build -B cmake_build \
  ... \
  -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib:$ORIGIN/../../sundials/lib' \
  ...
```

**Test in CI**:
```bash
# Check RPATH doesn't leak build paths
readelf -d bin/summa | grep RPATH || echo "No RPATH (good)"
otool -L bin/summa | grep -v /usr/lib || echo "External deps detected"
```

#### 2.2 Document Minimum Requirements

**Create** `docs/SYSTEM_REQUIREMENTS.md`:

```markdown
# SYMFLUENCE Binary System Requirements

## Linux (x86_64)
- **OS**: Ubuntu 22.04+ / RHEL 9+ / Debian 12+
- **glibc**: ≥ 2.35 (check with `ldd --version`)
- **Libraries** (system-provided):
  - libnetcdf (≥ 4.8)
  - libhdf5 (≥ 1.10)
  - MPI: optional (OpenMPI 4.1+ or MPICH 4.0+)

## macOS (ARM64)
- **OS**: macOS 12 (Monterey)+ for M1/M2
- **Libraries** (Homebrew):
  - netcdf, netcdf-fortran
  - hdf5
  - open-mpi (optional)

## Verification
```bash
symfluence --doctor  # Coming in Phase 4
```
```

#### 2.3 Static vs Dynamic Linking Decision

**Recommendation**: **Dynamic linking for scientific libraries**

**Rationale**:
- NetCDF/HDF5 are stable ABIs across minor versions
- Users may need specific NetCDF builds (e.g., with OPeNDAP)
- HPC environments often have optimized builds
- Smaller download size (50-100MB vs 200-300MB)

**Escape hatch**: Document how to swap system libraries (Phase 5)

**Estimated Effort**: 1-2 days
**Deliverables**:
- ✅ RPATH verification in CI
- ✅ System requirements documentation
- ✅ Dynamic linking strategy documented

---

### Phase 3: npm Wrapper Package 📦 **Distribution**

**Goal**: Deliver binaries via npm without any compilation.

#### 3.1 Create npm Package Structure

```
npm/
├── package.json
├── install.js
├── bin/
│   └── symfluence      # Node.js wrapper script
├── lib/
│   └── platform.js     # Platform detection
└── README.md
```

#### 3.2 `package.json`

```json
{
  "name": "symfluence",
  "version": "0.7.0",
  "description": "Structure for Unifying Multiple Modeling Alternatives - Hydrological modeling framework",
  "bin": {
    "symfluence": "./bin/symfluence"
  },
  "scripts": {
    "postinstall": "node install.js"
  },
  "os": ["darwin", "linux"],
  "cpu": ["x64", "arm64"],
  "engines": {
    "node": ">=14"
  },
  "keywords": ["hydrology", "modeling", "summa", "scientific"],
  "license": "GPL-3.0",
  "repository": {
    "type": "git",
    "url": "https://github.com/DarriEy/SYMFLUENCE.git"
  },
  "homepage": "https://github.com/DarriEy/SYMFLUENCE",
  "preferGlobal": true
}
```

#### 3.3 `install.js` - Platform Detection & Download

```javascript
#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const https = require('https');
const crypto = require('crypto');
const { execSync } = require('child_process');

const PACKAGE_VERSION = require('./package.json').version;
const GITHUB_REPO = 'DarriEy/SYMFLUENCE';

// Platform mapping
const PLATFORMS = {
  'darwin-arm64': 'macos-arm64',
  'darwin-x64': 'macos-x64',
  'linux-x64': 'linux-x86_64',
};

function getPlatform() {
  const platform = process.platform;
  const arch = process.arch;
  const key = `${platform}-${arch}`;

  if (!PLATFORMS[key]) {
    console.error(`❌ Unsupported platform: ${platform} ${arch}`);
    console.error(`Supported platforms: ${Object.keys(PLATFORMS).join(', ')}`);
    process.exit(1);
  }

  return PLATFORMS[key];
}

function getDownloadUrl(platform) {
  const tag = `v${PACKAGE_VERSION}`;
  const filename = `symfluence-tools-${tag}-${platform}.tar.gz`;
  return `https://github.com/${GITHUB_REPO}/releases/download/${tag}/${filename}`;
}

async function downloadFile(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, {
      headers: { 'User-Agent': 'symfluence-npm-installer' }
    }, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Follow redirect
        downloadFile(response.headers.location, dest).then(resolve).catch(reject);
        return;
      }

      if (response.statusCode !== 200) {
        reject(new Error(`Download failed: ${response.statusCode} ${response.statusMessage}`));
        return;
      }

      const totalBytes = parseInt(response.headers['content-length'], 10);
      let downloadedBytes = 0;

      response.on('data', (chunk) => {
        downloadedBytes += chunk.length;
        const percent = ((downloadedBytes / totalBytes) * 100).toFixed(1);
        process.stdout.write(`\r📥 Downloading... ${percent}%`);
      });

      response.pipe(file);

      file.on('finish', () => {
        file.close();
        console.log('\n✅ Download complete');
        resolve();
      });
    }).on('error', (err) => {
      fs.unlink(dest, () => {});
      reject(err);
    });
  });
}

async function verifyChecksum(file, checksumUrl) {
  console.log('🔐 Verifying checksum...');

  try {
    const checksumData = await new Promise((resolve, reject) => {
      let data = '';
      https.get(checksumUrl, (res) => {
        res.on('data', chunk => data += chunk);
        res.on('end', () => resolve(data));
      }).on('error', reject);
    });

    const expectedHash = checksumData.split(' ')[0];

    const fileBuffer = fs.readFileSync(file);
    const hash = crypto.createHash('sha256');
    hash.update(fileBuffer);
    const actualHash = hash.digest('hex');

    if (expectedHash !== actualHash) {
      throw new Error('Checksum mismatch! File may be corrupted.');
    }

    console.log('✅ Checksum verified');
  } catch (err) {
    console.warn('⚠️  Could not verify checksum:', err.message);
    console.warn('   Proceeding anyway...');
  }
}

function extractTarball(tarball, destDir) {
  console.log('📦 Extracting binaries...');

  const extractCmd = process.platform === 'win32'
    ? `tar -xzf "${tarball}" -C "${destDir}"`
    : `tar -xzf "${tarball}" -C "${destDir}" --strip-components=1`;

  try {
    execSync(extractCmd, { stdio: 'inherit' });
    console.log('✅ Extraction complete');
  } catch (err) {
    console.error('❌ Extraction failed:', err.message);
    process.exit(1);
  }
}

async function install() {
  console.log('🚀 Installing SYMFLUENCE binaries...\n');

  const platform = getPlatform();
  console.log(`📍 Platform: ${platform}`);

  const url = getDownloadUrl(platform);
  const checksumUrl = `${url}.sha256`;

  console.log(`🔗 Download URL: ${url}\n`);

  const distDir = path.join(__dirname, 'dist');
  const tarballPath = path.join(__dirname, 'symfluence-tools.tar.gz');

  // Clean dist directory
  if (fs.existsSync(distDir)) {
    fs.rmSync(distDir, { recursive: true });
  }
  fs.mkdirSync(distDir, { recursive: true });

  try {
    // Download
    await downloadFile(url, tarballPath);

    // Verify checksum
    await verifyChecksum(tarballPath, checksumUrl);

    // Extract
    extractTarball(tarballPath, distDir);

    // Cleanup
    fs.unlinkSync(tarballPath);

    console.log('\n🎉 SYMFLUENCE installation complete!');
    console.log('   Run: symfluence --version');

  } catch (err) {
    console.error('\n❌ Installation failed:', err.message);
    console.error('\n📖 Troubleshooting:');
    console.error('   1. Check your internet connection');
    console.error('   2. Verify the release exists: https://github.com/DarriEy/SYMFLUENCE/releases');
    console.error('   3. Try manual installation: https://github.com/DarriEy/SYMFLUENCE#installation');
    process.exit(1);
  }
}

install();
```

#### 3.4 `bin/symfluence` - CLI Wrapper

```javascript
#!/usr/bin/env node
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const distDir = path.join(__dirname, '..', 'dist');
const pythonBin = path.join(distDir, 'bin', 'python');  // If bundled
const symfluenceScript = path.join(distDir, 'run_symfluence.py');

// Check installation
if (!fs.existsSync(distDir)) {
  console.error('❌ SYMFLUENCE binaries not found. Try reinstalling:');
  console.error('   npm install --force');
  process.exit(1);
}

// Add dist/bin to PATH for subprocess
process.env.PATH = `${path.join(distDir, 'bin')}:${process.env.PATH}`;

// Forward all arguments to Python script
const args = process.argv.slice(2);
const proc = spawn('python3', [symfluenceScript, ...args], {
  stdio: 'inherit',
  env: process.env
});

proc.on('exit', (code) => {
  process.exit(code);
});
```

**Estimated Effort**: 2-3 days
**Deliverables**:
- ✅ npm package structure
- ✅ Platform detection and download logic
- ✅ Checksum verification
- ✅ CLI wrapper

---

### Phase 4: CLI Integration 🔧 **User Experience**

#### 4.1 `symfluence doctor` Command

Add to `cli_argument_manager.py`:

```python
def handle_doctor_command(self):
    """Check SYMFLUENCE installation health."""
    print("🔍 SYMFLUENCE System Diagnostics\n")

    # Check binaries
    print("📦 Checking binaries...")
    binaries = {
        'summa': 'installs/summa/bin/summa.exe',
        'mizuroute': 'installs/mizuRoute/route/bin/mizuRoute.exe',
        'fuse': 'installs/fuse/bin/fuse.exe',
        'ngen': 'installs/ngen/cmake_build/ngen',
    }

    for name, rel_path in binaries.items():
        full_path = Path(os.getenv('SYMFLUENCE_DATA')) / rel_path
        if full_path.exists():
            print(f"   ✅ {name}: {full_path}")
        else:
            print(f"   ❌ {name}: Not found")

    # Check toolchain metadata
    toolchain_path = Path(os.getenv('SYMFLUENCE_DATA')) / 'installs' / 'toolchain.json'
    if toolchain_path.exists():
        import json
        with open(toolchain_path) as f:
            toolchain = json.load(f)
        print(f"\n🔧 Toolchain:")
        print(f"   Platform: {toolchain.get('platform', 'unknown')}")
        print(f"   Build date: {toolchain.get('build_date', 'unknown')}")
        print(f"   Fortran: {toolchain['compilers'].get('fortran', 'unknown')}")

    # Check system libraries
    print(f"\n📚 System Libraries:")
    for lib in ['nc-config', 'h5cc', 'mpirun']:
        if shutil.which(lib):
            print(f"   ✅ {lib}: {shutil.which(lib)}")
        else:
            print(f"   ❌ {lib}: Not found")
```

#### 4.2 `symfluence tools info`

```python
def handle_tools_info(self):
    """Display installed tools information."""
    toolchain_path = Path(os.getenv('SYMFLUENCE_DATA')) / 'installs' / 'toolchain.json'

    if not toolchain_path.exists():
        print("❌ No toolchain metadata found. Install tools first.")
        return

    import json
    with open(toolchain_path) as f:
        toolchain = json.load(f)

    print("🔧 SYMFLUENCE Installed Tools\n")
    print(f"Platform: {toolchain.get('platform')}")
    print(f"Build Date: {toolchain.get('build_date')}\n")

    for tool_name, tool_info in toolchain.get('tools', {}).items():
        print(f"{tool_name.upper()}:")
        print(f"  Commit: {tool_info.get('commit', 'unknown')[:8]}")
        print(f"  Branch: {tool_info.get('branch', 'N/A')}")
        print(f"  Executable: {tool_info.get('executable')}")
```

**Estimated Effort**: 1 day
**Deliverables**:
- ✅ `--doctor` command
- ✅ `tools info` command
- ✅ Clear error messages

---

### Phase 5: MPI Escape Hatch 🚪 **HPC Flexibility**

#### 5.1 Detect MPI at Runtime

```python
# In BinaryManager or initialization
def detect_mpi():
    """Check if MPI is available and compatible."""
    if shutil.which('mpirun'):
        try:
            result = subprocess.run(['mpirun', '--version'],
                                  capture_output=True, text=True)
            if 'Open MPI' in result.stdout:
                return 'openmpi'
            elif 'MPICH' in result.stdout:
                return 'mpich'
            return 'unknown'
        except:
            pass
    return None
```

#### 5.2 Config Option for System Tools

**Add to `config_template.yaml`**:

```yaml
# Tool installation preferences
TOOLS_PREFERENCE:
  summa: bundled      # bundled, system, or /path/to/summa.exe
  mizuroute: bundled
  ngen: system        # Use system-installed NGEN with MPI
```

#### 5.3 `symfluence use system-tools`

```bash
#!/bin/bash
# Script to configure SYMFLUENCE to use system tools

cat > ~/.symfluence/tools_config.yaml <<EOF
tools_preference:
  summa: system  # Will use 'which summa.exe' or 'module load summa'
  mizuroute: system

# Optionally specify paths
tool_paths:
  summa: /opt/summa/bin/summa.exe
  mizuroute: /home/user/mizuRoute/bin/mizuRoute.exe
EOF

echo "✅ Configured to use system tools"
echo "   Edit ~/.symfluence/tools_config.yaml to customize"
```

**Estimated Effort**: 1-2 days
**Deliverables**:
- ✅ MPI detection
- ✅ System tools configuration
- ✅ Documentation for HPC users

---

### Phase 6: Release Workflow 🚀 **Automation**

#### 6.1 Release Checklist

1. **Version Bump**
   ```bash
   # In repo root
   ./scripts/bump_version.sh 0.8.0
   ```
   - Updates `package.json` (npm)
   - Updates `__init__.py` (`__version__`)
   - Updates `toolchain.json` template

2. **Tag Repository**
   ```bash
   git tag -a v0.8.0 -m "Release v0.8.0"
   git push origin v0.8.0
   ```

3. **CI Builds Binaries**
   - Triggered automatically on tag push
   - Builds Linux and macOS artifacts
   - Generates `toolchain.json`
   - Creates tarballs with checksums

4. **CI Publishes GitHub Release**
   - Uploads artifacts to GitHub Release
   - Adds release notes

5. **npm Package Update**
   ```bash
   cd npm/
   npm version 0.8.0
   npm publish
   ```

#### 6.2 Automated Release Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-binaries:
    # ... (from Phase 1) ...

  publish-npm:
    needs: build-binaries
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '18'
          registry-url: 'https://registry.npmjs.org'

      - name: Publish to npm
        working-directory: npm
        run: |
          # Version already bumped in package.json
          npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

**Estimated Effort**: 1-2 days
**Deliverables**:
- ✅ Version bumping script
- ✅ Automated release workflow
- ✅ npm publish integration

---

### Phase 7: Documentation & Trust 📚 **Adoption**

#### 7.1 Updated Installation Guide

```markdown
# Installation

## Quick Start (npm)

```bash
npm install -g symfluence
symfluence --version
```

This installs pre-built binaries for:
- SUMMA (hydrological model)
- mizuRoute (routing)
- FUSE, NGEN, TauDEM

**What's bundled**: Model binaries, tool executables
**What's NOT bundled**: NetCDF, HDF5 (use system libraries)
**MPI behavior**: Uses OpenMPI if available, serial otherwise

## System Requirements

### Linux
- Ubuntu 22.04+ / RHEL 9+ / Debian 12+
- glibc ≥ 2.35
- NetCDF ≥ 4.8, HDF5 ≥ 1.10

### macOS
- macOS 12+ (Monterey) for ARM (M1/M2)
- Homebrew packages: `netcdf netcdf-fortran hdf5`

## HPC Installation

On HPC systems with modules:

```bash
module load netcdf hdf5 openmpi
npm install -g symfluence

# Configure to use system-compiled tools if needed
symfluence use system-tools
```

## Reproducibility

Check tool versions and build info:

```bash
symfluence tools info
```

Output includes commit hashes for SUMMA, mizuRoute, etc.
```

#### 7.2 Reproducibility Documentation

```markdown
# Reproducibility Guide

SYMFLUENCE ensures reproducible modeling through:

## Binary Provenance

Every npm installation includes `toolchain.json`:

```json
{
  "symfluence_version": "v0.8.0",
  "build_date": "2025-01-15T10:30:00Z",
  "platform": "linux-x86_64",
  "tools": {
    "summa": {
      "commit": "abc123ef",
      "branch": "develop_sundials"
    },
    ...
  },
  "compilers": {
    "fortran": "GNU Fortran 11.4.0"
  }
}
```

## Citing SYMFLUENCE

```bibtex
@software{symfluence2025,
  author = {Darri Eythorsson},
  title = {SYMFLUENCE: Structure for Unifying Multiple Modeling Alternatives},
  version = {0.8.0},
  year = {2025},
  url = {https://github.com/DarriEy/SYMFLUENCE},
  doi = {10.5281/zenodo.XXXXXX}
}
```

Include `toolchain.json` in supplementary materials for publications.
```

#### 7.3 Trust & Security

- **Checksums**: All artifacts have SHA256 checksums
- **Reproducible builds**: Same inputs → same outputs
- **License compliance**: All licenses aggregated in `LICENSES/`
- **Transparency**: Full build logs in CI, commit hashes tracked

**Estimated Effort**: 2-3 days
**Deliverables**:
- ✅ Installation documentation
- ✅ Reproducibility guide
- ✅ Citation guidelines

---

### Phase 8: Hardening 🛡️ **Production Ready**

#### 8.1 Enhanced Checksums

```javascript
// In install.js, add integrity check
const KNOWN_CHECKSUMS = {
  'v0.8.0-linux-x86_64': 'abc123...',
  'v0.8.0-macos-arm64': 'def456...',
};

function verifyIntegrity(version, platform, file) {
  const key = `${version}-${platform}`;
  const expected = KNOWN_CHECKSUMS[key];

  if (!expected) {
    console.warn('⚠️  No known checksum for this version/platform');
    return;
  }

  // ... hash verification ...
}
```

#### 8.2 npm Installation E2E Test

```yaml
# .github/workflows/test-npm-install.yml
name: Test npm Installation

on:
  release:
    types: [published]

jobs:
  test-install:
    strategy:
      matrix:
        os: [ubuntu-22.04, macos-latest]

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/setup-node@v4
        with:
          node-version: '18'

      - name: Install from npm
        run: |
          npm install -g symfluence@${{ github.event.release.tag_name }}

      - name: Run toy model
        run: |
          # Create minimal test case
          symfluence setup-test-case --name toy_basin
          cd toy_basin
          symfluence run --duration 1day

          # Verify outputs
          [ -f simulations/summa_output.nc ] || exit 1
```

#### 8.3 Backward Compatibility Test

```python
# tests/test_backward_compat.py
def test_v0_7_0_config_still_works():
    """Ensure v0.8.0 can run v0.7.0 configs."""
    old_config = load_config('tests/fixtures/v0.7.0_config.yaml')

    sym = SYMFLUENCE(old_config)
    result = sym.run_workflow()

    assert result.success
```

**Estimated Effort**: 2-3 days
**Deliverables**:
- ✅ Integrity verification
- ✅ E2E installation tests
- ✅ Backward compatibility tests

---

## Implementation Timeline

### Sprint 1 (Week 1-2): **Foundation**
- ✅ Phase 1: Generate toolchain.json, create tarballs, GitHub Releases
- ✅ Phase 2: RPATH verification, system requirements docs

**Milestone**: Can download relocatable tarball from GitHub Release

### Sprint 2 (Week 3-4): **Distribution**
- ✅ Phase 3: npm package creation
- ✅ Phase 4: CLI improvements (doctor, tools info)

**Milestone**: `npm install symfluence` works on clean machine

### Sprint 3 (Week 5-6): **Polish & HPC**
- ✅ Phase 5: MPI escape hatch
- ✅ Phase 6: Automated release workflow
- ✅ Phase 7: Documentation

**Milestone**: Production-ready npm package with documentation

### Sprint 4 (Week 7-8): **Hardening** (Optional)
- ✅ Phase 8: Enhanced verification, E2E tests

**Milestone**: Enterprise-grade reliability

---

## Success Criteria

### Technical
- ✅ Users can run `npm install -g symfluence` and have working binaries
- ✅ Binaries run on Ubuntu 22.04, 24.04, macOS 12+
- ✅ No hard-coded build paths in artifacts
- ✅ Toolchain metadata tracks all tool versions
- ✅ HPC users can override with system-compiled tools

### Scientific
- ✅ Reproducibility: `toolchain.json` provides full provenance
- ✅ Citability: Clear citation guidelines with version tracking
- ✅ Institutional confidence: Documented build process, license compliance

### Operational
- ✅ Automated releases: Tag → CI builds → npm publish
- ✅ No manual heroics required
- ✅ Version consistency across repo, npm, toolchain

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RPATH leakage breaks relocation | Medium | High | Add CI smoke tests for extraction+run |
| Library version mismatches | Medium | Medium | Document minimum versions, dynamic linking |
| npm package size limits | Low | Medium | Tarballs ~50-100MB (under 500MB limit) |
| HPC users can't use npm | Low | Medium | Provide tarball download + `use system-tools` |
| Breaking changes in NetCDF/HDF5 | Low | High | Pin minimum versions, test matrix |

---

## Open Questions

1. **Python bundling**: Should we bundle Python interpreter or require system Python?
   - **Recommendation**: Require Python 3.11+ (already a dependency)

2. **MPI binary variants**: Provide both MPI and non-MPI builds?
   - **Recommendation**: Ship non-MPI by default, detect MPI at runtime

3. **Windows support**: Include in Phase 1 or defer?
   - **Recommendation**: Defer (WSL2 works, native Windows complex)

4. **npm vs standalone installer**: Support both?
   - **Recommendation**: npm primary, document tarball manual install

---

## Next Steps

**Immediate Actions**:
1. Review this roadmap with team
2. Create GitHub project board with Phase 1 tasks
3. Start with `scripts/generate_toolchain.sh` script
4. Add toolchain generation to Linux CI workflow

**Questions to Resolve**:
- Preferred timeline (aggressive 4 weeks vs comfortable 8 weeks)?
- Who handles npm package publishing (credentials)?
- Should we create `@symfluence/tools` npm org or use personal account?

---

*Generated: 2025-01-XX*
*Version: 1.0*
