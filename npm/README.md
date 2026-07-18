# SYMFLUENCE - npm Package

Pre-compiled hydrological modeling tools for SYMFLUENCE framework.

## What's Included

This package provides pre-built binaries for:

- **SUMMA** - Structure for Unifying Multiple Modeling Alternatives
- **mizuRoute** - Multi-scale routing model
- **FUSE** - Framework for Understanding Structural Errors
- **NGEN** - NOAA Next Generation Water Resources Modeling Framework
- **TauDEM** - Terrain Analysis Using Digital Elevation Models

## Installation

### Global Installation (Recommended)

```bash
npm install -g symfluence
```

This will:
1. Download platform-specific pre-compiled binaries (~50-100 MB)
2. Extract them to your global npm directory
3. Make the `symfluence` command available
4. Install the SYMFLUENCE Python package automatically, pinned to the same
   version as the npm package — preferring a pixi-managed environment, and
   falling back to `uv`/`pip3`/`pip` if pixi is unavailable
5. Verify the installed Python CLI version matches the npm package

No separate `pip install symfluence` is needed. Opt-out environment variables:

| Variable | Effect |
| --- | --- |
| `SYMFLUENCE_SKIP_PIXI=1` | Skip the pixi environment, use system pip directly |
| `SYMFLUENCE_SKIP_SYSTEM_DEPS=1` | Skip auto-install of NetCDF/HDF5/GDAL system libraries |
| `SYMFLUENCE_OPTIONAL_PYTHON=1` | Install the binary bundle only (built-in commands only) |

### Local Installation

```bash
npm install symfluence
```

## Supported Platforms

- **Linux**: x86_64 (Ubuntu 22.04+, RHEL 9+, Debian 12+)
- **macOS**: ARM64 (Apple Silicon M1/M2/M3, macOS 12+)

## System Requirements

### Linux

- **OS**: Ubuntu 22.04+, RHEL 9+, or Debian 12+
- **glibc**: ≥ 2.35
- **Libraries** (must be installed):
  ```bash
  sudo apt-get install libnetcdf19 libnetcdff7 libhdf5-103 libgdal32
  ```

### macOS

- **OS**: macOS 12 (Monterey) or later
- **Architecture**: Apple Silicon (ARM64)
- **Libraries** (install via Homebrew):
  ```bash
  brew install netcdf netcdf-fortran hdf5 gdal
  ```

For detailed requirements, see [SYSTEM_REQUIREMENTS.md](https://github.com/symfluence-org/SYMFLUENCE/blob/main/docs/SYSTEM_REQUIREMENTS.md).

## Usage

### Check Installation

```bash
symfluence info
```

This shows:
- Installed version
- Platform information
- Available tools
- Build metadata
- Binary directory path

### Use Tools Directly

#### Option 1: Add to PATH

```bash
# Bash/Zsh
export PATH="$(npm root -g)/symfluence/dist/bin:$PATH"

# Fish
set -x PATH (npm root -g)/symfluence/dist/bin $PATH
```

Then run tools directly:
```bash
summa --version
mizuroute --help
ngen --version
```

#### Option 2: Use Full Path

```bash
$(npm root -g)/symfluence/dist/bin/summa --version
```

#### Option 3: Use with SYMFLUENCE Python Package

The Python package is installed automatically by `npm install` (see above),
and the `symfluence` command forwards all non-built-in commands to it with
the npm-shipped binaries already on PATH. Manual setup is only needed if you
opted out with `SYMFLUENCE_OPTIONAL_PYTHON=1`:

```bash
# Install Python package manually (match the npm package version)
pip install "symfluence==$(symfluence version)"

# Configure to use npm-installed binaries
export SYMFLUENCE_DATA="$(npm root -g)/symfluence/dist"
```

### Get Binary Directory

```bash
symfluence path
```

## Commands

```bash
symfluence info       # Show installation info, available tools, Python CLI version
symfluence version    # Show version
symfluence path       # Show binary directory path
symfluence help       # Show help
```

All other commands (`workflow`, `binary`, ...) are forwarded to the Python
CLI. If the Python package version ever drifts from the npm package (e.g. an
old pip install survived an npm upgrade), every forwarded command prints a
warning with the exact command to re-sync.

## Upgrading

```bash
npm update -g symfluence
```

This re-runs the installer: binaries are replaced with the new release and
the Python package is upgraded to the matching pinned version (the pixi
environment is rebuilt; a pip fallback install is upgraded in place).

## Uninstalling

```bash
npm uninstall -g symfluence
```

This removes the binaries, the tool shims, and the pixi-managed Python
environment (it lives inside the package directory). If the Python package
was installed with system pip/uv (the fallback path), remove it separately —
the uninstaller prints a reminder when it detects one:

```bash
pip uninstall symfluence
```

## Troubleshooting

### Installation Fails

1. **Check platform support**:
   ```bash
   node -e "console.log(process.platform, process.arch)"
   ```
   Must be `linux x64` or `darwin arm64`

2. **Check internet connection**: Downloads from GitHub Releases

3. **Verify release exists**:
   https://github.com/symfluence-org/SYMFLUENCE/releases

4. **Try manual installation**: See repository README

### "libnetcdf.so.19: not found" (Linux)

Install required libraries:
```bash
sudo apt-get install libnetcdf19 libnetcdff7 libhdf5-103
```

### "dyld: Library not loaded" (macOS)

Install required libraries:
```bash
brew install netcdf netcdf-fortran hdf5
```

### "version `GLIBC_2.35' not found" (Linux)

Your system has an older glibc. Options:
- Upgrade to Ubuntu 22.04+ / RHEL 9+ / Debian 12+
- Build from source (see repository docs)
- Use Docker (see repository docs)

## Development

### Local Testing

```bash
# In the npm/ directory
npm install .          # Test installation
node install.js        # Test download manually
./bin/symfluence info  # Test CLI
```

### Publishing

```bash
# Update version in package.json to match release tag
npm publish
```

## Documentation

- **Repository**: https://github.com/symfluence-org/SYMFLUENCE
- **System Requirements**: [docs/SYSTEM_REQUIREMENTS.md](https://github.com/symfluence-org/SYMFLUENCE/blob/main/docs/SYSTEM_REQUIREMENTS.md)
- **Dynamic Linking Strategy**: [docs/DYNAMIC_LINKING_STRATEGY.md](https://github.com/symfluence-org/SYMFLUENCE/blob/main/docs/DYNAMIC_LINKING_STRATEGY.md)
- **Issues**: https://github.com/symfluence-org/SYMFLUENCE/issues

## License

GPL-3.0 - See repository for details.

## Contributing

This package provides pre-built binaries only. For contributing to the tools themselves or the Python framework, see the main repository.

## Credits

- **SUMMA**: Martyn Clark and NCAR
- **mizuRoute**: Naoki Mizukami and NCAR
- **FUSE**: Martyn Clark
- **NGEN**: NOAA-OWP
- **TauDEM**: David Tarboton, Utah State University

SYMFLUENCE framework developed by Darri Eythorsson.
