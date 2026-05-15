# SYMFLUENCE
**SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration**

[![PyPI version](https://badge.fury.io/py/symfluence.svg)](https://badge.fury.io/py/symfluence)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-symfluence.org-brightgreen)](https://symfluence.org)
[![Build Status](https://img.shields.io/github/actions/workflow/status/symfluence-org/SYMFLUENCE/ci.yml?branch=main)](https://github.com/symfluence-org/SYMFLUENCE/actions)
[![Tests](https://img.shields.io/badge/tests-99%20files-green)](tests/)

---

## Overview
**SYMFLUENCE** is a computational environmental modeling platform that streamlines the hydrological modeling workflow—from domain setup to evaluation. It provides an integrated framework for multi-model comparison, parameter optimization, and automated workflow management across spatial scales.

---

## Quick Links

- **Install:** `pip install symfluence` or `uv pip install symfluence`
- **Documentation:** [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- **Website:** [symfluence.org](https://symfluence.org)
- **Discussions:** [GitHub Discussions](https://github.com/symfluence-org/SYMFLUENCE/discussions)
- **Issues:** [GitHub Issues](https://github.com/symfluence-org/SYMFLUENCE/issues)

---

## Installation

### Quick Start (Recommended)

**Option 1: pip**
```bash
pip install symfluence
```

**Option 2: uv (Fast Python installer)**
```bash
# Into current environment
uv pip install symfluence

# As an isolated CLI tool
uv tool install symfluence
```

**Option 3: pipx (Isolated CLI)**
```bash
pipx install symfluence
```

After installation, install external model binaries:
```bash
symfluence binary install
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/symfluence-org/SYMFLUENCE.git
cd SYMFLUENCE

# Use built-in installer
./scripts/symfluence-bootstrap --install
```

This creates a clean Python 3.11 virtual environment, installs dependencies, and builds binaries.
For detailed instructions (ARC, FIR, Anvil, custom builds), see the [installation guide](https://symfluence.readthedocs.io/en/latest/installation.html).

### npm (Optional — Experimental)

The npm package bundles pre-built binaries (SUMMA, mizuRoute, FUSE, NGEN, TauDEM) for supported platforms:

```bash
npm install -g symfluence

# Verify bundled binaries
symfluence binary info

# Check system compatibility
symfluence binary doctor
```

**Supported platforms:**
- **Linux**: Ubuntu 22.04+, RHEL 9+, or Debian 12+ (x86_64)
- **macOS**: macOS 12+ (Apple Silicon M1/M2/M3)

> **Note:** The npm package is an alternative distribution channel for pre-built binaries.
> The Python package (`pip`/`uv`) is the primary installation method.

### System Requirements

- **Build dependencies**: See the installation guide at https://symfluence.readthedocs.io/en/latest/installation.html
- **npm installation**: See [tools/npm/README.md](tools/npm/README.md) for platform-specific requirements

### System Dependencies (Important)

SYMFLUENCE requires several system-level libraries that must be installed before pip installation:

**GDAL (Required)**

GDAL is a complex geospatial library that requires system-level installation. The Python bindings (`gdal` package) will fail to install if the system library is not present.

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y gdal-bin libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# macOS (Homebrew)
brew install gdal

# Windows (conda recommended)
conda install -c conda-forge gdal

# Verify installation
gdalinfo --version
```

**Other System Libraries**

```bash
# Ubuntu/Debian
sudo apt-get install -y libnetcdf-dev libhdf5-dev libproj-dev libgeos-dev

# macOS (Homebrew)
brew install netcdf hdf5 proj geos

# Windows
# Use conda-forge channel for these dependencies
conda install -c conda-forge netcdf4 hdf5 proj geos
```

**R (Required for rpy2)**

Some hydrological models require R integration via rpy2:

```bash
# Ubuntu/Debian
sudo apt-get install -y r-base r-base-dev

# macOS
brew install r

# Windows
# Download and install from https://cran.r-project.org/
```

**Troubleshooting**

If you encounter GDAL installation issues:
1. Ensure GDAL system library version matches the Python package version
2. On Windows, prefer conda installation over pip for geospatial packages
3. Run `symfluence binary doctor` to diagnose system dependencies

**macOS Apple Silicon (M1/M2/M3):**
```bash
# Recommended: use Homebrew
brew install gdal
pip install gdal==$(gdal-config --version)

# Alternative: use conda-forge
conda install -c conda-forge gdal geopandas rasterio
```

**Windows:**
```bash
# Use conda-forge for all geospatial dependencies
conda create -n symfluence python=3.11
conda activate symfluence
conda install -c conda-forge gdal geopandas rasterio netcdf4 hdf5
pip install symfluence
```

For detailed troubleshooting, see the [installation guide](https://symfluence.readthedocs.io/en/latest/installation.html#troubleshooting)

---

## Containerised installs

The `docker/` directory contains a Dockerfile per documented install method. Each method has two files:

- **`Dockerfile`** — the install method **as described in the docs**, verbatim. No workarounds. Useful for reproducing what a user following the README/installation.html would actually get.
- **`Dockerfile.fixed`** — a copy with the minimum set of workarounds needed for a fully-working image. Each workaround is annotated with the upstream bug it papers over so it can be removed once fixed upstream.

### Method matrix

| Method | Base image | `Dockerfile` (doc-faithful) | `Dockerfile.fixed` |
|---|---|---|---|
| pip | `python:3.11-slim-bookworm` | builds; 5/12 model binaries succeed | **12/12** ✅ |
| uv | `python:3.11-slim-bookworm` | builds; 5/12 model binaries succeed | **12/12** ✅ |
| uv-tool | `python:3.11-slim-bookworm` | builds; 4/12 model binaries succeed | 11/12 (troute upstream-flaky) |
| pipx | `python:3.11-slim-bookworm` | builds; 5/12 model binaries succeed | **12/12** ✅ |
| npm | `node:20-bookworm-slim` | **fails** at `binary info` (npm shim needs a Python CLI install — undocumented) | 21/23 binaries (ngiab + troute not bundled by npm) ✅ |
| conda | `condaforge/miniforge3:24.11.3-2` | **fails** at `pip install symfluence` (no compilers) | 8/12 (conda-vs-system HDF5 ABI conflict) |
| source | `python:3.11-slim-bookworm` | builds; 0/11 binaries (bootstrap doesn't compile binaries) | **12/12** ✅ |

### Build & run

For each method, run from the repo root. Add `--platform=linux/amd64` to npm if you're on Apple Silicon (npm publishes binaries for Linux x86_64 / macOS ARM64 only).

```bash
# pip
docker build -f docker/pip/Dockerfile.fixed -t symfluence:pip-fixed .
docker run --rm symfluence:pip-fixed --help

# uv
docker build -f docker/uv/Dockerfile.fixed -t symfluence:uv-fixed .
docker run --rm symfluence:uv-fixed --help

# uv tool (isolated CLI)
docker build -f docker/uv-tool/Dockerfile.fixed -t symfluence:uv-tool-fixed .
docker run --rm symfluence:uv-tool-fixed --help

# pipx (isolated CLI)
docker build -f docker/pipx/Dockerfile.fixed -t symfluence:pipx-fixed .
docker run --rm symfluence:pipx-fixed --help

# npm (pre-built binaries, no compilation)
docker build --platform=linux/amd64 -f docker/npm/Dockerfile.fixed -t symfluence:npm-fixed .
docker run --rm --platform=linux/amd64 symfluence:npm-fixed --help

# conda (Windows install path / macOS ARM64 GDAL workaround)
docker build -f docker/conda/Dockerfile.fixed -t symfluence:conda-fixed .
docker run --rm symfluence:conda-fixed --help

# source — bootstrap from upstream clone
docker build -f docker/source/Dockerfile.fixed -t symfluence:source-fixed .
docker run --rm symfluence:source-fixed --help

# source — manual editable install of local checkout
docker build --target manual -f docker/source/Dockerfile.fixed -t symfluence:source-manual-fixed .
docker run --rm symfluence:source-manual-fixed --help
```

### Which one should I use?

- **Most users**: `pip-fixed` or `uv-fixed`. Both produce 12/12 working binaries on Linux. uv is faster to install.
- **Need pre-compiled binaries (no host build toolchain)**: `npm-fixed`. Linux x86_64 and macOS ARM64 only.
- **Developing on the project**: `source --target manual` so the venv contains an editable install of your local checkout.
- **Avoid for now**: the unmodified `Dockerfile` files. They are kept as a record of what the docs literally say; they're not meant to be used directly.

### Upstream issues the workarounds paper over

These are the bugs `Dockerfile.fixed` works around. When they are fixed upstream, the corresponding workaround can be removed.

- `_build.sh` host-libc probe omits aarch64 paths (`/lib/aarch64-linux-gnu/libc.so.6`); falsely triggers a static-link "workaround" that breaks `cmake`.
- `_build.sh` static-link fallback passes `-static-libstdc++` as a top-level CMake argument instead of via `CMAKE_EXE_LINKER_FLAGS`.
- `ngen` `_build.sh` doesn't set `Boost_NO_SYSTEM_PATHS=ON` despite specifying `BOOST_ROOT`, so cmake picks up Debian's libboost 1.74 over the 1.79 the script downloads.
- `fuse` Makefile assumes `-L/usr/lib -lhdf5` finds plain-named `libhdf5.so` (Debian multi-arch installs it as `libhdf5_serial.so` under `/usr/lib/<arch>/`).
- `symfluence binary install` exits 0 even with tool failures, masking partial-build problems in CI.
- `symfluence binary info` reports "No toolchain metadata found" even when `binary doctor` reads the same file successfully.
- The npm package is presented in the README as a self-contained install but is actually a wrapper that needs the Python CLI installed separately.
- The `bootstrap` install path documented in installation.html doesn't run `symfluence binary install`; users get a working CLI with no model binaries.
- Pip-installed `h5py` and `netCDF4` wheels bundle different `libhdf5` builds, causing runtime warnings (avoidable only by switching to conda).

---

## Quick Start

### Basic CLI Usage
```bash
# Show options
symfluence --help

# Run full workflow
symfluence workflow run --config my_config.yaml

# Run specific steps
symfluence workflow steps setup_project calibrate_model

# Define domain from pour point
symfluence project pour-point 51.1722/-115.5717 --domain-name MyDomain --definition semidistributed

# Check workflow status
symfluence workflow status

# Validate configuration
symfluence config validate --config my_config.yaml
```

### First Project
```bash
# Initialize project from template
symfluence project init

# Or copy template manually
cp src/symfluence/resources/config_templates/config_template.yaml my_project.yaml

# Run setup
symfluence workflow step setup_project --config my_project.yaml

# Run full workflow
symfluence workflow run --config my_project.yaml
```

---

## Python API
For programmatic control or integration:

```python
from pathlib import Path
from symfluence import SYMFLUENCE

cfg = Path('my_config.yaml')
symfluence = SYMFLUENCE(cfg)
symfluence.run_individual_steps(['setup_project', 'calibrate_model'])
```

---

## Configuration
YAML configuration files define:
- Domain boundaries and discretization
- Model selection and parameters
- Optimization targets
- Output and visualization options

See [`src/symfluence/resources/config_templates/config_template.yaml`](src/symfluence/resources/config_templates/config_template.yaml) for a full example.

---

## Project Structure
```
SYMFLUENCE/
├── src/symfluence/           # Main Python package
│   ├── core/                 # Core system, configuration, mixins
│   ├── cli/                  # Command-line interface
│   ├── project/              # Project and workflow management
│   ├── data/                 # Data acquisition and preprocessing
│   ├── geospatial/           # Domain discretization and geofabric
│   ├── models/               # Model integrations (SUMMA, FUSE, GR4J, etc.)
│   ├── optimization/         # Calibration algorithms (DDS, DE, PSO, NSGA-II)
│   ├── evaluation/           # Performance metrics and evaluation
│   ├── reporting/            # Visualization and plotting
│   └── resources/            # Configuration templates and base settings
├── examples/                 # Progressive tutorial examples
├── docs/                     # Sphinx documentation source
├── scripts/                  # Build and release scripts
├── tools/                    # NPM packaging and utilities
└── tests/                    # Unit, integration, and E2E tests
```

---

## Branching Strategy
- **main**: Stable releases only — every commit is a published version.
- **develop**: Ongoing integration — merges from feature branches and then tested before release.
- Feature branches: `feature/<description>`, PR to `develop`.

---

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code standards and testing
- Branching and pull request process
- Issue reporting

---

## License
Licensed under the GPL-3.0 License.
See [LICENSE](LICENSE) for details.

## Commercial Licensing
SYMFLUENCE is free and open-source software under GPL-3.0-or-later.
For organizations that require alternative licensing terms — including
proprietary integration, redistribution without copyleft obligations,
or operational deployment support — commercial licenses are available.

For commercial licensing, derivative-platform inquiries, and the
Foundation's dual-licensing policy, see [LICENSING.md](LICENSING.md).

Contact: licensing@symfluence.org (licensing) · dev@symfluence.org (general)

---

Happy modelling!
The SYMFLUENCE Team
