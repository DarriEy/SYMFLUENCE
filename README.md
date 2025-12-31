# SYMFLUENCE
**SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration**

[![PyPI version](https://badge.fury.io/py/symfluence.svg)](https://badge.fury.io/py/symfluence)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-symfluence.org-brightgreen)](https://symfluence.readthedocs.io)
[![Build Status](https://img.shields.io/github/actions/workflow/status/DarriEy/SYMFLUENCE/ci.yml?branch=main)](https://github.com/DarriEy/SYMFLUENCE/actions)  

---

## Overview
**SYMFLUENCE** is a computational environmental modeling platform that streamlines the hydrological modeling workflow—from domain setup to evaluation. It provides an integrated framework for multi-model comparison, parameter optimization, and automated workflow management across spatial scales.

---

## Quick Links

- **Install:** `npm install -g symfluence` or `pip install symfluence`
- **Documentation:** [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- **Website:** [symfluence.org](https://symfluence.org)
- **Discussions:** [GitHub Discussions](https://github.com/DarriEy/SYMFLUENCE/discussions)
- **Issues:** [GitHub Issues](https://github.com/DarriEy/SYMFLUENCE/issues)

---

## Installation

### Quick Start (Recommended)

**Option 1: npm (Includes pre-built binaries)**
```bash
# Install globally (includes SUMMA, mizuRoute, FUSE, NGEN, TauDEM)
npm install -g symfluence

# Verify installation
symfluence info

# Check system compatibility
symfluence --doctor
```

**Requirements:**
- **Linux**: Ubuntu 22.04+, RHEL 9+, or Debian 12+ (x86_64)
- **macOS**: macOS 12+ (Apple Silicon M1/M2/M3)
- **System libraries**: NetCDF, HDF5 (install via package manager)

**Option 2: Python only**
```bash
# Install Python framework
pip install symfluence

# Install modeling tools separately (if npm not used)
python -m symfluence.cli --get_executables
```

### Development Installation

For development or custom builds:

```bash
# Clone repository
git clone https://github.com/DarriEy/SYMFLUENCE.git
cd SYMFLUENCE

# Use built-in installer
./symfluence --install
```

This creates a clean Python 3.11 virtual environment, installs dependencies, and builds binaries.
For detailed instructions (ARC, FIR, Anvil, custom builds), see [INSTALL.md](INSTALL.md).

### System Requirements

- **npm installation**: See [npm/README.md](npm/README.md) for platform-specific requirements
- **Development**: See [docs/SYSTEM_REQUIREMENTS.md](docs/SYSTEM_REQUIREMENTS.md) for build dependencies

---

## Quick Start

### Basic CLI Usage
```bash
# Show options
./symfluence --help

# Run default workflow
./symfluence

# Run specific steps
./symfluence --setup_project --calibrate_model

# Define domain from pour point
./symfluence --pour_point 51.1722/-115.5717 --domain_def delineate

# Preview workflow
./symfluence --dry_run
```

### First Project
```bash
cp 0_config_files/config_template.yaml my_project.yaml
./symfluence --config my_project.yaml --setup_project
./symfluence --config my_project.yaml
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

See [`0_config_files/config_template.yaml`](0_config_files/config_template.yaml) for a full example.

---

## Project Structure
```
SYMFLUENCE/
├── SYMFLUENCE.py         # Main entry point
├── symfluence            # Shell wrapper
├── utils/                # Core framework modules
│   ├── project/
│   ├── geospatial/
│   ├── models/
│   ├── optimization/
│   └── evaluation/
├── 0_config_files/       # Configuration templates
├── examples/             # Example workflows
├── docs/                 # Documentation
└── installs/             # Auto-generated tool installs
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

---

Happy modelling!  
The SYMFLUENCE Team  