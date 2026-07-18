# SYMFLUENCE
**SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration**

[![PyPI version](https://badge.fury.io/py/symfluence.svg)](https://badge.fury.io/py/symfluence)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-brightgreen)](https://symfluence.readthedocs.io)
[![CI](https://github.com/symfluence-org/SYMFLUENCE/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/symfluence-org/SYMFLUENCE/actions/workflows/ci.yml?query=branch%3Amain)

---

## Overview
**SYMFLUENCE** is a computational environmental modeling platform that streamlines the hydrological modeling workflow—from domain setup to evaluation. It provides an integrated framework for multi-model comparison, parameter optimization, and automated workflow management across spatial scales.

---

## Quick Links

- **Install:** `pip install symfluence`
- **Documentation:** [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- **Website:** [symfluence.org](https://symfluence.org)
- **Discussions:** [GitHub Discussions](https://github.com/symfluence-org/SYMFLUENCE/discussions)
- **Issues:** [GitHub Issues](https://github.com/symfluence-org/SYMFLUENCE/issues)

---

## Installation

```bash
pip install symfluence
```

After installation, install external model binaries:
```bash
symfluence binary install
```

For development setup (editable install / bootstrap) and all other install methods — Docker, conda, npm, uv, HPC modules — see the [installation guide](https://symfluence.readthedocs.io/en/latest/installation.html).

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
symfluence project pour-point 51.1722/-115.5717 --domain-name MyDomain --definition delineate

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
│   ├── tui/                  # Terminal user interface
│   ├── gui/                  # Graphical user interface
│   ├── agent/                # Agentic / assistant integration
│   ├── project/              # Project and workflow management
│   ├── data/                 # Data acquisition and preprocessing
│   ├── geospatial/           # Domain discretization and geofabric
│   ├── models/               # Model integrations (SUMMA, FUSE, GR, etc.)
│   ├── coupling/             # Model coupling
│   ├── optimization/         # Calibration algorithms (DDS, DE, PSO, NSGA-II)
│   ├── evaluation/           # Performance metrics and evaluation
│   ├── fews/                 # Delft-FEWS integration
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
- Feature branches: `feat/<description>`, PR to `develop`.

---

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Code standards and testing
- Branching and pull request process
- Issue reporting

---

## License
SYMFLUENCE is free and open-source software under **GPL-3.0-or-later**.
See [LICENSE](LICENSE) for the full text.

Commercial and dual-licensing options are available for organizations that
need alternative terms (proprietary integration, redistribution without
copyleft obligations, or operational deployment support). See the
[licensing guide](https://symfluence.readthedocs.io/en/latest/licensing.html)
for details, or the full [Licensing Policy](LICENSING.md). Inquiries:
licensing@symfluence.org.

---

Happy modelling!
The SYMFLUENCE Team
