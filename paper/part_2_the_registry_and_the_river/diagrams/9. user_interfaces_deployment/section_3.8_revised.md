## 3.8 User Interfaces and Deployment

SYMFLUENCE provides three access modalities---a Python API, a command-line interface, and an AI-assisted agent---that share a common configuration and execution architecture (Figure 9). All three modalities operate through `SymfluenceConfig`, a type-validated structure comprising over 346 parameters organized into logical groupings: system settings, domain definition, geospatial data sources, forcing datasets, model-specific parameters, optimization algorithms, and evaluation targets. Factory methods support initialization from YAML files with automatic validation (`from_file()`), predefined templates for common use cases (`from_preset()`), and sensible defaults for rapid prototyping (`from_minimal()`). Environment variable substitution enables credential management without hardcoding sensitive values.

The `WorkflowOrchestrator` coordinates execution of sixteen workflow steps organized into four phases (Figure 9): domain setup (project initialization through domain discretization), data acquisition (observation processing, forcing retrieval, and model-agnostic preprocessing), modeling (model-specific preprocessing, execution, calibration, and emulation), and analysis (benchmarking, decision analysis, sensitivity analysis, and post-processing). Steps can be executed individually, in specified sequences, or as complete workflows, with dependencies automatically resolved and intermediate outputs cached to enable resumption after interruption. Internal managers for domain processing, data acquisition, model execution, and analysis are instantiated on-demand through a `LazyManagerDict` pattern, loading only the components required by the requested workflow.

### 3.8.1 Python API

The programmatic interface centers on the `SYMFLUENCE` class, which serves as the primary entry point. Users instantiate the class with a configuration object and invoke methods corresponding to individual workflow steps or complete workflows. Debug, visualization, and diagnostic flags can be enabled at instantiation for development and troubleshooting.

### 3.8.2 Command-Line Interface

The CLI organizes commands into seven categories: `workflow` (step execution, status, validation, resumption, cleaning), `project` (initialization from presets, pour-point specification), `binary` (compilation and validation of external modeling tools including SUMMA, mizuRoute, FUSE, HYPE, MESH, TauDEM, and others), `config` (configuration management and validation), `job` (SLURM submission and distributed computing coordination), `agent` (AI-assisted interaction), and `docs` (documentation generation). All commands accept common flags for configuration path, debug output, visualization, diagnostic plots, dry-run mode, and I/O profiling. The Rich library provides structured terminal output with progress indicators and formatted error reporting.

### 3.8.3 AI-Assisted Workflows

The `AgentManager` orchestrates AI-assisted workflows through a multi-provider interface supporting OpenAI, Groq, and local Ollama deployments. Provider selection follows an automatic fallback cascade, enabling both cloud-based and fully offline operation. The agent provides access to over 50 registered tools spanning codebase analysis, file operations, workflow management, testing, validation, and GitHub integration, with independent operations executing concurrently (up to 4 parallel workers).

Two interaction paradigms are supported: interactive mode (`agent start`) maintains conversation context across multiple turns for iterative refinement and debugging, while single-prompt mode (`agent run PROMPT`) executes isolated requests suitable for scripted automation and CI/CD integration.

### 3.8.4 Deployment Infrastructure

Automated CI/CD pipelines (Figure 9, L2) ensure code quality and cross-platform compatibility. Every push triggers Ruff linting and MyPy type checking; a matrix testing workflow validates functionality across Ubuntu 22.04, macOS (Apple Silicon), and Windows with Python 3.11. Release workflows compile external modeling tools from source for Linux x86_64 and macOS ARM64 targets. Sphinx documentation builds and deploys to ReadTheDocs automatically.

Code quality is enforced through automated pre-commit hooks (Ruff linting, Bandit security scanning, MyPy type checking, file hygiene, and notebook output stripping) and a comprehensive test suite comprising over 99 test files organized into unit, integration, end-to-end, and live categories. Over 70 pytest markers enable selective execution by speed, data requirements, component, model, and dataset. The contribution workflow follows a fork-and-branch model with `develop` as the integration branch and `main` reserved for stable releases, governed by semantic versioning.

Multiple distribution channels (Figure 9, L1) accommodate different user needs: NPM packages include pre-compiled binaries for immediate use, pip provides the Python package, uv supports external package management, and a development bootstrap script enables source compilation.
