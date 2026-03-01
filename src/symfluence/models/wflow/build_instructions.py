# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Wflow build instructions for SYMFLUENCE."""
from symfluence.cli.services import BuildInstructionsRegistry


@BuildInstructionsRegistry.register('wflow')
def get_wflow_build_instructions():
    """Get Wflow install instructions.

    Wflow.jl does not distribute pre-compiled binaries.  This build script:
    1. Checks for a Julia installation.
    2. Creates a Julia project that depends on Wflow.jl.
    3. Builds a wrapper script (``bin/wflow_cli``) that invokes the Julia
       entry-point, so the rest of SYMFLUENCE can call it like any other
       external binary.
    """
    return {
        'description': 'Wflow Distributed Hydrological Model (Deltares)',
        'config_path_key': 'WFLOW_INSTALL_PATH',
        'config_exe_key': 'WFLOW_EXE',
        'default_path_suffix': 'installs/wflow/bin',
        'default_exe': 'wflow_cli',
        'repository': None,  # We manage the Julia project ourselves
        'branch': None,
        'install_dir': 'wflow',
        'build_commands': [
            r"""
# Wflow Install Script for SYMFLUENCE
set -e
echo "=== Wflow Installation Starting ==="

INSTALL_DIR="$(pwd)"

# --- Julia check ---
if ! command -v julia >/dev/null 2>&1; then
    echo "ERROR: Julia is required to install Wflow."
    echo "Install Julia from https://julialang.org/downloads/"
    exit 1
fi
JULIA_VERSION=$(julia --version 2>&1)
echo "Found: $JULIA_VERSION"

# --- Create Julia project ---
mkdir -p project bin

echo "Installing Wflow.jl package (this may take a few minutes)..."
julia --project=project -e '
    using Pkg
    Pkg.add("Wflow")
    Pkg.precompile()
    println("Wflow.jl installed successfully")
'

# --- Create wrapper script ---
cat > bin/wflow_cli << WRAPEOF
#!/bin/bash
# Wflow CLI wrapper — installed by SYMFLUENCE
# Invokes Wflow.jl via Julia with the project environment
exec julia --project="${INSTALL_DIR}/project" -e '
using Wflow
Wflow.run(ARGS[1])
' -- "\$@"
WRAPEOF
chmod +x bin/wflow_cli

echo "=== Wflow Installation Complete ==="
echo "Installed: bin/wflow_cli"
echo "Julia project: project/"
            """.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/wflow_cli'],
            'check_type': 'exists',
        },
        'order': 20,
        'optional': True,
    }
