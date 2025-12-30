#!/usr/bin/env python3
"""
SYMFLUENCE Development Wrapper

This script allows running SYMFLUENCE from the source tree without installation.
It delegates all execution to the actual package entry point in src/symfluence/cli.py.
"""
import sys
from pathlib import Path

# Check if package is already installed
package_installed = False
try:
    import symfluence
    package_installed = True
except ImportError:
    pass

# Only add src to path if package is not installed
if not package_installed:
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))

try:
    from symfluence.cli import main
except ImportError as e:
    print(f"Error importing symfluence: {e}")
    print(f"PYTHONPATH: {sys.path}")
    sys.exit(1)

if __name__ == "__main__":
    main()